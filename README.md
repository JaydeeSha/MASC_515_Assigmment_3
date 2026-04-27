# microgpt — Extended with GELU, LoRA, RoPE, and Mixture of Experts

**MASC 515 Assignment 3**

This repository contains Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a 200-line, dependency-free Python implementation of a GPT — extended with four modern LLM algorithms.

## Files

| File | Description |
|---|---|
| `microgpt.py` | Original, unmodified microgpt by Karpathy |
| `microgpt_modified.py` | microgpt extended with GELU, LoRA, RoPE, and MoE |
| `README.md` | This documentation |

## How to Run

```bash
# Original
python microgpt.py

# Extended version
python microgpt_modified.py
```

No pip installs needed — pure Python only.

---

## Algorithm 1 — GELU (Gaussian Error Linear Units)

**Paper:** [Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)

### Underlying Idea

The original GPT uses **ReLU** as its activation function: `ReLU(x) = max(0, x)`. ReLU is a hard gate — it passes positive values unchanged and kills negative ones completely. This means that any neuron with a negative pre-activation produces zero output *and* zero gradient, a phenomenon called "dying ReLU."

**GELU** (Gaussian Error Linear Unit) replaces this hard gate with a *soft, probabilistic* gate based on the Gaussian cumulative distribution function (CDF) Φ(x):

```
GELU(x) = x · Φ(x)
```

The intuition: instead of asking "is x positive?" (ReLU's hard threshold), GELU asks "how likely is x to be positive under a standard normal distribution?" and scales x by that probability. Neurons near zero are partially activated rather than fully on or off. This provides smoother gradients and empirically improves learning, especially for deeper networks.

**Tanh approximation** (used in code for efficiency):
```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

### Implementation in this Code

A `tanh()` method is added to the `Value` autograd class with the correct local gradient `1 - tanh²(x)`. A standalone `gelu()` function then composes `Value` operations, so gradients flow automatically through backprop:

```python
def gelu(x):
    coeff = math.sqrt(2.0 / math.pi)
    inner = (x + x**3 * 0.044715) * coeff
    return x * (inner.tanh() + 1) * 0.5
```

GELU replaces `relu()` inside every expert MLP in the model.

---

## Algorithm 2 — LoRA (Low-Rank Adaptation)

**Paper:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

### Underlying Idea

When fine-tuning a large pretrained model, updating every weight W ∈ ℝⁿˣᵐ requires n·m gradient computations — which becomes prohibitive at scale (GPT-3 has 175 billion parameters). LoRA is motivated by the observation that **weight updates during fine-tuning have low intrinsic rank**: most of the meaningful change in W lies in a much smaller subspace.

LoRA exploits this by decomposing the weight update ΔW into two small matrices:

```
W_effective = W_base  +  B · A

where A ∈ ℝ^(r × m),  B ∈ ℝ^(n × r),  and r << min(n, m)
```

- **W_base** is the frozen pretrained weight (never updated).
- **A** and **B** are the only trainable parameters.
- Parameter count drops from n·m to r·(n + m). For n=m=4096 and r=8, that is 33,554,432 → 65,536 — a 512× reduction.

**Initialisation convention from the paper:**
- A is initialised with small random values (N(0, σ²))
- B is initialised to **zero**, so ΔW = B·A = 0 at step 0. Training starts from exactly the pretrained behaviour.

### Implementation in this Code

LoRA is applied to the **Wq** and **Wv** attention matrices (the standard choice in the original paper — value and query projections benefit most):

```python
# W_base for Wq is frozen (excluded from `params`)
# Only lora_wq_A and lora_wq_B are trained

def lora_linear(x, w_base, lora_a, lora_b):
    base_out   = linear(x, w_base)        # frozen base output
    low_rank   = linear(x, lora_a)        # project down to rank r
    lora_delta = linear(low_rank, lora_b) # project back up to n_embd
    return [a + b for a, b in zip(base_out, lora_delta)]
```

Since we train from scratch (not fine-tuning), Wq and Wv base weights are frozen at their random initialisation and only the LoRA matrices adapt. This demonstrates the structural mechanism; in a real fine-tuning scenario W_base would hold pretrained knowledge.

---

## Algorithm 3 — RoPE (Rotary Position Embedding)

**Paper:** [Su et al., 2021](https://arxiv.org/abs/2104.09864)

### Underlying Idea

The original GPT adds a **learned position embedding** `wpe[pos]` to the token embedding before the first layer. This has two limitations:

1. It encodes *absolute* position, while attention naturally benefits from *relative* position information ("how far apart are these two tokens?").
2. It cannot generalise beyond the training sequence length (no entries exist in the table for unseen positions).

**RoPE** encodes position as a **rotation** applied to the Query (Q) and Key (K) vectors *inside* each attention head, at attention computation time — no learned parameters are needed.

For a head of dimension d, each pair of dimensions (2i, 2i+1) is rotated by angle θᵢ that depends on both the token position and the dimension index:

```
θᵢ = position / 10000^(2i/d)

[q'₂ᵢ  ]   [ cos θᵢ  -sin θᵢ ] [q₂ᵢ  ]
[q'₂ᵢ₊₁] = [ sin θᵢ   cos θᵢ ] [q₂ᵢ₊₁]
```

The key insight: when two rotated vectors are dot-producted (as in attention), the result depends only on the **difference** of their angles, i.e., the *relative distance* between their positions. The model gets relative positional information for free, from the geometry of rotations. RoPE is now used in virtually all modern LLMs (LLaMA, Mistral, Gemini, etc.).

### Implementation in this Code

`wpe` is completely removed from the state dict. Instead, `apply_rope()` is called on each head's Q vector at the current position, and on each cached K vector at its original cached position:

```python
def apply_rope(vec, pos, dim):
    result = list(vec)
    for i in range(0, dim, 2):
        theta  = pos / (10000.0 ** (i / dim))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        v0, v1 = vec[i], vec[i + 1]
        result[i]     = v0 * cos_t - v1 * sin_t
        result[i + 1] = v0 * sin_t + v1 * cos_t
    return result
```

The rotation angles are computed from pure Python math (no Value nodes needed for cos/sin since these are constants w.r.t. the model parameters), so the dot products in attention still flow correct gradients through the Q and K values.

---

## Algorithm 4 — Mixture of Experts (MoE)

**Reference:** [Hugging Face MoE Guide](https://huggingface.co/blog/moe)

### Underlying Idea

A standard transformer MLP applies the **same feed-forward network** to every token. MoE challenges this: why use the same computation for every possible input? Different tokens may require different kinds of processing.

**MoE** replaces the single MLP with a **pool of E expert MLPs** plus a lightweight **router (gating network)**:

```
router probs  = softmax(Router(x))          # shape: (E,)
top-k indices = argtopk(router probs, k)    # only k experts fire
output        = Σᵢ∈top-k  w̃ᵢ · Expertᵢ(x)  # weighted combination
```

where w̃ᵢ are the router probabilities re-normalised over the selected k experts.

The critical property: **only k out of E experts run per token**. This means:
- **Compute cost** scales with k (constant, cheap)
- **Model capacity** (total parameters) scales with E (large, expressive)

By choosing k=2 and E=8 (as in Mixtral), you get 4× the parameters of a dense model at only 2× the compute. Modern frontier models (GPT-4, Mixtral, DeepSeek-V3) are believed to use MoE for exactly this reason.

A key engineering challenge is **load balancing**: naive routing tends to collapse onto 1-2 experts. Production implementations add auxiliary losses to encourage even expert utilisation, which is omitted here for clarity.

### Implementation in this Code

Each transformer layer has `N_EXPERTS = 2` expert MLPs and a router that selects `MOE_TOP_K = 2` of them. The router is a linear projection from `n_embd → N_EXPERTS`, and expert outputs are re-normalised before combining:

```python
def moe_forward(x, li):
    router_logits = linear(x, state_dict[f'layer{li}.moe_router'])
    router_probs  = softmax(router_logits)

    # Select top-k experts
    ranked = sorted(enumerate(router_probs), key=lambda t: t[1].data, reverse=True)
    top_k  = ranked[:MOE_TOP_K]
    weight_sum = sum(w for _, w in top_k)

    out = None
    for e_idx, raw_weight in top_k:
        w_norm = raw_weight / weight_sum   # re-normalise
        h = linear(x, state_dict[f'layer{li}.expert{e_idx}_fc1'])
        h = [gelu(hi) for hi in h]         # GELU activation (Algorithm 1)
        h = linear(h, state_dict[f'layer{li}.expert{e_idx}_fc2'])
        contribution = [w_norm * hi for hi in h]
        out = contribution if out is None else [a + b for a, b in zip(out, contribution)]

    return out
```

---

## Summary of Changes vs. Original microgpt

| Component | Original | This Version |
|---|---|---|
| Activation | `relu()` | `gelu()` (Algorithm 1) |
| Position encoding | Learned `wpe` table | RoPE rotation on Q and K (Algorithm 3) |
| Attention Wq, Wv | Full weight matrix | Frozen base + LoRA low-rank update (Algorithm 2) |
| MLP block | Single MLP per layer | N_EXPERTS expert MLPs + router (Algorithm 4) |
| Autograd | `relu`, `exp`, `log` | Added `tanh` for GELU |

---

## References

- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. https://arxiv.org/abs/1606.08415
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864
- Hugging Face (2024). *Mixture of Experts Explained*. https://huggingface.co/blog/moe
- Karpathy, A. (2026). *microgpt*. https://karpathy.github.io/2026/02/12/microgpt/
