"""
microgpt — extended with four modern LLM algorithms:
  1. GELU  (Gaussian Error Linear Units)   https://arxiv.org/abs/1606.08415
  2. LoRA  (Low-Rank Adaptation)           https://arxiv.org/abs/2106.09685
  3. RoPE  (Rotary Position Embedding)     https://arxiv.org/abs/2104.09864
  4. MoE   (Mixture of Experts)            https://huggingface.co/blog/moe

Base code by @karpathy:
  https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

All additions are pure-Python, dependency-free, and run through the same
scalar autograd Value engine used in the original.
"""

import os
import math
import random
random.seed(42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Autograd  (unchanged from original, with tanh added for GELU)
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):  return Value(math.log(self.data),  (self,), (1 / self.data,))
    def exp(self):  return Value(math.exp(self.data),  (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data),    (self,), (float(self.data > 0),))

    # NEW — needed by GELU
    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1.0 - t * t,))   # d/dx tanh(x) = 1 - tanh²(x)

    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ===========================================================================
# Algorithm 1 — GELU (Gaussian Error Linear Units)
# Paper: https://arxiv.org/abs/1606.08415
#
# Underlying idea:
#   ReLU gates activations with a hard 0/1 step: f(x) = max(0, x).
#   GELU gates them softly using the Gaussian CDF Φ(x):
#       GELU(x) = x · Φ(x)
#   Intuitively this means: "pass x through, but weight it by how
#   likely x is to be positive under a standard normal distribution."
#   This soft gate is differentiable everywhere, which provides better
#   gradient flow, especially for negative inputs (unlike ReLU which
#   kills gradients entirely for x < 0).
#
#   We use the standard tanh approximation:
#       GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
#
# Usage in this file: replaces relu() inside every expert MLP.
# ===========================================================================
_GELU_COEFF = math.sqrt(2.0 / math.pi)

def gelu(x):
    """GELU activation (tanh approximation). x is a Value."""
    inner = (x + x**3 * 0.044715) * _GELU_COEFF
    return x * (inner.tanh() + 1) * 0.5


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
head_dim   = n_embd // n_head   # = 4  (must be even for RoPE)

# LoRA rank — smaller → fewer trainable params; r << n_embd
LORA_RANK  = 4

# Number of MoE experts per layer
N_EXPERTS  = 2
# How many experts are activated per token (top-k)
MOE_TOP_K  = 2

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def rand_matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def zero_matrix(nout, nin):
    return [[Value(0.0) for _ in range(nin)] for _ in range(nout)]

# ===========================================================================
# Algorithm 3 — RoPE (Rotary Position Embedding)
# Paper: https://arxiv.org/abs/2104.09864
#
# Underlying idea:
#   Original transformers learn a separate position embedding table (wpe)
#   that is simply added to the token embedding. This couples absolute
#   positions to specific directions in embedding space and does not
#   generalise well beyond the training sequence length.
#
#   RoPE instead encodes position as a ROTATION applied to Query and Key
#   vectors at attention time. For each pair of dimensions (2i, 2i+1) in a
#   head of dimension d, a rotation by angle θᵢ is applied:
#
#       θᵢ = pos / 10000^(2i/d)
#
#       [q'₂ᵢ ]   [ cos θᵢ  -sin θᵢ ] [q₂ᵢ ]
#       [q'₂ᵢ₊₁] = [ sin θᵢ   cos θᵢ ] [q₂ᵢ₊₁]
#
#   The relative position between two tokens t₁ and t₂ then appears as the
#   *difference* of their rotation angles, which means attention scores
#   depend only on (t₁ - t₂), giving the model a natural sense of relative
#   distance without needing learned position parameters.
#
# Usage in this file:
#   - wpe is REMOVED from state_dict (no learned position table).
#   - apply_rope() is called on q and k per attention head.
# ===========================================================================
def apply_rope(vec, pos, dim):
    """
    Apply RoPE at sequence position `pos` to a vector of length `dim`.
    `dim` must be even; each pair (2i, 2i+1) is independently rotated.
    The rotation angles are deterministic (no learnable parameters).
    """
    result = list(vec)
    for i in range(0, dim, 2):
        theta   = pos / (10000.0 ** (i / dim))
        cos_t   = math.cos(theta)
        sin_t   = math.sin(theta)
        v0, v1  = vec[i], vec[i + 1]
        result[i]     = v0 * cos_t - v1 * sin_t
        result[i + 1] = v0 * sin_t + v1 * cos_t
    return result


# ===========================================================================
# Algorithm 2 — LoRA (Low-Rank Adaptation)
# Paper: https://arxiv.org/abs/2106.09685
#
# Underlying idea:
#   When fine-tuning a large pretrained model it is wasteful to update every
#   weight W ∈ ℝ^(n×m) with a full-rank gradient matrix.  LoRA observes that
#   the weight update ΔW has low intrinsic rank in practice, so we
#   decompose it as:
#
#       W_effective = W_base  +  B · A
#
#   where  A ∈ ℝ^(r×m)  and  B ∈ ℝ^(n×r)  with r << min(n, m).
#   W_base is frozen; only A and B are trained.
#   This reduces trainable parameters from n·m to r·(n+m).
#
#   Initialisation convention (from the paper):
#       A ~ N(0, σ²)    (random)
#       B = 0           (so ΔW = B·A = 0 at the start → training starts
#                         from the pretrained behaviour)
#
# Usage in this file:
#   - Applied to attention Wq and Wv matrices (standard choice in the paper).
#   - W_base weights for Wq and Wv are NOT included in `params` (frozen).
#   - lora_A and lora_B matrices ARE in `params` (trainable).
# ===========================================================================
def lora_linear(x, w_base, lora_a, lora_b):
    """
    Compute  W_base @ x  +  B @ (A @ x).
    w_base is frozen (not in params); lora_a and lora_b are trainable.
    """
    base_out   = linear(x, w_base)          # (n_embd,)
    low_rank   = linear(x,        lora_a)   # (rank,)
    lora_delta = linear(low_rank, lora_b)   # (n_embd,)
    return [a + b for a, b in zip(base_out, lora_delta)]


# ---------------------------------------------------------------------------
# State dict — all weights
# ---------------------------------------------------------------------------
state_dict = {
    'wte':      rand_matrix(vocab_size, n_embd),   # token embedding table
    # NOTE: 'wpe' (positional embedding table) is intentionally OMITTED —
    # position is now encoded via RoPE inside the attention heads.
    'lm_head':  rand_matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    # Attention projections — Wq and Wv will use LoRA; Wk and Wo are fully trained
    state_dict[f'layer{i}.attn_wq'] = rand_matrix(n_embd, n_embd)   # frozen base
    state_dict[f'layer{i}.attn_wk'] = rand_matrix(n_embd, n_embd)   # fully trained
    state_dict[f'layer{i}.attn_wv'] = rand_matrix(n_embd, n_embd)   # frozen base
    state_dict[f'layer{i}.attn_wo'] = rand_matrix(n_embd, n_embd)   # fully trained

    # LoRA matrices for Wq  (A: rank×n_embd, B: n_embd×rank, B init = 0)
    state_dict[f'layer{i}.lora_wq_A'] = rand_matrix(LORA_RANK, n_embd, std=0.01)
    state_dict[f'layer{i}.lora_wq_B'] = zero_matrix(n_embd,    LORA_RANK)

    # LoRA matrices for Wv
    state_dict[f'layer{i}.lora_wv_A'] = rand_matrix(LORA_RANK, n_embd, std=0.01)
    state_dict[f'layer{i}.lora_wv_B'] = zero_matrix(n_embd,    LORA_RANK)

    # ===========================================================================
    # Algorithm 4 — Mixture of Experts (MoE)
    # Reference: https://huggingface.co/blog/moe
    #
    # Underlying idea:
    #   A vanilla transformer MLP runs the same feed-forward network for every
    #   token.  MoE replaces that single MLP with a *pool* of E expert MLPs
    #   plus a lightweight *router* (gating) network:
    #
    #       router logits  = Router(x)                 shape (E,)
    #       router probs   = softmax(router logits)
    #       top-k indices  = argtopk(router probs, k)  only k experts fire
    #       output         = Σᵢ∈top-k  w̃ᵢ · Expertᵢ(x)
    #
    #   where w̃ᵢ = wᵢ / Σⱼ∈top-k wⱼ  (re-normalised weights).
    #
    #   Only k out of E experts run per token, so *compute cost scales with k*
    #   while *model capacity scales with E*.  Modern LLMs (Mixtral, DeepSeek,
    #   GPT-4) use MoE to get much larger effective parameter counts without
    #   proportionally larger training/inference costs.
    #
    # Usage in this file:
    #   - N_EXPERTS = 2, MOE_TOP_K = 2 (both experts fire; illustrative).
    #   - Each expert is a 2-layer MLP with GELU (same structure as original).
    #   - The router is a simple linear layer (no bias).
    # ===========================================================================
    state_dict[f'layer{i}.moe_router'] = rand_matrix(N_EXPERTS, n_embd)

    for e in range(N_EXPERTS):
        state_dict[f'layer{i}.expert{e}_fc1'] = rand_matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}_fc2'] = rand_matrix(n_embd, 4 * n_embd)

# ---------------------------------------------------------------------------
# Collect *trainable* parameters
#   Frozen: attn_wq and attn_wv base weights (LoRA keeps them fixed).
#   Everything else (LoRA A/B, Wk, Wo, experts, router, wte, lm_head) is trained.
# ---------------------------------------------------------------------------
_frozen = {f'layer{i}.attn_wq' for i in range(n_layer)} | \
          {f'layer{i}.attn_wv' for i in range(n_layer)}

params = [p for key, mat in state_dict.items()
            if key not in _frozen
            for row in mat for p in row]
print(f"num trainable params: {len(params)}")


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps    = [(val - max_val).exp() for val in logits]
    total   = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms    = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def moe_forward(x, li):
    """
    Mixture-of-Experts MLP block for layer `li`.
    Router selects top-k experts; their outputs are re-normalised and summed.
    """
    # 1. Router: n_embd → N_EXPERTS logits
    router_logits = linear(x, state_dict[f'layer{li}.moe_router'])
    router_probs  = softmax(router_logits)

    # 2. Rank experts by probability, pick top-k
    ranked = sorted(enumerate(router_probs), key=lambda t: t[1].data, reverse=True)
    top_k  = ranked[:MOE_TOP_K]

    # 3. Re-normalise the selected weights so they sum to 1
    weight_sum = sum(w for _, w in top_k)

    # 4. Weighted sum of selected expert outputs
    out = None
    for e_idx, raw_weight in top_k:
        w_norm = raw_weight / weight_sum    # differentiable re-normalisation

        # Expert MLP: fc1 → GELU → fc2
        h = linear(x, state_dict[f'layer{li}.expert{e_idx}_fc1'])
        h = [gelu(hi) for hi in h]          # GELU replaces ReLU here
        h = linear(h, state_dict[f'layer{li}.expert{e_idx}_fc2'])

        contribution = [w_norm * hi for hi in h]
        out = contribution if out is None else [a + b for a, b in zip(out, contribution)]

    return out


# ---------------------------------------------------------------------------
# GPT forward pass — integrates all four algorithms
# ---------------------------------------------------------------------------
def gpt(token_id, pos_id, keys, values):
    # Token embedding only (no position table — RoPE handles position)
    x = list(state_dict['wte'][token_id])
    x = rmsnorm(x)

    for li in range(n_layer):

        # ------------------------------------------------------------------ #
        # 1) Multi-head Attention  (LoRA on Wq/Wv, RoPE on Q and K)
        # ------------------------------------------------------------------ #
        x_residual = x
        x = rmsnorm(x)

        # Wq with LoRA, Wk fully trained, Wv with LoRA
        q = lora_linear(x,
                        state_dict[f'layer{li}.attn_wq'],
                        state_dict[f'layer{li}.lora_wq_A'],
                        state_dict[f'layer{li}.lora_wq_B'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        val = lora_linear(x,
                          state_dict[f'layer{li}.attn_wv'],
                          state_dict[f'layer{li}.lora_wv_A'],
                          state_dict[f'layer{li}.lora_wv_B'])

        keys[li].append(k)
        values[li].append(val)

        x_attn = []
        for h in range(n_head):
            hs  = h * head_dim
            q_h = q[hs:hs + head_dim]

            # RoPE: rotate the current query at its position
            q_h_rot = apply_rope(q_h, pos_id, head_dim)

            # RoPE: rotate each cached key at its original position
            k_cache  = [ki[hs:hs + head_dim] for ki in keys[li]]
            k_h_rots = [apply_rope(k_cache[t], t, head_dim)
                        for t in range(len(k_cache))]

            v_h = [vi[hs:hs + head_dim] for vi in values[li]]

            attn_logits  = [sum(q_h_rot[j] * k_h_rots[t][j]
                               for j in range(head_dim)) / head_dim**0.5
                            for t in range(len(k_h_rots))]
            attn_weights = softmax(attn_logits)
            head_out     = [sum(attn_weights[t] * v_h[t][j]
                               for t in range(len(v_h)))
                            for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]   # residual connection

        # ------------------------------------------------------------------ #
        # 2) MoE MLP block  (Mixture of Experts with GELU activations)
        # ------------------------------------------------------------------ #
        x_residual = x
        x = rmsnorm(x)
        x = moe_forward(x, li)
        x = [a + b for a, b in zip(x, x_residual)]   # residual connection

    logits = linear(x, state_dict['lm_head'])
    return logits


# ---------------------------------------------------------------------------
# Training  (identical structure to original)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)    # Adam first moment
v_buf = [0.0] * len(params)    # Adam second moment

num_steps = 1000
for step in range(num_steps):

    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits  = gpt(token_id, pos_id, keys, values)
        probs   = softmax(logits)
        loss_t  = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        m_hat    = m_buf[i] / (1 - beta1 ** (step + 1))
        v_hat    = v_buf[i] / (1 - beta2 ** (step + 1))
        p.data  -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad   = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample   = []
    for pos_id in range(block_size):
        logits   = gpt(token_id, pos_id, keys, values)
        probs    = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
