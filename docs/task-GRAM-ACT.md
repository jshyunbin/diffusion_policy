# Task: Implement GRAM-ACT (`GRAMHybridImagePolicy`) for Multi-Goal PushT

You are extending an existing prototype `GRAMHybridImagePolicy` class in a fork of the
`real-stanford/diffusion_policy` repository. The prototype was created by copying the
ACT (Action Chunking Transformer) architecture and removing the CVAE encoder. Your job
is to evolve this prototype into a faithful adaptation of GRAM (Generative Recursive
reAsoning Models) for continuous visuomotor policy learning on Multi-Goal PushT.

You have NOT read the GRAM, TRM, or HRM papers. All necessary architectural and
algorithmic details are specified below. Follow them exactly. Do not introduce design
choices from your training data unless explicitly asked.

---

## Background context (read carefully)

GRAM is a recursive latent reasoning architecture originally designed for discrete
symbolic tasks (Sudoku, ARC-AGI, MNIST). It works by maintaining two latent states
— a "prediction" latent `y` and a "reasoning" latent `z` — and iteratively refining
them through repeated application of a small (2-layer) transformer block. The key
novelty over its predecessor TRM is that GRAM injects stochastic noise into the
latent updates at each recursion step, producing a generative model that can sample
diverse reasoning trajectories.

We are adapting GRAM to continuous action prediction for robot policies. The novelty
of our work is the *domain transfer* — we preserve GRAM's architecture and
training recipe as faithfully as possible and change only what is strictly necessary
for continuous visuomotor control. Reviewers will scrutinize every deviation, so
do not add design choices that aren't specified here.

---

## Architecture specification

### Overall structure
Encoder-decoder transformer with cross-attention:
- **Encoder**: Standard transformer encoder over observation tokens (image features
  from each camera view + proprioception). Runs ONCE per inference call. Reuse the
  existing observation encoder from the ACT prototype — do not modify it.
- **Decoder**: GRAM-style recursive block. Generates action chunks of length
  `T_action` (default 64–128) in one shot via cross-attention to encoder tokens.
  The decoder is what you are building.

### Decoder (GRAM recursive block)
The decoder is a **single tiny network** of **2 transformer layers**, applied
recursively. Each layer contains, in this order:
1. Self-attention over action tokens (with causal masking — see below)
2. Cross-attention from action tokens to observation tokens
3. SwiGLU MLP

Use **Post-Norm** (LayerNorm AFTER the residual connection, not before).
Use **RMSNorm** (not LayerNorm).
Use **RoPE** (rotary positional embeddings) on the self-attention.
No bias terms in linear layers.
Hidden size: 512 (default — make configurable).

The same 2-layer network is applied repeatedly during recursion. Do NOT instantiate
multiple copies — there is one shared 2-layer block that gets reused.

### Causal masking decision
Apply causal masking on the self-attention. Even though the action chunk is generated
in one shot (not autoregressively), causal masking respects temporal ordering of
actions and matches GRAM's positional handling. The whole chunk is produced in
parallel — causal masking just shapes the attention pattern.

### Latent states
Maintain two latent tensors per recursion:
- `y`: shape `[B, T_action, D]` — the predicted action chunk in latent space
- `z`: shape `[B, T_action, D]` — the reasoning latent

Both are initialized at the start of inference from a fixed truncated normal
(std=1, truncation=2). Initialize once per forward pass; do NOT re-initialize per
recursion step.

### Recursion step (deterministic TRM-style for reference)
```
def latent_recursion(obs_tokens, y, z, n=6):
    for i in range(n):
        # Update reasoning latent z, conditioned on obs, y, z
        z = block(z + y + cross_attn_input(obs_tokens))
    # Update prediction latent y, conditioned on z and y (NOT obs)
    y = block(y + z)
    return y, z
```

Note: `z` updates see the observation context; `y` updates do NOT. This asymmetry
is load-bearing — it forces `y` to integrate observation information through `z`.
Do not break it.

### GRAM stochastic guidance (the key generative addition)
GRAM modifies the deterministic update by injecting learned-scale Gaussian noise
into the latent transition. Replace `z = block(...)` with:

```
u = block(z + y + cross_attn_input(obs_tokens))   # deterministic update
ε ~ N(0, σ² I)                                      # sampled noise
z = u + ε                                           # residual stochastic guidance
```

The noise is added to the OUTPUT of the block (residual formulation), not to the
input. The noise scale `σ` should be a learned parameter (one scalar, or one per
hidden dim — start with a scalar). Initialize `σ` to a small value like 0.1.

Critically: do NOT replace `z = u + ε` with `z ~ N(μ_θ, σ²_θ I)` (i.e., direct
sampling from a learned Gaussian). The residual formulation `u + ε` is what makes
GRAM stable. Direct sampling is known to underperform.

### Deep supervision (recursion across multiple supervision steps)
GRAM uses "deep supervision": the recursion is wrapped in an outer loop where the
final `(y, z)` from one recursion is detached and used as the initialization for
the next, with a loss applied at each supervision step.

```
def deep_recursion(obs_tokens, y, z, n=6, T=3):
    # Run T-1 recursions WITHOUT gradients
    with torch.no_grad():
        for _ in range(T-1):
            y, z = latent_recursion(obs_tokens, y, z, n)
    # Final recursion WITH gradients
    y, z = latent_recursion(obs_tokens, y, z, n)
    return y, z
```

For training, wrap this in an outer supervision loop (`N_sup = 16`), where each
iteration applies the loss and detaches latents. See the loss section.

### Output head
A single linear projection from the final `y` to action dimension. No Gaussian head
with learned variance — variance comes from the recursive sampling itself.

```
action_pred = output_head(y)   # [B, T_action, action_dim]
```

### Q-head (halting head)
Keep a Q-head, but adapted for continuous outputs. It is a linear projection from
the mean-pooled final `y` to a scalar logit, trained with binary cross-entropy
against a thresholded loss signal:

```
q_logit = q_head(y.mean(dim=1))   # [B, 1]
success_target = (l2_loss < threshold).float()
q_loss = BCE(q_logit, success_target)
```

The `threshold` is a hyperparameter. Default to a value that yields ~30–50%
"success" rate at convergence — make it configurable, default to 0.05.

At inference, optionally use the Q-head for early stopping (halt when sigmoid(q) > 0.5
AND minimum supervision step reached). Make this an inference-time flag.

---

## Optional CVAE encoder

Make the CVAE encoder OPTIONAL via a config flag `use_cvae: bool` (default `False`).

When `use_cvae=False` (default, GRAM-faithful):
- Initialize `(y, z)` from fixed truncated normal as described above.
- No CVAE KL term in loss.

When `use_cvae=True`:
- Reuse the CVAE encoder from the original ACT codebase (it takes ground-truth
  action chunks during training and produces a latent z_style).
- Use `z_style` to initialize `y` instead of the fixed truncated normal.
  (Project z_style to `[B, T_action, D]` if dimensions don't match — broadcast
  across the time dimension.)
- Add the standard CVAE KL term to the loss with weight `kl_weight` (default 10
  matching ACT, but make it configurable).
- At inference time, sample z_style from the prior (zero mean, unit variance)
  as ACT does.

The user wants to be able to flip this flag and run both variants for comparison.
Implement cleanly so this is a single config change.

---

## Loss specification

Total loss per supervision step `m`:

```
L_total = L_action + L_q + L_kl_gram + L_cvae_kl (if use_cvae)
```

Where:
- `L_action`: L2 loss between predicted and ground-truth action chunks.
  `F.mse_loss(action_pred, action_gt)` with reduction='mean'.
- `L_q`: BCE loss for Q-head as described above. Weight: 0.5 (matches GRAM/HRM).
- `L_kl_gram`: KL regularization on the noise scale, encouraging it to remain
  non-trivial. Add a small term: `kl_weight_gram * (sigma**2 - log(sigma**2) - 1)` 
  to prevent noise collapse. Default `kl_weight_gram = 0.01`. (This is a simple
  free-bits-style regularizer to prevent σ → 0.)
- `L_cvae_kl`: standard KL between CVAE encoder posterior and N(0, I). Only when
  `use_cvae=True`.

**Deep supervision loop** (this is the outer training loop, conceptually):

```python
# Inside training step
y, z = init_latents()  # truncated normal, OR from CVAE encoder if use_cvae
for sup_step in range(N_sup):  # default N_sup=16
    y, z = deep_recursion(obs_tokens, y, z, n=n_recursion, T=T_recursion)
    action_pred = output_head(y)
    q_logit = q_head(y.mean(dim=1))
    
    loss = mse(action_pred, action_gt) \
         + 0.5 * bce(q_logit, (mse_per_sample < threshold).float()) \
         + kl_weight_gram * sigma_kl_term()
    if use_cvae:
        loss += kl_weight * cvae_kl
    
    loss.backward()
    
    # Detach latents for next supervision step
    y = y.detach()
    z = z.detach()
    
    # Optional early stopping if Q-head says halt
    if sigmoid(q_logit).mean() > 0.5 and sup_step >= 1:
        break
```

Note: each supervision step has its own backward pass — gradients do NOT flow
across supervision steps because of the detach. This is intentional.

---

## EMA

Wrap the model parameters in an Exponential Moving Average with decay 0.999.
This is critical for stability on small datasets. The diffusion_policy codebase
already has EMA infrastructure (`diffusion_policy/model/diffusion/ema_model.py`
or similar) — reuse it.

---

## Inference

At inference time:
1. Run encoder once on observation.
2. Initialize `(y, z)` from truncated normal (or CVAE prior if `use_cvae`).
3. Run the supervision loop, but WITHOUT loss computation, WITHOUT detaching
   (or with detaching — doesn't matter at inference).
4. Use Q-head for early stopping if `inference_use_q_halting=True`, else run for
   a fixed number of steps (`inference_n_sup`, default 8).
5. Return `output_head(y)` from the final step.

**For multimodal inference** (this is the key novelty): support a mode where N
parallel samples are drawn (different noise sequences ε), each producing a
different action chunk. The user will use this to evaluate goal coverage on
Multi-Goal PushT. Add a method `predict_action_samples(obs, n_samples=N)` that
returns `[N, B, T_action, action_dim]`.

---

## Hyperparameters for Multi-Goal PushT config

Create a config file at `diffusion_policy/config/task/multigoal_pusht_gram_act.yaml`
(adjust path as appropriate for the repo structure) with the following:

```yaml
name: gram_act_multigoal_pusht

policy:
  _target_: diffusion_policy.policy.gram_hybrid_image_policy.GRAMHybridImagePolicy
  
  # Architecture
  hidden_dim: 512
  n_layers: 2                    # GRAM uses 2 layers
  n_heads: 8
  ffn_expansion: 4               # SwiGLU expansion ratio
  use_rope: true
  norm_type: rmsnorm
  norm_position: post            # Post-Norm
  use_bias: false
  
  # Recursion
  n_recursion: 6                 # n in TRM/GRAM (latent recursion steps)
  T_recursion: 3                 # T in TRM/GRAM (deep recursion outer loop)
  N_sup: 16                      # max supervision steps
  
  # Stochastic guidance (GRAM-specific)
  sigma_init: 0.1
  kl_weight_gram: 0.01
  
  # Q-head
  use_q_head: true
  q_loss_weight: 0.5
  success_threshold: 0.05
  
  # CVAE (optional)
  use_cvae: false
  kl_weight: 10.0                # only used if use_cvae=true
  
  # Action chunk
  horizon: 16                    # action chunk length — start with 16, can scale up
  n_obs_steps: 2
  n_action_steps: 8
  
  # Inference
  inference_n_sup: 8
  inference_use_q_halting: false
  
training:
  lr: 1.0e-4
  weight_decay: 1.0              # GRAM/TRM use high weight decay
  optimizer: AdamW
  betas: [0.9, 0.95]             # GRAM uses 0.95 not 0.999 for beta2
  warmup_steps: 2000
  batch_size: 64                 # adjust for GPU memory; GRAM uses 768 but PushT smaller
  num_epochs: 200                # tune based on dataset size
  ema_decay: 0.999
  grad_clip: 1.0
```

Notes on the config:
- The high weight decay (1.0) and Adam beta2=0.95 are GRAM's choices, not standard
  for robot learning. Keep them — this is part of "preserve GRAM's recipe."
- Batch size 64 is a starting guess; user has Multi-Goal PushT which is small.
- Horizon 16 to start; user may want to test 64–128 once the architecture works.

---

## What you should NOT do

- Do not add LayerNorm in places GRAM doesn't have it.
- Do not switch to Pre-Norm unless I explicitly tell you the user reported instability.
- Do not add dropout (GRAM doesn't use it).
- Do not change SwiGLU to GELU.
- Do not add positional embeddings beyond RoPE on self-attention and whatever
  the existing observation encoder uses on its tokens.
- Do not add a Gaussian output head with learned variance.
- Do not implement direct sampling `z ~ N(μ, σ²)` instead of residual `z = u + ε`.
- Do not use BPTT across supervision steps — always detach.

## What you SHOULD do

- Keep the implementation tight and readable. The recursive block + recursion
  logic should fit in roughly 200–400 lines total, not 1000+.
- Add inline comments referencing this spec where you make a design choice.
- Print the parameter count at model construction so we can verify it's "tiny"
  (should be in the low millions, not tens of millions).
- Make the CVAE option a single boolean flag — clean toggling matters for
  ablations.
- Add a `predict_action_samples` method for multimodal evaluation.
- Reuse diffusion_policy's existing EMA, optimizer, and training loop
  infrastructure where possible.

---

## Deliverables

1. Updated `GRAMHybridImagePolicy` class implementing the above.
2. The config file at the path specified.
3. Any new modules (e.g., `RMSNorm`, `SwiGLU`, `RoPE` helpers if not already
   present) in appropriate locations under `diffusion_policy/model/`.
4. A brief README section at the top of the policy file summarizing the
   architecture and citing GRAM as the inspiration.

Ask clarifying questions ONLY if something in this spec contradicts something
in the existing codebase. Otherwise, proceed with implementation.
