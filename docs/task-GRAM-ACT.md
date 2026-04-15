**Status:** v2 — fixed spec, not implemented yet

**Key Changes:** Causal masking removal, latent states, recursion steps, y update, loss

**Last updated:** [2026-04-15]

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

### No causal masking on self-attention
The action chunk is generated in one shot via parallel refinement, not autoregressively. 
All action positions attend to all other action positions bidirectionally. This matches 
GRAM's handling of structured outputs (Sudoku grids) and ACT's parallel action chunk 
generation.

### Latent states
Maintain two latent tensors per recursion:
- `y`: shape `[B, T_action, D]` — the predicted action chunk in latent space
  (this is the STOCHASTIC latent — gets sampled at each recursion step)
- `z`: shape `[B, T_action, D]` — the reasoning latent (deterministic)

Both are initialized at the start of inference from a fixed truncated normal
(std=1, truncation=2). Initialize once per forward pass; do NOT re-initialize per
recursion step.

### Recursion step (GRAM with variational stochastic guidance)

GRAM is a **variational generative model**. It has TWO networks predicting the
distribution over the next `y`:

- **Prior network** `p_θ`: predicts `(μ_p, σ_p)` WITHOUT access to ground truth.
  Used at both training and inference.
- **Posterior network** `q_φ`: predicts `(μ_q, σ_q)` WITH access to ground truth
  action chunk. Used at training time only.

Both networks share the same 2-layer transformer block backbone (this is GRAM's
"single tiny network" property — DO NOT instantiate two separate transformers).
The difference is in their inputs:

```
# Prior path (no ground truth)
u_prior = block(y + z + cross_attn(obs_tokens))
μ_p = linear_mu_prior(u_prior)
σ_p = softplus(linear_sigma_prior(u_prior)) + ε_min   # ensure positive

# Posterior path (with ground truth action embedding a_gt_emb)
u_post = block(y + z + cross_attn(obs_tokens) + a_gt_emb)
μ_q = linear_mu_post(u_post)
σ_q = softplus(linear_sigma_post(u_post)) + ε_min
```

Where `a_gt_emb` is an embedding of the ground truth action chunk (project
ground truth actions through a linear layer to dimension `D`, then add to the
block input). At inference time, posterior path is not used.

### Stochastic update of y (residual formulation)

The new `y` is sampled using reparameterization, in residual form:

```
# At training time, sample from posterior
ε ~ N(0, I)
y_new = u_post + σ_q * ε        # residual: deterministic update + scaled noise

# At inference time, sample from prior
ε ~ N(0, I)
y_new = u_prior + σ_p * ε
```

Critical: this is a RESIDUAL formulation. The deterministic update `u` is
preserved and noise is added on top. Do NOT replace with direct sampling
`y_new ~ N(μ, σ²)` — this is known to be unstable.

The reasoning latent z is updated DETERMINISTICALLY using the same shared block:

```
z_new = block(z + y_new + cross_attn(obs_tokens))
```

(Note: z update happens AFTER y is sampled, and uses the new y.)

### Full recursion step

```python
def latent_recursion(obs_tokens, y, z, n=6, a_gt_emb=None, training=True):
    for i in range(n):
        # Predict prior parameters (always)
        u_prior = block(y + z + cross_attn(obs_tokens))
        mu_p = linear_mu_prior(u_prior)
        sigma_p = softplus(linear_sigma_prior(u_prior)) + 1e-4
        
        if training and a_gt_emb is not None:
            # Predict posterior parameters (training only)
            u_post = block(y + z + cross_attn(obs_tokens) + a_gt_emb)
            mu_q = linear_mu_post(u_post)
            sigma_q = softplus(linear_sigma_post(u_post)) + 1e-4
            
            # Sample from posterior using reparameterization
            eps = torch.randn_like(mu_q)
            y = mu_q + sigma_q * eps
            
            # Accumulate KL divergence between posterior and prior
            kl = kl_divergence_gaussian(mu_q, sigma_q, mu_p, sigma_p)
            kl_total += kl.mean()
        else:
            # Inference: sample from prior
            eps = torch.randn_like(mu_p)
            y = mu_p + sigma_p * eps
        
        # Update reasoning latent deterministically
        z = block(z + y + cross_attn(obs_tokens))
    
    return y, z, kl_total
```

The KL term per step:
```
KL(q || p) = log(σ_p / σ_q) + (σ_q² + (μ_q - μ_p)²) / (2 σ_p²) - 0.5
```
Sum across recursion steps and across the action chunk dimension; mean across batch.


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

Total loss per supervision step:

```
L_total = L_action + L_q + β_kl * L_kl_variational + L_cvae_kl (if use_cvae)
```

Where:
- `L_action`: L2 loss between predicted and ground-truth action chunks.
- `L_q`: BCE loss for Q-head, weight 0.5.
- `L_kl_variational`: accumulated KL(q_φ || p_θ) across all recursion steps,
  with weight `β_kl` (default 0.001 — start small to avoid posterior collapse,
  may need to anneal up).
- `L_cvae_kl`: only when `use_cvae=True`, standard CVAE KL term.

Remove the previous `L_kl_gram` (free-bits regularizer on σ) — it was a hack
that's no longer needed now that we have proper variational training. The
KL(q || p) term naturally regularizes the noise scale.

**Posterior collapse warning**: Variational models with strong reconstruction
losses tend to collapse the posterior to match the prior (so KL→0) and ignore
the latent. Watch for this by monitoring KL value during training. If KL → 0
and reconstruction is poor, you're collapsing. Mitigations:
- KL annealing: start `β_kl = 0`, ramp up to 0.001 over first 10k steps
- Free bits: clip KL to a minimum value per dimension (e.g., max(KL, 0.1))
- Lower `β_kl` further

Start without these mitigations and add them only if you see collapse.

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
