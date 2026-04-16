**Status:** v4 — Adding K inner loop. Using truncated ELBO

**Key Changes from v3:**
- Using K inner loops when updating `z`
- Using stop grad when running the first n-1 outer loop

**Last updated:** [2026-04-16]

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
novelty over its predecessor TRM is that GRAM is a **variational generative model**:
it has a posterior network (sees ground truth at training) and a prior network (used
at inference), and samples the prediction latent stochastically using reparameterization.
This produces diverse reasoning trajectories, which is the central capability we want
for multimodal robot policies.

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
  `T_action` in one shot via cross-attention to encoder tokens.
  The decoder is what you are building.

### Decoder (GRAM recursive block)
The decoder is a **single tiny network** of **2 transformer layers**, applied
recursively. Each layer contains, in this order:
1. Self-attention over action tokens (bidirectional, no causal mask)
2. Cross-attention from action tokens to observation tokens
3. SwiGLU MLP

Use **Post-Norm** (RMSNorm AFTER the residual connection, not before).
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

The four heads (`linear_mu_prior`, `linear_sigma_prior`, `linear_mu_post`,
`linear_sigma_post`) are separate `nn.Linear(D, D)` layers — NOT shared.
Only the 2-layer transformer block is shared.

### Stochastic update of y (residual formulation)

The new `y` is sampled using reparameterization, in residual form:

```
# At training time, sample from posterior
ε ~ N(0, I)
y_new = u_post + μ_q + σ_q * ε   # residual u + learned mean delta + noise

# At inference time, sample from prior
ε ~ N(0, I)
y_new = u_prior + μ_p + σ_p * ε
```

Critical: this is a RESIDUAL formulation. The block output `u` is preserved as
the base; the Gaussian head (μ, σ) adds a learned correction on top. The KL term
regularizes only the Gaussian delta N(μ_q, σ_q²) vs N(μ_p, σ_p²), not the full
distribution including `u`. Do NOT replace with direct sampling `y_new ~ N(μ, σ²)`
— omitting the residual `u` is known to be unstable.

The reasoning latent z is updated DETERMINISTICALLY using the same shared block:

```
z_new = block(z + y_new + cross_attn(obs_tokens))
```

(Note: z update happens AFTER y is sampled, and uses the new y.)

### Full recursion step

The recursion has two nested loops per outer step:
- **Inner loop (K times)**: deterministic updates to the low-level latent `z`, 
  conditioned on the current high-level latent `y`.
- **Outer update (once per step)**: stochastic update to the high-level latent `y`,
  using the refined `z`.

Per GRAM's truncated ELBO (Appendix C.1 of the paper), gradients only flow through
the **final outer step**. The first `n-1` outer steps run under `torch.no_grad()`,
and the KL term is computed ONLY for the final step. This is mathematically
equivalent to the full ELBO because the earlier KL terms contribute zero gradient
under truncation (the detach blocks both direct and indirect gradient paths to
θ and φ for t < T — see the paper's equations 26-27).

````python
def latent_recursion(obs_tokens, y, z, n=3, K=4, a_gt_emb=None, training=True):
    """
    n outer steps = (n-1) no-grad warm-up + 1 final step with gradients.
    Each outer step = K deterministic z-updates + 1 stochastic y-update.
    KL is computed only on the final outer step (truncated ELBO).
    """
    # ---------------------------------------------------------------
    # Warm-up: n-1 outer steps without gradients
    # These refine (y, z) but do not contribute to the loss gradient.
    # ---------------------------------------------------------------
    with torch.no_grad():
        for i in range(n - 1):
            # K deterministic z-updates
            for k in range(K):
                z = block(z + y + cross_attn(obs_tokens))
            
            # Stochastic y-update (sample from posterior if training,
            # prior otherwise — matches what the final step will do,
            # so the warm-up reaches a realistic starting state)
            u_prior = block(y + z + cross_attn(obs_tokens))
            mu_p = linear_mu_prior(u_prior)
            sigma_p = softplus(linear_sigma_prior(u_prior)) + 1e-4
            
            if training and a_gt_emb is not None:
                u_post = block(y + z + cross_attn(obs_tokens) + a_gt_emb)
                mu_q = linear_mu_post(u_post)
                sigma_q = softplus(linear_sigma_post(u_post)) + 1e-4
                eps = torch.randn_like(mu_q)
                y = u_post + mu_q + sigma_q * eps
            else:
                eps = torch.randn_like(mu_p)
                y = u_prior + mu_p + sigma_p * eps
    
    # ---------------------------------------------------------------
    # Final outer step WITH gradients — this is where the loss lives
    # ---------------------------------------------------------------
    # K deterministic z-updates (gradients flow through f_L)
    for k in range(K):
        z = block(z + y + cross_attn(obs_tokens))
    
    # Stochastic y-update (gradients flow through f_H and reparam)
    u_prior = block(y + z + cross_attn(obs_tokens))
    mu_p = linear_mu_prior(u_prior)
    sigma_p = softplus(linear_sigma_prior(u_prior)) + 1e-4
    
    if training and a_gt_emb is not None:
        u_post = block(y + z + cross_attn(obs_tokens) + a_gt_emb)
        mu_q = linear_mu_post(u_post)
        sigma_q = softplus(linear_sigma_post(u_post)) + 1e-4
        
        eps = torch.randn_like(mu_q)
        y = u_post + mu_q + sigma_q * eps
        
        # KL ONLY at the final step (truncated ELBO)
        kl = kl_balanced(mu_q, sigma_q, mu_p, sigma_p, alpha=0.8).mean()
    else:
        eps = torch.randn_like(mu_p)
        y = u_prior + mu_p + sigma_p * eps
        kl = torch.tensor(0.0, device=y.device)
    
    return y, z, kl
````

Key points:
- The first `n-1` outer steps are wrapped in `torch.no_grad()` (equivalent to
  but more efficient than detaching after the fact — no graph is built).
- The final outer step runs with gradients and produces the single KL term 
  that goes into the loss. This is the truncated ELBO from equation (13) of 
  the paper.
- The warm-up steps sample from the same distribution (posterior during 
  training, prior during inference) so that the final step receives a 
  realistic `(y, z)` initialization. This is what the paper's Appendix C.1
  derivation assumes.
- At inference, the distinction between warm-up and final step is moot since 
  no gradients are computed anywhere — but the structure is preserved for 
  code simplicity. You may want to unify into a single loop for inference 
  in the actual implementation.
- `n` here is what the paper calls `T` (number of outer recursion steps per
  supervision step). Default n=3 matches our earlier setting.
- `K` is the number of low-level refinement inner steps per outer step.
  Default K=4.

### KL divergence with KL balancing

GRAM uses **KL balancing** (Hafner et al. 2020/2023) with coefficient α=0.8 to
prevent posterior collapse. This trains the prior more aggressively than the
posterior toward a stop-gradient version of the other:

```python
def kl_balanced(mu_q, sigma_q, mu_p, sigma_p, alpha=0.8):
    # Standard KL between two Gaussians
    def kl_gauss(mu1, sig1, mu2, sig2):
        return (torch.log(sig2 / sig1) 
                + (sig1**2 + (mu1 - mu2)**2) / (2 * sig2**2) 
                - 0.5).sum(dim=-1)
    
    # Train prior toward stopgrad(posterior) with weight alpha
    # Train posterior toward stopgrad(prior) with weight (1 - alpha)
    kl_lhs = kl_gauss(mu_q.detach(), sigma_q.detach(), mu_p, sigma_p)
    kl_rhs = kl_gauss(mu_q, sigma_q, mu_p.detach(), sigma_p.detach())
    return alpha * kl_lhs + (1 - alpha) * kl_rhs
```

Sum across recursion steps and across the action chunk dimension; mean across batch.

### Output head
A single linear projection from the final `y` to action dimension. No Gaussian head
with learned variance — variance comes from the recursive sampling itself.

```
action_pred = output_head(y)   # [B, T_action, action_dim]
```

### No Q-head, no halting

GRAM's paper does not include a halt loss in its training objective. The Q-head
that appears in their architecture table is inherited from TRM/HRM but is not
used in GRAM's reported training or inference. We follow GRAM's actual practice:

- No Q-head module.
- No halt loss in the objective.
- No early stopping during training.
- No early stopping at inference.
- Training uses fixed `N_sup = 16` supervision steps every iteration.
- Inference uses fixed `inference_n_sup` supervision steps (configurable).

A V-head / LPRM (value head for ranking parallel samples at inference) is part
of GRAM's design but is left as a v2 feature. When implemented later, it must be
**conditioned on observation tokens**, not just on the prediction latent — a
deviation from GRAM's symbolic-domain design that is necessary because action
quality depends on observation context.

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

Note: when `use_cvae=True` you have TWO sources of stochasticity (CVAE-style
latent at init + GRAM stochastic recursion). This is intentional for ablation
purposes but may require lower KL weights on both terms to avoid posterior
collapse. Start with the defaults; tune if needed.

---

## Loss specification

Total loss per supervision step:

```
L_total = L_action + β_kl * L_kl_variational + L_cvae_kl (if use_cvae)
```

Where:
- `L_action`: L2 loss between predicted and ground-truth action chunks.
  `F.mse_loss(action_pred, action_gt)` with reduction='mean'.
- `L_kl_variational`: accumulated KL-balanced(q_φ || p_θ) across all recursion
  steps within the supervision step, with weight `β_kl` (default 1.0 — GRAM's
  KL balancing handles the collapse-vs-information tradeoff via its α coefficient,
  so β_kl can stay at standard ELBO weight).
- `L_cvae_kl`: only when `use_cvae=True`, standard CVAE KL term with weight
  `kl_weight`.

**Posterior collapse warning**: Variational models with strong reconstruction
losses tend to collapse the posterior to match the prior (so KL→0) and ignore
the latent. GRAM's KL balancing (α=0.8) is the primary mitigation. If you
still see KL → 0 with poor reconstruction:
- Try KL annealing: ramp `β_kl` from 0 to 1.0 over the first 10k steps.
- Try free bits: clip per-dimension KL to a minimum (e.g., max(KL, 0.1)).
- Reduce α toward 0.5 (more symmetric KL).

Start with KL balancing α=0.8 alone and add other mitigations only if needed.

**Deep supervision loop** (this is the outer training loop, conceptually):

```python
# Inside training step
y, z = init_latents()  # truncated normal, OR from CVAE encoder if use_cvae
a_gt_emb = action_embedding(action_gt)   # for posterior conditioning

for sup_step in range(N_sup):  # default N_sup=16
    y, z, kl = latent_recursion(
        obs_tokens, y, z, 
        n=n_recursion, 
        a_gt_emb=a_gt_emb, 
        training=True,
    )
    action_pred = output_head(y)
    
    loss = mse(action_pred, action_gt) + beta_kl * kl
    if use_cvae:
        loss += kl_weight * cvae_kl
    
    loss.backward()
    
    # Detach latents for next supervision step (no BPTT across sup steps)
    y = y.detach()
    z = z.detach()

optimizer.step()
optimizer.zero_grad()
```

Note: each supervision step has its own backward pass — gradients do NOT flow
across supervision steps because of the detach. This is intentional (matches
GRAM's deep supervision with truncated ELBO).

The optimizer step is taken ONCE after all `N_sup` supervision steps have
accumulated gradients into `.grad`. (Alternatively, take an optimizer step
per supervision step — TRM/GRAM's papers are slightly ambiguous on this; default
to one optimizer step per N_sup supervision steps to match standard
gradient-accumulation practice.)

Note on `T_recursion`: TRM has a separate `T` for the deep recursion outer loop
that runs the full latent recursion T times (T-1 without gradients, 1 with).
GRAM's training section describes a single segment of T recursion steps per
supervision step. For simplicity and to match GRAM more closely, set
`T_recursion = 1` and treat `n_recursion` as the only recursion knob. The
gradient truncation is already handled by the supervision-step detach.

---

## EMA

Wrap the model parameters in an Exponential Moving Average with decay **0.9999**
(this is GRAM's value, not TRM's 0.999). This is critical for stability on small
datasets. The diffusion_policy codebase already has EMA infrastructure
(`diffusion_policy/model/diffusion/ema_model.py` or similar) — reuse it.

---

## Inference

At inference time:
1. Run encoder once on observation.
2. Initialize `(y, z)` from truncated normal (or CVAE prior if `use_cvae`).
3. Run the supervision loop for `inference_n_sup` steps. No loss computation.
   Posterior network is not used (no ground truth available); sample from prior.
4. Return `output_head(y)` from the final step.

```python
def predict_action(self, obs):
    obs_tokens = self.encoder(obs)
    y, z = self.init_latents(obs.shape[0])
    
    for _ in range(self.inference_n_sup):
        y, z, _ = self.latent_recursion(
            obs_tokens, y, z, 
            n=self.n_recursion, 
            K=self.k_recursion,
            a_gt_emb=None,         # no ground truth at inference
            training=False,
        )
    
    return self.output_head(y)
```

**For multimodal inference** (this is the key novelty): support a mode where N
parallel samples are drawn (different noise sequences ε), each producing a
different action chunk. The user will use this to evaluate goal coverage on
Multi-Goal PushT. Add a method:

```python
def predict_action_samples(self, obs, n_samples=N):
    # Returns [N, B, T_action, action_dim]
    # Implement by tiling obs along a new sample dimension and running
    # inference once with batch size N*B, then reshaping. This is more 
    # efficient than a Python for-loop over samples.
```

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
  n_recursion: 6                 # n in GRAM (latent recursion steps per sup step)
  N_sup: 16                      # supervision steps (fixed, no early stopping)
  
  # Variational training (GRAM-specific)
  beta_kl: 1.0                   # ELBO weight on KL term
  kl_balance_alpha: 0.8          # GRAM's KL balancing coefficient (Hafner et al.)
  sigma_min: 1.0e-4              # numerical floor on σ
  
  # CVAE (optional)
  use_cvae: false
  kl_weight: 10.0                # only used if use_cvae=true
  
  # Action chunk
  horizon: 16                    # action chunk length — start with 16, can scale up
  n_obs_steps: 2
  n_action_steps: 8
  
  # Inference
  inference_n_sup: 8             # can differ from training N_sup
  inference_n_samples: 1         # set >1 for multimodal eval
  
training:
  lr: 1.0e-4
  weight_decay: 1.0              # GRAM/TRM use high weight decay
  optimizer: AdamW
  betas: [0.9, 0.95]             # GRAM uses 0.95 not 0.999 for beta2
  warmup_steps: 2000
  batch_size: 64                 # adjust for GPU memory; GRAM uses 768 but PushT smaller
  num_epochs: 200                # tune based on dataset size
  ema_decay: 0.9999              # GRAM's value, not TRM's 0.999
  grad_clip: 1.0
```

Notes on the config:
- The high weight decay (1.0) and Adam beta2=0.95 are GRAM's choices, not standard
  for robot learning. Keep them — this is part of "preserve GRAM's recipe."
- Batch size 64 is a starting guess; user has Multi-Goal PushT which is small.
- Horizon 16 to start; user may want to test 64–128 once the architecture works.
- `inference_n_sup` can be set larger than training `N_sup` to test inference-time
  scaling (one of GRAM's claims).

---

## What you should NOT do

- Do not add a Q-head, halt logic, or any early stopping (training or inference).
- Do not add LayerNorm in places GRAM doesn't have it.
- Do not switch to Pre-Norm unless I explicitly tell you the user reported instability.
- Do not add dropout (GRAM doesn't use it).
- Do not change SwiGLU to GELU.
- Do not add positional embeddings beyond RoPE on self-attention and whatever
  the existing observation encoder uses on its tokens.
- Do not add a Gaussian output head with learned variance.
- Do not implement direct sampling `y_new ~ N(μ, σ²)` instead of residual `y = u + σ·ε`.
- Do not use BPTT across supervision steps — always detach.
- Do not share the four (μ_prior, σ_prior, μ_post, σ_post) linear heads. Only
  the 2-layer transformer block is shared.

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
- Log the KL value (per supervision step and aggregated) prominently during
  training. Posterior collapse is the most likely failure mode and it's
  invisible without this monitoring.

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