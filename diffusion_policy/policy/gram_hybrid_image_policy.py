"""
GRAM-ACT v4: Generative Recursive reAsoning Model for visuomotor policy learning.

Variational recursive latent reasoning architecture adapted from GRAM for continuous
action prediction. Key components:

- Observation encoder: robomimic ResNet18 with GroupNorm (reused from ACT, unmodified)
- GRAM decoder: shared 2-layer transformer block (RMSNorm post-norm, SwiGLU, RoPE,
  bidirectional self-attention) applied recursively
- Dual latents:
    y [B, T, D]: stochastic prediction latent — sampled once per outer step
    z [B, T, D]: deterministic reasoning latent — refined K times per outer step
- Recursion structure (per latent_recursion call):
    Outer loop (n steps): K deterministic z-updates → 1 stochastic y-update
    Truncated ELBO: first n-1 outer steps under torch.no_grad() (warm-up);
    KL loss computed only at the final outer step (paper Appendix C.1, Eq. 26-27)
- Variational training: prior p_θ(y) vs posterior q_φ(y|a_gt); only the 2-layer
  block is shared — the four (μ_p, σ_p, μ_q, σ_q) heads are separate Linear layers
- KL balancing (Hafner et al. 2020) with α=0.8 prevents posterior collapse
- Deep supervision: N_sup steps with per-step backward; latents detached between steps
- Optional CVAE for latent initialization (single flag for ablation)

Reference: GRAM (Generative Recursive reAsoning Models), adapted for continuous
visuomotor control on Multi-Goal PushT (v4 — K inner z-loop + truncated ELBO).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmoc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.gram.gram_modules import GRAMBlock, precompute_freqs_cis


class GRAMHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # GRAM decoder architecture
            hidden_dim=512,
            n_decoder_layers=2,
            n_heads=8,
            ffn_expansion=4,
            # recursion
            n_recursion=3,
            k_recursion=4,
            N_sup=16,
            # variational (GRAM-specific)
            beta_kl=1.0,
            kl_balance_alpha=0.8,
            sigma_min=1e-4,
            # CVAE (optional — single flag for ablation)
            use_cvae=False,
            latent_dim=32,
            kl_weight=10.0,
            # inference
            inference_n_sup=8,
            ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # ========= Observation encoder (reused from ACT, unmodified) =========
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')

        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]

        # ========= Obs projection =========
        self.obs_proj = nn.Linear(obs_feature_dim, hidden_dim, bias=False)

        # ========= GRAM recursive decoder block (SHARED — single instance) =========
        self.block = GRAMBlock(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=n_decoder_layers, ffn_expansion=ffn_expansion)

        # Precompute RoPE frequencies (no causal mask in v3 — bidirectional self-attn)
        head_dim = hidden_dim // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=horizon)
        self.register_buffer('freqs_cis', freqs_cis)

        # ========= Fixed initial latent states (buffers, not trained) =========
        # Sampled once at construction from truncated normal; same starting point
        # for every forward pass, duplicated to batch size at runtime.
        y_init = torch.empty(1, horizon, hidden_dim)
        z_init = torch.empty(1, horizon, hidden_dim)
        nn.init.trunc_normal_(y_init, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z_init, std=1.0, a=-2.0, b=2.0)
        self.register_buffer('y_init', y_init)
        self.register_buffer('z_init', z_init)

        # ========= Variational heads (4 separate — only the block is shared) =========
        # Prior: predicts (μ_p, σ_p) without ground truth
        self.linear_mu_prior = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_sigma_prior = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Posterior: predicts (μ_q, σ_q) conditioned on ground truth action embedding
        self.linear_mu_post = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_sigma_post = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Action embedding: projects ground truth actions to hidden_dim for posterior
        self.action_embedding = nn.Linear(action_dim, hidden_dim, bias=False)

        # ========= Output head =========
        self.output_head = nn.Linear(hidden_dim, action_dim, bias=False)

        # ========= Optional CVAE encoder =========
        if use_cvae:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
            self.encoder_obs_proj = nn.Linear(obs_feature_dim, hidden_dim)
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, 1 + horizon, hidden_dim))
            nn.init.normal_(self.encoder_pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=4*hidden_dim, dropout=0.0,
                activation='gelu', batch_first=True,
                norm_first=True)
            self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
            self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # ========= Store config =========
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_recursion = n_recursion
        self.k_recursion = k_recursion
        self.N_sup = N_sup
        self.beta_kl = beta_kl
        self.kl_balance_alpha = kl_balance_alpha
        self.sigma_min = sigma_min
        self.use_cvae = use_cvae
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.inference_n_sup = inference_n_sup

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_gram = sum(p.numel() for p in self.block.parameters())
        print(f"GRAMHybridImagePolicy: {n_params:,} total params, "
              f"{n_gram:,} GRAM block params")

    # ========= helpers =========

    def encode_obs(self, nobs, B, To):
        """Encode observations through robomimic obs encoder → (B, To, obs_feature_dim)."""
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def init_latents(self, B):
        """Return learned initial latents expanded to batch size B."""
        return self.y_init.expand(B, -1, -1), self.z_init.expand(B, -1, -1)

    def encode_to_latent(self, obs_features, actions):
        """CVAE encoder: encode actions conditioned on obs → (mu, logvar, z_style).
        Only used when use_cvae=True."""
        B = actions.shape[0]
        action_tokens = self.encoder_action_proj(actions)
        cls = self.cls_token.expand(B, -1, -1)
        encoder_input = torch.cat([cls, action_tokens], dim=1)
        encoder_input = encoder_input + self.encoder_pos_embed
        obs_cond = self.encoder_obs_proj(obs_features.mean(dim=1, keepdim=True))
        encoder_input[:, :1] = encoder_input[:, :1] + obs_cond
        encoder_output = self.cvae_encoder(encoder_input)
        cls_output = encoder_output[:, 0]
        latent_params = self.latent_proj(cls_output)
        mu, logvar = latent_params.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_style = mu + std * eps
        return mu, logvar, z_style

    def _kl_balanced(self, mu_q, sigma_q, mu_p, sigma_p):
        """KL balancing (Hafner et al. 2020) — trains prior more aggressively.

        Returns [B, T] tensor (sum over hidden dim, not mean, per standard ELBO).
        alpha=0.8: 80% gradient flows through the prior, 20% through the posterior.
        This prevents posterior collapse by keeping the prior flexible.
        """
        alpha = self.kl_balance_alpha

        def kl_gauss(mu1, sig1, mu2, sig2):
            # KL(N(mu1,sig1²) || N(mu2,sig2²)), summed over last dim
            return (torch.log(sig2 / sig1)
                    + (sig1**2 + (mu1 - mu2)**2) / (2 * sig2**2)
                    - 0.5).sum(dim=-1)

        # Train prior toward stopgrad(posterior)
        kl_lhs = kl_gauss(mu_q.detach(), sigma_q.detach(), mu_p, sigma_p)
        # Train posterior toward stopgrad(prior)
        kl_rhs = kl_gauss(mu_q, sigma_q, mu_p.detach(), sigma_p.detach())
        return alpha * kl_lhs + (1 - alpha) * kl_rhs  # [B, T]

    # ========= GRAM recursion =========

    def latent_recursion(self, obs_tokens, y, z, n, K, a_gt_emb=None):
        """GRAM v4 variational recursion with K inner z-loop and truncated ELBO.

        Structure per outer step:
          - K deterministic z-updates: z = block(z+y, obs)  [low-level refinement]
          - 1 stochastic y-update: sample from posterior (training) or prior (inference)

        Truncated ELBO (Appendix C.1): first n-1 outer steps run under torch.no_grad()
        as warm-up — they refine (y, z) to a realistic state but contribute no gradient.
        KL is computed ONLY at the final outer step; earlier KL terms contribute zero
        gradient under truncation (paper Eq. 26-27).

        y update (residual): y = u + μ + σ * ε  (NOT y = μ + σ*ε)
        μ heads are used only for the KL term, not for the y sample itself.

        Returns: (y, z, kl_scalar)
        """
        # ------------------------------------------------------------------
        # Warm-up: first n-1 outer steps under no_grad (truncated ELBO)
        # ------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(n - 1):
                # K deterministic z-updates (low-level latent refinement)
                for _ in range(K):
                    z = self.block(z + y, memory=obs_tokens, freqs_cis=self.freqs_cis)

                # Stochastic y-update — same distribution as the final step so
                # the warm-up reaches a realistic (y, z) starting state
                u = self.block(y + z, memory=obs_tokens, freqs_cis=self.freqs_cis)
                mu_p = self.linear_mu_prior(u)
                sigma_p = F.softplus(self.linear_sigma_prior(u)) + self.sigma_min

                if a_gt_emb is not None:
                    mu_q = self.linear_mu_post(u + a_gt_emb)
                    sigma_q = F.softplus(self.linear_sigma_post(u + a_gt_emb)) + self.sigma_min
                    eps = torch.randn_like(mu_q)
                    y = u + mu_q + sigma_q * eps   # residual + learned mean + noise
                else:
                    eps = torch.randn_like(mu_p)
                    y = u + mu_p + sigma_p * eps

        # ------------------------------------------------------------------
        # Final outer step WITH gradients — loss and KL live here
        # ------------------------------------------------------------------
        for _ in range(K):
            z = self.block(z + y, memory=obs_tokens, freqs_cis=self.freqs_cis)

        u = self.block(y + z, memory=obs_tokens, freqs_cis=self.freqs_cis)
        mu_p = self.linear_mu_prior(u)
        sigma_p = F.softplus(self.linear_sigma_prior(u)) + self.sigma_min

        if a_gt_emb is not None:
            mu_q = self.linear_mu_post(u + a_gt_emb)
            sigma_q = F.softplus(self.linear_sigma_post(u + a_gt_emb)) + self.sigma_min
            eps = torch.randn_like(mu_q)
            y = u + mu_q + sigma_q * eps   # residual + learned mean + noise

            # KL only at the final step — truncated ELBO
            kl = self._kl_balanced(mu_q, sigma_q, mu_p, sigma_p).mean()
        else:
            eps = torch.randn_like(mu_p)
            y = u + mu_p + sigma_p * eps
            kl = torch.tensor(0.0, device=obs_tokens.device)

        return y, z, kl

    # ========= inference =========

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        device = self.device

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Initialize latents (truncated normal, or CVAE prior if use_cvae)
        y, z = self.init_latents(B)
        if self.use_cvae:
            z_style = torch.randn(B, self.latent_dim, device=device)
            y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                B, self.horizon, -1).clone()

        # Fixed supervision steps — no halting (removed in v3 per GRAM paper)
        for _ in range(self.inference_n_sup):
            y, z, _ = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion, a_gt_emb=None)

        naction_pred = self.output_head(y)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],
            'action_pred': action_pred,
        }

    def predict_action_samples(
        self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 16
    ) -> torch.Tensor:
        """Draw n_samples diverse action chunks via different noise sequences.

        Implemented via batch tiling (more efficient than a Python loop):
        obs is tiled to [n_samples*B, ...], inference runs once, then reshaped.

        Returns: (n_samples, B, n_action_steps, action_dim)
        """
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        device = self.device
        NB = n_samples * B

        # Tile: each batch element repeated n_samples times
        tiled_nobs = dict_apply(nobs, lambda x: x.repeat_interleave(n_samples, dim=0))
        obs_features = self.encode_obs(tiled_nobs, NB, To)
        obs_tokens = self.obs_proj(obs_features)

        y, z = self.init_latents(NB)
        if self.use_cvae:
            z_style = torch.randn(NB, self.latent_dim, device=device)
            y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                NB, self.horizon, -1).clone()

        for _ in range(self.inference_n_sup):
            y, z, _ = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion, a_gt_emb=None)

        naction_pred = self.output_head(y)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        actions = action_pred[:, start:end]  # [N*B, n_action_steps, action_dim]

        # Reshape: repeat_interleave order → [B, n_samples, ...] → [n_samples, B, ...]
        actions = actions.view(B, n_samples, *actions.shape[1:])
        return actions.transpose(0, 1).contiguous()

    # ========= training =========

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'obs_encoder' in name:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": transformer_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": self.obs_encoder.parameters(),
             "weight_decay": obs_encoder_weight_decay},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        """GRAM training with deep supervision.

        Runs N_sup supervision steps, each with its own backward pass.
        Gradients do NOT flow across supervision steps (latents detached between steps).
        Posterior is used at training time; prior is used at inference.

        Returns a dict {'loss': float, 'kl_loss': float} for logging.
        The KL loss monitors posterior collapse — it should be non-zero if training
        correctly. KL → 0 indicates collapse.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        # Encode obs once (shared across all supervision steps)
        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Ground truth action embedding for posterior conditioning
        a_gt_emb = self.action_embedding(nactions)  # [B, T_horizon, D]

        # Initialize latents
        y, z = self.init_latents(B)

        # Optional CVAE initialization
        cvae_kl = None
        if self.use_cvae:
            mu, logvar, z_style = self.encode_to_latent(
                obs_features.detach(), nactions)
            y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                B, self.horizon, -1).clone()
            cvae_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = 0.0
        total_kl = 0.0

        for sup_step in range(self.N_sup):
            # Variational recursion — uses posterior at training, prior at inference
            y, z, kl = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion, a_gt_emb=a_gt_emb)

            action_pred = self.output_head(y)
            l2_loss = F.mse_loss(action_pred, nactions)

            # ELBO: reconstruction + KL (normalized by N_sup for gradient accumulation)
            step_loss = (l2_loss + self.beta_kl * kl) / self.N_sup

            # CVAE KL added once (on first step — graph shared with y init)
            if self.use_cvae and cvae_kl is not None and sup_step == 0:
                step_loss = step_loss + self.kl_weight * cvae_kl / self.N_sup

            # Backward only when gradients are enabled (skipped during validation)
            if torch.is_grad_enabled():
                # retain_graph=True for all but the last step — obs_tokens and
                # a_gt_emb graphs are shared across all N_sup backward passes
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)

            total_loss += step_loss.item()
            total_kl += kl.item()

            # Detach: no BPTT across supervision steps
            y = y.detach()
            z = z.detach()

        return {'loss': total_loss, 'kl_loss': total_kl / self.N_sup}
