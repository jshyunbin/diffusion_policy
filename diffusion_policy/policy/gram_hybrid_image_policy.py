"""
GRAM-ACT: Generative Recursive reAsoning Model for visuomotor policy learning.

Adapts the GRAM architecture (originally for discrete symbolic tasks) to continuous
action prediction for robot policies. Key components:

- Observation encoder: robomimic ResNet18 with GroupNorm (reused from ACT)
- Decoder: 2-layer transformer block (RMSNorm post-norm, SwiGLU, RoPE, causal mask)
  applied recursively with stochastic guidance
- Dual latents: y (prediction) and z (reasoning), iteratively refined
  - z updates see observation context via cross-attention
  - y updates do NOT see observations (forced to integrate through z)
- Stochastic guidance: learned-scale Gaussian noise on z transitions
- Deep supervision: multiple supervision steps with detached latent propagation
- Q-head: halting head trained with BCE against thresholded loss
- Optional CVAE encoder for latent initialization (toggleable for ablation)

Reference: GRAM (Generative Recursive reAsoning Models), adapted for continuous
visuomotor control on Multi-Goal PushT.
"""

from typing import Dict, Tuple
import math
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
            n_recursion=6,
            T_recursion=3,
            N_sup=16,
            # stochastic guidance
            sigma_init=0.1,
            kl_weight_gram=0.01,
            # Q-head
            use_q_head=True,
            q_loss_weight=0.5,
            success_threshold=0.05,
            # CVAE (optional)
            use_cvae=False,
            latent_dim=32,
            kl_weight=10.0,
            # inference
            inference_n_sup=8,
            inference_use_q_halting=False,
            ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
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

        # ========= GRAM recursive decoder block =========
        self.block = GRAMBlock(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=n_decoder_layers, ffn_expansion=ffn_expansion)

        # Precompute RoPE frequencies and causal mask (registered as buffers)
        head_dim = hidden_dim // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=horizon)
        self.register_buffer('freqs_cis', freqs_cis)

        causal_mask = torch.full((horizon, horizon), float('-inf'))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

        # Learned noise scale for stochastic guidance (parameterized as log_sigma)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))

        # ========= Output head =========
        self.output_head = nn.Linear(hidden_dim, action_dim, bias=False)

        # ========= Q-head (halting head) =========
        self.q_head = nn.Linear(hidden_dim, 1, bias=False) if use_q_head else None

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
            self.cvae_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=2)
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
        self.T_recursion = T_recursion
        self.N_sup = N_sup
        self.kl_weight_gram = kl_weight_gram
        self.use_q_head = use_q_head
        self.q_loss_weight = q_loss_weight
        self.success_threshold = success_threshold
        self.use_cvae = use_cvae
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.inference_n_sup = inference_n_sup
        self.inference_use_q_halting = inference_use_q_halting

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_gram = sum(p.numel() for p in self.block.parameters())
        print(f"GRAMHybridImagePolicy: {n_params:,} total params, "
              f"{n_gram:,} GRAM block params")

    # ========= helpers =========

    def encode_obs(self, nobs, B, To):
        """Encode observations through robomimic obs encoder.
        Returns (B, To, obs_feature_dim)."""
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def init_latents(self, B, T, device):
        """Initialize y and z from truncated normal (std=1, truncation=2)."""
        y = torch.empty(B, T, self.hidden_dim, device=device)
        z = torch.empty(B, T, self.hidden_dim, device=device)
        nn.init.trunc_normal_(y, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z, std=1.0, a=-2.0, b=2.0)
        return y, z

    def encode_to_latent(self, obs_features, actions):
        """CVAE encoder: encode actions conditioned on obs into latent.
        Only used when use_cvae=True. Returns (mu, logvar, z_style)."""
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

    # ========= GRAM recursion =========

    def latent_recursion(self, obs_tokens, y, z, n):
        """Inner GRAM recursion: n steps of z refinement + 1 y update.

        z updates see obs via cross-attention; y update does NOT (asymmetry
        is load-bearing — forces y to integrate obs info through z).
        """
        sigma = self.log_sigma.exp()
        for _ in range(n):
            # z update: conditioned on obs, y, z
            u = self.block(z + y, memory=obs_tokens,
                           freqs_cis=self.freqs_cis, causal_mask=self.causal_mask)
            # Stochastic guidance: residual noise injection
            eps = sigma * torch.randn_like(u)
            z = u + eps
        # y update: conditioned on z and y, NOT obs
        y = self.block(y + z, memory=None,
                       freqs_cis=self.freqs_cis, causal_mask=self.causal_mask)
        return y, z

    def deep_recursion(self, obs_tokens, y, z, n, T):
        """Deep recursion: T calls to latent_recursion, gradients only on last."""
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self.latent_recursion(obs_tokens, y, z, n)
                y = y.detach()
                z = z.detach()
        # Last recursion with gradients
        y, z = self.latent_recursion(obs_tokens, y, z, n)
        return y, z

    # ========= inference =========

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        device = self.device

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Initialize latents
        y, z = self.init_latents(B, self.horizon, device)
        if self.use_cvae:
            z_style = torch.randn(B, self.latent_dim, device=device)
            y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                B, self.horizon, -1).clone()

        # Supervision loop (no loss, inference only)
        for step in range(self.inference_n_sup):
            y, z = self.deep_recursion(obs_tokens, y, z,
                                       self.n_recursion, self.T_recursion)
            # Optional Q-halting
            if self.inference_use_q_halting and self.q_head is not None and step >= 1:
                q_logit = self.q_head(y.mean(dim=1))
                if torch.sigmoid(q_logit).mean() > 0.5:
                    break

        naction_pred = self.output_head(y)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    def predict_action_samples(
        self, obs_dict: Dict[str, torch.Tensor], n_samples: int = 16
    ) -> torch.Tensor:
        """Draw n_samples diverse action chunks via different noise sequences.
        Returns: (n_samples, B, n_action_steps, action_dim)."""
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps
        device = self.device

        # Encode obs once (shared across all samples)
        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        all_actions = []
        for _ in range(n_samples):
            y, z = self.init_latents(B, self.horizon, device)
            if self.use_cvae:
                z_style = torch.randn(B, self.latent_dim, device=device)
                y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                    B, self.horizon, -1).clone()

            for step in range(self.inference_n_sup):
                y, z = self.deep_recursion(obs_tokens, y, z,
                                           self.n_recursion, self.T_recursion)
                if self.inference_use_q_halting and self.q_head is not None and step >= 1:
                    q_logit = self.q_head(y.mean(dim=1))
                    if torch.sigmoid(q_logit).mean() > 0.5:
                        break

            naction_pred = self.output_head(y)
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            start = To - 1
            end = start + self.n_action_steps
            all_actions.append(action_pred[:, start:end])

        return torch.stack(all_actions, dim=0)

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
        Gradients do NOT flow across supervision steps (detached).
        Returns a float (total loss, already backpropagated).
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        # Encode obs (shared, computed once)
        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Initialize latents
        y, z = self.init_latents(B, self.horizon, nactions.device)

        # Optional CVAE initialization
        cvae_kl = None
        if self.use_cvae:
            mu, logvar, z_style = self.encode_to_latent(
                obs_features.detach(), nactions)
            y = self.latent_out_proj(z_style).unsqueeze(1).expand(
                B, self.horizon, -1).clone()
            cvae_kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = 0.0

        for sup_step in range(self.N_sup):
            # Deep recursion (gradients only on last inner recursion)
            y, z = self.deep_recursion(
                obs_tokens, y, z, self.n_recursion, self.T_recursion)

            # Predict actions
            action_pred = self.output_head(y)

            # L2 action loss
            l2_loss = F.mse_loss(action_pred, nactions)

            # Q-head loss
            q_loss_val = torch.tensor(0.0, device=nactions.device)
            if self.use_q_head and self.q_head is not None:
                q_logit = self.q_head(y.mean(dim=1))
                per_sample_mse = F.mse_loss(
                    action_pred, nactions, reduction='none').mean(dim=(1, 2))
                success_target = (per_sample_mse < self.success_threshold).float()
                q_loss_val = self.q_loss_weight * F.binary_cross_entropy_with_logits(
                    q_logit.squeeze(-1), success_target)

            # KL regularizer on sigma (recomputed per step for fresh graph)
            sigma_sq = torch.exp(2 * self.log_sigma)
            kl_sigma = self.kl_weight_gram * (sigma_sq - 2 * self.log_sigma - 1)

            # Assemble step loss (normalized by N_sup)
            step_loss = (l2_loss + q_loss_val + kl_sigma) / self.N_sup

            # Add CVAE KL only on first step (graph shared with y init)
            if self.use_cvae and cvae_kl is not None and sup_step == 0:
                step_loss = step_loss + self.kl_weight * cvae_kl

            # Backward only when gradients are enabled (skip during validation)
            if torch.is_grad_enabled():
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)
            total_loss += step_loss.item()

            # Detach latents for next supervision step
            y = y.detach()
            z = z.detach()

        return total_loss
