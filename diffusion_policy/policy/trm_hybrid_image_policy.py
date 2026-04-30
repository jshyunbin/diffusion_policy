"""
TRM (Tiny Recursive Model) hybrid image policy.

Originates from DETGRAMHybridImagePolicy. Everything is identical except
`latent_recursion`, which uses a TRM-style memory construction:

  memory = [x, z_cvae_token, obs_tokens]   where x = z_gram + y

The only difference between the two update types is an attention mask:
  - z_gram (low) update: no mask — cross-attends to all memory tokens
  - y (high) update: obs_mask — obs_tokens positions set to -inf, so y
    cannot attend to raw observations during its update

This mirrors the ACTRM pattern (z_H never sees raw obs) but via concat +
masking rather than the additive conditioning used in the original ACTRM.
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


class TRMHybridImagePolicy(BaseImagePolicy):
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
            # decoder architecture
            hidden_dim=512,
            n_decoder_layers=2,
            n_heads=8,
            ffn_expansion=4,
            # recursion
            n_recursion=3,
            k_recursion=4,
            N_sup=16,
            # CVAE encoder
            latent_dim=32,
            kl_weight=10.0,
            n_encoder_layers=2,
            encoder_dropout=0.1,
            # inference
            inference_n_sup=16,
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

        # ========= Observation encoder (same as DET-GRAM) =========
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

        # ========= Shared recursive block =========
        self.block = GRAMBlock(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=n_decoder_layers, ffn_expansion=ffn_expansion)

        head_dim = hidden_dim // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=horizon)
        self.register_buffer('freqs_cis', freqs_cis)

        # ========= Fixed initial latent states =========
        y_init = torch.empty(1, horizon, hidden_dim)
        z_init = torch.empty(1, horizon, hidden_dim)
        nn.init.trunc_normal_(y_init, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z_init, std=1.0, a=-2.0, b=2.0)
        self.register_buffer('y_init', y_init)
        self.register_buffer('z_init', z_init)

        # ========= Obs mask for high-latent (y) update =========
        # memory layout: [x (horizon), z_cvae_token (1), obs_tokens (n_obs_steps)]
        # y cannot attend to the last n_obs_steps positions.
        mem_len = horizon + 1 + n_obs_steps
        obs_mask = torch.zeros(1, 1, 1, mem_len)
        obs_mask[:, :, :, horizon + 1:] = float('-inf')
        self.register_buffer('obs_mask', obs_mask)

        # ========= CVAE encoder (same as DET-GRAM, actions only) =========
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_embed, std=0.02)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + horizon, hidden_dim))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=4 * hidden_dim, dropout=encoder_dropout,
            activation='relu', batch_first=True, norm_first=False)
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # ========= Output head =========
        self.output_head = nn.Linear(hidden_dim, action_dim, bias=False)

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
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.inference_n_sup = inference_n_sup

        n_params = sum(p.numel() for p in self.parameters())
        n_block = sum(p.numel() for p in self.block.parameters())
        print(f"TRMHybridImagePolicy: {n_params:,} total params, "
              f"{n_block:,} block params")

    # ========= helpers =========

    def encode_obs(self, nobs, B, To):
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def init_latents(self, B):
        return self.y_init.expand(B, -1, -1), self.z_init.expand(B, -1, -1)

    def encode_to_latent(self, actions):
        """CVAE encoder: [CLS, action_tokens] → (mu, logvar, z). Actions only."""
        B = actions.shape[0]
        action_tokens = self.encoder_action_proj(actions)
        cls = self.cls_embed.expand(B, -1, -1)
        encoder_input = torch.cat([cls, action_tokens], dim=1)
        encoder_input = encoder_input + self.encoder_pos_embed
        encoder_output = self.cvae_encoder(encoder_input)
        cls_output = encoder_output[:, 0]
        mu, logvar = self.latent_proj(cls_output).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z

    # ========= TRM recursion =========

    def latent_recursion(self, obs_tokens, z_cvae_token, y, z_gram, n, K):
        """TRM-style latent recursion with masked high-latent update.

        Memory for both updates: [x, z_cvae_token, obs_tokens]
          - x = z_gram + y  (commutative, same value for both updates)
        Difference: y update applies obs_mask (blocks last n_obs_steps positions).

        Truncated warm-up: first n-1 outer steps under no_grad.
        """
        def _step(y, z_gram):
            x = z_gram + y
            memory = torch.cat([x, z_cvae_token, obs_tokens], dim=1)

            # K low-latent updates: no mask
            for _ in range(K):
                z_gram = self.block(x, memory=memory, freqs_cis=self.freqs_cis,
                                    cross_attn_mask=None)
                x = z_gram + y
                memory = torch.cat([x, z_cvae_token, obs_tokens], dim=1)

            # 1 high-latent update: obs masked
            y = self.block(x, memory=memory, freqs_cis=self.freqs_cis,
                           cross_attn_mask=self.obs_mask)
            return y, z_gram

        with torch.no_grad():
            for _ in range(n - 1):
                y, z_gram = _step(y, z_gram)

        y, z_gram = _step(y, z_gram)
        return y, z_gram

    # ========= inference =========

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Prior: z = 0
        z_prior = torch.zeros(B, self.latent_dim, device=self.device, dtype=obs_tokens.dtype)
        z_cvae_token = self.latent_out_proj(z_prior).unsqueeze(1)  # (B, 1, D)

        y, z_gram = self.init_latents(B)
        for _ in range(self.inference_n_sup):
            y, z_gram = self.latent_recursion(
                obs_tokens, z_cvae_token, y, z_gram, self.n_recursion, self.k_recursion)

        naction_pred = self.output_head(y)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],
            'action_pred': action_pred,
        }

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
        """TRM training with CVAE z-token memory conditioning and deep supervision.

        Returns a dict {'loss': float, 'mse_loss': float, 'kl_loss': float}.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        mu, logvar, z_latent = self.encode_to_latent(nactions)
        z_cvae_token = self.latent_out_proj(z_latent).unsqueeze(1)  # (B, 1, D)

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        y, z_gram = self.init_latents(B)
        total_loss = 0.0
        total_mse = 0.0

        for sup_step in range(self.N_sup):
            y, z_gram = self.latent_recursion(
                obs_tokens, z_cvae_token, y, z_gram, self.n_recursion, self.k_recursion)

            action_pred = self.output_head(y)
            mse = F.mse_loss(action_pred, nactions)
            step_loss = mse / self.N_sup

            if sup_step == 0:
                step_loss = step_loss + self.kl_weight * kl_loss / self.N_sup

            if torch.is_grad_enabled():
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)

            total_loss += step_loss.item()
            total_mse += mse.item()

            y = y.detach()
            z_gram = z_gram.detach()

        return {
            'loss': total_loss,
            'mse_loss': total_mse / self.N_sup,
            'kl_loss': kl_loss.item(),
        }
