"""
DET-GRAM: Deterministic GRAM-ACT with ACT-style CVAE latent initialization.

Removes all within-recursion stochasticity (prior/posterior heads, KL balancing,
sigma sampling). Instead, a CVAE encoder (identical to ACT) provides a single
stochastic latent z that initializes y before the recursive loop:

  Training: z ~ q(z | a_gt, obs)  via TransformerEncoder on [CLS, action_tokens]
  Inference: z = 0  (prior mean of N(0, I))

The recursive structure (K inner z-loop, n outer steps, N_sup deep supervision,
truncated warm-up) is preserved unchanged. y update: y = block(y+z, obs).

Loss = MSE + kl_weight * KL(q || N(0,I))
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


class DETGRAMHybridImagePolicy(BaseImagePolicy):
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
            # CVAE encoder
            latent_dim=32,
            kl_weight=10.0,
            n_encoder_layers=2,
            encoder_dropout=0.1,
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

        # Precompute RoPE frequencies (bidirectional self-attn, no causal mask)
        head_dim = hidden_dim // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=horizon)
        self.register_buffer('freqs_cis', freqs_cis)

        # ========= Fixed initial latent states (buffers, not trained) =========
        y_init = torch.empty(1, horizon, hidden_dim)
        z_init = torch.empty(1, horizon, hidden_dim)
        nn.init.trunc_normal_(y_init, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z_init, std=1.0, a=-2.0, b=2.0)
        self.register_buffer('y_init', y_init)
        self.register_buffer('z_init', z_init)

        # ========= CVAE encoder (ACT-style) =========
        # Encodes ground truth actions into a latent z that initializes y.
        # At inference z=0 (prior mean); at training z ~ q(z|a_gt, obs).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_obs_proj = nn.Linear(obs_feature_dim, hidden_dim)
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + horizon, hidden_dim))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=4 * hidden_dim, dropout=encoder_dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)   # → (mu, logvar)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)   # z → y_init offset

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
        n_gram = sum(p.numel() for p in self.block.parameters())
        print(f"DETGRAMHybridImagePolicy: {n_params:,} total params, "
              f"{n_gram:,} GRAM block params")

    # ========= helpers =========

    def encode_obs(self, nobs, B, To):
        """Encode observations through robomimic obs encoder → (B, To, obs_feature_dim)."""
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def init_latents(self, B):
        """Return fixed initial latents expanded to batch size B."""
        return self.y_init.expand(B, -1, -1), self.z_init.expand(B, -1, -1)

    def encode_to_latent(self, obs_features, actions):
        """CVAE encoder: encode actions conditioned on obs → (mu, logvar, z).

        Identical to ACT: a TransformerEncoder on [CLS, action_tokens] with obs
        mean added to the CLS token as conditioning context.

        Returns mu, logvar, z each of shape (B, latent_dim).
        """
        B = actions.shape[0]
        action_tokens = self.encoder_action_proj(actions)           # (B, T, D)
        cls = self.cls_token.expand(B, -1, -1)                      # (B, 1, D)
        encoder_input = torch.cat([cls, action_tokens], dim=1)      # (B, 1+T, D)
        encoder_input = encoder_input + self.encoder_pos_embed
        obs_cond = self.encoder_obs_proj(obs_features.mean(dim=1, keepdim=True))
        encoder_input[:, :1] = encoder_input[:, :1] + obs_cond
        encoder_output = self.cvae_encoder(encoder_input)
        cls_output = encoder_output[:, 0]                           # (B, D)
        mu, logvar = self.latent_proj(cls_output).chunk(2, dim=-1)  # (B, latent_dim)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z

    # ========= GRAM recursion (deterministic) =========

    def latent_recursion(self, obs_tokens, y, z, n, K):
        """Deterministic GRAM recursion: y = block(y + z, obs), no sampling.

        Truncated warm-up: first n-1 outer steps under torch.no_grad();
        final outer step runs with gradients. Structure per outer step:
          K deterministic z-updates: z = block(z + y, obs)
          1 deterministic y-update:  y = block(y + z, obs)

        Returns: (y, z)
        """
        with torch.no_grad():
            for _ in range(n - 1):
                for _ in range(K):
                    z = self.block(z + y, memory=obs_tokens, freqs_cis=self.freqs_cis)
                y = self.block(y + z, memory=obs_tokens, freqs_cis=self.freqs_cis)

        for _ in range(K):
            z = self.block(z + y, memory=obs_tokens, freqs_cis=self.freqs_cis)
        y = self.block(y + z, memory=obs_tokens, freqs_cis=self.freqs_cis)

        return y, z

    # ========= inference =========

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # Prior: z = 0 (mean of N(0,I)), projected to y initialization
        z_prior = torch.zeros(B, self.latent_dim, device=self.device, dtype=obs_tokens.dtype)
        y_offset = self.latent_out_proj(z_prior).unsqueeze(1)   # (B, 1, D)
        y_base, z = self.init_latents(B)
        y = (y_base + y_offset).clone()

        for _ in range(self.inference_n_sup):
            y, z = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion)

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
        """DET-GRAM training with CVAE initialization and deep supervision.

        CVAE encodes ground truth actions → z ~ q(z|a_gt, obs), which is
        projected and added to y_init before the recursive loop.
        KL(q || N(0,I)) is added once to the first supervision step.

        Returns a dict {'loss': float, 'mse_loss': float, 'kl_loss': float}.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        # CVAE: encode actions → z, initialize y with z projection
        mu, logvar, z_latent = self.encode_to_latent(obs_features.detach(), nactions)
        y_offset = self.latent_out_proj(z_latent).unsqueeze(1)   # (B, 1, D)
        y_base, z = self.init_latents(B)
        y = (y_base + y_offset).clone()

        # KL(q(z|a,o) || N(0,I)) — computed once, added to first step loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = 0.0
        total_mse = 0.0

        for sup_step in range(self.N_sup):
            y, z = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion)

            action_pred = self.output_head(y)
            mse = F.mse_loss(action_pred, nactions)
            step_loss = mse / self.N_sup

            # Add KL on the first step (graph is alive; y_offset shares the graph)
            if sup_step == 0:
                step_loss = step_loss + self.kl_weight * kl_loss / self.N_sup

            if torch.is_grad_enabled():
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)

            total_loss += step_loss.item()
            total_mse += mse.item()

            y = y.detach()
            z = z.detach()

        return {
            'loss': total_loss,
            'mse_loss': total_mse / self.N_sup,
            'kl_loss': kl_loss.item(),
        }
