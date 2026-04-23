"""
DET-GRAM: Deterministic ablation of GRAM-ACT v4.

Identical to GRAMHybridImagePolicy except all stochasticity and variational
components are removed:

- No prior/posterior heads (μ_p, σ_p, μ_q, σ_q)
- No action embedding for posterior conditioning
- No KL divergence loss
- y update: y = u  (block output replaces y directly, no sampling)
- Loss: MSE only

The recursive structure (K inner z-loop, n outer steps, N_sup deep supervision,
truncated warm-up) is preserved unchanged to isolate the effect of stochasticity.
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

        y, z = self.init_latents(B)
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
        """DET-GRAM training with deep supervision (MSE only, no KL).

        Runs N_sup supervision steps, each with its own backward pass.
        Gradients do NOT flow across supervision steps (latents detached between steps).

        Returns a dict {'loss': float, 'mse_loss': float}.
        """
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)
        obs_tokens = self.obs_proj(obs_features)

        y, z = self.init_latents(B)

        total_loss = 0.0

        for sup_step in range(self.N_sup):
            y, z = self.latent_recursion(
                obs_tokens, y, z, self.n_recursion, self.k_recursion)

            action_pred = self.output_head(y)
            mse = F.mse_loss(action_pred, nactions)
            step_loss = mse / self.N_sup

            if torch.is_grad_enabled():
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)

            total_loss += step_loss.item()

            y = y.detach()
            z = z.detach()

        return {
            'loss': total_loss,
            'mse_loss': total_loss,
        }
