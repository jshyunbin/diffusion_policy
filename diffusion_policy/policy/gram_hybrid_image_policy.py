from typing import Dict, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmoc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


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
            # arch
            n_layer=4,
            n_head=8,
            n_emb=256,
            p_drop=0.1,
            # GRAM params
            n_reasoning_steps=8,
            n_reasoning_steps_train=16,
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

        # get raw robomimic config for obs encoder
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

        # ========= GRAM reasoning module =========

        # Learnable initial state for reasoning
        self.z_init = nn.Parameter(torch.zeros(n_emb))
        nn.init.trunc_normal_(self.z_init, std=0.02)

        # Obs projection for decoder memory
        self.decoder_obs_proj = nn.Linear(obs_feature_dim, n_emb)

        # Reasoning module: transformer encoder layers that iteratively refine z
        # conditioned on obs tokens via input injection (following ACTRM pattern)
        reasoning_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True,
            norm_first=True)
        self.reasoning_module = nn.TransformerEncoder(
            reasoning_layer, num_layers=n_layer)

        # ========= Decoder =========

        # Learned action queries
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.normal_(self.action_queries, std=0.02)

        # Decoder: cross-attention from action queries to refined reasoning state
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layer)

        self.action_head = nn.Linear(n_emb, action_dim)

        # Store modules and params
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_emb = n_emb
        self.n_reasoning_steps = n_reasoning_steps
        self.n_reasoning_steps_train = n_reasoning_steps_train

    def encode_obs(self, nobs, B, To):
        """Encode observations through robomimic obs encoder.
        Returns (B, To, obs_feature_dim)."""
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def gram_reasoning(self, obs_features, n_steps):
        """GRAM recursive reasoning on the decoder side.

        Iteratively refines a state representation z conditioned on
        observation features. Following the ACTRM pattern: gradients
        are only kept for the last iteration during training.

        Args:
            obs_features: (B, To, obs_feature_dim)
            n_steps: number of reasoning iterations

        Returns:
            z: refined state (B, To, n_emb) — used as decoder memory
        """
        B = obs_features.shape[0]
        To = obs_features.shape[1]

        # Project obs features to embedding space
        obs_tokens = self.decoder_obs_proj(obs_features)  # (B, To, n_emb)

        # Initialize reasoning state from learnable buffer
        z = self.z_init.unsqueeze(0).unsqueeze(0).expand(B, To, -1)

        # Iterative refinement with detachment (prevent BPTT explosion)
        # All but the last step run without gradients
        if n_steps > 1:
            with torch.no_grad():
                for _ in range(n_steps - 1):
                    # GRAM placeholder: input injection + self-attention refinement
                    z_input = z + obs_tokens
                    z = self.reasoning_module(z_input)
                    z = z.detach()

        # Last step with gradients
        z_input = z + obs_tokens
        z = self.reasoning_module(z_input)

        return z

    def decode_actions(self, memory):
        """Decode action sequence from refined reasoning state.

        Args:
            memory: (B, To, n_emb) — refined state from GRAM reasoning

        Returns:
            actions: (B, horizon, action_dim)
        """
        B = memory.shape[0]
        queries = self.action_queries.expand(B, -1, -1)
        decoder_output = self.decoder(queries, memory)
        return self.action_head(decoder_output)

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)

        # GRAM reasoning with fixed steps at inference
        memory = self.gram_reasoning(obs_features, self.n_reasoning_steps)

        # Decode actions
        naction_pred = self.decode_actions(memory)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred
        }

    # ========= training ============
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
            {"params": self.obs_encoder.parameters(), "weight_decay": obs_encoder_weight_decay},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        obs_features = self.encode_obs(nobs, B, To)

        # Random reasoning steps during training (1 to n_reasoning_steps_train)
        n_steps = random.randint(1, self.n_reasoning_steps_train)

        # GRAM reasoning
        memory = self.gram_reasoning(obs_features, n_steps)

        # Decode actions
        pred_actions = self.decode_actions(memory)

        # L1 reconstruction loss only (no KL — GRAM handles multimodality)
        loss = F.l1_loss(pred_actions, nactions)
        return loss
