from typing import Dict, Tuple
import math
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
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmoc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class ACTHybridImagePolicy(BaseImagePolicy):
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
            latent_dim=32,
            kl_weight=10.0,
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

        # CVAE encoder (training only): encodes actions into latent z
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_emb))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_action_proj = nn.Linear(action_dim, n_emb)
        self.encoder_obs_proj = nn.Linear(obs_feature_dim, n_emb)
        # positional embedding for encoder: [CLS] + horizon action tokens
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + horizon, n_emb))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layer)
        self.latent_proj = nn.Linear(n_emb, latent_dim * 2)  # mu and logvar

        # Decoder: predicts actions from obs features + latent z
        self.latent_out_proj = nn.Linear(latent_dim, n_emb)
        self.decoder_obs_proj = nn.Linear(obs_feature_dim, n_emb)
        # learned action queries for decoder
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.normal_(self.action_queries, std=0.02)
        # positional embedding for decoder memory: [z_token] + n_obs_steps obs tokens
        self.decoder_memory_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_obs_steps, n_emb))
        nn.init.normal_(self.decoder_memory_pos_embed, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(n_emb, action_dim)

        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def encode_obs(self, nobs, B, To):
        """Encode observations through robomimic obs encoder.
        Returns (B, To, obs_feature_dim)."""
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(B, To, -1)

    def encode_to_latent(self, obs_features, actions):
        """CVAE encoder: encode actions conditioned on obs into latent z.
        Returns (mu, logvar, z) each of shape (B, latent_dim)."""
        B = actions.shape[0]
        # project actions: (B, T, Da) -> (B, T, n_emb)
        action_tokens = self.encoder_action_proj(actions)
        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        # encoder input: [CLS, action_1, ..., action_T]
        encoder_input = torch.cat([cls, action_tokens], dim=1)
        encoder_input = encoder_input + self.encoder_pos_embed
        # obs features as additional context via cross-attention
        # For simplicity, concatenate obs conditioning:
        # We use the mean obs feature as additive bias to CLS
        obs_cond = self.encoder_obs_proj(obs_features.mean(dim=1, keepdim=True))
        encoder_input[:, :1] = encoder_input[:, :1] + obs_cond

        encoder_output = self.encoder(encoder_input)
        cls_output = encoder_output[:, 0]  # (B, n_emb)
        latent_params = self.latent_proj(cls_output)  # (B, latent_dim*2)
        mu, logvar = latent_params.chunk(2, dim=-1)
        # reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return mu, logvar, z

    def decode_actions(self, obs_features, z):
        """Decoder: predict action sequence from obs features and latent z.
        Returns (B, T, Da)."""
        B = obs_features.shape[0]
        # project z to token: (B, 1, n_emb)
        z_token = self.latent_out_proj(z).unsqueeze(1)
        # project obs features: (B, To, n_emb)
        obs_tokens = self.decoder_obs_proj(obs_features)
        # decoder memory: [z_token, obs_1, ..., obs_To]
        memory = torch.cat([z_token, obs_tokens], dim=1)
        memory = memory + self.decoder_memory_pos_embed

        # action queries: (B, T, n_emb)
        queries = self.action_queries.expand(B, -1, -1)

        decoder_output = self.decoder(queries, memory)
        actions = self.action_head(decoder_output)  # (B, T, Da)
        return actions

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        # encode observations
        obs_features = self.encode_obs(nobs, B, To)

        # use prior: z = 0 (mean of standard normal)
        z = torch.zeros(B, self.latent_dim, device=self.device, dtype=self.dtype)

        # decode actions
        naction_pred = self.decode_actions(obs_features, z)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # select action steps to execute
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
        # separate weight decay for transformer vs obs encoder
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'obs_encoder' in name:
                continue  # handled separately
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

        # encode observations
        obs_features = self.encode_obs(nobs, B, To)

        # CVAE encode: get latent from actions
        mu, logvar, z = self.encode_to_latent(obs_features.detach(), nactions)

        # decode actions
        pred_actions = self.decode_actions(obs_features, z)

        # L1 reconstruction loss
        l1_loss = F.l1_loss(pred_actions, nactions)

        # KL divergence loss: KL(q(z|a,o) || N(0,I))
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp())

        loss = l1_loss + self.kl_weight * kl_loss
        return loss
