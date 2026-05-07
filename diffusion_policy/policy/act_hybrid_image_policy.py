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
        obs_config = {'low_dim': [], 'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        rgb_keys = []
        lowdim_keys = []
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
                rgb_keys.append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

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

        # Compute spatial token count from backbone output shape
        with torch.no_grad():
            _backbone = obs_encoder.obs_nets[rgb_keys[0]].nets[0]
            _h, _w = (crop_shape[0], crop_shape[1]) if crop_shape is not None else obs_key_shapes[rgb_keys[0]][1:]
            _feat = _backbone(torch.zeros(1, 3, _h, _w))
            backbone_dim = _feat.shape[1]      # 512
            n_spatial = _feat.shape[2] * _feat.shape[3]  # e.g. 9

        n_tokens_per_step = len(rgb_keys) * n_spatial + len(lowdim_keys)
        n_obs_tokens = n_obs_steps * n_tokens_per_step

        # Spatial projection: backbone features → n_emb
        self.spatial_proj = nn.Linear(backbone_dim, n_emb, bias=False)
        # Low-dim projections: one per key
        self.lowdim_projs = nn.ModuleDict({
            key: nn.Linear(obs_key_shapes[key][0], n_emb, bias=False)
            for key in lowdim_keys
        })

        # CVAE encoder: [CLS, action_tokens] → z
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_emb))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_action_proj = nn.Linear(action_dim, n_emb)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + horizon, n_emb))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.latent_proj = nn.Linear(n_emb, latent_dim * 2)

        # Decoder
        self.latent_out_proj = nn.Linear(latent_dim, n_emb)
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.normal_(self.action_queries, std=0.02)
        # positional embedding for memory: [z_token] + n_obs_tokens obs tokens
        self.decoder_memory_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_obs_tokens, n_emb))
        nn.init.normal_(self.decoder_memory_pos_embed, std=0.02)
        # memory encoder: contextualizes [z_token, obs_tokens] before cross-attention
        mem_enc_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=True)
        self.mem_encoder = nn.TransformerEncoder(mem_enc_layer, num_layers=n_layer)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(n_emb, action_dim)

        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_obs_tokens = n_obs_tokens
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

    def encode_obs_tokens(self, nobs, B, To):
        """Encode observations to spatial tokens: (B, To*n_tokens_per_step, n_emb).

        For each rgb key: crop → backbone → (B*To, h*w, backbone_dim) → spatial_proj
        For each low_dim key: (B*To, dim) → lowdim_proj → (B*To, 1, n_emb)
        Returns: (B, n_obs_tokens, n_emb)
        """
        all_tokens = []

        for key in self.rgb_keys:
            imgs = nobs[key][:, :To].reshape(B * To, *nobs[key].shape[2:])
            crop_rand = self.obs_encoder.obs_randomizers[key][0]
            imgs = crop_rand.forward_in(imgs)
            backbone = self.obs_encoder.obs_nets[key].nets[0]
            feat = backbone(imgs)                                     # (B*To, C, h, w)
            BTo, C, h, w = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(BTo, h * w, C)  # (B*To, h*w, C)
            feat = self.spatial_proj(feat)                            # (B*To, h*w, n_emb)
            all_tokens.append(feat)

        for key in self.lowdim_keys:
            ld = nobs[key][:, :To].reshape(B * To, -1)
            tok = self.lowdim_projs[key](ld).unsqueeze(1)            # (B*To, 1, n_emb)
            all_tokens.append(tok)

        tokens = torch.cat(all_tokens, dim=1)                        # (B*To, n_per_step, n_emb)
        return tokens.reshape(B, To * tokens.shape[1], tokens.shape[2])  # (B, n_obs_tokens, n_emb)

    def encode_to_latent(self, actions):
        """CVAE encoder: [CLS, action_tokens] → (mu, logvar, z). Actions only."""
        B = actions.shape[0]
        action_tokens = self.encoder_action_proj(actions)
        cls = self.cls_token.expand(B, -1, -1)
        encoder_input = torch.cat([cls, action_tokens], dim=1) + self.encoder_pos_embed
        encoder_output = self.encoder(encoder_input)
        cls_output = encoder_output[:, 0]
        mu, logvar = self.latent_proj(cls_output).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return mu, logvar, z

    def decode_actions(self, obs_tokens, z):
        """Decoder: predict action sequence from obs tokens and latent z.

        obs_tokens: (B, n_obs_tokens, n_emb) — already projected spatial tokens
        Returns: (B, T, Da)
        """
        B = obs_tokens.shape[0]
        z_token = self.latent_out_proj(z).unsqueeze(1)               # (B, 1, n_emb)
        memory = torch.cat([z_token, obs_tokens], dim=1)             # (B, 1+n_obs_tokens, n_emb)
        memory = memory + self.decoder_memory_pos_embed
        memory = self.mem_encoder(memory)

        queries = self.action_queries.expand(B, -1, -1)
        decoder_output = self.decoder(queries, memory)
        return self.action_head(decoder_output)                       # (B, T, Da)

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps

        obs_tokens = self.encode_obs_tokens(nobs, B, To)
        z = torch.zeros(B, self.latent_dim, device=self.device, dtype=self.dtype)
        naction_pred = self.decode_actions(obs_tokens, z)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}

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
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps

        obs_tokens = self.encode_obs_tokens(nobs, B, To)
        mu, logvar, z = self.encode_to_latent(nactions)
        pred_actions = self.decode_actions(obs_tokens, z)

        l1_loss = F.l1_loss(pred_actions, nactions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return l1_loss + self.kl_weight * kl_loss
