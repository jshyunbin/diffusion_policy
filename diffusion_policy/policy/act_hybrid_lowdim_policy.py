from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class ACTHybridLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            obs_dim,
            action_dim,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            # arch
            n_layer=4,
            n_cvae_enc_layers=4,
            n_mem_enc_layers=4,
            n_head=8,
            n_emb=256,
            p_drop=0.1,
            latent_dim=32,
            kl_weight=10.0,
            ):
        super().__init__()

        n_obs_tokens = n_obs_steps

        # Obs projection: flat obs → n_emb per timestep
        self.obs_proj = nn.Linear(obs_dim, n_emb)

        # CVAE encoder: [CLS, action_tokens] → z
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_emb))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_action_proj = nn.Linear(action_dim, n_emb)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + horizon, n_emb))
        nn.init.normal_(self.encoder_pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cvae_enc_layers)
        self.latent_proj = nn.Linear(n_emb, latent_dim * 2)

        # Decoder
        self.latent_out_proj = nn.Linear(latent_dim, n_emb)
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, n_emb))
        nn.init.normal_(self.action_queries, std=0.02)
        self.decoder_memory_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_obs_tokens, n_emb))
        nn.init.normal_(self.decoder_memory_pos_embed, std=0.02)
        mem_enc_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=False)
        self.mem_encoder = nn.TransformerEncoder(mem_enc_layer, num_layers=n_mem_enc_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4*n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.action_head = nn.Linear(n_emb, action_dim)

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_obs_tokens = n_obs_tokens
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ACTHybridLowdimPolicy: {n_params:,} parameters")

    def encode_obs_tokens(self, nobs, B, To):
        """(B, To, obs_dim) → (B, To, n_emb)"""
        return self.obs_proj(nobs[:, :To])

    def encode_to_latent(self, actions):
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
        B = obs_tokens.shape[0]
        z_token = self.latent_out_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, obs_tokens], dim=1)
        memory = memory + self.decoder_memory_pos_embed
        memory = self.mem_encoder(memory)
        queries = self.action_queries.expand(B, -1, -1)
        decoder_output = self.decoder(queries, memory)
        return self.action_head(decoder_output)

    def predict_action(self, obs_dict: torch.Tensor) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer['obs'].normalize(obs_dict)
        B = nobs.shape[0]
        To = self.n_obs_steps
        obs_tokens = self.encode_obs_tokens(nobs, B, To)
        z = torch.zeros(B, self.latent_dim, device=self.device, dtype=nobs.dtype)
        naction_pred = self.decode_actions(obs_tokens, z)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        return {'action': action_pred[:, start:end], 'action_pred': action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self,
            transformer_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": transformer_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        nobs = self.normalizer['obs'].normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]
        To = self.n_obs_steps
        obs_tokens = self.encode_obs_tokens(nobs, B, To)
        mu, logvar, z = self.encode_to_latent(nactions)
        pred_actions = self.decode_actions(obs_tokens, z)
        l1_loss = F.l1_loss(pred_actions, nactions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = l1_loss + self.kl_weight * kl_loss
        return {
            'loss': loss,
            'l1_loss': l1_loss.item(),
            'kl_loss': kl_loss.item(),
        }
