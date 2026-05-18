from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.gram.gram_modules import RActionBlock, precompute_freqs_cis


class HRAMHybridLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            obs_dim,
            action_dim,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            # decoder architecture
            hidden_dim=256,
            n_decoder_layers=2,
            n_heads=8,
            ffn_expansion=4,
            # recursion
            n_recursion=3,
            k_recursion=4,
            N_sup=8,
            use_z_L=True,
            # CVAE encoder
            use_cvae=True,
            latent_dim=32,
            kl_weight=10.0,
            n_encoder_layers=4,
            n_mem_enc_layers=2,
            encoder_dropout=0.1,
            # inference
            inference_n_sup=8,
            ):
        super().__init__()

        n_obs_tokens = n_obs_steps

        # Obs projection: flat obs → hidden_dim per timestep
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        # ========= Shared recursive block =========
        self.block = RActionBlock(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=n_decoder_layers, ffn_expansion=ffn_expansion)

        head_dim = hidden_dim // n_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=horizon)
        self.register_buffer('freqs_cis', freqs_cis)
        n_z_tokens = n_obs_tokens + (1 if use_cvae else 0)
        obs_freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=n_z_tokens)
        self.register_buffer('obs_freqs_cis', obs_freqs_cis)

        # ========= Fixed initial latent states =========
        y_init = torch.empty(1, horizon, hidden_dim)
        z_init = torch.empty(1, n_z_tokens, hidden_dim)
        nn.init.trunc_normal_(y_init, std=1.0, a=-2.0, b=2.0)
        nn.init.trunc_normal_(z_init, std=1.0, a=-2.0, b=2.0)
        self.register_buffer('y_init', y_init)
        self.register_buffer('z_init', z_init)

        # ========= Context encoder =========
        self.context_encoder = RActionBlock(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=n_mem_enc_layers, ffn_expansion=ffn_expansion
        )
        self.use_z_L = use_z_L
        self.use_cvae = use_cvae

        # ========= CVAE encoder =========
        if use_cvae:
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

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_obs_tokens = n_obs_tokens
        self.n_recursion = n_recursion
        self.k_recursion = k_recursion
        self.N_sup = N_sup
        self.latent_dim = latent_dim if use_cvae else None
        self.kl_weight = kl_weight
        self.inference_n_sup = inference_n_sup

        n_params = sum(p.numel() for p in self.parameters())
        n_block = sum(p.numel() for p in self.block.parameters())
        print(f"HRAMHybridLowdimPolicy: {n_params:,} total params, {n_block:,} block params")

    def encode_obs_tokens(self, nobs, B, To):
        """(B, To, obs_dim) → (B, To, hidden_dim)"""
        return self.obs_proj(nobs[:, :To])

    def init_latents(self, B):
        return self.y_init.expand(B, -1, -1), self.z_init.expand(B, -1, -1)

    def encode_to_latent(self, actions):
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

    def latent_recursion(self, obs_tokens, z_cvae_token, y, z_L, n, K):
        x = torch.cat([z_cvae_token, obs_tokens], dim=1) if self.use_cvae else obs_tokens

        def _step(y, z_L):
            for _ in range(K):
                z_L = self.context_encoder(
                    (z_L + x) if self.use_z_L else z_L,
                    memory=y, freqs_cis=self.obs_freqs_cis, cross_attn_mask=None)
            y = self.block(y, memory=z_L, freqs_cis=self.freqs_cis, cross_attn_mask=None)
            return y, z_L

        with torch.no_grad():
            for _ in range(n - 1):
                y, z_L = _step(y, z_L)

        y, z_L = _step(y, z_L)
        return y, z_L

    def predict_action(self, obs_dict: torch.Tensor) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer['obs'].normalize(obs_dict)
        B = nobs.shape[0]
        To = self.n_obs_steps
        obs_tokens = self.encode_obs_tokens(nobs, B, To)

        if self.use_cvae:
            z_prior = torch.zeros(B, self.latent_dim, device=self.device, dtype=obs_tokens.dtype)
            z_cvae_token = self.latent_out_proj(z_prior).unsqueeze(1)
        else:
            z_cvae_token = None

        y, z_L = self.init_latents(B)
        if not self.use_z_L:
            z_L = torch.cat([z_cvae_token, obs_tokens], dim=1) if self.use_cvae else obs_tokens
        for _ in range(self.inference_n_sup):
            y, z_L = self.latent_recursion(
                obs_tokens, z_cvae_token, y, z_L, self.n_recursion, self.k_recursion)

        naction_pred = self.output_head(y)
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

        if self.use_cvae:
            mu, logvar, z_latent = self.encode_to_latent(nactions)
            z_cvae_token = self.latent_out_proj(z_latent).unsqueeze(1)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            z_cvae_token = None
            kl_loss = None

        y, z_L = self.init_latents(B)
        if not self.use_z_L:
            z_L = torch.cat([z_cvae_token, obs_tokens], dim=1) if self.use_cvae else obs_tokens

        total_loss = 0.0
        total_mse = 0.0

        for sup_step in range(self.N_sup):
            y, z_L = self.latent_recursion(
                obs_tokens, z_cvae_token, y, z_L, self.n_recursion, self.k_recursion)

            action_pred = self.output_head(y)
            mse = F.mse_loss(action_pred, nactions)
            step_loss = mse / self.N_sup

            if sup_step == 0 and self.use_cvae:
                step_loss = step_loss + self.kl_weight * kl_loss / self.N_sup

            if torch.is_grad_enabled():
                is_last = (sup_step == self.N_sup - 1)
                step_loss.backward(retain_graph=not is_last)

            total_loss += step_loss.item()
            total_mse += mse.item()

            y = y.detach()
            z_L = z_L.detach()

        ret = {
            'loss': total_loss,
            'mse_loss': total_mse / self.N_sup,
        }
        if self.use_cvae:
            ret['kl_loss'] = kl_loss.item()
        return ret
