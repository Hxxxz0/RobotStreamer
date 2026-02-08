"""
Transformer Encoder-Decoder for Diffusion Models.

Architecture:
    Encoder: timestep + history tokens + text tokens -> memory
    Decoder: noisy samples + memory -> denoised samples
    
Changes from v1:
    - Text: single pooled vector (1 token) -> token-level (N tokens)
    - Added text_mask support for padding
    - CFG uses zero vectors instead of empty string encoding
"""

import math
import logging
import torch
import torch.nn as nn
from typing import Union, Optional

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        text_dim: int = 512,
        text_max_length: int = 60,
        n_layer: int = 6,
        n_head: int = 8,
        n_emb: int = 512,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = True,
        n_cond_layers: int = 4
    ) -> None:
        """
        Transformer Encoder-Decoder for diffusion-based motion generation.
        
        Args:
            input_dim: Input motion dimension (e.g., 38)
            output_dim: Output motion dimension (e.g., 38)
            horizon: Prediction horizon (e.g., 5 frames)
            n_obs_steps: Number of observation/history steps (e.g., 60 frames)
            cond_dim: Dimension of observation condition (e.g., 16 for motion tokens)
            text_dim: Dimension of text token embedding (e.g., 512 for T5)
            text_max_length: Max text token length (e.g., 60)
            n_layer: Number of decoder layers
            n_head: Number of attention heads
            n_emb: Embedding dimension
            p_drop_emb: Embedding dropout probability
            p_drop_attn: Attention dropout probability
            causal_attn: Whether to use causal attention
            time_as_cond: Whether to use timestep as condition (should be True)
            obs_as_cond: Whether to use observations as condition (should be True)
            n_cond_layers: Number of encoder layers
        """
        super().__init__()

        # Compute number of tokens
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon  # Decoder sequence length
        T_cond = 1   # Start with timestep token
        
        assert time_as_cond, "time_as_cond must be True for diffusion"
        assert obs_as_cond, "obs_as_cond must be True for motion history"
        
        if obs_as_cond:
            T_cond += n_obs_steps  # Add history tokens
        T_cond += text_max_length  # Add text tokens (was +1 before)
        
        # Input embedding for decoder (noisy samples)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # Condition encoder components
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if cond_dim > 0 else None
        self.text_emb_proj = nn.Linear(text_dim, n_emb)
        
        # Condition positional embedding
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        
        # Encoder for conditions
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
                enable_nested_tensor=False
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb)
            )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # Attention masks
        self.mask = None
        self.memory_mask = None
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

        # Output head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # Constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.n_obs_steps = n_obs_steps
        self.text_max_length = text_max_length

        # Initialize weights
        self.apply(self._init_weights)
        logger.info(
            "TransformerForDiffusion: %e parameters", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, SinusoidalPosEmb, nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer, nn.TransformerEncoder,
                        nn.TransformerDecoder, nn.ModuleList, nn.Mish,
                        nn.Sequential, nn.GELU)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass

    def forward(
        self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass for diffusion denoising.
        
        Args:
            sample: (B, T, input_dim) - Noisy samples to denoise
            timestep: (B,) or int - Diffusion timestep
            cond: (B, n_obs_steps, cond_dim) - History condition tokens
            text_emb: (B, text_max_length, text_dim) - Token-level text embedding
            history_mask: (B, n_obs_steps) - Bool mask, True=valid, False=padding
            text_mask: (B, text_max_length) - Bool mask, True=valid, False=padding
        
        Returns:
            output: (B, T, output_dim) - Denoised samples
        """
        B = sample.shape[0]
        device = sample.device
        
        # 1. Process timestep
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.expand(B)
        time_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_emb)

        # 2. Build encoder input: [timestep, history, text_tokens]
        cond_embeddings = time_emb
        
        if self.obs_as_cond and cond is not None:
            cond_obs_emb = self.cond_obs_emb(cond)  # (B, n_obs_steps, n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        
        if text_emb is not None:
            text_emb_proj = self.text_emb_proj(text_emb)  # (B, text_max_length, n_emb)
            cond_embeddings = torch.cat([cond_embeddings, text_emb_proj], dim=1)
        
        # 3. Build padding mask: [timestep(always valid), history_mask, text_mask]
        # Transformer src_key_padding_mask: True=IGNORE, False=ATTEND
        src_key_padding_mask = None
        if history_mask is not None or text_mask is not None:
            # Timestep is always valid
            timestep_pad = torch.zeros((B, 1), dtype=torch.bool, device=device)
            
            # History mask
            if history_mask is not None:
                history_pad = ~history_mask  # Invert: True=padding
            else:
                history_pad = torch.zeros((B, self.n_obs_steps), dtype=torch.bool, device=device)
            
            # Text mask
            if text_mask is not None:
                text_pad = ~text_mask  # Invert: True=padding
            else:
                text_pad = torch.zeros((B, self.text_max_length), dtype=torch.bool, device=device)
            
            src_key_padding_mask = torch.cat([timestep_pad, history_pad, text_pad], dim=1)
        
        # 4. Encoder: process conditions
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_embeddings + position_embeddings)
        
        if isinstance(self.encoder, nn.TransformerEncoder):
            memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            memory = self.encoder(x)
        
        # 5. Decoder: process noisy samples
        token_embeddings = self.input_emb(sample)  # (B, T, n_emb)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # 6. Output head
        x = self.ln_f(x)
        x = self.head(x)
        
        return x

    def forward_with_cfg(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        history_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with Classifier-Free Guidance.
        
        Args:
            sample: (B, T, input_dim) - Noisy samples (NOT duplicated)
            timestep: (B,) or (2B,) - Timesteps
            cond: (2B, n_obs_steps, cond_dim) - [cond, uncond] concatenated
            text_emb: (2B, text_max_length, text_dim) - [cond_text, zero_text] concatenated
            cfg_scale: CFG guidance scale
            history_mask: (2B, n_obs_steps) - Bool mask for history tokens
            text_mask: (2B, text_max_length) - Bool mask for text tokens
        
        Returns:
            output: (B, T, output_dim) - Guided prediction
        """
        if cfg_scale == 1.0:
            return self.forward(sample, timestep, cond, text_emb, history_mask, text_mask)
        
        B = sample.shape[0]
        
        # Duplicate sample for conditional and unconditional paths
        sample_combined = torch.cat([sample, sample], dim=0)
        
        # Ensure timestep is properly formatted
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None]
        
        if timestep.shape[0] == B:
            timestep = timestep.repeat(2)
        
        # Forward pass for both paths
        model_out = self.forward(sample_combined, timestep, cond, text_emb, history_mask, text_mask)
        
        # Split and apply CFG
        cond_out, uncond_out = torch.split(model_out, B, dim=0)
        guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return guided_out
