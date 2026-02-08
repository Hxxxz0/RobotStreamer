"""
MotionDiffusionCore: Main model with Transformer Encoder-Decoder Diffusion.

Architecture (v2 - token-level text):
    Input: history (60, 38) + text_tokens (60, 512)
           |
    [MotionTokenMLP] -> history_tokens (60, 16)
           |
    [Diffusion Head with Encoder-Decoder]
        Encoder: [timestep(1), history_tokens(60), text_tokens(60)] -> memory (121, n_emb)
        Decoder: noisy_samples (5, 38) -> denoised (5, 38)
           |
    Output: future 5 frames (5, 38)
"""

import torch
from torch import nn

from .token_mlp import MotionTokenMLP
from .diffloss import DiffLoss


class MotionDiffusionModel(nn.Module):
    """
    Motion prediction model using Transformer Encoder-Decoder Diffusion.
    
    Args:
        input_dim: Motion feature dimension (default: 38)
        hidden_size: Embedding dimension for transformer (default: 512)
        latent_dim: Token dimension (default: 16)
        text_encoder_dim: Text encoder token dimension (default: 512)
        text_max_length: Max text token length (default: 60)
        history_len: History window length (default: 60)
        pred_len: Prediction length (default: 5)
        device: Device to run model on
    """
    
    def __init__(
        self,
        input_dim=38,
        hidden_size=512,
        latent_dim=16,
        text_encoder_dim=512,
        text_max_length=60,
        history_len=60,
        pred_len=5,
        device="cuda",
        # Diffusion config
        num_sampling_steps=10,
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        diffusion_width=512,
        grad_checkpointing=False,
        # Transformer encoder-decoder config
        n_decoder_layers=6,
        n_encoder_layers=4,
        n_heads=8,
        # Root loss weighting
        root_loss_weight=1.0,
    ):
        super().__init__()
        
        # 1. Motion Token MLP
        self.token_mlp = MotionTokenMLP(input_dim, hidden_size, latent_dim)
        
        # 2. Diffusion Head with Transformer Encoder-Decoder
        self.action_diffusion = DiffLoss(
            motion_dim=input_dim,
            root_loss_weight=root_loss_weight,
            pred_len=pred_len,
            history_len=history_len,
            history_token_dim=latent_dim,
            text_dim=text_encoder_dim,
            text_max_length=text_max_length,
            width=diffusion_width,
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            num_sampling_steps=num_sampling_steps,
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            grad_checkpointing=grad_checkpointing,
        )
        
        self.history_len = history_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.device = device

    def forward(self, history, feat_text, target=None, history_mask=None, text_mask=None):
        """
        Forward pass for training.
        
        Args:
            history: (B, history_len, input_dim) - Historical motion frames
            feat_text: (B, text_max_length, text_encoder_dim) - Token-level text features
            target: (B, pred_len, input_dim) - Ground truth future frames
            history_mask: (B, history_len) - Bool mask, True=valid, False=padding
            text_mask: (B, text_max_length) - Bool mask, True=valid, False=padding
        
        Returns:
            loss: Diffusion loss
            pred: Predicted motion (B, pred_len * input_dim)
        """
        history_tokens = self.token_mlp(history)
        
        target_flat = target.reshape(target.shape[0], -1)
        loss, pred = self.action_diffusion(
            target=target_flat,
            history_tokens=history_tokens,
            text_emb=feat_text,
            history_mask=history_mask,
            text_mask=text_mask
        )
        return loss, pred

    def predict(self, history, feat_text, cfg_scale=1.0, empty_feat_text=None,
                temperature=1.0, history_mask=None, text_mask=None):
        """
        Inference with Classifier-Free Guidance.
        
        CFG uses zero vectors as the unconditional text (not empty string encoding).
        
        Args:
            history: (B, history_len, input_dim)
            feat_text: (B, text_max_length, text_encoder_dim) - Token-level text features
            cfg_scale: CFG guidance scale (1.0=off)
            empty_feat_text: Ignored (kept for backward compat). Zero vectors used automatically.
            temperature: Sampling temperature
            history_mask: (B, history_len) - Bool mask
            text_mask: (B, text_max_length) - Bool mask for text tokens
        
        Returns:
            pred_motion: (B, pred_len, input_dim)
        """
        with torch.no_grad():
            history_tokens = self.token_mlp(history)
            
            if cfg_scale != 1.0:
                # CFG: conditional + unconditional (zero text)
                B = history_tokens.shape[0]
                history_tokens_combined = torch.cat([history_tokens, history_tokens], dim=0)
                
                # Unconditional = zero text embeddings (not empty string!)
                zero_text = torch.zeros_like(feat_text)
                text_emb_combined = torch.cat([feat_text, zero_text], dim=0)
                
                # Text mask: conditional uses real mask, unconditional marks all as padding
                if text_mask is not None:
                    zero_text_mask = torch.zeros_like(text_mask)  # All False = all padding
                    text_mask_combined = torch.cat([text_mask, zero_text_mask], dim=0)
                else:
                    text_mask_combined = None
                
                # History mask
                history_mask_combined = torch.cat([history_mask, history_mask], dim=0) if history_mask is not None else None
                
                pred_flat = self.action_diffusion.sample(
                    history_tokens=history_tokens_combined,
                    text_emb=text_emb_combined,
                    temperature=temperature,
                    cfg=cfg_scale,
                    history_mask=history_mask_combined,
                    text_mask=text_mask_combined
                )
            else:
                pred_flat = self.action_diffusion.sample(
                    history_tokens=history_tokens,
                    text_emb=feat_text,
                    temperature=temperature,
                    cfg=cfg_scale,
                    history_mask=history_mask,
                    text_mask=text_mask
                )
            
            pred_motion = pred_flat.reshape(-1, self.pred_len, self.input_dim)
            
        return pred_motion
