"""
MotionDiffusionCore: Main model with Transformer Encoder-Decoder Diffusion.

Architecture:
    Input: history (60, 38) + text (512)
           ↓
    [MotionTokenMLP] → history_tokens (60, 16)
           ↓
    [Diffusion Head with Encoder-Decoder]
        Encoder: timestep + history_tokens + text → memory
        Decoder: noisy_samples → denoised samples
           ↓
    Output: future 5 frames (5, 38)
"""

import torch
from torch import nn

from .token_mlp import MotionTokenMLP
from .diffloss import DiffLoss


class MotionDiffusionModel(nn.Module):
    """
    Motion prediction model using Transformer Encoder-Decoder Diffusion.
    
    Forward Pass:
        1. MotionTokenMLP: Convert history (B, 60, 38) -> tokens (B, 60, 16)
        2. Diffusion Head with Encoder-Decoder:
           - Encoder: [timestep, history_tokens, text] -> memory (B, 62, n_emb)
           - Decoder: noisy_samples (B, 5, 38) -> denoised (B, 5, 38)
    
    Args:
        input_dim: Motion feature dimension (default: 38)
        hidden_size: Embedding dimension for transformer (default: 512)
        latent_dim: Token dimension (default: 16)
        text_encoder_dim: Text encoder output dimension (default: 512)
        history_len: History window length (default: 60)
        pred_len: Prediction length (default: 5)
        device: Device to run model on
        num_sampling_steps: DDIM sampling steps for inference (default: 10)
        num_train_timesteps: Diffusion training timesteps (default: 1000)
        beta_schedule: Noise schedule type (default: "squaredcos_cap_v2")
        prediction_type: Model prediction target - "sample" (x0) or "epsilon" (noise) (default: "sample")
        diffusion_width: Diffusion transformer width (default: 512)
        grad_checkpointing: Enable gradient checkpointing (default: False)
        n_decoder_layers: Number of decoder layers (default: 6)
        n_encoder_layers: Number of encoder layers (default: 4)
        n_heads: Number of attention heads (default: 8)
    """
    
    def __init__(
        self,
        input_dim=38,
        hidden_size=512,
        latent_dim=16,
        text_encoder_dim=512,
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
    ):
        super().__init__()
        
        # 1. Motion Token MLP: Map raw motion to latent tokens
        self.token_mlp = MotionTokenMLP(input_dim, hidden_size, latent_dim)
        
        # 2. Diffusion Head with Transformer Encoder-Decoder
        self.action_diffusion = DiffLoss(
            motion_dim=input_dim,
            pred_len=pred_len,
            history_len=history_len,
            history_token_dim=latent_dim,
            text_dim=text_encoder_dim,
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

    def forward(self, history, feat_text, target=None, history_mask=None):
        """
        Forward pass for training.
        
        Args:
            history: (B, history_len, input_dim) - Historical motion frames
            feat_text: (B, text_encoder_dim) - Text condition features
            target: (B, pred_len, input_dim) - Ground truth future frames (for training)
            history_mask: (B, history_len) - Not used in new architecture
        
        Returns:
            loss: Diffusion loss
            pred: Predicted motion (B, pred_len * input_dim)
        """
        # Step 1: Convert motion history to tokens
        history_tokens = self.token_mlp(history)  # (B, 60, 38) -> (B, 60, 16)
        
        # Step 2: Compute diffusion loss with transformer encoder-decoder
        target_flat = target.reshape(target.shape[0], -1)  # (B, 5, 38) -> (B, 190)
        loss, pred = self.action_diffusion(
            target=target_flat,
            history_tokens=history_tokens,
            text_emb=feat_text
        )
        return loss, pred

    def predict(self, history, feat_text, cfg_scale=1.0, empty_feat_text=None, temperature=1.0, history_mask=None):
        """
        Inference with Classifier-Free Guidance.
        
        Args:
            history: (B, history_len, input_dim) - Historical motion frames
            feat_text: (B, text_encoder_dim) - Text condition features
            cfg_scale: CFG guidance scale (1.0 = no guidance, !=1.0 = use CFG)
            empty_feat_text: (B, text_encoder_dim) - Empty text features for CFG (required if cfg_scale != 1.0)
            temperature: Sampling temperature for noise scaling (default: 1.0)
            history_mask: (B, history_len) - Not used in new architecture
        
        Returns:
            pred_motion: (B, pred_len, input_dim) - Predicted future motion
        """
        with torch.no_grad():
            # Step 1: Convert motion history to tokens
            history_tokens = self.token_mlp(history)
            
            # Step 2: Prepare conditions for CFG
            if cfg_scale != 1.0:
                if empty_feat_text is None:
                    raise ValueError("empty_feat_text is required when cfg_scale != 1.0")
                # Concatenate conditional and unconditional
                history_tokens_combined = torch.cat([history_tokens, history_tokens], dim=0)
                text_emb_combined = torch.cat([feat_text, empty_feat_text], dim=0)
                
                pred_flat = self.action_diffusion.sample(
                    history_tokens=history_tokens_combined,
                    text_emb=text_emb_combined,
                    temperature=temperature,
                    cfg=cfg_scale
                )
            else:
                pred_flat = self.action_diffusion.sample(
                    history_tokens=history_tokens,
                    text_emb=feat_text,
                    temperature=temperature,
                    cfg=cfg_scale
                )
            
            pred_motion = pred_flat.reshape(-1, self.pred_len, self.input_dim)
            
        return pred_motion
