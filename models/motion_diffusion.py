"""
MotionDiffusionCore: Main model combining MLP, Transformer, and Diffusion.

Architecture:
    Input (60 frames history + text) -> MotionTokenMLP -> LLaMAHF Transformer -> DiffLoss -> Output (5 frames)
"""

import torch
from torch import nn

from .token_mlp import MotionTokenMLP
from .transformer import LLaMAHF, LLaMAHFConfig
from .diffloss import DiffLoss


class MotionDiffusionModel(nn.Module):
    """
    Motion prediction model using Transformer + Diffusion.
    
    Forward Pass:
        1. MotionTokenMLP: Convert 60-frame history (B, 60, 38) -> tokens (B, 60, 16)
        2. LLaMAHF: Fuse history tokens + text condition -> context (B, 60, 512)
        3. DiffLoss: Use last token as condition -> generate 5 future frames (B, 5, 38)
    
    Args:
        input_dim: Motion feature dimension (default: 38)
        hidden_size: MLP hidden size (default: 512)
        latent_dim: Token dimension (default: 16)
        text_encoder_dim: Text encoder output dimension
        num_diffusion_head_layers: Number of diffusion head layers (default: 4)
        history_len: History window length (default: 60)
        pred_len: Prediction length (default: 5)
        device: Device to run model on
        num_sampling_steps: DDIM sampling steps for inference (default: 10)
        num_train_timesteps: Diffusion training timesteps (default: 1000)
        beta_schedule: Noise schedule type (default: "squaredcos_cap_v2")
        prediction_type: Model prediction target - "sample" (x0) or "epsilon" (noise) (default: "sample")
        diffusion_width: Diffusion head hidden dimension (default: 512)
        grad_checkpointing: Enable gradient checkpointing (default: False)
    """
    
    def __init__(
        self,
        input_dim=38,
        hidden_size=512,
        latent_dim=16,
        text_encoder_dim=512,
        num_diffusion_head_layers=4,
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
    ):
        super().__init__()
        
        # 1. Motion Token MLP: Map raw motion to latent tokens
        self.token_mlp = MotionTokenMLP(input_dim, hidden_size, latent_dim)
        
        # 2. LLaMAHF Transformer: Fuse history tokens + text condition
        # Use hidden_size for transformer dimension to respect user config
        # Default to 768 if hidden_size is small/default (compatibility), but better to follow config explicitly.
        # Given config/default.yaml has hidden_size: 512, we should use it.
        config = LLaMAHFConfig(
            n_embd=hidden_size,  # Use hidden_size from args (e.g. 512)
            n_layer=8,           # Keep default depth
            n_head=8,            # Keep default heads (ensure 512 % 8 == 0)
        )
        config.text_encoder_dim = text_encoder_dim
        config.block_size = history_len + 1  # +1 for extra token
        self.trans_encoder = LLaMAHF(config, input_token_dim=latent_dim)
        
        # 3. Diffusion Head: Generate future motion from condition
        self.action_diffusion = DiffLoss(
            target_channels=input_dim * pred_len,  # 38 * 5 = 190
            z_channels=config.n_embd,               # Matches transformer output
            depth=num_diffusion_head_layers,        # 4 layers
            width=diffusion_width,
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
        Forward pass for training or inference.
        
        Args:
            history: (B, history_len, input_dim) - Historical motion frames
            feat_text: (B, text_encoder_dim) - Text condition features
            target: (B, pred_len, input_dim) - Ground truth future frames (for training)
            history_mask: (B, history_len) - True for valid frames, False for padding
        
        Returns:
            If target is provided (training):
                loss: Diffusion loss
                pred: Predicted motion (B, pred_len * input_dim)
            If target is None (inference):
                z: Condition vector (B, 512) for external diffusion sampling
        """
        # Step 1: Convert motion history to tokens
        history_tokens = self.token_mlp(history)  # (B, 60, 38) -> (B, 60, 16)
        
        # Step 2: Fuse history tokens + text with Transformer
        # Pass history_mask to mask out padding positions in attention
        conditions = self.trans_encoder(history_tokens, feat_text, mask=history_mask)  # (B, 60, 512)
        
        # Step 3: Extract condition vector from last token
        z = conditions[:, -1, :]  # (B, 512)
        
        # Step 4: Generate future motion with diffusion
        if target is None:
            # Inference: return condition vector for external sampling
            return z
        
        # Training: compute diffusion loss
        target_flat = target.reshape(target.shape[0], -1)  # (B, 5, 38) -> (B, 190)
        loss, pred = self.action_diffusion(target_flat, z)
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
            history_mask: (B, history_len) - True for valid frames, False for padding
        
        Returns:
            pred_motion: (B, pred_len, input_dim) - Predicted future motion
        """
        with torch.no_grad():
            # Get condition vector
            history_tokens = self.token_mlp(history)
            conditions = self.trans_encoder(history_tokens, feat_text, mask=history_mask)
            z = conditions[:, -1, :]
            
            # Apply CFG if requested (consistent with diffloss.py)
            if cfg_scale != 1.0:
                if empty_feat_text is None:
                    raise ValueError("empty_feat_text is required when cfg_scale != 1.0")
                # Get unconditional condition vector
                empty_conditions = self.trans_encoder(history_tokens, empty_feat_text, mask=history_mask)
                empty_z = empty_conditions[:, -1, :]
                # Concatenate [cond, uncond] for CFG
                z = torch.cat([z, empty_z], dim=0)
            
            # Sample using the diffusion model
            pred_flat = self.action_diffusion.sample(z, temperature=temperature, cfg=cfg_scale)  # (B, 190)
            pred_motion = pred_flat.reshape(-1, self.pred_len, self.input_dim)  # (B, 5, 38)
            
        return pred_motion
