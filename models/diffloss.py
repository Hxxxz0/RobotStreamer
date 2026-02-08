import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusers import DDPMScheduler, DDIMScheduler
from .transformer_diffusion import TransformerForDiffusion


class DiffLoss(nn.Module):
    def __init__(
        self,
        motion_dim=38,
        pred_len=5,
        history_len=60,
        history_token_dim=16,
        text_dim=512,
        text_max_length=60,
        width=512,
        n_decoder_layers=6,
        n_encoder_layers=4,
        n_heads=8,
        num_sampling_steps=10,
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        grad_checkpointing=False,
        root_loss_weight=1.0,
    ):
        """
        Diffusion loss module with Transformer Encoder-Decoder.
        
        Args:
            motion_dim: Single frame motion dimension (default: 38)
            pred_len: Prediction horizon (default: 5)
            history_len: History window length (default: 60)
            history_token_dim: History token dimension (default: 16)
            text_dim: Text token embedding dimension (default: 512)
            text_max_length: Max text token length (default: 60)
            width: Transformer hidden dimension (default: 512)
            n_decoder_layers: Number of decoder layers (default: 6)
            n_encoder_layers: Number of encoder layers (default: 4)
            n_heads: Number of attention heads (default: 8)
            num_sampling_steps: DDIM sampling steps for inference (default: 10)
            num_train_timesteps: Diffusion training timesteps (default: 1000)
            beta_schedule: Noise schedule type
            prediction_type: "sample" (x0) or "epsilon" (noise)
            grad_checkpointing: Enable gradient checkpointing
            root_loss_weight: Weight for root features (dims 29-37)
        """
        super().__init__()
        self.num_sampling_steps = num_sampling_steps
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.motion_dim = motion_dim
        self.pred_len = pred_len
        self.root_loss_weight = root_loss_weight
        
        # Create loss weight tensor
        loss_weights = torch.ones(motion_dim)
        loss_weights[29:38] = root_loss_weight
        self.register_buffer('loss_weights', loss_weights)
        
        # Transformer Encoder-Decoder Network
        self.net = TransformerForDiffusion(
            input_dim=motion_dim,
            output_dim=motion_dim,
            horizon=pred_len,
            n_obs_steps=history_len,
            cond_dim=history_token_dim,
            text_dim=text_dim,
            text_max_length=text_max_length,
            n_layer=n_decoder_layers,
            n_head=n_heads,
            n_emb=width,
            n_cond_layers=n_encoder_layers,
            time_as_cond=True,
            obs_as_cond=True,
        )

        # Training scheduler: DDPM
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,
            variance_type="fixed_small"
        )
        
        # Inference scheduler: DDIM
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False
        )

    def forward(self, target, history_tokens, text_emb, mask=None, history_mask=None, text_mask=None):
        """
        Training forward pass with diffusion loss.
        
        Args:
            target: (B, motion_dim*pred_len) - Ground truth (flattened)
            history_tokens: (B, history_len, token_dim) - History tokens
            text_emb: (B, text_max_length, text_dim) - Token-level text embedding
            mask: Optional mask for loss weighting
            history_mask: (B, history_len) - Bool mask (True=valid)
            text_mask: (B, text_max_length) - Bool mask (True=valid)
            
        Returns:
            loss: Scalar loss
            pred_x0: Predicted (B, motion_dim*pred_len)
        """
        batch_size = target.shape[0]
        device = target.device
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # 2. Reshape target
        if target.ndim == 2:
            target = target.reshape(batch_size, self.pred_len, self.motion_dim)
        
        # 3. Add noise
        noise = torch.randn_like(target)
        noisy_target = self.train_scheduler.add_noise(target, noise, t)
        
        # 4. Model prediction
        model_output = self.net(
            sample=noisy_target,
            timestep=t,
            cond=history_tokens,
            text_emb=text_emb,
            history_mask=history_mask,
            text_mask=text_mask
        )
        
        # 5. Compute loss
        if self.prediction_type == "epsilon":
            target_loss = noise
        elif self.prediction_type == "sample":
            target_loss = target
        elif self.prediction_type == "v_prediction":
            target_loss = self.train_scheduler.get_velocity(target, noise, t)
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        loss = F.mse_loss(model_output, target_loss, reduction='none')
        
        # Apply dimension-wise loss weights
        loss_weights_broadcast = self.loss_weights.view(1, 1, -1)
        loss = loss * loss_weights_broadcast
        
        if mask is not None:
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            model_output = model_output.reshape(batch_size, -1)
            return loss, model_output
        
        model_output = model_output.reshape(batch_size, -1)
        return loss.mean(), model_output

    def sample(self, history_tokens, text_emb, temperature=1.0, cfg=1.0, history_mask=None, text_mask=None):
        """
        Inference sampling using DDIM.
        
        Args:
            history_tokens: (B, history_len, token_dim) or (2B, ...) for CFG
            text_emb: (B, text_max_length, text_dim) or (2B, ...) for CFG
            temperature: Noise temperature
            cfg: CFG scale
            history_mask: (B, history_len) or (2B, ...) for CFG
            text_mask: (B, text_max_length) or (2B, ...) for CFG
            
        Returns:
            sample: (B, motion_dim*pred_len) - Generated samples (flattened)
        """
        device = history_tokens.device
        
        self.inference_scheduler.set_timesteps(self.num_sampling_steps, device=device)
        
        if cfg != 1.0:
            batch_size = history_tokens.shape[0] // 2
        else:
            batch_size = history_tokens.shape[0]
        
        # Initialize noise
        sample = torch.randn(batch_size, self.pred_len, self.motion_dim, device=device) * temperature
        
        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            if cfg != 1.0:
                t_batch = t.repeat(batch_size * 2)
                model_output = self.net.forward_with_cfg(
                    sample=sample,
                    timestep=t_batch,
                    cond=history_tokens,
                    text_emb=text_emb,
                    cfg_scale=cfg,
                    history_mask=history_mask,
                    text_mask=text_mask
                )
            else:
                t_batch = t.repeat(batch_size)
                model_output = self.net(
                    sample=sample,
                    timestep=t_batch,
                    cond=history_tokens,
                    text_emb=text_emb,
                    history_mask=history_mask,
                    text_mask=text_mask
                )
            
            sample = self.inference_scheduler.step(model_output, t, sample).prev_sample
        
        sample = sample.reshape(batch_size, -1)
        return sample
