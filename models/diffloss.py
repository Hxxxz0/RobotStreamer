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
        width=512,
        n_decoder_layers=6,
        n_encoder_layers=4,
        n_heads=8,
        num_sampling_steps=10,
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        grad_checkpointing=False,
    ):
        """
        Diffusion loss module with Transformer Encoder-Decoder.
        
        Architecture:
            Encoder: [timestep, history_tokens, text] -> memory
            Decoder: noisy_samples -> denoised_samples
        
        Args:
            motion_dim: Single frame motion dimension (default: 38)
            pred_len: Prediction horizon (default: 5)
            history_len: History window length (default: 60)
            history_token_dim: History token dimension (default: 16)
            text_dim: Text embedding dimension (default: 512)
            width: Transformer hidden dimension (default: 512)
            n_decoder_layers: Number of decoder layers (default: 6)
            n_encoder_layers: Number of encoder layers (default: 4)
            n_heads: Number of attention heads (default: 8)
            num_sampling_steps: Number of DDIM sampling steps for inference (default: 10)
            num_train_timesteps: Number of diffusion timesteps for training (default: 1000)
            beta_schedule: Noise schedule type (default: "squaredcos_cap_v2")
            prediction_type: What the model predicts - "sample" (x0) or "epsilon" (noise) (default: "sample")
            grad_checkpointing: Enable gradient checkpointing to save memory
        """
        super().__init__()
        self.num_sampling_steps = num_sampling_steps
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.motion_dim = motion_dim
        self.pred_len = pred_len
        
        # Transformer Encoder-Decoder Network
        self.net = TransformerForDiffusion(
            input_dim=motion_dim,
            output_dim=motion_dim,
            horizon=pred_len,
            n_obs_steps=history_len,
            cond_dim=history_token_dim,
            text_dim=text_dim,
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
        
        # Inference scheduler: DDIM for fast sampling
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False
        )

    def forward(self, target, history_tokens, text_emb, mask=None):
        """
        Training forward pass with diffusion loss.
        
        Args:
            target: (B, motion_dim*pred_len) - Ground truth samples (x0), flattened
            history_tokens: (B, history_len, token_dim) - History tokens
            text_emb: (B, text_dim) - Text embedding
            mask: Optional mask for loss weighting
            
        Returns:
            loss: Scalar loss
            pred_x0: Predicted clean sample (B, motion_dim*pred_len), flattened
        """
        batch_size = target.shape[0]
        device = target.device
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # 2. Reshape target from (B, motion_dim*pred_len) to (B, pred_len, motion_dim)
        if target.ndim == 2:
            target = target.reshape(batch_size, self.pred_len, self.motion_dim)
        
        # 3. Add noise to clean samples
        noise = torch.randn_like(target)
        noisy_target = self.train_scheduler.add_noise(target, noise, t)
        
        # 4. Model prediction
        model_output = self.net(
            sample=noisy_target,
            timestep=t,
            cond=history_tokens,
            text_emb=text_emb
        )
        
        # 5. Compute Loss based on prediction type
        if self.prediction_type == "epsilon":
            target_loss = noise
        elif self.prediction_type == "sample":
            target_loss = target
        elif self.prediction_type == "v_prediction":
            target_loss = self.train_scheduler.get_velocity(target, noise, t)
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")

        loss = F.mse_loss(model_output, target_loss, reduction='none')
        if mask is not None:
            # Ensure mask broadcasts correctly with loss
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            # Flatten output for compatibility
            model_output = model_output.reshape(batch_size, -1)
            return loss, model_output
        
        # Flatten output for compatibility
        model_output = model_output.reshape(batch_size, -1)
        return loss.mean(), model_output

    def sample(self, history_tokens, text_emb, temperature=1.0, cfg=1.0):
        """
        Inference sampling using DDIM.
        
        Args:
            history_tokens: (B, history_len, token_dim) or (2B, ...) - History tokens
            text_emb: (B, text_dim) or (2B, text_dim) - Text embedding
            temperature: Noise temperature scaling
            cfg: Classifier-free guidance scale
            
        Returns:
            sample: (B, motion_dim*pred_len) - Generated clean samples (flattened)
        """
        device = history_tokens.device
        
        # Set inference timesteps
        self.inference_scheduler.set_timesteps(self.num_sampling_steps, device=device)
        
        # Determine batch size
        if cfg != 1.0:
            batch_size = history_tokens.shape[0] // 2
        else:
            batch_size = history_tokens.shape[0]
        
        # Initialize noise: (B, pred_len, motion_dim)
        sample = torch.randn(batch_size, self.pred_len, self.motion_dim, device=device) * temperature
        
        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            # Model prediction
            if cfg != 1.0:
                # CFG: use forward_with_cfg
                t_batch = t.repeat(batch_size * 2)
                model_output = self.net.forward_with_cfg(
                    sample=sample,
                    timestep=t_batch,
                    cond=history_tokens,
                    text_emb=text_emb,
                    cfg_scale=cfg
                )
            else:
                # Standard prediction
                t_batch = t.repeat(batch_size)
                model_output = self.net(
                    sample=sample,
                    timestep=t_batch,
                    cond=history_tokens,
                    text_emb=text_emb
                )
            
            # DDIM step: compute x_{t-1} from predicted x0
            sample = self.inference_scheduler.step(model_output, t, sample).prev_sample
        
        # Flatten output: (B, pred_len, motion_dim) -> (B, motion_dim*pred_len)
        sample = sample.reshape(batch_size, -1)
        
        return sample
