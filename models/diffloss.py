import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusers import DDPMScheduler, DDIMScheduler


class DiffLoss(nn.Module):
    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps=10,
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        grad_checkpointing=False,
        learn_sigma=False,
    ):
        """
        Diffusion loss module with configurable schedulers.
        
        Args:
            target_channels: Output channels (motion dimension)
            z_channels: Conditioning vector dimension
            depth: Number of residual blocks
            width: Hidden dimension
            num_sampling_steps: Number of DDIM sampling steps for inference (default: 10)
            num_train_timesteps: Number of diffusion timesteps for training (default: 1000)
            beta_schedule: Noise schedule type (default: "squaredcos_cap_v2")
            prediction_type: What the model predicts - "sample" (x0) or "epsilon" (noise) (default: "sample")
            grad_checkpointing: Enable gradient checkpointing to save memory
            learn_sigma: Whether to learn variance (not commonly used)
        """
        super().__init__()
        self.in_channels = target_channels
        self.num_sampling_steps = num_sampling_steps
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2 if learn_sigma else target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
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

    def forward(self, target, z, mask=None):
        """
        Training forward pass with diffusion loss.
        
        Args:
            target: (B, D) - Ground truth samples (x0)
            z: (B, z_dim) - Condition vector
            mask: Optional mask for loss weighting
            
        Returns:
            loss: Scalar loss
            pred_x0: Predicted clean sample
        """
        batch_size = target.shape[0]
        device = target.device
        
        # 1. Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # 2. Add noise to clean samples: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noise = torch.randn_like(target)
        noisy_target = self.train_scheduler.add_noise(target, noise, t)
        
        # 3. Model prediction
        model_output = self.net(noisy_target, t, c=z)
        
        # 4. Compute Loss based on prediction type
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
            # mask: [B] or [B, 1] -> expand to match loss shape [B, D]
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            # Normalize by number of valid elements (not just frames)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            return loss, model_output
        
        return loss.mean(), model_output

    def sample(self, z, temperature=1.0, cfg=1.0):
        """
        Inference sampling using DDIM.
        
        Args:
            z: (B, z_dim) or (2B, z_dim) - Condition vector
                - If cfg == 1.0: z is (B, z_dim)
                - If cfg != 1.0: z is (2B, z_dim) = [cond, uncond] concatenated
            temperature: Noise temperature scaling
            cfg: Classifier-free guidance scale
            
        Returns:
            sample: (B, D) - Generated clean samples
        """
        device = z.device
        
        # Set inference timesteps
        self.inference_scheduler.set_timesteps(self.num_sampling_steps, device=device)
        
        # Determine batch size
        if cfg != 1.0:
            batch_size = z.shape[0] // 2  # z contains [cond, uncond]
        else:
            batch_size = z.shape[0]
        
        # Initialize noise
        sample = torch.randn(batch_size, self.in_channels, device=device) * temperature
        
        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            # Model prediction
            if cfg != 1.0:
                # CFG: use forward_with_cfg (handles duplication internally)
                t_batch = t.repeat(batch_size * 2)  # [2B] for both cond and uncond
                model_output = self.net.forward_with_cfg(sample, t_batch, c=z, cfg_scale=cfg)
            else:
                # Standard prediction
                t_batch = t.repeat(batch_size)
                model_output = self.net(sample, t_batch, c=z)
            
            # DDIM step: compute x_{t-1} from predicted x0
            sample = self.inference_scheduler.step(model_output, t, sample).prev_sample
        
        return sample


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = x.float()
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        """
        Classifier-Free Guidance forward pass.
        
        Args:
            x: [B, D] - noisy sample (NOT [2B, D])
            t: [2B] - timesteps (duplicated for cond and uncond)
            c: [2B, z_dim] - conditions [cond, uncond] concatenated
            cfg_scale: guidance scale
        
        Returns:
            pred: [B, D] - guided prediction (and variance if learn_sigma=True)
        """
        # Duplicate input for both conditional and unconditional paths
        x_combined = torch.cat([x, x], dim=0)  # [2B, D]
        
        # Forward pass for both paths
        model_out = self.forward(x_combined, t, c)  # [2B, out_channels]
        
        # Split conditional and unconditional outputs
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        
        # Apply CFG only to main prediction (not variance)
        if model_out.shape[1] == self.in_channels * 2:  # learn_sigma=True
            # Separate prediction and variance
            cond_pred, cond_var = cond_out[:, :self.in_channels], cond_out[:, self.in_channels:]
            uncond_pred, uncond_var = uncond_out[:, :self.in_channels], uncond_out[:, self.in_channels:]
            
            # Apply CFG only to prediction
            guided_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            
            # Use unconditional variance (more stable than guiding variance)
            guided_out = torch.cat([guided_pred, uncond_var], dim=1)
        else:  # learn_sigma=False (current default)
            # Apply CFG to entire output
            guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return guided_out
