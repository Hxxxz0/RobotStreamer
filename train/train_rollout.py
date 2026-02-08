"""
Rollout Training for MotionDiffusionCore
实现 Stage 2 (Scheduled Sampling) 和 Stage 3 (Full Rollout)
"""

import os
import sys
import json
import random
import yaml

import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import MotionDiffusionModel
from utils.text_encoder import load_text_encoder, resolve_text_encoder_path
from utils.logger import get_logger
from utils.rollout_utils import (
    RolloutScheduler,
    scheduled_sampling_history,
    rollout_history,
    add_history_noise
)
from datasets.motion_dataset import get_dataloader, cycle


# Import training utilities (copy from train.py instead of importing)
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


class WarmupCosineScheduler:
    """Warmup + Cosine LR Scheduler"""
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_iters - warmup_iters, eta_min=min_lr
        )

    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()


def load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None):
    """Load checkpoint from Stage 1"""
    if ckpt_path is None:
        return 0
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "token_mlp" in ckpt:
        model.token_mlp.load_state_dict(ckpt["token_mlp"], strict=True)
        if is_main: print("Loaded token_mlp")

    if "action_diffusion" in ckpt:
        model.action_diffusion.load_state_dict(ckpt["action_diffusion"], strict=True)
        if is_main: print("Loaded action_diffusion")
    
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main: print("Loaded optimizer state")
            
    start_iter = ckpt.get("iter", 0)
    if is_main: print(f"Resuming from iteration {start_iter}")
    
    return start_iter


def parse_args():
    """Load YAML config and merge with command-line arguments."""
    import argparse
    
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/rollout.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    
    config_path = os.path.join(repo_root, pre_args.config)
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if is_main:
            print(f"[Config] Loaded from {config_path}")
    else:
        config = {}
        if is_main:
            print(f"[Config] File not found: {config_path}, using defaults")
    
    parser = argparse.ArgumentParser()
    
    # Add config argument (already parsed by pre_parser, but need to add for help text)
    parser.add_argument("--config", type=str, default="configs/rollout.yaml", help="Path to config file")
    
    # Add all config keys as arguments (skip dicts for now)
    for key, value in config.items():
        if isinstance(value, dict):
            continue  # Handle dicts separately after parsing
        elif isinstance(value, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() == 'true', default=value)
        elif isinstance(value, (int, float)):
            parser.add_argument(f"--{key}", type=type(value), default=value)
        elif isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                float_val = float(value)
                parser.add_argument(f"--{key}", type=float, default=float_val)
            except ValueError:
                # Not a number, keep as string
                parser.add_argument(f"--{key}", type=str, default=value)
        elif isinstance(value, list):
            parser.add_argument(f"--{key}", nargs='+', default=value)
    
    args = parser.parse_args()
    
    # Add nested dicts from config as attributes
    for key, value in config.items():
        if isinstance(value, dict):
            setattr(args, key, value)
    
    # Set defaults for missing args
    if not hasattr(args, 'use_fp16'):
        args.use_fp16 = False
    if not hasattr(args, 'grad_accum_steps'):
        args.grad_accum_steps = 1
    if not hasattr(args, 'grad_clip'):
        args.grad_clip = 1.0
    if not hasattr(args, 'log_every'):
        args.log_every = 100
    if not hasattr(args, 'save_every'):
        args.save_every = 5000
    
    return args


def train_rollout(args):
    """Rollout Training 主函数"""
    
    # Setup
    accelerator = Accelerator(
        mixed_precision="fp16" if args.use_fp16 else "no",
        gradient_accumulation_steps=args.grad_accum_steps
    )
    device = accelerator.device
    is_main = accelerator.is_main_process
    
    if is_main:
        log_dir = os.path.join(repo_root, "logs", args.exp_name)
        out_dir = os.path.join(repo_root, "outputs", args.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        logger = get_logger(log_dir)  # get_logger 只接受一个参数
        writer = SummaryWriter(log_dir)
        logger.info("="*60)
        logger.info(f"Rollout Training - Stage {args.rollout_stage}")
        logger.info("="*60)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load text encoder
    text_encoder_path = resolve_text_encoder_path(args.text_encoder_type, args.text_encoder, repo_root)
    text_encoder, text_encoder_dim = load_text_encoder(
        args.text_encoder_type, text_encoder_path, args.text_encoder_device,
        max_length=args.text_max_length
    )
    args.text_encoder_dim = text_encoder_dim
    
    if is_main:
        logger.info(f"Text encoder: {args.text_encoder_type}, dim: {text_encoder_dim}")
    
    # Create model
    model = MotionDiffusionModel(
        input_dim=args.motion_dim,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        text_encoder_dim=text_encoder_dim,
        text_max_length=args.text_max_length,
        history_len=args.history_len,
        pred_len=args.pred_len,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        diffusion_width=args.diffusion_width,
        grad_checkpointing=args.grad_checkpointing,
        n_decoder_layers=args.n_decoder_layers,
        n_encoder_layers=args.n_encoder_layers,
        n_heads=args.n_heads,
        root_loss_weight=args.root_loss_weight,
    )
    
    # Load checkpoint (必须加载 Stage 1 的模型)
    if not args.resume_from:
        raise ValueError("Rollout training requires --resume_from checkpoint from Stage 1!")
    
    start_iter = load_checkpoint(model, args.resume_from)
    
    if is_main:
        logger.info(f"Loaded Stage 1 checkpoint from {args.resume_from}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler (no warmup for rollout training, model is already trained)
    # Use simple cosine decay from initial LR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=args.lr * 0.1)
    
    # Rollout Scheduler
    if args.rollout_stage == 2:
        rollout_scheduler = RolloutScheduler(
            stage=2,
            total_epochs=args.rollout_stage2_epochs,
            rollout_ratio_start=args.rollout_ratio_start,
            rollout_ratio_end=args.rollout_ratio_end,
            schedule_type=args.rollout_schedule_type
        )
    else:  # Stage 3
        rollout_scheduler = RolloutScheduler(
            stage=3,
            total_epochs=args.rollout_stage3_epochs,
            rollout_ratio_start=1.0,
            rollout_ratio_end=1.0,
            schedule_type="linear"
        )
    
    # Build dataset paths mapping
    dataset_paths = {
        "humanml3d": args.humanml3d_path,
        "babel_stream": args.babel_stream_path,
        "humanml3d_stream": args.humanml3d_stream_path,
    }
    
    # Data
    dataloader = get_dataloader(
        datasets=args.datasets,
        dataset_paths=dataset_paths,
        meta_dir=args.meta_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_len=args.history_len,
        pred_len=args.pred_len,
    )
    data_iter = cycle(dataloader)
    
    # Accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_text_encoder = text_encoder
    
    # Load statistics
    mean = np.load(os.path.join(args.mean_std_dir, "Mean.npy"))
    std = np.load(os.path.join(args.mean_std_dir, "Std.npy"))
    mean = torch.from_numpy(mean).float().to(device)
    std = torch.from_numpy(std).float().to(device)
    
    # Training loop
    model.train()
    global_step = 0  # Reset to 0 for Rollout Training (don't continue from Stage 1 step count)
    epoch = 0
    steps_per_epoch = len(dataloader)
    
    if is_main:
        logger.info(f"Starting Rollout Training from step 0 (model weights from step {start_iter})")
        logger.info(f"Stage {args.rollout_stage}, Total iterations: {args.total_iter}")
        logger.info(f"Rollout steps: {args.rollout_steps}, Sampling steps: {args.rollout_sampling_steps}")
    
    while global_step < args.total_iter:
        # Get rollout ratio for current epoch
        rollout_ratio = rollout_scheduler.get_rollout_ratio(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.total_iter:
                break
            
            # Unpack batch (tuple, not dict)
            caption, history, target, history_mask = batch
            caption = list(caption)
            history = history.to(device).float()
            target = target.to(device).float()
            history_mask = history_mask.to(device)  # (B, history_len), bool
            
            B = history.shape[0]
            
            # Encode text
            feat_text_list = []
            text_mask_list = []
            for cap in caption:
                feat_np, mask_np = unwrapped_text_encoder.encode(cap)
                feat_text_list.append(torch.from_numpy(feat_np))
                text_mask_list.append(torch.from_numpy(mask_np))
            
            feat_text = torch.stack(feat_text_list, dim=0).float().to(device)
            text_mask = torch.stack(text_mask_list, dim=0).to(device)
            
            # ===== Rollout Training Core Logic =====
            
            # 根据 rollout_ratio 决定使用 GT history 还是 rollout history
            if rollout_ratio > 0.0:
                # Scheduled Sampling or Full Rollout
                with torch.no_grad():
                    history, history_mask = scheduled_sampling_history(
                        gt_history=history,
                        gt_history_mask=history_mask,
                        model=unwrapped_model,
                        text_emb=feat_text,
                        text_mask=text_mask,
                        rollout_ratio=rollout_ratio,
                        rollout_steps=args.rollout_steps,
                        pred_len=args.pred_len,
                        cfg_scale=args.rollout_cfg_scale,
                        temperature=args.rollout_temperature
                    )
            
            # Optional: 添加噪声（额外的鲁棒性）
            if hasattr(args, 'history_noise_level') and args.history_noise_level > 0:
                history = add_history_noise(history, history_mask, args.history_noise_level)
            
            # Forward pass
            loss, _ = model(
                history,
                feat_text,
                target,
                history_mask=history_mask,
                text_mask=text_mask
            )
            
            loss_dict = {"loss": loss}
            
            # Backward
            accelerator.backward(loss)
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step(global_step)
                optimizer.zero_grad()
            
            # Logging
            if is_main and global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch:03d}] Step {global_step:06d}/{args.total_iter} | "
                    f"Rollout Ratio: {rollout_ratio:.2f} | "
                    f"Loss: {loss.item():.4f} | LR: {lr:.2e}"
                )
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/rollout_ratio", rollout_ratio, global_step)
                
                # Log detailed losses
                for key, val in loss_dict.items():
                    if key != "loss" and isinstance(val, torch.Tensor):
                        writer.add_scalar(f"train/{key}", val.item(), global_step)
            
            # Save checkpoint
            if is_main and global_step % args.save_every == 0 and global_step > 0:
                ckpt_path = os.path.join(out_dir, f"checkpoint_{global_step:06d}.pth")
                torch.save({
                    "token_mlp": unwrapped_model.token_mlp.state_dict(),
                    "action_diffusion": unwrapped_model.action_diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": global_step,
                    "epoch": epoch,
                    "rollout_ratio": rollout_ratio,
                    "args": vars(args)
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
                
                # Save latest
                latest_path = os.path.join(out_dir, "latest.pth")
                torch.save({
                    "token_mlp": unwrapped_model.token_mlp.state_dict(),
                    "action_diffusion": unwrapped_model.action_diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": global_step,
                    "epoch": epoch,
                    "rollout_ratio": rollout_ratio,
                    "args": vars(args)
                }, latest_path)
            
            global_step += 1
        
        epoch += 1
    
    # Final save
    if is_main:
        final_path = os.path.join(out_dir, "final.pth")
        torch.save({
            "token_mlp": unwrapped_model.token_mlp.state_dict(),
            "action_diffusion": unwrapped_model.action_diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": global_step,
            "epoch": epoch,
            "args": vars(args)
        }, final_path)
        logger.info(f"Training complete! Saved final checkpoint to {final_path}")
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate rollout training args
    if not hasattr(args, 'rollout_training') or not args.rollout_training:
        raise ValueError("This script requires rollout_training to be enabled in config!")
    
    # Set rollout-specific args
    args.rollout_stage = args.rollout_training.get("stage", 2)
    args.rollout_steps = args.rollout_training.get("rollout_steps", 2)
    args.rollout_sampling_steps = args.rollout_training.get("rollout_sampling_steps", 5)
    args.rollout_cfg_scale = args.rollout_training.get("rollout_cfg_scale", 1.0)
    args.rollout_temperature = args.rollout_training.get("rollout_temperature", 1.0)
    
    # Stage 2 args
    args.rollout_stage2_epochs = args.rollout_training["stage2"].get("total_epochs", 80)
    args.rollout_ratio_start = args.rollout_training["stage2"].get("rollout_ratio_start", 0.0)
    args.rollout_ratio_end = args.rollout_training["stage2"].get("rollout_ratio_end", 1.0)
    args.rollout_schedule_type = args.rollout_training["stage2"].get("schedule_type", "linear")
    
    # Stage 3 args
    args.rollout_stage3_epochs = args.rollout_training["stage3"].get("total_epochs", 40)
    
    # Optional noise
    args.history_noise_level = args.rollout_training.get("history_noise_level", 0.0)
    
    train_rollout(args)
