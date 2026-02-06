"""Train MotionDiffusionCore: 60-frame history + text -> 5-frame future motion."""

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
from datasets.motion_dataset import get_dataloader, cycle


class WarmupCosineScheduler:
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
    if ckpt_path is None:
        return 0
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Note: trans_encoder is now integrated into action_diffusion (Transformer Encoder-Decoder)
    # Old checkpoints with separate "trans" key are not compatible with new architecture
    
    if "token_mlp" in ckpt:
        model.token_mlp.load_state_dict(ckpt["token_mlp"], strict=True)
        if is_main: print("Loaded token_mlp")

    if "action_diffusion" in ckpt:
        model.action_diffusion.load_state_dict(ckpt["action_diffusion"], strict=True)
        if is_main: print("Loaded action_diffusion")
    
    # Load optimizer and scheduler state if available and requested
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if is_main: print("Loaded optimizer state")
            
    # Load iteration count
    start_iter = ckpt.get("iter", 0)
    if is_main: print(f"Resuming from iteration {start_iter}")
    
    return start_iter


def parse_args():
    """Load YAML config and merge with command-line arguments."""
    import argparse
    
    # First pass: get config file path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load YAML config
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
            print(f"[Config] {config_path} not found, using command-line defaults")
    
    # Main parser with YAML defaults
    parser = argparse.ArgumentParser(description="Train MotionDiffusionCore")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML file")
    
    # Dataset paths (from YAML or command line)
    parser.add_argument("--humanml3d_path", type=str, default=config.get("humanml3d_path"))
    parser.add_argument("--babel_stream_path", type=str, default=config.get("babel_stream_path"))
    parser.add_argument("--humanml3d_stream_path", type=str, default=config.get("humanml3d_stream_path"))
    parser.add_argument("--meta_dir", type=str, default=config.get("meta_dir"))
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=config.get("datasets", ["humanml3d", "babel_stream", "humanml3d_stream"]),
        choices=["humanml3d", "babel_stream", "humanml3d_stream"],
    )
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size", 256))
    parser.add_argument("--num_workers", type=int, default=config.get("num_workers", 8))
    parser.add_argument("--total_iter", type=int, default=config.get("total_iter", 100000))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.0))
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=config.get("hidden_size", 512))
    parser.add_argument("--latent_dim", type=int, default=config.get("latent_dim", 16))
    parser.add_argument("--motion_dim", type=int, default=config.get("motion_dim", 38))
    parser.add_argument("--history_len", type=int, default=config.get("history_len", 60))
    parser.add_argument("--pred_len", type=int, default=config.get("pred_len", 5))
    
    # Transformer encoder-decoder parameters
    parser.add_argument("--n_decoder_layers", type=int, default=config.get("n_decoder_layers", 6))
    parser.add_argument("--n_encoder_layers", type=int, default=config.get("n_encoder_layers", 4))
    parser.add_argument("--n_heads", type=int, default=config.get("n_heads", 8))
    
    # Diffusion parameters
    parser.add_argument("--num_sampling_steps", type=int, default=config.get("num_sampling_steps", 10))
    parser.add_argument("--num_train_timesteps", type=int, default=config.get("num_train_timesteps", 1000))
    parser.add_argument("--beta_schedule", type=str, default=config.get("beta_schedule", "squaredcos_cap_v2"))
    parser.add_argument("--prediction_type", type=str, default=config.get("prediction_type", "sample"),
                        choices=["sample", "epsilon", "v_prediction"])
    parser.add_argument("--diffusion_width", type=int, default=config.get("diffusion_width", 512))
    
    # Classifier-Free Guidance
    parser.add_argument("--cfg_mask_prob", type=float, default=config.get("cfg_mask_prob", 0.0),
                        help="Probability of masking text during training (0.0=no CFG, 0.1=10%% masking)")
    parser.add_argument("--grad_checkpointing", type=bool, default=config.get("grad_checkpointing", False))
    
    # Root loss weighting
    parser.add_argument("--root_loss_weight", type=float, default=config.get("root_loss_weight", 1.0),
                        help="Loss weight multiplier for root features (dims 29-37). Higher = more root movement")
    
    # Text encoder
    parser.add_argument("--text_encoder_type", type=str, default=config.get("text_encoder_type", "t5"), 
                        choices=["bge", "t5"])
    parser.add_argument("--text_encoder", type=str, default=config.get("text_encoder", ""))
    parser.add_argument("--text_encoder_device", type=str, default=config.get("text_encoder_device", "cuda"))
    
    # Other
    parser.add_argument("--resume_trans", type=str, default=config.get("resume_trans"))
    parser.add_argument("--seed", type=int, default=config.get("seed", 123))
    parser.add_argument("--exp_name", type=str, default=config.get("exp_name", "motion_diff"))
    
    args = parser.parse_args()
    
    # Validate required paths (stable_utils_dir is optional, will auto-detect)
    required_paths = ["humanml3d_path", "babel_stream_path", "humanml3d_stream_path", "meta_dir"]
    for path_name in required_paths:
        if getattr(args, path_name) is None:
            raise ValueError(f"Missing required path: {path_name}. Set it in config YAML or command line.")
    
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Separate outputs (checkpoints) and logs (training logs & tensorboard)
    output_dir = os.path.join(repo_root, "outputs", args.exp_name)
    log_dir = os.path.join(repo_root, "logs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Disable finding unused parameters for DDP (all parameters are used in forward pass)
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    if accelerator.is_main_process:
        logger = get_logger(log_dir)
        writer = SummaryWriter(log_dir)
        logger.info(f"Process PID: {os.getpid()}")
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    else:
        logger = None
        writer = None

    text_encoder_path = resolve_text_encoder_path(
        args.text_encoder_type, args.text_encoder, repo_root
    )
    text_encoder, text_encoder_dim = load_text_encoder(
        args.text_encoder_type, text_encoder_path, args.text_encoder_device
    )

    history_len = args.history_len
    pred_len = args.pred_len

    model = MotionDiffusionModel(
        input_dim=args.motion_dim,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        text_encoder_dim=text_encoder_dim,
        history_len=history_len,
        pred_len=pred_len,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        diffusion_width=args.diffusion_width,
        grad_checkpointing=args.grad_checkpointing,
        # Transformer encoder-decoder config
        n_decoder_layers=args.n_decoder_layers,
        n_encoder_layers=args.n_encoder_layers,
        n_heads=args.n_heads,
        # Root loss weighting
        root_loss_weight=args.root_loss_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )

    nb_iter = load_checkpoint(model, args.resume_trans, optimizer)
    model.train()
    model.to(device)

    # Build dataset paths mapping
    dataset_paths = {
        "humanml3d": args.humanml3d_path,
        "babel_stream": args.babel_stream_path,
        "humanml3d_stream": args.humanml3d_stream_path,
    }
    
    train_loader = get_dataloader(
        datasets=args.datasets,
        dataset_paths=dataset_paths,
        meta_dir=args.meta_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_len=history_len,
        pred_len=pred_len,
        unit_length=1,
    )

    scheduler = WarmupCosineScheduler(optimizer, args.total_iter // 10, args.total_iter)

    text_encoder, model, optimizer, train_loader = accelerator.prepare(
        text_encoder, model, optimizer, train_loader
    )
    
    # Ensure text encoder stays in eval mode (critical for stable text encoding)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    train_loader_iter = cycle(train_loader)

    # Log CFG configuration
    if accelerator.is_main_process:
        if args.cfg_mask_prob > 0.0:
            logger.info(f"CFG enabled: masking {args.cfg_mask_prob*100:.1f}% of captions during training")
        else:
            logger.info("CFG disabled: training without caption masking (following Pi0's approach)")

    avg_loss = 0.0
    while nb_iter <= args.total_iter:
        batch = next(train_loader_iter)
        caption, history, target, history_mask = batch
        caption = list(caption)
        history = history.to(device).float()
        target = target.to(device).float()
        history_mask = history_mask.to(device)  # [B, history_len], bool

        bs = len(caption)
        # Classifier-Free Guidance: mask captions based on cfg_mask_prob
        # cfg_mask_prob = 0.0: no masking (follows Pi0's approach)
        # cfg_mask_prob > 0.0: mask captions for CFG training
        if args.cfg_mask_prob > 0.0:
            num_masked = max(1, int(bs * args.cfg_mask_prob))
            mask_indices = random.sample(range(bs), num_masked)
            for idx in mask_indices:
                caption[idx] = ""

        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        feat_text = torch.from_numpy(unwrapped_text_encoder.encode(caption)).float()
        feat_text = feat_text.to(device)

        loss, _ = model(history, feat_text, target, history_mask=history_mask)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(nb_iter)

        avg_loss += loss.item()
        nb_iter += 1

        if nb_iter % 100 == 0:
            if accelerator.is_main_process:
                avg_loss = avg_loss / 100
                writer.add_scalar("Loss/train", avg_loss, nb_iter)
                writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], nb_iter)
                logger.info(f"Iter {nb_iter} : Loss {avg_loss:.5f}")
            avg_loss = 0.0

        if nb_iter % 10000 == 0:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_dict = {
                        "token_mlp": unwrapped_model.token_mlp.state_dict(),
                        "action_diffusion": unwrapped_model.action_diffusion.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": nb_iter,
                        "args": vars(args),
                    }
                torch.save(
                    save_dict,
                    os.path.join(output_dir, "latest.pth"),
                )
                logger.info(f"Saved checkpoint at iter {nb_iter}")

        # Save permanent checkpoints every 100K iterations
        if nb_iter % 100000 == 0:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_dict = {
                        "token_mlp": unwrapped_model.token_mlp.state_dict(),
                        "action_diffusion": unwrapped_model.action_diffusion.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": nb_iter,
                        "args": vars(args),
                    }
                torch.save(
                    save_dict,
                    os.path.join(output_dir, f"ckpt_{nb_iter}.pth"),
                )
                logger.info(f"Saved permanent checkpoint at iter {nb_iter}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
