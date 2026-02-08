"""Train MotionDiffusionCore on a single data sample to test overfitting capability.

This script is useful for debugging and verifying that the model can learn.
It loads only one specific data file (e.g., 000000.npz) and trains until loss approaches zero.
"""

import os
import sys
import json
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import MotionDiffusionModel
from utils.text_encoder import load_text_encoder, resolve_text_encoder_path
from utils.logger import get_logger


class SingleSampleDataset:
    """Dataset that loads a single motion file and creates multiple samples from it."""
    
    def __init__(self, npz_path, text_path, meta_dir, history_len=60, pred_len=5):
        self.history_len = history_len
        self.pred_len = pred_len
        
        # Load normalization statistics
        self.mean = np.load(os.path.join(meta_dir, "Mean.npy"))
        self.std = np.load(os.path.join(meta_dir, "Std.npy"))
        
        # Load motion data
        from utils.robot_process import process_robot_npz
        npz = np.load(npz_path)
        motion = process_robot_npz(npz, root_idx=0)
        
        # Normalize
        motion = (motion - self.mean) / self.std
        
        # Load text annotations
        with open(text_path, 'r') as f:
            lines = f.readlines()
        
        # Parse text annotations
        self.samples = []
        full_text_list = []
        
        for line in lines:
            line_split = line.strip().split("#")
            caption = line_split[0]
            
            if len(line_split) >= 4:
                try:
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                except (ValueError, IndexError):
                    f_tag = 0.0
                    to_tag = 0.0
                
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                
                if f_tag == 0.0 and to_tag == 0.0:
                    full_text_list.append(caption)
                else:
                    # Segment-specific caption
                    start = int(f_tag * 50)  # fps=50
                    end = int(to_tag * 50)
                    if end > start and end <= len(motion):
                        motion_segment = motion[start:end]
                        if len(motion_segment) >= self.pred_len:
                            self.samples.append({
                                "motion": motion_segment.astype(np.float32),
                                "caption": caption
                            })
            else:
                full_text_list.append(caption)
        
        # Add full motion with all captions
        if len(motion) >= self.pred_len:
            for caption in full_text_list:
                self.samples.append({
                    "motion": motion.astype(np.float32),
                    "caption": caption
                })
        
        if not self.samples:
            raise ValueError(f"No valid samples found in {npz_path}")
        
        print(f"Loaded {len(self.samples)} samples from {npz_path}")
        for i, sample in enumerate(self.samples):
            print(f"  Sample {i}: motion_len={len(sample['motion'])}, caption='{sample['caption']}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Sample a random window from the motion sequence."""
        sample = self.samples[idx % len(self.samples)]
        motion = sample["motion"]
        caption = sample["caption"]
        
        # Sample a random step
        motion_len = len(motion)
        max_step = motion_len - self.pred_len - 1
        
        if max_step < 0:
            # Motion too short, use what we have
            step_idx = 0
        else:
            step_idx = np.random.randint(0, max_step + 1)
        
        # Build history with zero-padding if needed
        start = max(0, step_idx - self.history_len + 1)
        history = motion[start:step_idx + 1]
        
        # Create mask: True for valid frames, False for padding
        history_mask = np.ones(self.history_len, dtype=bool)
        
        if len(history) < self.history_len:
            pad_count = self.history_len - len(history)
            pad = np.zeros((pad_count, motion.shape[1]), dtype=motion.dtype)
            history = np.concatenate([pad, history], axis=0)
            history_mask[:pad_count] = False
        
        # Get target
        target = motion[step_idx + 1:step_idx + 1 + self.pred_len]
        if len(target) < self.pred_len:
            # Pad target if needed (edge case)
            pad_count = self.pred_len - len(target)
            pad = np.zeros((pad_count, motion.shape[1]), dtype=motion.dtype)
            target = np.concatenate([target, pad], axis=0)
        
        return caption, history.astype(np.float32), target.astype(np.float32), history_mask


def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Overfit MotionDiffusionCore on single sample")
    
    # Required: path to single data file
    parser.add_argument("--npz_path", type=str, required=True,
                        help="Path to single .npz file (e.g., 000000.npz)")
    parser.add_argument("--text_path", type=str, required=True,
                        help="Path to corresponding .txt file (e.g., 000000.txt)")
    parser.add_argument("--meta_dir", type=str, required=True,
                        help="Path to statistics directory (Mean.npy, Std.npy)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (small for overfitting)")
    parser.add_argument("--total_iter", type=int, default=5000,
                        help="Total training iterations")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (higher for faster overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (0 for overfitting)")
    
    # Model parameters (should match your trained model)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--motion_dim", type=int, default=38)
    parser.add_argument("--history_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=5)
    
    # Transformer encoder-decoder parameters
    parser.add_argument("--n_decoder_layers", type=int, default=6)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    
    # Diffusion parameters
    parser.add_argument("--num_sampling_steps", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", type=str, default="sample")
    parser.add_argument("--diffusion_width", type=int, default=512)
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    
    # Root loss weighting
    parser.add_argument("--root_loss_weight", type=float, default=1.0)
    
    # Text encoder
    parser.add_argument("--text_encoder_type", type=str, default="t5")
    parser.add_argument("--text_encoder", type=str, default="")
    parser.add_argument("--text_encoder_device", type=str, default="cuda")
    parser.add_argument("--text_max_length", type=int, default=60)
    
    # Other
    parser.add_argument("--resume_trans", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--exp_name", type=str, default="overfit_single")
    
    # Logging frequency
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N iterations")
    
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup directories
    output_dir = os.path.join(repo_root, "outputs", args.exp_name)
    log_dir = os.path.join(repo_root, "logs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Simple setup without DDP for single-GPU debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = get_logger(log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(f"Process PID: {os.getpid()}")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Text encoder config
    text_encoder_path = resolve_text_encoder_path(
        args.text_encoder_type, args.text_encoder, repo_root
    )
    text_encoder, text_encoder_dim = load_text_encoder(
        args.text_encoder_type, text_encoder_path, args.text_encoder_device,
        max_length=args.text_max_length
    )
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

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
    model.to(device)
    model.train()

    # Optimizer with higher LR for faster overfitting
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Load checkpoint if provided
    if args.resume_trans:
        logger.info(f"Loading checkpoint from {args.resume_trans}")
        ckpt = torch.load(args.resume_trans, map_location="cpu")
        if "token_mlp" in ckpt:
            model.token_mlp.load_state_dict(ckpt["token_mlp"])
        if "action_diffusion" in ckpt:
            model.action_diffusion.load_state_dict(ckpt["action_diffusion"])
        logger.info("Checkpoint loaded")

    # Create single-sample dataset
    logger.info(f"Loading single sample from {args.npz_path}")
    dataset = SingleSampleDataset(
        npz_path=args.npz_path,
        text_path=args.text_path,
        meta_dir=args.meta_dir,
        history_len=args.history_len,
        pred_len=args.pred_len,
    )

    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Single process for debugging
        drop_last=False,
    )

    logger.info(f"Starting overfitting training on {len(dataset)} samples")
    logger.info(f"Target: Loss should approach 0 if model can learn")

    # Training loop
    nb_iter = 0
    losses = []
    
    while nb_iter < args.total_iter:
        for batch in dataloader:
            caption, history, target, history_mask = batch
            caption = list(caption)
            history = history.to(device).float()
            target = target.to(device).float()
            history_mask = history_mask.to(device)

            # Encode text
            feat_text_np, text_mask_np = text_encoder.encode(caption)
            feat_text = torch.from_numpy(feat_text_np).float().to(device)
            text_mask = torch.from_numpy(text_mask_np).to(device)

            # Forward pass
            loss, _ = model(history, feat_text, target, history_mask=history_mask, text_mask=text_mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            nb_iter += 1

            # Log frequently to monitor overfitting
            if nb_iter % args.log_every == 0:
                avg_loss = np.mean(losses)
                writer.add_scalar("Loss/train", avg_loss, nb_iter)
                writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], nb_iter)
                logger.info(f"Iter {nb_iter}/{args.total_iter} : Loss {avg_loss:.6f}")
                losses = []

            # Save checkpoint every 5000 iterations
            if nb_iter % 5000 == 0:
                save_dict = {
                    "token_mlp": model.token_mlp.state_dict(),
                    "action_diffusion": model.action_diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": nb_iter,
                    "args": vars(args),
                }
                torch.save(save_dict, os.path.join(output_dir, f"ckpt_{nb_iter}.pth"))
                logger.info(f"Saved checkpoint at iter {nb_iter}")

            if nb_iter >= args.total_iter:
                break

    # Final save
    save_dict = {
        "token_mlp": model.token_mlp.state_dict(),
        "action_diffusion": model.action_diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": nb_iter,
        "args": vars(args),
    }
    torch.save(save_dict, os.path.join(output_dir, "final.pth"))
    logger.info("Training completed!")
    logger.info(f"Final checkpoint saved to {output_dir}/final.pth")


if __name__ == "__main__":
    main()
