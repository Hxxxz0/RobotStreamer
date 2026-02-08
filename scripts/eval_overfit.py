"""Evaluate overfitting model by generating predictions on the training sample."""

import os
import sys
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import MotionDiffusionModel
from utils.text_encoder import load_text_encoder, resolve_text_encoder_path
from utils.robot_process import process_robot_npz
from utils.motion_npz import save_motion_npz


def evaluate_overfit(ckpt_path, npz_path, text_path, meta_dir, output_dir=None, generate_full=False, max_frames=None):
    """Generate predictions and compare with ground truth.
    
    Args:
        ckpt_path: Path to model checkpoint
        npz_path: Path to ground truth motion npz
        text_path: Path to text description
        meta_dir: Directory containing Mean.npy and Std.npy
        output_dir: Directory to save generated motion (if generate_full=True)
        generate_full: If True, generate full sequence like streaming inference
        max_frames: Maximum frames to generate (if None, generates same length as GT)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_dict = ckpt["args"]
    
    # Load text encoder
    text_encoder_path = resolve_text_encoder_path(
        args_dict["text_encoder_type"], args_dict["text_encoder"], repo_root
    )
    text_encoder, text_encoder_dim = load_text_encoder(
        args_dict["text_encoder_type"], text_encoder_path, 
        args_dict["text_encoder_device"], max_length=args_dict["text_max_length"]
    )
    text_encoder.eval()
    
    # Create model
    model = MotionDiffusionModel(
        input_dim=args_dict["motion_dim"],
        hidden_size=args_dict["hidden_size"],
        latent_dim=args_dict["latent_dim"],
        text_encoder_dim=text_encoder_dim,
        text_max_length=args_dict["text_max_length"],
        history_len=args_dict["history_len"],
        pred_len=args_dict["pred_len"],
        device=device,
        num_sampling_steps=args_dict["num_sampling_steps"],
        num_train_timesteps=args_dict["num_train_timesteps"],
        beta_schedule=args_dict["beta_schedule"],
        prediction_type=args_dict["prediction_type"],
        diffusion_width=args_dict["diffusion_width"],
        grad_checkpointing=False,
        n_decoder_layers=args_dict["n_decoder_layers"],
        n_encoder_layers=args_dict["n_encoder_layers"],
        n_heads=args_dict["n_heads"],
        root_loss_weight=args_dict["root_loss_weight"],
    )
    model.to(device)
    model.token_mlp.load_state_dict(ckpt["token_mlp"])
    model.action_diffusion.load_state_dict(ckpt["action_diffusion"])
    model.eval()
    
    # Load motion data
    print(f"Loading motion: {npz_path}")
    mean = np.load(os.path.join(meta_dir, "Mean.npy"))
    std = np.load(os.path.join(meta_dir, "Std.npy"))
    
    npz = np.load(npz_path)
    motion_raw = process_robot_npz(npz, root_idx=0)  # Keep raw for comparison
    motion = (motion_raw - mean) / std
    
    # Load text
    with open(text_path, 'r') as f:
        lines = f.readlines()
    caption = lines[0].strip().split("#")[0]
    
    print(f"Caption: '{caption}'")
    print(f"GT Motion shape: {motion.shape}")
    
    history_len = args_dict["history_len"]
    pred_len = args_dict["pred_len"]
    
    if generate_full:
        # Generate full sequence like streaming inference
        print("\n" + "="*60)
        print("Generating full sequence (streaming mode)")
        print("="*60)
        
        # Encode text once
        feat_text_np, text_mask_np = text_encoder.encode([caption])
        feat_text = torch.from_numpy(feat_text_np).float().to(device)
        text_mask = torch.from_numpy(text_mask_np).to(device)
        
        # Start with ground truth history
        current_history = motion[:history_len].copy()
        current_mask = np.ones(history_len, dtype=bool)
        
        all_predictions = []
        target_length = max_frames if max_frames is not None else len(motion)
        num_iterations = (target_length - history_len + pred_len - 1) // pred_len
        
        print(f"Starting from GT history: {history_len} frames")
        print(f"Target length: {target_length} frames")
        print(f"Iterations: {num_iterations}")
        
        with torch.no_grad():
            for iter_idx in range(num_iterations):
                # Get last history_len frames
                history_tensor = torch.from_numpy(current_history[-history_len:]).unsqueeze(0).to(device).float()
                mask_tensor = torch.from_numpy(current_mask[-history_len:]).unsqueeze(0).to(device)
                
                # Generate prediction
                pred_tensor = model.predict(
                    history_tensor, feat_text,
                    cfg_scale=1.0,
                    temperature=1.0,
                    history_mask=mask_tensor, text_mask=text_mask
                )
                
                pred = pred_tensor.squeeze(0).cpu().numpy()  # (pred_len, D)
                
                # Denormalize
                pred_denorm = pred * std + mean
                all_predictions.append(pred_denorm)
                
                # Update history for next iteration
                current_history = np.concatenate([current_history, pred], axis=0)
                new_mask = np.ones(pred_len, dtype=bool)
                current_mask = np.concatenate([current_mask, new_mask], axis=0)
                
                current_len = len(current_history)
                print(f"Iteration {iter_idx + 1}/{num_iterations}: Generated {current_len} frames")
        
        # Combine: initial GT history + all predictions
        initial_history_denorm = motion[:history_len] * std + mean
        generated_motion = np.concatenate([initial_history_denorm] + all_predictions, axis=0)
        generated_motion = generated_motion[:target_length]  # Trim to target length
        
        print(f"\nGenerated motion shape: {generated_motion.shape}")
        
        # Save generated motion
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save raw npy
            out_npy = os.path.join(output_dir, "generated_motion_38d.npy")
            np.save(out_npy, generated_motion.astype(np.float32))
            print(f"Saved raw motion to {out_npy}")
            
            # Save npz with metadata
            out_npz = os.path.join(output_dir, "generated_motion.npz")
            save_motion_npz(generated_motion, out_npz, fps=50, smooth=False)
            print(f"Saved motion npz to {out_npz}")
            
            # Also save ground truth for comparison
            gt_npz = os.path.join(output_dir, "ground_truth_motion.npz")
            save_motion_npz(motion_raw[:target_length], gt_npz, fps=50, smooth=False)
            print(f"Saved ground truth to {gt_npz}")
        
        # Calculate overall error
        overlap_len = min(len(generated_motion), len(motion_raw))
        mse = np.mean((generated_motion[:overlap_len] - motion_raw[:overlap_len]) ** 2)
        mae = np.mean(np.abs(generated_motion[:overlap_len] - motion_raw[:overlap_len]))
        
        print("\n" + "="*60)
        print("Overall Comparison (Generated vs Ground Truth)")
        print("="*60)
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Generated range: [{generated_motion.min():.3f}, {generated_motion.max():.3f}]")
        print(f"GT range: [{motion_raw[:overlap_len].min():.3f}, {motion_raw[:overlap_len].max():.3f}]")
        
    else:
        # Original spot-check evaluation
        test_steps = [30, 60, 90, 120]  # Different starting points
        
        print("\n" + "="*60)
        print("Testing predictions at different time steps")
        print("="*60)
        
        errors = []
        
        for step_idx in test_steps:
            if step_idx + pred_len >= len(motion):
                continue
                
            # Build history
            start = max(0, step_idx - history_len + 1)
            history = motion[start:step_idx + 1]
            history_mask = np.ones(history_len, dtype=bool)
            
            if len(history) < history_len:
                pad_count = history_len - len(history)
                pad = np.zeros((pad_count, motion.shape[1]), dtype=motion.dtype)
                history = np.concatenate([pad, history], axis=0)
                history_mask[:pad_count] = False
            
            # Get ground truth
            gt_target = motion[step_idx + 1:step_idx + 1 + pred_len]
            
            # Encode text
            feat_text_np, text_mask_np = text_encoder.encode([caption])
            feat_text = torch.from_numpy(feat_text_np).float().to(device)
            text_mask = torch.from_numpy(text_mask_np).to(device)
            
            # Prepare inputs
            history_tensor = torch.from_numpy(history[None]).float().to(device)  # [1, H, D]
            history_mask_tensor = torch.from_numpy(history_mask[None]).to(device)  # [1, H]
            
            # Generate prediction
            with torch.no_grad():
                pred = model.predict(
                    history_tensor, feat_text, 
                    cfg_scale=1.0,
                    temperature=1.0,
                    history_mask=history_mask_tensor, text_mask=text_mask
                )  # [1, pred_len, D]
            
            pred_np = pred.cpu().numpy()[0]  # [pred_len, D]
            
            # Calculate error
            mse = np.mean((pred_np - gt_target) ** 2)
            mae = np.mean(np.abs(pred_np - gt_target))
            
            errors.append(mse)
            
            print(f"\nStep {step_idx}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Pred range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
            print(f"  GT range: [{gt_target.min():.3f}, {gt_target.max():.3f}]")
        
        print("\n" + "="*60)
        print(f"Average MSE: {np.mean(errors):.6f}")
        print("="*60)
        
        # Interpretation
        avg_mse = np.mean(errors)
        if avg_mse < 0.01:
            print("\n✅ EXCELLENT: Model has successfully overfit (MSE < 0.01)")
        elif avg_mse < 0.05:
            print("\n✅ GOOD: Model has learned well (MSE < 0.05)")
        elif avg_mse < 0.1:
            print("\n⚠️  FAIR: Model is learning but not perfect (MSE < 0.1)")
        else:
            print("\n❌ POOR: Model may not be overfitting properly (MSE >= 0.1)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--npz_path", type=str, 
                        default="/limx_embap/tos/user/Jensen/dataset/robot_humanml_data/npz/000000.npz")
    parser.add_argument("--text_path", type=str,
                        default="/limx_embap/tos/user/Jensen/dataset/robot_humanml_data/texts/000000.txt")
    parser.add_argument("--meta_dir", type=str,
                        default="/limx_embap/tos/user/Jensen/project/dataset/statistics")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for generated motion (only used with --generate_full)")
    parser.add_argument("--generate_full", action="store_true",
                        help="Generate full sequence like streaming inference (default: spot-check only)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to generate (default: same as GT length)")
    
    args = parser.parse_args()
    
    evaluate_overfit(args.ckpt, args.npz_path, args.text_path, args.meta_dir, 
                     args.output_dir, args.generate_full, args.max_frames)
