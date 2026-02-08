"""Streaming inference: generate motion from 60-frame history + text condition."""

import os
import sys
import argparse

import numpy as np
import torch

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.motion_diffusion import MotionDiffusionModel
from utils.text_encoder import load_text_encoder, resolve_text_encoder_path
from utils.motion_npz import save_motion_npz


def build_history_window(history, history_len, input_dim):
    """Build history window with padding if needed.
    
    Returns:
        history: [history_len, input_dim] - History frames
        mask: [history_len] - True for valid frames, False for padding
    """
    mask = np.ones(history_len, dtype=bool)
    
    if history is None:
        # All padding
        mask[:] = False
        return np.zeros((history_len, input_dim), dtype=np.float32), mask
    
    if history.shape[0] >= history_len:
        # No padding needed
        return history[-history_len:], mask
    
    # Need padding
    pad_count = history_len - history.shape[0]
    pad = np.zeros((pad_count, history.shape[1]), dtype=history.dtype)
    history_padded = np.concatenate([pad, history], axis=0)
    mask[:pad_count] = False  # Mark padding positions
    return history_padded, mask


def encode_text(text_encoder, text, device):
    """Encode text to token-level features.
    
    Returns:
        feat: (1, text_max_length, text_dim) tensor
        mask: (1, text_max_length) bool tensor
    """
    feat_np, mask_np = text_encoder.encode(text)
    # Single string returns (text_max_length, dim) and (text_max_length,)
    feat = torch.from_numpy(feat_np).float().to(device)
    mask = torch.from_numpy(mask_np).to(device)
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
        mask = mask.unsqueeze(0)
    return feat, mask


def main():
    parser = argparse.ArgumentParser("Streaming motion generation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--history_npy", type=str, default="", help="Path to history motion (optional)")
    parser.add_argument("--mean", type=str, required=True, help="Mean.npy path")
    parser.add_argument("--std", type=str, required=True, help="Std.npy path")
    parser.add_argument("--text", type=str, default="A robot walks forward.")
    parser.add_argument("--text_encoder", type=str, default="")
    parser.add_argument("--text_encoder_type", type=str, default="t5", choices=["bge", "t5"])
    parser.add_argument("--text_encoder_device", type=str, default="cuda")
    parser.add_argument("--text_max_length", type=int, default=60, help="Max text token length")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--history_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=512)
    
    # Transformer encoder-decoder
    parser.add_argument("--n_decoder_layers", type=int, default=6)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    
    # Diffusion config
    parser.add_argument("--num_sampling_steps", type=int, default=10, help="DDIM sampling steps (10=fast, 50=quality)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Training timesteps (must match training)")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2", help="Noise schedule")
    parser.add_argument("--prediction_type", type=str, default="sample", choices=["sample", "epsilon"], help="Prediction type")
    parser.add_argument("--diffusion_width", type=int, default=512, help="Diffusion head hidden dim")
    
    # Sampling config
    parser.add_argument("--cfg", type=float, default=7.5, help="Classifier-free guidance scale (1.0=off, 7.5=recommended)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_motion_length", type=int, default=150, help="Total frames to generate (ignored if --num_iterations is set)")
    parser.add_argument("--num_iterations", type=int, default=None, help="Total iteration count (overrides max_motion_length if set)")
    parser.add_argument("--slide_step", type=int, default=None, help="Sliding step: keep first N predicted frames per iter (default=pred_len, set 1 for single-frame sliding)")
    parser.add_argument("--fps", type=int, default=50, help="FPS for output")
    parser.add_argument("--smooth", action="store_true", help="Apply smoothing")
    parser.add_argument("--slide_window_size", type=int, default=None, help="[Deprecated] Use --slide_step instead")
    parser.add_argument("--out_dir", type=str, default="output")
    args = parser.parse_args()

    device = torch.device(args.device)
    text_encoder_path = resolve_text_encoder_path(args.text_encoder_type, args.text_encoder, repo_root)
    text_encoder, text_encoder_dim = load_text_encoder(
        args.text_encoder_type, text_encoder_path, args.text_encoder_device,
        max_length=args.text_max_length
    )

    mean = np.load(args.mean)
    std = np.load(args.std)
    input_dim = int(mean.shape[0])
    
    # Load Checkpoint to check for config
    print(f"[Info] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    if "args" in ckpt:
        print("[Info] Found config in checkpoint, overwriting default args...")
        ckpt_args = ckpt["args"]
        # Whitelist of args to overwrite from checkpoint
        model_keys = [
            "motion_dim", "hidden_size", "latent_dim", 
            "n_decoder_layers", "n_encoder_layers", "n_heads",
            "history_len", "pred_len", "diffusion_width", "prediction_type", "beta_schedule",
            "num_train_timesteps", "text_max_length"
        ]
        for key in model_keys:
            if key in ckpt_args:
                val = ckpt_args[key]
                print(f"  - {key}: {getattr(args, key, 'None')} -> {val}")
                setattr(args, key, val)
    
    # Use motion_dim from args if available, otherwise input_dim from mean
    motion_dim = getattr(args, "motion_dim", input_dim)
    if motion_dim != input_dim:
        print(f"[Warning] args.motion_dim ({motion_dim}) != input_dim ({input_dim}). Using input_dim.")
        motion_dim = input_dim

    history = None
    if args.history_npy:
        history = np.load(args.history_npy).astype(np.float32)
        if history.ndim == 3:
            history = history.squeeze(0)
        if history.shape[1] != input_dim:
            raise ValueError(f"history_npy dim mismatch: {history.shape[1]} vs {input_dim}")
        history = (history - mean) / std

    history, history_mask = build_history_window(history, args.history_len, input_dim)

    # Initialize Model
    text_max_length = getattr(args, 'text_max_length', 60)
    model = MotionDiffusionModel(
        input_dim=motion_dim,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        text_encoder_dim=text_encoder_dim,
        text_max_length=text_max_length,
        history_len=args.history_len,
        pred_len=args.pred_len,
        device=device,
        num_sampling_steps=args.num_sampling_steps,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        diffusion_width=args.diffusion_width,
        n_decoder_layers=args.n_decoder_layers,
        n_encoder_layers=args.n_encoder_layers,
        n_heads=args.n_heads,
    ).to(device)
    
    # Load weights
    # New architecture: no trans_encoder, only token_mlp and action_diffusion
    model_state = model.state_dict()
    loaded_state = {}
    
    # Checkpoint structure: {"token_mlp": ..., "action_diffusion": ...}
    key_mapping = {
        "token_mlp": "token_mlp",
        "action_diffusion": "action_diffusion"
    }
    
    for ckpt_prefix, model_prefix in key_mapping.items():
        if ckpt_prefix in ckpt:
            sub_state = ckpt[ckpt_prefix]
            for key, val in sub_state.items():
                # Handle "module." prefix if it exists in sub_state keys (DDP artifact)
                key_clean = ".".join(key.split(".")[1:]) if key.startswith("module.") else key
                full_key = f"{model_prefix}.{key_clean}"
                if full_key in model_state:
                    loaded_state[full_key] = val
                else:
                    print(f"[Warning] Key {full_key} not found in model")
    
    model.load_state_dict(loaded_state, strict=False)
    model.eval()

    feat_text, text_mask = encode_text(text_encoder, args.text, device)

    # Sliding step: keep first N predicted frames per iteration
    slide_step = args.slide_step if args.slide_step is not None else args.pred_len
    if slide_step > args.pred_len:
        print(f"[Warning] slide_step ({slide_step}) > pred_len ({args.pred_len}), clamping to pred_len")
        slide_step = args.pred_len
    
    # Determine number of iterations
    if args.num_iterations is not None:
        num_iterations = args.num_iterations
        estimated_frames = num_iterations * slide_step
        print(f"[Info] Using --num_iterations={num_iterations}, estimated output: ~{estimated_frames} frames")
    else:
        num_iterations = (args.max_motion_length + slide_step - 1) // slide_step
        print(f"[Info] Generating {num_iterations} iterations (~{args.max_motion_length} frames)")

    all_predictions = []
    current_history = history.copy()
    current_mask = history_mask.copy()

    print(f"[Info] Sliding config: pred_len={args.pred_len}, slide_step={slide_step} (keep first {slide_step} frames per iter)")

    with torch.no_grad():
        for iter_idx in range(num_iterations):
            # Get last history_len frames and corresponding mask
            history_tensor = torch.from_numpy(current_history[-args.history_len:]).unsqueeze(0).to(device).float()
            mask_tensor = torch.from_numpy(current_mask[-args.history_len:]).unsqueeze(0).to(device)
            
            # Predict using model method (always predicts pred_len frames)
            # CFG uses zero vectors internally (no need for empty_feat_text)
            pred_tensor = model.predict(
                history_tensor, 
                feat_text, 
                cfg_scale=args.cfg, 
                temperature=args.temperature,
                history_mask=mask_tensor,
                text_mask=text_mask
            )
            
            pred = pred_tensor.squeeze(0).cpu().numpy()  # (pred_len, 38)
            
            # Keep only first slide_step frames
            pred_to_keep = pred[:slide_step]  # (slide_step, 38)
            pred_denorm = pred_to_keep * std + mean

            all_predictions.append(pred_denorm)
            
            # Slide history window by slide_step frames
            current_history = np.concatenate([current_history, pred_to_keep], axis=0)
            new_mask = np.ones(slide_step, dtype=bool)
            current_mask = np.concatenate([current_mask, new_mask], axis=0)

            print(f"[Info] Iteration {iter_idx + 1}/{num_iterations}: Predicted {pred.shape[0]} frames, kept {pred_to_keep.shape[0]} frames")

    final_motion = np.concatenate(all_predictions, axis=0)
    
    # Trim to max_motion_length only if num_iterations was not explicitly set
    if args.num_iterations is None:
        final_motion = final_motion[:args.max_motion_length]
    
    print(f"[Info] Final motion shape: {final_motion.shape}")

    os.makedirs(args.out_dir, exist_ok=True)

    out_npy = os.path.join(args.out_dir, "motion_38d.npy")
    np.save(out_npy, final_motion.astype(np.float32))
    print(f"[Info] Saved raw motion to {out_npy}")

    final_npz = os.path.join(args.out_dir, "motion_g1_minimal.npz")
    save_motion_npz(final_motion, final_npz, fps=args.fps, smooth=args.smooth)
    print(f"[Info] Saved final motion to {final_npz}")


if __name__ == "__main__":
    main()
