"""
验证 Exposure Bias：对比 Teacher Forcing vs Autoregressive Rollout
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.motion_diffusion import MotionDiffusionModel
from utils.text_encoder import load_text_encoder, resolve_text_encoder_path
from utils.motion_npz import save_motion_npz


def load_model(args, device):
    """加载模型"""
    print(f"[Info] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # 从 checkpoint 获取配置
    if "args" in ckpt:
        ckpt_args = ckpt["args"]
        for key in ["motion_dim", "hidden_size", "latent_dim", "history_len", "pred_len",
                    "n_decoder_layers", "n_encoder_layers", "n_heads", "diffusion_width",
                    "prediction_type", "beta_schedule", "num_train_timesteps", "text_max_length"]:
            if key in ckpt_args:
                setattr(args, key, ckpt_args[key])
    
    # 初始化模型
    model = MotionDiffusionModel(
        input_dim=args.motion_dim,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        text_encoder_dim=args.text_encoder_dim,
        text_max_length=getattr(args, 'text_max_length', 60),
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
    
    # 加载权重
    model_state = model.state_dict()
    loaded_state = {}
    
    for ckpt_prefix, model_prefix in [("token_mlp", "token_mlp"), ("action_diffusion", "action_diffusion")]:
        if ckpt_prefix in ckpt:
            sub_state = ckpt[ckpt_prefix]
            for key, val in sub_state.items():
                key_clean = ".".join(key.split(".")[1:]) if key.startswith("module.") else key
                full_key = f"{model_prefix}.{key_clean}"
                if full_key in model_state:
                    loaded_state[full_key] = val
    
    model.load_state_dict(loaded_state, strict=False)
    model.eval()
    print("[Info] Model loaded")
    
    return model


def read_motion_data(npz_path, mean, std):
    """读取动作数据并归一化"""
    from utils.robot_process import process_robot_npz
    
    data = np.load(npz_path)
    
    # 使用 robot_process 转换为 38D
    motion_38d = process_robot_npz(data, root_idx=0)
    
    # 归一化
    motion_normalized = (motion_38d - mean) / std
    
    return motion_normalized


def read_text(txt_path):
    """读取文本标注"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # HumanML3D 格式: caption#tokens#start#end
    # 取第一行的 caption
    caption = lines[0].split('#')[0].strip()
    return caption


def teacher_forcing_rollout(model, motion_gt, text_emb, text_mask, args, device, mean, std):
    """
    Teacher Forcing：从零历史开始，每次用 GT 历史预测
    
    关键：每次的历史窗口都从 GT 中提取（但初始是 padding）
    
    Returns:
        predictions: list of (pred_len, 38) - 每次预测的 5 帧
        errors: list of float - 每次预测的 MSE 误差
        losses: list of float - 每次预测的训练 loss（包括 root_loss_weight）
    """
    T = motion_gt.shape[0]
    history_len = args.history_len
    pred_len = args.pred_len
    
    predictions = []
    errors = []
    
    # 计算需要生成的步数：生成完整的 T 帧
    num_steps = (T + pred_len - 1) // pred_len
    
    print(f"[Teacher Forcing] Total steps: {num_steps}, Target frames: {T}")
    
    predictions = []
    errors = []
    losses = []
    
    with torch.no_grad():
        for step in range(num_steps):
            # 计算当前已生成的帧数
            generated_frames = step * pred_len
            
            if generated_frames >= T:
                break
            
            # 提取历史：从 GT 中取最近的 history_len 帧
            if generated_frames >= history_len:
                # 足够的历史
                history = motion_gt[generated_frames - history_len : generated_frames]
                history_mask_np = np.ones(history_len, dtype=bool)
            else:
                # 不足 history_len 帧，需要 padding
                pad_count = history_len - generated_frames
                if generated_frames > 0:
                    history = motion_gt[:generated_frames]
                    pad = np.zeros((pad_count, motion_gt.shape[1]), dtype=np.float32)
                    history = np.concatenate([pad, history], axis=0)
                else:
                    # 第一步：完全是 padding
                    history = np.zeros((history_len, motion_gt.shape[1]), dtype=np.float32)
                
                history_mask_np = np.zeros(history_len, dtype=bool)
                history_mask_np[pad_count:] = True  # 只有非 padding 部分是 True
            
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(device).float()
            history_mask_tensor = torch.from_numpy(history_mask_np).unsqueeze(0).to(device)
            
            # 预测
            pred_tensor = model.predict(
                history_tensor,
                text_emb,
                cfg_scale=args.cfg,
                temperature=args.temperature,
                history_mask=history_mask_tensor,
                text_mask=text_mask
            )
            
            pred = pred_tensor.squeeze(0).cpu().numpy()  # (5, 38)
            
            # 反归一化
            pred_denorm = pred * std + mean
            
            # GT（归一化和反归一化）
            gt_start = generated_frames
            gt_end = min(gt_start + pred_len, T)
            gt_segment = motion_gt[gt_start:gt_end]  # 归一化空间
            gt_denorm = gt_segment * std + mean  # 反归一化
            
            # 计算误差（反归一化空间的 MSE）
            pred_valid = pred_denorm[:len(gt_denorm)]
            error = np.mean((pred_valid - gt_denorm) ** 2)
            
            # 计算训练 loss（归一化空间 + root_loss_weight）
            pred_norm_valid = pred[:len(gt_segment)]  # (valid_len, 38)
            loss_per_dim = (pred_norm_valid - gt_segment) ** 2  # (valid_len, 38)
            
            # 应用 root_loss_weight（dims 29-37，与训练一致）
            loss_weights = np.ones(38, dtype=np.float32)
            if hasattr(model, 'action_diffusion'):
                root_loss_weight = float(model.action_diffusion.root_loss_weight)
                loss_weights[29:38] = root_loss_weight
            
            weighted_loss = loss_per_dim * loss_weights.reshape(1, -1)
            loss = float(weighted_loss.mean())
            
            predictions.append(pred_denorm)
            errors.append(error)
            losses.append(loss)
            
            if (step + 1) % 10 == 0 or step == 0:
                valid_frames = int(history_mask_np.sum())
                print(f"  Step {step+1}/{num_steps}, History: {valid_frames}/{history_len} valid, Error: {error:.6f}, Loss: {loss:.6f}")
    
    return predictions, errors, losses


def autoregressive_rollout(model, motion_gt, text_emb, text_mask, args, device, mean, std):
    """
    Autoregressive Rollout：从零历史开始，每次用模型预测的历史
    
    真实推理场景：完全依赖模型预测滚动
    
    Returns:
        predictions: list of (pred_len, 38) - 每次预测的 5 帧
        errors: list of float - 每次预测的 MSE 误差（与 GT 对比）
        losses: list of float - 每次预测的训练 loss（包括 root_loss_weight）
    """
    T = motion_gt.shape[0]
    history_len = args.history_len
    pred_len = args.pred_len
    
    predictions = []
    errors = []
    losses = []
    
    # 初始历史：全 padding（从零开始）
    current_history = np.zeros((0, motion_gt.shape[1]), dtype=np.float32)
    current_mask = np.zeros(0, dtype=bool)
    
    num_steps = (T + pred_len - 1) // pred_len
    
    print(f"[Autoregressive] Total steps: {num_steps}, Target frames: {T}")
    
    with torch.no_grad():
        for step in range(num_steps):
            generated_frames = step * pred_len
            
            if generated_frames >= T:
                break
            
            # 提取最近的 history_len 帧（初始全是 padding）
            if current_history.shape[0] >= history_len:
                history = current_history[-history_len:]
                mask = current_mask[-history_len:]
            else:
                # 不足 history_len，需要 padding
                pad_count = history_len - current_history.shape[0]
                pad = np.zeros((pad_count, motion_gt.shape[1]), dtype=np.float32)
                pad_mask = np.zeros(pad_count, dtype=bool)
                
                if current_history.shape[0] > 0:
                    history = np.concatenate([pad, current_history], axis=0)
                    mask = np.concatenate([pad_mask, current_mask], axis=0)
                else:
                    # 第一步：完全是 padding
                    history = pad
                    mask = pad_mask
            
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(device).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
            
            # 预测
            pred_tensor = model.predict(
                history_tensor,
                text_emb,
                cfg_scale=args.cfg,
                temperature=args.temperature,
                history_mask=mask_tensor,
                text_mask=text_mask
            )
            
            pred = pred_tensor.squeeze(0).cpu().numpy()  # (5, 38)
            
            # 反归一化
            pred_denorm = pred * std + mean
            
            # GT（归一化和反归一化）
            gt_start = generated_frames
            gt_end = min(gt_start + pred_len, T)
            gt_segment = motion_gt[gt_start:gt_end]  # 归一化空间
            gt_denorm = gt_segment * std + mean  # 反归一化
            
            # 计算误差（反归一化空间的 MSE）
            pred_valid = pred_denorm[:len(gt_denorm)]
            error = np.mean((pred_valid - gt_denorm) ** 2)
            
            # 计算训练 loss（归一化空间 + root_loss_weight）
            pred_norm_valid = pred[:len(gt_segment)]  # (valid_len, 38)
            loss_per_dim = (pred_norm_valid - gt_segment) ** 2  # (valid_len, 38)
            
            # 应用 root_loss_weight（dims 29-37，与训练一致）
            loss_weights = np.ones(38, dtype=np.float32)
            if hasattr(model, 'action_diffusion'):
                root_loss_weight = float(model.action_diffusion.root_loss_weight)
                loss_weights[29:38] = root_loss_weight
            
            weighted_loss = loss_per_dim * loss_weights.reshape(1, -1)
            loss = float(weighted_loss.mean())
            
            predictions.append(pred_denorm)
            errors.append(error)
            losses.append(loss)
            
            # 更新历史：追加预测（归一化空间）
            current_history = np.concatenate([current_history, pred], axis=0)
            new_mask = np.ones(pred_len, dtype=bool)
            current_mask = np.concatenate([current_mask, new_mask], axis=0)
            
            if (step + 1) % 10 == 0 or step == 0:
                valid_frames = int(mask.sum())
                print(f"  Step {step+1}/{num_steps}, History: {valid_frames}/{history_len} valid, Error: {error:.6f}, Loss: {loss:.6f}")
    
    return predictions, errors, losses


def main():
    parser = argparse.ArgumentParser("验证 Exposure Bias")
    parser.add_argument("--ckpt", type=str, required=True, help="模型checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="HumanML3D 数据根目录")
    parser.add_argument("--sample_id", type=str, default="000000", help="样本ID（如 000000）")
    parser.add_argument("--mean", type=str, required=True)
    parser.add_argument("--std", type=str, required=True)
    parser.add_argument("--text_encoder_type", type=str, default="t5")
    parser.add_argument("--text_encoder", type=str, default="")
    parser.add_argument("--text_encoder_device", type=str, default="cuda")
    parser.add_argument("--text_max_length", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    
    # 模型参数（会从 checkpoint 覆盖）
    parser.add_argument("--history_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--motion_dim", type=int, default=38)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_decoder_layers", type=int, default=6)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--diffusion_width", type=int, default=512)
    parser.add_argument("--num_sampling_steps", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", type=str, default="sample")
    
    # 推理参数
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="outputs/exposure_bias_test")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载统计信息
    mean = np.load(args.mean)
    std = np.load(args.std)
    
    # 加载文本编码器
    text_encoder_path = resolve_text_encoder_path(args.text_encoder_type, args.text_encoder, repo_root)
    text_encoder, text_encoder_dim = load_text_encoder(
        args.text_encoder_type, text_encoder_path, args.text_encoder_device,
        max_length=args.text_max_length
    )
    args.text_encoder_dim = text_encoder_dim
    
    # 加载模型
    model = load_model(args, device)
    
    # 读取动作数据
    npz_path = os.path.join(args.data_path, "npz", f"{args.sample_id}.npz")
    txt_path = os.path.join(args.data_path, "texts", f"{args.sample_id}.txt")
    
    print(f"[Info] Loading motion from {npz_path}")
    motion_gt = read_motion_data(npz_path, mean, std)
    print(f"[Info] Motion shape: {motion_gt.shape}")
    
    print(f"[Info] Loading text from {txt_path}")
    text = read_text(txt_path)
    print(f"[Info] Text: {text}")
    
    # 编码文本
    text_emb_np, text_mask_np = text_encoder.encode(text)
    text_emb = torch.from_numpy(text_emb_np).unsqueeze(0).float().to(device)
    text_mask = torch.from_numpy(text_mask_np).unsqueeze(0).to(device)
    
    # ========== Teacher Forcing ==========
    print("\n" + "="*60)
    print("Running Teacher Forcing Rollout...")
    print("="*60)
    tf_predictions, tf_errors, tf_losses = teacher_forcing_rollout(
        model, motion_gt, text_emb, text_mask, args, device, mean, std
    )
    
    # ========== Autoregressive Rollout ==========
    print("\n" + "="*60)
    print("Running Autoregressive Rollout...")
    print("="*60)
    ar_predictions, ar_errors, ar_losses = autoregressive_rollout(
        model, motion_gt, text_emb, text_mask, args, device, mean, std
    )
    
    # ========== 统计分析 ==========
    print("\n" + "="*60)
    print("Analysis")
    print("="*60)
    
    tf_errors = np.array(tf_errors)
    ar_errors = np.array(ar_errors)
    tf_losses = np.array(tf_losses)
    ar_losses = np.array(ar_losses)
    
    print(f"\nTeacher Forcing:")
    print(f"  Mean Error: {tf_errors.mean():.6f}")
    print(f"  Std Error:  {tf_errors.std():.6f}")
    print(f"  Max Error:  {tf_errors.max():.6f}")
    print(f"  Mean Loss:  {tf_losses.mean():.6f}")
    print(f"  Std Loss:   {tf_losses.std():.6f}")
    
    print(f"\nAutoregressive:")
    print(f"  Mean Error: {ar_errors.mean():.6f}")
    print(f"  Std Error:  {ar_errors.std():.6f}")
    print(f"  Max Error:  {ar_errors.max():.6f}")
    print(f"  Mean Loss:  {ar_losses.mean():.6f}")
    print(f"  Std Loss:   {ar_losses.std():.6f}")
    
    print(f"\nExposure Bias Indicator:")
    print(f"  Error Ratio (AR/TF): {ar_errors.mean() / tf_errors.mean():.2f}x")
    print(f"  Error Increase:      {ar_errors.mean() - tf_errors.mean():.6f}")
    print(f"  Loss Ratio (AR/TF):  {ar_losses.mean() / tf_losses.mean():.2f}x")
    print(f"  Loss Increase:       {ar_losses.mean() - tf_losses.mean():.6f}")
    
    # 误差增长趋势
    if len(ar_errors) >= 20:
        early_ar = ar_errors[:10].mean()
        late_ar = ar_errors[-10:].mean()
        early_ar_loss = ar_losses[:10].mean()
        late_ar_loss = ar_losses[-10:].mean()
        print(f"  AR Early (first 10) Error: {early_ar:.6f}, Loss: {early_ar_loss:.6f}")
        print(f"  AR Late (last 10) Error:   {late_ar:.6f}, Loss: {late_ar_loss:.6f}")
        print(f"  Error Drift: {late_ar - early_ar:.6f} ({(late_ar/early_ar - 1)*100:.1f}%)")
        print(f"  Loss Drift:  {late_ar_loss - early_ar_loss:.6f} ({(late_ar_loss/early_ar_loss - 1)*100:.1f}%)")
    
    # ========== 保存结果 ==========
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 拼接完整序列（都从零开始生成，不包含初始历史）
    tf_motion = np.concatenate(tf_predictions, axis=0)
    ar_motion = np.concatenate(ar_predictions, axis=0)
    gt_motion_full = motion_gt * std + mean
    
    print(f"\n[Info] Generated sequences:")
    print(f"  Teacher Forcing: {tf_motion.shape[0]} frames")
    print(f"  Autoregressive:  {ar_motion.shape[0]} frames")
    print(f"  Ground Truth:    {gt_motion_full.shape[0]} frames")
    
    # 保存 npz
    save_motion_npz(tf_motion, os.path.join(args.out_dir, f"{args.sample_id}_teacher_forcing.npz"), fps=50)
    save_motion_npz(ar_motion, os.path.join(args.out_dir, f"{args.sample_id}_autoregressive.npz"), fps=50)
    save_motion_npz(gt_motion_full, os.path.join(args.out_dir, f"{args.sample_id}_ground_truth.npz"), fps=50)
    
    # 保存误差和loss曲线
    np.savez(
        os.path.join(args.out_dir, f"{args.sample_id}_errors.npz"),
        teacher_forcing_errors=tf_errors,
        autoregressive_errors=ar_errors,
        teacher_forcing_losses=tf_losses,
        autoregressive_losses=ar_losses
    )
    
    print(f"\n[Info] Results saved to {args.out_dir}")
    print(f"  - {args.sample_id}_teacher_forcing.npz")
    print(f"  - {args.sample_id}_autoregressive.npz")
    print(f"  - {args.sample_id}_ground_truth.npz")
    print(f"  - {args.sample_id}_errors.npz (includes errors and losses)")
    print(f"  - {args.sample_id}_autoregressive.npz")
    print(f"  - {args.sample_id}_ground_truth.npz")
    print(f"  - {args.sample_id}_errors.npz")
    
    # ========== 结论 ==========
    print("\n" + "="*60)
    print("Conclusion")
    print("="*60)
    
    error_ratio = ar_errors.mean() / tf_errors.mean()
    
    if error_ratio > 2.0:
        print("❌ SEVERE Exposure Bias detected!")
        print("   Autoregressive error is >2x teacher forcing.")
        print("   → Strongly recommend implementing Rollout Training.")
    elif error_ratio > 1.5:
        print("⚠️  MODERATE Exposure Bias detected.")
        print("   Autoregressive error is 1.5-2x teacher forcing.")
        print("   → Consider implementing Rollout Training.")
    elif error_ratio > 1.2:
        print("✅ MILD Exposure Bias.")
        print("   Autoregressive error is 1.2-1.5x teacher forcing.")
        print("   → Current model is acceptable. Monitor long sequences.")
    else:
        print("✅ NO significant Exposure Bias!")
        print("   Autoregressive error is similar to teacher forcing.")
        print("   → Model generalizes well to autoregressive generation.")


if __name__ == "__main__":
    main()
