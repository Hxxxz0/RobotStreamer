"""
Rollout Training Utilities for Exposure Bias Mitigation

实现 DART 风格的三阶段训练策略：
- Stage 1: Teacher Forcing (已在 train.py 中实现)
- Stage 2: Scheduled Sampling (本模块)
- Stage 3: Full Rollout (本模块)
"""

import torch
import numpy as np
from typing import Tuple, Optional


class RolloutScheduler:
    """管理 Rollout Training 的进度调度"""
    
    def __init__(
        self,
        stage: int = 2,
        total_epochs: int = 100,
        rollout_ratio_start: float = 0.0,
        rollout_ratio_end: float = 1.0,
        schedule_type: str = "linear"
    ):
        """
        Args:
            stage: 训练阶段 (2=Scheduled Sampling, 3=Full Rollout)
            total_epochs: 总训练轮数
            rollout_ratio_start: 起始 rollout 比例
            rollout_ratio_end: 结束 rollout 比例
            schedule_type: 调度类型 ("linear", "cosine", "exp")
        """
        self.stage = stage
        self.total_epochs = total_epochs
        self.start_ratio = rollout_ratio_start
        self.end_ratio = rollout_ratio_end
        self.schedule_type = schedule_type
        
    def get_rollout_ratio(self, epoch: int) -> float:
        """获取当前 epoch 的 rollout 比例"""
        if self.stage == 3:  # Full Rollout
            return 1.0
        
        # Stage 2: Scheduled Sampling
        progress = min(epoch / self.total_epochs, 1.0)
        
        if self.schedule_type == "linear":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        elif self.schedule_type == "cosine":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (1 - np.cos(progress * np.pi)) / 2
        elif self.schedule_type == "exp":
            ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (progress ** 2)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return float(ratio)


def rollout_history(
    model,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    text_emb: torch.Tensor,
    text_mask: torch.Tensor,
    rollout_steps: int,
    pred_len: int,
    cfg_scale: float = 1.0,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    执行 rollout：使用模型预测来生成历史
    
    Args:
        model: MotionDiffusionModel
        history: (B, history_len, 38) 当前历史（可能包含 GT）
        history_mask: (B, history_len) 历史有效性 mask
        text_emb: (B, text_max_length, dim) 文本嵌入
        text_mask: (B, text_max_length) 文本 mask
        rollout_steps: 要 rollout 多少步
        pred_len: 每步预测长度（通常是 5）
        cfg_scale: CFG 强度
        temperature: 采样温度
    
    Returns:
        rollout_history: (B, history_len, 38) rollout 后的历史
        rollout_mask: (B, history_len) rollout 后的 mask
    
    Note: 采样步数由模型初始化时的 num_sampling_steps 决定
    """
    B, history_len, D = history.shape
    device = history.device
    
    # 复制当前历史
    current_history = history.clone()
    current_mask = history_mask.clone()
    
    with torch.no_grad():
        for step in range(rollout_steps):
            # 使用当前历史预测下一段
            pred = model.predict(
                current_history,
                text_emb,
                cfg_scale=cfg_scale,
                temperature=temperature,
                history_mask=current_mask,
                text_mask=text_mask
            )  # (B, pred_len, 38)
            
            # 滑动窗口：移除最旧的帧，添加新预测
            if current_history.shape[1] >= history_len:
                # 移除最旧的 pred_len 帧
                current_history = current_history[:, pred_len:, :]
                current_mask = current_mask[:, pred_len:]
            
            # 添加新预测
            current_history = torch.cat([current_history, pred], dim=1)
            new_mask = torch.ones(B, pred_len, dtype=torch.bool, device=device)
            current_mask = torch.cat([current_mask, new_mask], dim=1)
            
            # 确保不超过 history_len
            if current_history.shape[1] > history_len:
                current_history = current_history[:, -history_len:, :]
                current_mask = current_mask[:, -history_len:]
    
    return current_history, current_mask


def scheduled_sampling_history(
    gt_history: torch.Tensor,
    gt_history_mask: torch.Tensor,
    model,
    text_emb: torch.Tensor,
    text_mask: torch.Tensor,
    rollout_ratio: float,
    rollout_steps: int,
    pred_len: int,
    cfg_scale: float = 1.0,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scheduled Sampling: 以一定概率混合 GT 历史和 Rollout 历史
    
    Args:
        gt_history: (B, history_len, 38) Ground Truth 历史
        gt_history_mask: (B, history_len) GT 历史 mask
        model: MotionDiffusionModel
        text_emb: (B, text_max_length, dim)
        text_mask: (B, text_max_length)
        rollout_ratio: [0, 1] 使用 rollout 的比例
        rollout_steps: rollout 步数
        pred_len: 每步预测长度
        cfg_scale: CFG 强度
        temperature: 采样温度
    
    Returns:
        mixed_history: (B, history_len, 38)
        mixed_mask: (B, history_len)
    
    Note: 采样步数由模型初始化时的 num_sampling_steps 决定
    """
    B = gt_history.shape[0]
    
    # 决定哪些样本用 rollout，哪些用 GT
    use_rollout = torch.rand(B, device=gt_history.device) < rollout_ratio
    
    if use_rollout.sum() == 0:
        # 全部用 GT
        return gt_history, gt_history_mask
    
    if use_rollout.sum() == B:
        # 全部用 rollout
        return rollout_history(
            model, gt_history, gt_history_mask, text_emb, text_mask,
            rollout_steps, pred_len, cfg_scale, temperature
        )
    
    # 混合：部分 GT，部分 Rollout
    mixed_history = gt_history.clone()
    mixed_mask = gt_history_mask.clone()
    
    # 只对需要 rollout 的样本执行 rollout
    rollout_indices = torch.where(use_rollout)[0]
    if len(rollout_indices) > 0:
        rollout_hist, rollout_msk = rollout_history(
            model,
            gt_history[rollout_indices],
            gt_history_mask[rollout_indices],
            text_emb[rollout_indices],
            text_mask[rollout_indices],
            rollout_steps,
            pred_len,
            cfg_scale,
            temperature
        )
        mixed_history[rollout_indices] = rollout_hist
        mixed_mask[rollout_indices] = rollout_msk
    
    return mixed_history, mixed_mask


def add_history_noise(
    history: torch.Tensor,
    history_mask: torch.Tensor,
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    给历史添加噪声（简单但有效的 Exposure Bias 缓解方法）
    
    Args:
        history: (B, history_len, 38)
        history_mask: (B, history_len)
        noise_level: 噪声强度（相对于标准差）
    
    Returns:
        noisy_history: (B, history_len, 38)
    """
    noise = torch.randn_like(history) * noise_level
    # 只对有效的历史帧添加噪声
    mask_expanded = history_mask.unsqueeze(-1).expand_as(history)
    noisy_history = history + noise * mask_expanded.float()
    return noisy_history
