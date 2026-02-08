# Rollout Training for Exposure Bias Mitigation

## 📋 概述

实现 DART 风格的 **Rollout Training**，用于缓解 Autoregressive 推理中的 Exposure Bias 问题。

### 问题诊断

运行 `scripts/verify_exposure_bias.py` 发现：
- **Teacher Forcing Loss**: 0.0545
- **Autoregressive Loss**: 1.4550
- **Loss Ratio**: **26.68x** ❌ (严重的 Exposure Bias)

### 解决方案

三阶段训练策略：
1. **Stage 1: Teacher Forcing** (已完成) - 使用 GT 历史
2. **Stage 2: Scheduled Sampling** (本次实现) - 0% → 100% rollout
3. **Stage 3: Full Rollout** (可选) - 100% rollout 强化

---

## 🚀 快速开始

### 1. 准备工作

确保已有 Stage 1 训练好的模型：
```bash
ls outputs/motion_diff/latest.pth  # 应该存在
```

### 2. Stage 2: Scheduled Sampling

直接运行：
```bash
bash scripts/train_rollout.sh
```

或手动启动：
```bash
python train/train_rollout.py --config configs/rollout.yaml
```

### 3. 监控训练

```bash
# 查看日志
tail -f logs/motion_diff_rollout/train_rollout.log

# TensorBoard
tensorboard --logdir logs/motion_diff_rollout
```

### 4. 验证改善

每训练 20 epochs，运行：
```bash
python scripts/verify_exposure_bias.py \
  --ckpt outputs/motion_diff_rollout/latest.pth \
  --data_path /limx_embap/tos/user/Jensen/dataset/robot_humanml_data \
  --sample_id 000000 \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text_encoder_type t5 \
  --text_max_length 60 \
  --num_sampling_steps 10 \
  --cfg 1.0 \
  --out_dir outputs/exposure_bias_test_rollout
```

---

## ⚙️ 配置说明

### `configs/rollout.yaml`

关键参数：

```yaml
# 训练参数
lr: 5e-5                      # 降低学习率（微调）
total_iter: 100000            # 总迭代次数

# Checkpoint 加载
resume_from: outputs/motion_diff/latest.pth

# Rollout Training
rollout_training:
  stage: 2                    # 2=Scheduled Sampling, 3=Full Rollout
  
  stage2:
    total_epochs: 80
    rollout_ratio_start: 0.0  # 起始 0%
    rollout_ratio_end: 1.0    # 结束 100%
    schedule_type: linear     # linear, cosine, exp
  
  rollout_steps: 2            # 只 rollout 2 步（节省时间）
  rollout_sampling_steps: 5   # 快速采样（训练时）
```

### 调整训练策略

#### 方案 A: 标准（稳妥，推荐）
```yaml
total_iter: 100000
lr: 5e-5
rollout_steps: 2
rollout_ratio_start: 0.0
rollout_ratio_end: 1.0
```
- 预计时间：约 80 epochs，+30% 时间
- 预期效果：Loss ratio 26x → 3-5x

#### 方案 B: 激进（快速验证）
```yaml
total_iter: 50000
lr: 1e-5
rollout_steps: 2
rollout_ratio_start: 0.5   # 直接从 50% 开始
rollout_ratio_end: 1.0
```
- 预计时间：约 40 epochs，+15% 时间
- 预期效果：Loss ratio 26x → 5-8x

#### 方案 C: 噪声预热（最快）
```yaml
rollout_training:
  stage: 2
  rollout_ratio_start: 0.0
  rollout_ratio_end: 0.0     # 不做 rollout
  history_noise_level: 0.2   # 只加噪声
total_iter: 20000
```
- 预计时间：约 15 epochs，+0% 时间
- 预期效果：Loss ratio 26x → 10-15x

---

## 📊 预期改善时间线

| Epoch | Rollout Ratio | 预期 Loss Ratio | 状态 |
|-------|--------------|----------------|------|
| 0 | 0% | 26.68x ❌ | Stage 1 结束 |
| 20 | 25% | 15-20x ⚠️ | 开始适应 |
| 50 | 62% | 8-12x ⚠️ | 明显改善 |
| 80 | 100% | 5-8x ⚠️ | Stage 2 结束 |
| 120 | 100% | **2-4x** ✅ | Stage 3 结束 |

---

## 🔄 Stage 3: Full Rollout（可选）

如果 Stage 2 后 loss ratio 仍 > 5x，继续 Stage 3：

### 修改 `configs/rollout.yaml`：
```yaml
rollout_training:
  stage: 3                    # 改为 3

resume_from: outputs/motion_diff_rollout/latest.pth  # 从 Stage 2 继续
total_iter: 50000             # Stage 3 迭代次数
lr: 1e-5                      # 进一步降低学习率
```

### 重新训练：
```bash
bash scripts/train_rollout.sh
```

---

## 📁 文件结构

```
MotionDiffusionCore/
├── configs/
│   ├── default.yaml          # Stage 1 配置
│   └── rollout.yaml          # Stage 2+3 配置 ✨
├── train/
│   ├── train.py              # Stage 1 训练
│   └── train_rollout.py      # Stage 2+3 训练 ✨
├── utils/
│   └── rollout_utils.py      # Rollout 工具函数 ✨
├── scripts/
│   ├── train.sh              # Stage 1 脚本
│   ├── train_rollout.sh      # Stage 2+3 脚本 ✨
│   └── verify_exposure_bias.py  # 验证脚本
└── outputs/
    ├── motion_diff/          # Stage 1 checkpoints
    └── motion_diff_rollout/  # Stage 2+3 checkpoints ✨
```

---

## 🐛 常见问题

### Q1: 训练崩溃 / Loss 爆炸
**A**: 降低学习率
```yaml
lr: 1e-5  # 或更低
```

### Q2: 训练太慢
**A**: 减少 rollout 步数
```yaml
rollout_steps: 1  # 从 2 降到 1
rollout_sampling_steps: 3  # 从 5 降到 3
```

### Q3: 显存不足
**A**: 减小 batch size 或使用梯度累积
```yaml
batch_size: 128  # 从 256 降到 128
```

### Q4: Loss ratio 改善不明显
**A**: 尝试更激进的 rollout ratio
```yaml
rollout_ratio_start: 0.3  # 直接从 30% 开始
```

---

## 📈 成功标准

训练成功的指标：
- ✅ Loss ratio < 5x（可接受）
- ✅ Loss ratio < 3x（良好）
- ✅ Loss ratio < 2x（优秀，接近 DART）

---

## 🔗 相关文件

- **验证脚本**: `scripts/verify_exposure_bias.py`
- **推理脚本**: `scripts/infer_stream.py`
- **配置文件**: `configs/rollout.yaml`
- **训练脚本**: `train/train_rollout.py`
- **工具函数**: `utils/rollout_utils.py`

---

**祝训练顺利！如有问题请查看日志或修改配置参数。**
