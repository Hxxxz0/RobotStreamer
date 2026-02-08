# Rollout Training 实现总结

## ✅ 已完成的文件

### 1. **核心工具模块**
- **`utils/rollout_utils.py`** ✨
  - `RolloutScheduler`: 管理 rollout ratio 的调度（linear/cosine/exp）
  - `rollout_history()`: 执行 rollout（用模型预测生成历史）
  - `scheduled_sampling_history()`: 混合 GT 和 rollout 历史
  - `add_history_noise()`: 给历史添加噪声（简单但有效）

### 2. **训练脚本**
- **`train/train_rollout.py`** ✨
  - 完整的 Rollout Training 主循环
  - 支持 Stage 2 (Scheduled Sampling) 和 Stage 3 (Full Rollout)
  - 从 Stage 1 checkpoint 继续训练
  - 集成 `rollout_utils` 的所有功能

### 3. **配置文件**
- **`configs/rollout.yaml`** ✨
  - 完整的 rollout training 配置
  - 继承自 `default.yaml`
  - 包含 Stage 2/3 所有参数
  - 性能优化参数（rollout_steps, sampling_steps）

### 4. **Shell 脚本**
- **`scripts/train_rollout.sh`** ✨
  - 一键启动 rollout training
  - 自动设置环境变量
  - 包含训练后的提示信息

### 5. **文档**
- **`docs/ROLLOUT_TRAINING.md`** ✨
  - 完整的使用指南
  - 快速开始步骤
  - 配置说明和调优建议
  - 常见问题解答
  - 预期改善时间线

---

## 🎯 使用流程

### **Step 1: 验证当前模型的 Exposure Bias**
```bash
bash scripts/test_exposure_bias.sh
# 结果: Loss Ratio 26.68x ❌
```

### **Step 2: 启动 Rollout Training (Stage 2)**
```bash
bash scripts/train_rollout.sh
```

### **Step 3: 监控训练**
```bash
# 查看日志
tail -f logs/motion_diff_rollout/train_rollout.log

# TensorBoard
tensorboard --logdir logs/motion_diff_rollout
```

### **Step 4: 定期验证改善**
```bash
# 每 20 epochs 运行一次
python scripts/verify_exposure_bias.py \
  --ckpt outputs/motion_diff_rollout/latest.pth \
  ...
```

### **Step 5 (可选): Stage 3 Full Rollout**
```bash
# 修改 configs/rollout.yaml 中的 stage: 3
# 然后重新运行
bash scripts/train_rollout.sh
```

---

## 📊 核心实现原理

### **Scheduled Sampling (Stage 2)**

```python
# 每个 batch，根据 rollout_ratio 决定使用哪种历史
rollout_ratio = scheduler.get_rollout_ratio(epoch)  # 0.0 → 1.0

# 以 rollout_ratio 的概率使用模型预测的历史
if random() < rollout_ratio:
    history = model.predict(history, text)  # Rollout
else:
    history = ground_truth_history           # GT
```

### **Rollout 机制**

```python
# 从当前历史开始，执行 rollout_steps 步
for step in range(rollout_steps):
    pred = model.predict(current_history, text)  # 预测 5 帧
    current_history = slide_window(current_history, pred)  # 滑动窗口
```

### **性能优化**

```yaml
rollout_steps: 2              # 只 rollout 2 步（10 帧），而非 12 步（60 帧）
                              # → 节省 ~80% rollout 时间
rollout_sampling_steps: 5     # 训练时用 5 步 DDIM，而非 10 步
                              # → 再节省 50% 采样时间
```

**总时间节省：约 90%**（相比 full rollout with 10 sampling steps）

---

## 🔧 可调参数

### **训练速度 vs 效果权衡**

| 参数 | 快速 | 标准 | 高质量 |
|------|------|------|--------|
| `rollout_steps` | 1 | 2 | 4 |
| `rollout_sampling_steps` | 3 | 5 | 10 |
| `total_iter` | 50k | 100k | 150k |
| `rollout_ratio_start` | 0.3 | 0.0 | 0.0 |
| 训练时间增长 | +10% | +30% | +80% |
| 预期 Loss Ratio | 5-8x | 3-5x | 2-3x |

### **学习率调优**

```yaml
# 从 Stage 1 继续：降低学习率
lr: 5e-5   # 标准（原 1e-4 的 1/2）
lr: 1e-5   # 保守（原 1e-4 的 1/10）
lr: 1e-6   # 极保守（微调）
```

---

## 📈 预期训练曲线

### **Loss Ratio 改善趋势**

```
Epoch  Rollout  Loss Ratio  状态
  0      0%       26.68x    Stage 1 结束
 20     25%       15-20x    开始适应噪声
 40     50%       10-15x    中期改善
 60     75%        6-10x    接近目标
 80    100%        5-8x     Stage 2 结束 ✅
120    100%        2-4x     Stage 3 结束 ✅✅
```

### **TensorBoard 监控指标**

- `train/loss`: 训练 loss（应该逐渐稳定）
- `train/rollout_ratio`: rollout 比例（0 → 1）
- `train/lr`: 学习率（warmup + cosine decay）

---

## 🎓 技术要点

### **为什么需要 Rollout Training？**

**问题：Exposure Bias**
- 训练：模型见的都是完美的 GT 历史
- 推理：模型自己预测的历史（有误差）
- 结果：分布不匹配 → 性能暴跌（26x）

**解决：Scheduled Sampling**
- 逐渐引入模型自己的预测作为历史
- 让模型学会从"带噪声的历史"中恢复
- 缩小训练和推理的分布差异

### **为什么不直接 Full Rollout？**

直接 100% rollout 会导致：
- 初期 loss 爆炸（模型完全不认识噪声历史）
- 训练不稳定，甚至崩溃
- 收敛速度极慢

**Scheduled Sampling 提供平滑过渡：**
- 0% → 25% → 50% → 75% → 100%
- 模型逐步适应噪声，稳定收敛

---

## 🚀 下一步

### **立即可做：**
1. ✅ 运行 `bash scripts/train_rollout.sh`
2. ✅ 监控 TensorBoard
3. ✅ 每 20 epochs 验证一次 exposure bias

### **后续优化（可选）：**
1. 尝试不同的 `schedule_type`: cosine, exp
2. 调整 `rollout_steps`: 1-4
3. 添加 `history_noise_level`: 0.1-0.2
4. 实验 `rollout_ratio_start`: 0.3-0.5（跳过早期）

---

## 📝 代码集成说明

### **无需修改原有代码**
- `train.py`: 保持不变（Stage 1）
- `models/`: 保持不变
- `infer_stream.py`: 保持不变

### **新增模块**
- `utils/rollout_utils.py`: 独立的工具模块
- `train/train_rollout.py`: 独立的训练脚本
- `configs/rollout.yaml`: 独立的配置文件

### **模块化设计**
- 可以随时切换回 Stage 1 训练
- 可以分别管理不同阶段的 checkpoints
- 配置文件分离，避免混淆

---

## ✨ 总结

**已实现功能：**
- ✅ 完整的 Rollout Training 框架
- ✅ Scheduled Sampling (Stage 2)
- ✅ Full Rollout (Stage 3)
- ✅ 性能优化（rollout_steps, sampling_steps）
- ✅ 灵活的配置系统
- ✅ 完整的文档和脚本

**预期效果：**
- ✅ Loss Ratio 从 26.68x 降低到 2-5x
- ✅ Autoregressive 推理质量显著提升
- ✅ 训练时间增加约 20-50%（可接受）

**准备就绪，可以开始训练！** 🚀
