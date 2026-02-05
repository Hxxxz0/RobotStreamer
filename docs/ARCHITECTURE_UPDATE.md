# 架构更新：Transformer 编码器-解码器扩散模型

## 🔄 变更概述

架构已从 **仅解码器 + MLP** 设计更新为扩散头中的 **Transformer 编码器-解码器** 设计。

---

## 📊 架构对比

### ❌ 旧架构（已移除）
```
输入: history (60, 38) + text (512)
         ↓
[MotionTokenMLP] → history_tokens (60, 16)
         ↓
[LLaMAHF Transformer 解码器] → 条件向量 z (512)
         ↓
[扩散头 - 带 AdaLN 的 MLP]
    输入: noisy_sample (190) + z (512) + timestep
    输出: denoised_sample (190)
```

### ✅ 新架构（当前）
```
输入: history (60, 38) + text (512)
         ↓
[MotionTokenMLP] → history_tokens (60, 16)
         ↓
[扩散头 - Transformer 编码器-解码器]
    编码器输入: [timestep(1), history_tokens(60), text(1)] = 62 tokens
         ↓
    编码器 (4层) → memory (62, 512)
         ↓
    解码器输入: noisy_samples (5, 38)
         ↓
    解码器 (6层) + 交叉注意力到 memory
         ↓
    输出: denoised_samples (5, 38)
```

---

## 🎯 关键差异

| 方面 | 旧架构 | 新架构 |
|--------|-----|-----|
| **条件融合** | 独立 transformer → 单个向量 | 编码器融合所有条件 |
| **扩散网络** | 带 AdaLN 的 MLP | Transformer 解码器 |
| **时间步** | AdaLN 调制 | 编码器中的 token |
| **序列建模** | 将5帧展平为190维 | 保持5帧作为序列 |
| **交叉注意力** | 无 | 有（解码器关注编码器 memory） |

---

## 🔧 配置变更

### `configs/default.yaml` 中的新参数

```yaml
# Transformer 编码器-解码器（在扩散头中）
n_decoder_layers: 6            # 解码器层数
n_encoder_layers: 4            # 编码器层数
n_heads: 8                     # 注意力头数
```

### 移除的参数
- `num_diffusion_head_layers`（已被 `n_decoder_layers` 替代）

---

## 📝 代码变更

### 修改的文件
1. **`models/transformer_diffusion.py`** - 新增：Transformer 编码器-解码器实现
2. **`models/diffloss.py`** - 用 Transformer 替换 MLP
3. **`models/motion_diffusion.py`** - 移除 `LLaMAHF` transformer 编码器
4. **`train/train_motiondiffusion.py`** - 更新模型初始化和检查点
5. **`scripts/infer_stream.py`** - 更新推理参数
6. **`configs/default.yaml`** - 添加新参数

### 移除的文件
- 无（从现有文件中移除旧代码）

---

## ⚠️ 破坏性变更

### 1. **检查点不兼容**
包含独立 `trans`（LLaMAHF 编码器）的旧检查点与新架构**不兼容**。

旧检查点结构：
```python
{
    "trans": {...},              # ❌ 不再使用
    "token_mlp": {...},          # ✅ 仍在使用
    "action_diffusion": {...}    # ✅ 结构已更改
}
```

新检查点结构：
```python
{
    "token_mlp": {...},          # ✅ 动作 token MLP
    "action_diffusion": {...}    # ✅ 现在包含 Transformer 编码器-解码器
}
```

### 2. **需要从头训练**
必须使用新架构从头训练新模型。

---

## 🚀 使用方法

### 训练
```bash
# 所有新参数都在 configs/default.yaml 中
bash scripts/train.sh

# 或使用自定义参数
accelerate launch train/train_motiondiffusion.py \
    --n_decoder_layers 6 \
    --n_encoder_layers 4 \
    --n_heads 8
```

### 推理
```bash
# 新检查点会自动包含正确的架构配置
python scripts/infer_stream.py \
    --ckpt outputs/motion_diff/latest.pth \
    --text "A robot walks forward"
```

### 测试
```bash
# 测试新架构
python test_new_architecture.py
```

---

## 💡 新架构的优势

1. **更强的条件控制**
   - 编码器可以对所有条件进行双向注意力
   - 更好地融合时间步、历史和文本

2. **更好的序列建模**
   - 解码器将5个输出帧视为序列
   - 交叉注意力允许每帧关注历史的不同部分

3. **更灵活**
   - 易于改变预测范围
   - 可以添加更多条件（如目标、约束）

4. **更清晰的设计**
   - 扩散头中的单一统一架构
   - 无需单独的 transformer 编码器

---

## 📊 模型大小对比

| 组件 | 旧大小 | 新大小 |
|-----------|----------|----------|
| MotionTokenMLP | ~30K | ~30K（相同） |
| 编码器 | ~6M (LLaMAHF) | ~4M（在扩散中） |
| 扩散头 | ~2M (MLP) | ~8M (Transformer) |
| **总计** | **~8M** | **~12M** |

新架构更大，但提供了显著更好的建模能力。

---

## 🔍 故障排除

### 问题："模型中未找到键"
**解决方案**：确保不要尝试加载旧检查点。从头训练。

### 问题："缺少参数：n_decoder_layers"
**解决方案**：使用新参数更新配置文件或使用命令行参数。

### 问题：训练时内存不足
**解决方案**： 
- 减少批次大小
- 在配置中启用 `grad_checkpointing: true`
- 减少 `n_decoder_layers` 或 `n_encoder_layers`

---

## 📚 参考资料

- `models/transformer_diffusion.py` - 实现细节
- `configs/default.yaml` - 默认配置
- `test_new_architecture.py` - 架构验证

---

**最后更新**: 2026-02-05  
**架构版本**: 2.0 (Transformer 编码器-解码器)
