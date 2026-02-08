# 架构更新：Transformer 编码器-解码器扩散模型

## 🔄 变更概述

架构已从 **仅解码器 + MLP** 设计更新为扩散头中的 **Transformer 编码器-解码器** 设计。

### 🆕 最新更新 (v2.1 - 2026-02-06)
1. **✅ 修复了关键的 Padding Mask 问题**：
   - 之前虽然生成了 `history_mask`，但模型完全没有使用
   - 现在 Encoder 和 Decoder 都正确使用 `src_key_padding_mask` 和 `memory_key_padding_mask`
   - 确保文本条件不会被 padding 污染

2. **✅ 可配置的历史窗口长度**：
   - 默认改为 5 帧（更快，适合短期预测）
   - 可配置为 10/30/60 等（更多时序信息，但计算更慢）
   - 通过 `configs/default.yaml` 中的 `history_len` 配置

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
输入: history (N, 38) + text (512) + history_mask (N,)
    （N=history_len，默认5，可配置）
         ↓
[MotionTokenMLP] → history_tokens (N, 16)
         ↓
[扩散头 - Transformer 编码器-解码器]
    编码器输入: [timestep(1), history_tokens(N), text(1)] = N+2 tokens
    + src_key_padding_mask: 标记 history 中的 padding 位置
         ↓
    编码器 (4层) → memory (N+2, 512)
    （padding 位置在 self-attention 中被忽略）
         ↓
    解码器输入: noisy_samples (5, 38)
    + memory_key_padding_mask: 同样的 padding mask
         ↓
    解码器 (6层) + 交叉注意力到 memory
    （cross-attention 忽略 memory 中的 padding）
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
| **Padding 处理** | 无 mask（padding 污染） | ✅ 完整的 mask 机制 |
| **历史长度** | 固定 60 帧 | 可配置（5/10/30/60） |

---

## 🔧 配置变更

### `configs/default.yaml` 中的新参数

```yaml
# Model
history_len: 5                 # 历史窗口长度（可配置：5/10/30/60）
pred_len: 5                    # 预测长度

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

3. **✅ 正确的 Padding 处理（重要更新）**
   - **Encoder**: 使用 `src_key_padding_mask` 在 self-attention 时忽略历史 padding
   - **Decoder**: 使用 `memory_key_padding_mask` 在 cross-attention 时忽略历史 padding
   - **文本条件不受影响**：timestep 和 text token 始终有效，不会被 mask
   - **避免 padding 污染**：特别重要对于短历史窗口（如5帧）

4. **可配置历史长度**
   - 支持 5/10/30/60 等不同长度
   - 短窗口（5帧）：计算快，但需要 mask 避免 padding 影响
   - 长窗口（60帧）：时序信息丰富，但计算量大

5. **更灵活**
   - 易于改变预测范围
   - 可以添加更多条件（如目标、约束）

6. **更清晰的设计**
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

**最后更新**: 2026-02-06  
**架构版本**: 2.1 (Transformer 编码器-解码器 + Padding Mask 修复)
