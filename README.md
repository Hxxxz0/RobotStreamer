# MotionDiffusionCore

基于 **Transformer 编码器-解码器扩散** 的流式动作生成模型，用于机器人动作预测和生成。

---

## 📖 项目简介

MotionDiffusionCore 是一个用于机器人动作序列生成的深度学习框架。该项目采用：
- **历史动作编码**：通过 MLP 将可配置长度的历史动作（默认60帧）映射为 token 序列
- **Token-level 文本编码**：T5 输出 60 个文本 token（而非单一向量），提供更细粒度的语义信息
- **统一的 Transformer 编码器-解码器**：在扩散头中融合时间步、历史和文本条件，**支持 padding mask**
- **扩散生成**：通过 Transformer 解码器直接生成未来 5 帧动作序列

**核心特点**：
- ✅ 无需 VAE/TAE 等重型编码器，直接端到端训练
- ✅ 统一的编码器-解码器架构，更强的序列建模能力
- ✅ **Token-level 文本编码** + padding mask，增强文本控制力
- ✅ 支持任意长度的流式生成（滚动预测）
- ✅ 支持 Classifier-Free Guidance (CFG) 控制生成质量（零向量 masking）
- ✅ 三个独立数据集，可灵活组合训练

---

## 🏗️ 模型架构

### 整体流程

```
输入：history_len帧历史动作 (38D) + 文本描述
    （默认 60 帧，可配置 5/10/30/60 等）
                ↓
    ┌───────────────────────────────────────┐
    │  MotionTokenMLP (38→512→16)           │  ← 动作特征映射
    └───────────────────────────────────────┘
                ↓
         60个 16维 history token
                ↓
    ┌───────────────────────────────────────┐
    │  T5 Text Encoder                      │  ← Token-level 文本编码 ✨
    │  输入: "A person walks" (原始文本)      │
    │  输出: 60个 512维 text token + mask    │
    └───────────────────────────────────────┘
                ↓
    ┌───────────────────────────────────────┐
    │  Transformer 编码器-解码器扩散头        │
    │                                       │
    │  [编码器 - 4层]                        │
    │  输入: timestep(1) +                   │
    │        history_tokens(60) +            │
    │        text_tokens(60)                 │
    │  + History Padding Mask ✅            │
    │  + Text Padding Mask ✅               │
    │         ↓                              │
    │  输出: memory (121, 512)               │
    │         ↓                              │
    │  [解码器 - 6层]                        │
    │  输入: noisy_samples (5, 38)           │
    │  交叉注意力: ← memory                  │
    │  + Memory Padding Mask ✅             │
    │         ↓                              │
    │  输出: denoised (5, 38)                │
    │                                       │
    │  - DDIM 快速采样（默认10步）             │
    │  - Squared Cosine 噪声调度             │
    └───────────────────────────────────────┘
                ↓
       输出：未来5帧动作 (38D×5)
```

### 关键组件说明

#### 1. **MotionTokenMLP** (`models/token_mlp.py`)
- **输入**：单帧 38 维动作向量
- **输出**：16 维 latent token
- **结构**：两层 MLP (38 → 512 → 16)
- **作用**：将原始动作映射到 Transformer 可处理的 token 空间

#### 2. **Transformer 编码器-解码器** (`models/transformer_diffusion.py`)

**编码器部分**（4 层，可配置）：
- **输入**：
  - 时间步嵌入 (1, 512D)
  - history_len 个历史 token (60, 16D)
  - **Token-level 文本嵌入 (60, 512D)** ✨ 新增
  - **History Padding Mask**：标记哪些历史 token 是 padding（序列开始时）
  - **Text Padding Mask**：标记哪些文本 token 是 padding（短文本补齐）
  - 总计：121 个 token（1 + 60 + 60）
- **输出**：memory 向量 (121, 512D)
- **作用**：双向融合所有条件信息，**同时正确忽略 padding 位置**

**解码器部分**（6 层，可配置）：
- **输入**：
  - 带噪声的动作样本 (5, 38D)
  - 交叉注意力到编码器 memory
  - **Memory Padding Mask**：在 cross-attention 时忽略历史和文本的 padding 位置
- **输出**：去噪后的动作 (5, 38D)
- **作用**：通过交叉注意力机制，让每一帧都能关注有效的历史帧和文本 token（不受 padding 污染）

**架构优势**：
- ✅ **更强的文本条件融合**：60 个文本 token 提供细粒度语义信息（vs 旧版 1 个 token）
- ✅ **更强的条件融合**：编码器可对所有条件进行双向注意力
- ✅ **更好的序列建模**：解码器将 5 帧作为序列处理，而非展平
- ✅ **正确的 Padding 处理**：通过 mask 机制，padding 不会污染 attention
- ✅ **可配置历史长度**：支持 5/10/30/60 等不同长度，权衡计算效率和时序信息
- ✅ **灵活扩展**：易于添加新的条件（如目标、约束）

#### 3. **扩散过程** (`models/diffloss.py`)
- **训练**：使用 DDPM 学习从高斯噪声逐步去噪到目标动作（1000 步）
- **推理**：使用 DDIM 快速采样（默认 10 步，可配置 50 步以提升质量）
- **特性**：
  - **Timestep Embedding**：将扩散时间步嵌入到编码器
  - **Squared Cosine Cap V2 Schedule**：改进的噪声调度，训练更稳定
  - **HuggingFace Diffusers 框架**：基于标准库实现，支持多种噪声调度和采样策略

---

## 📂 数据集

本项目支持三个独立的机器人动作数据集，每个数据集有独立的加载逻辑。

### ✨ 路径配置（无硬编码）

**所有数据集路径都可通过命令行参数或配置文件指定，核心代码无硬编码。**

可配置的路径参数：
- `--humanml3d_path`: HumanML3D 数据根目录
- `--babel_stream_path`: BABEL Stream 数据根目录
- `--humanml3d_stream_path`: HumanML3D Stream 数据根目录
- `--meta_dir`: 统计文件目录 (Mean.npy, Std.npy)
- `--stable_utils_dir`: StableMoFusion 工具目录

默认值见 `configs/default.yaml`，也可在命令行覆盖。

---

### 1. **HumanML3D** (`humanml3d`)
- **默认路径**：`/limx_embap/tos/user/Jensen/dataset/robot_humanml_data/`
- **数据格式**：
  - 动作：`npz/*.npz` (50fps, 38维)
  - 文本：`texts/*.txt`
- **文本格式**：`caption#tokens#start_time#end_time`
- **采样策略**：
  - 全局文本（start=0, end=0）→ 随机采样整个序列
  - 时间段文本 → 只从对应时间段采样
- **特点**：多样化的单人动作，文本描述丰富

### 2. **BABEL Stream** (`babel_stream`)
- **默认路径**：`/limx_embap/tos/user/Jensen/project/dataset/babel_stream_50/`
- **数据格式**：
  - 动作：`train_stream/*.npz`
  - 文本：`train_stream_text/*.txt`
- **文本格式**：`A_caption*B_caption#A_length`
  - A 段：过渡动作
  - B 段：主要动作
- **采样策略**：
  - **仅从 B 段采样**（跳过 A 段过渡）
  - 使用 schedule 动态匹配文本
- **特点**：长序列复杂动作组合

### 3. **HumanML3D Stream** (`humanml3d_stream`)
- **默认路径**：`/limx_embap/tos/user/Jensen/project/dataset/humanml3d_stream_50/`
- **数据格式**：
  - 动作：`train_stream/*.npz`
  - 文本：`train_stream_text/*.txt`
- **文本格式**：同 BABEL Stream
- **采样策略**：同 BABEL Stream
- **特点**：HumanML3D 动作的流式版本

### 数据预处理

所有数据集加载时会自动：
1. **标准化**：使用全局 `Mean.npy` 和 `Std.npy` 归一化
2. **过滤**：仅剔除长度 < 5 帧（预测长度）的样本，短历史序列会自动 0-padding
3. **缓存**：启动时一次性加载到内存，加速训练

### 训练时数据采样

每个训练 batch 的数据采样流程：

#### **1. HumanML3D 数据集**
- **随机选择样本**：从预加载的样本列表中随机选择一个
- **随机选择时间点**：在序列中随机选择一个位置作为预测起点
  ```python
  step_idx = random.randint(0, motion_len - pred_len - 1)
  # 限制：确保有足够的未来帧可以预测（至少 5 帧）
  ```
- **构建历史窗口**：截取 `[step_idx-59, step_idx]` 共 60 帧作为历史（不足则 0-padding）
- **提取预测目标**：截取 `[step_idx+1, step_idx+5]` 共 5 帧作为预测目标
- **随机选择文本**：从该样本关联的文本列表中随机选择一条

#### **2. BABEL/HumanML3D Stream 数据集**
- **约束采样**：对于包含文本时间段标注的序列（如 BABEL 的 A 段 + B 段）
- **起点限制**：采样起点必须在 B 段（主要动作段），避免 A 段（过渡动作）
  ```python
  min_step = first_segment_len  # B 段的起始位置
  max_step = motion_len - pred_len - 1
  step_idx = random.randint(min_step, max_step)
  ```
- **文本调度**：根据时间段返回当前帧对应的文本描述

#### **3. 历史窗口 0-Padding**
- **足够长的序列**：直接截取过去 60 帧
  ```python
  # 序列长度 100 帧，step_idx = 80
  history = motion[21:81]  # 60 帧历史
  ```
- **短序列**：前面补 0（在归一化空间）
  ```python
  # 序列长度 30 帧，step_idx = 25
  history = [0,0,...,0 (35个0), motion[0:25]]  # 35个0 + 25帧实际历史
           |← 零填充 →|←   实际历史   →|
  ```

#### **4. 文本条件 Masking（CFG 训练）**
- 默认关闭（`cfg_mask_prob=0.0`），遵循 Pi0 的方法
- 可选启用：设置 `cfg_mask_prob=0.1`，随机将 10% 的样本文本替换为**零向量 + 全 False mask**（真正的无条件）
  ```python
  feat_text[idx] = 0.0       # 零向量（而非空字符串编码）
  text_mask[idx] = False     # 全 False = 模型忽略所有文本 token
  ```
- 作用：训练模型学习无条件生成，用于推理时的 Classifier-Free Guidance

---

## 🚀 快速开始

### 环境配置

```bash
# 创建环境
conda env create -f environment.yaml

# 激活环境
conda activate mgpt
```

### 配置文件

所有默认配置在 `configs/default.yaml`，包括：
- 数据集路径
- 训练超参数
- 模型参数

**修改配置（二选一）**：
1. **直接编辑 YAML**（推荐）：修改 `configs/default.yaml`
2. **命令行覆盖**：通过 `--参数名` 覆盖特定配置

### 训练

#### 使用默认配置（所有数据集）
```bash
cd /limx_embap/tos/user/Jensen/project/MotionDiffusionCore
bash scripts/train.sh
```

#### 自定义配置（命令行覆盖）

**只用 HumanML3D：**
```bash
accelerate launch --mixed_precision no --num_processes 4 \
  train/train_motiondiffusion.py \
  --config configs/default.yaml \
  --datasets humanml3d \
  --exp_name humanml3d_only
```

**组合 BABEL + HumanML3D Stream：**
```bash
accelerate launch --mixed_precision no --num_processes 4 \
  train/train_motiondiffusion.py \
  --config configs/default.yaml \
  --datasets babel_stream humanml3d_stream \
  --exp_name babel_hml_stream
```

**自定义数据路径：**
```bash
accelerate launch --mixed_precision no --num_processes 4 \
  train/train_motiondiffusion.py \
  --config configs/default.yaml \
  --humanml3d_path /path/to/custom/data \
  --meta_dir /path/to/custom/statistics
```

**启用 CFG 训练：**
```bash
accelerate launch --mixed_precision no --num_processes 4 \
  train/train_motiondiffusion.py \
  --config configs/default.yaml \
  --cfg_mask_prob 0.1 \
  --exp_name motion_diff_cfg
# 10% 概率遮罩文本，训练无条件分支以支持更强的 CFG
```

### 训练参数说明

#### 数据集参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | `humanml3d babel_stream humanml3d_stream` | 选择训练数据集 |
| `--humanml3d_path` | `/limx_embap/.../robot_humanml_data` | HumanML3D 根目录 |
| `--babel_stream_path` | `/limx_embap/.../babel_stream_50` | BABEL Stream 根目录 |
| `--humanml3d_stream_path` | `/limx_embap/.../humanml3d_stream_50` | HumanML3D Stream 根目录 |
| `--meta_dir` | `/limx_embap/.../statistics` | 统计文件目录 |
| `--stable_utils_dir` | `/limx_embap/.../StableMoFusion/utils` | StableMoFusion 工具目录 |

#### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 256 | 批次大小 |
| `--total_iter` | 100000 | 总训练步数 |
| `--lr` | 1e-4 | 学习率 |
| `--num_workers` | 8 | DataLoader 工作线程数 |

#### 模型参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_size` | 512 | Token MLP 隐藏层维度 |
| `--latent_dim` | 16 | 动作 token 维度 |
| `--motion_dim` | 38 | 动作维度 |
| `--n_decoder_layers` | 6 | Transformer 解码器层数 |
| `--n_encoder_layers` | 4 | Transformer 编码器层数 |
| `--n_heads` | 8 | 注意力头数 |

#### 扩散模型参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_sampling_steps` | 10 | DDIM 推理采样步数（10=快速，50=高质量） |
| `--num_train_timesteps` | 1000 | 训练时扩散总步数 |
| `--beta_schedule` | `squaredcos_cap_v2` | 噪声调度类型（linear/scaled_linear/squaredcos_cap_v2） |
| `--prediction_type` | `sample` | 模型预测目标：`sample`(x0) 或 `epsilon`(噪声) |
| `--diffusion_width` | 512 | 扩散头 Transformer 维度 |
| `--grad_checkpointing` | false | 梯度检查点（节省显存） |
| `--cfg_mask_prob` | 0.0 | CFG 训练时文本遮罩概率（0.0=关闭，0.1=10%遮罩） |

#### 文本编码器参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--text_encoder_type` | `t5` | 文本编码器类型 (`t5` / `bge`) |
| `--text_encoder` | `flan-t5-small` | 具体模型名称 |
| `--text_encoder_device` | `cuda` | 文本编码器设备 |
| `--text_max_length` | `60` | 文本 token 最大长度（token-level 编码）|

#### 其他参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_name` | `motion_diff` | 实验名称 |
| `--resume_trans` | None | 恢复训练的检查点路径 |
| `--seed` | 123 | 随机种子 |

### 自定义数据路径

如果数据集位于不同路径，有两种方式配置：

**方式1：命令行参数**
```bash
accelerate launch train/train_motiondiffusion.py \
  --datasets humanml3d \
  --humanml3d_path /your/custom/path/humanml3d \
  --meta_dir /your/custom/path/statistics \
  --stable_utils_dir /your/custom/path/StableMoFusion/utils
```

**方式2：修改配置文件**

编辑 `configs/default.yaml`：
```yaml
humanml3d_path: /your/custom/path/humanml3d
meta_dir: /your/custom/path/statistics
stable_utils_dir: /your/custom/path/StableMoFusion/utils
```

### 训练监控

**查看文本日志：**
```bash
tail -f logs/motion_diff/train.log
```

**启动 TensorBoard：**
```bash
tensorboard --logdir logs/motion_diff --port 6006
# 然后访问 http://localhost:6006
```

**日志内容：**
- `Loss/train`：训练损失曲线
- `LR/train`：学习率变化曲线
- 每 100 步记录一次损失
- 每 10k 步保存检查点

---

## 🎯 推理

### 基础推理（从零历史开始）

```bash
python scripts/infer_stream.py \
  --ckpt outputs/motion_diff/latest.pth \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text "一个机器人向前走" \
  --text_encoder_type t5 \
  --text_encoder flan-t5-small \
  --max_motion_length 150 \
  --out_dir infer_output
```

### 带自定义历史的推理

```bash
python scripts/infer_stream.py \
  --ckpt outputs/motion_diff/latest.pth \
  --history_npy /path/to/history_38d.npy \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text "机器人向左转" \
  --cfg 7.5 \
  --temperature 1.0 \
  --out_dir infer_output
```

### 高质量推理（50步采样 + 强 CFG）

```bash
python scripts/infer_stream.py \
  --ckpt outputs/motion_diff/latest.pth \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text "机器人跳跃" \
  --num_sampling_steps 50 \
  --cfg 7.5 \
  --out_dir infer_output
```

**说明：**
- `--num_sampling_steps 10`（默认）：速度最快，适合实时应用
- `--num_sampling_steps 50`：质量更高，适合离线生成

### 推理参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt` | ✅ | - | 模型检查点路径 |
| `--mean` | ✅ | - | 归一化均值文件 |
| `--std` | ✅ | - | 归一化标准差文件 |
| `--text` | ✅ | - | 文本描述（中英文均可） |
| `--history_npy` | ❌ | None | 历史动作 (N, 38)，不提供则从零开始 |
| `--max_motion_length` | ❌ | 150 | 最大生成长度 |
| `--cfg` | ❌ | 7.5 | CFG 引导强度（1.0=关闭，7.5=推荐，>1.0 增强文本一致性） |
| `--temperature` | ❌ | 1.0 | 扩散温度（>1.0 增加随机性） |
| `--num_sampling_steps` | ❌ | 10 | DDIM 采样步数（10=快速，50=高质量） |
| `--num_train_timesteps` | ❌ | 1000 | 训练时扩散步数（需与训练时一致） |
| `--beta_schedule` | ❌ | `squaredcos_cap_v2` | 噪声调度类型 |
| `--prediction_type` | ❌ | `sample` | 预测类型（sample/epsilon） |
| `--diffusion_width` | ❌ | 512 | 扩散头 Transformer 维度 |
| `--n_decoder_layers` | ❌ | 6 | Transformer 解码器层数（需与训练时一致） |
| `--n_encoder_layers` | ❌ | 4 | Transformer 编码器层数（需与训练时一致） |
| `--n_heads` | ❌ | 8 | 注意力头数（需与训练时一致） |
| `--out_dir` | ❌ | `output` | 输出目录 |

### 流式生成原理

推理时采用**滚动预测**策略：

```
Step 1: [0, 0, ..., 0 (60个)] + text → 生成 frame[1-5]
Step 2: [0, 0, ..., 0 (55个), frame1-5] + text → 生成 frame[6-10]
Step 3: [0, ..., 0, frame1-10] + text → 生成 frame[11-15]
...
Step N: [frame(N-59) - frame(N)] + text → 生成 frame[N+1 - N+5]
```

每次生成 5 帧后，将其追加到历史窗口，继续预测下一个 5 帧，直到达到目标长度。

### Classifier-Free Guidance (CFG)

CFG 通过混合有条件和无条件预测来增强文本控制：

```python
# 有文本条件
pred_text = model(history, text_tokens, text_mask)      

# 无文本条件（使用零向量 + 全 False mask）✨ 重要改进
zero_text = torch.zeros_like(text_tokens)  # 零向量（而非空字符串编码）
zero_mask = torch.zeros_like(text_mask, dtype=torch.bool)  # 全 False = 忽略所有文本
pred_uncond = model(history, zero_text, zero_mask)

# CFG 混合
pred_final = pred_uncond + cfg_scale * (pred_text - pred_uncond)
```

**推荐配置**：
- `cfg=1.0`：关闭 CFG，标准生成
- `cfg=3.0-5.0`：**推荐值**（token-level text 下，CFG 效果更强，建议比旧版低）
- `cfg=1.5-3.0`：轻度引导
- `cfg>10.0`：可能过度拟合文本，导致动作不自然

**架构改进（v2）**：
- ✅ **零向量 masking**：真正的无条件（旧版用空字符串 `""` 产生非零嵌入）
- ✅ **Text mask 标记**：零向量 + 全 False mask，模型完全忽略文本输入
- ✅ **Token-level 增强**：60 个文本 token 提供更强的语义控制

**注意**：
- 默认训练时 `cfg_mask_prob=0.0`（关闭 CFG 训练），但推理时仍可使用 CFG
- 如需更强的 CFG 效果，可在训练时设置 `cfg_mask_prob=0.1`（10% 零向量 masking）

---

## 📁 项目结构

```
MotionDiffusionCore/
├── datasets/                    # 数据集加载
│   ├── __init__.py
│   └── motion_dataset.py       # 三个独立数据集类 + 组合加载器
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── motion_diffusion.py     # 🎯 主模型（MLP + Encoder-Decoder Diffusion）
│   ├── token_mlp.py            # 动作→token MLP (38→16)
│   ├── transformer_diffusion.py # Transformer 编码器-解码器（扩散头）
│   ├── diffloss.py             # 扩散损失封装（基于 HuggingFace Diffusers）
│   ├── transformer.py          # 遗留文件（已不再使用）
│   └── diffusion/              # 扩散模型工具（兼容旧代码）
│       ├── __init__.py
│       ├── gaussian_diffusion.py   # 高斯扩散过程
│       ├── respace.py             # 时间步重映射
│       └── diffusion_utils.py     # KL散度等工具函数
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── text_encoder.py         # T5 / BGE 文本编码器
│   ├── logger.py               # 日志工具
│   └── motion_npz.py           # npz 文件保存
├── train/                       # 训练脚本
│   ├── __init__.py
│   └── train_motiondiffusion.py   # 主训练入口
├── scripts/                     # 脚本
│   ├── train.sh                # 4卡训练启动脚本
│   └── infer_stream.py         # 流式推理脚本
├── configs/                     # 配置文件
│   └── default.yaml            # 默认超参数
├── outputs/                     # 模型检查点（自动创建）
│   └── {exp_name}/
│       ├── latest.pth
│       └── ckpt_100000.pth
├── logs/                        # 训练日志（自动创建）
│   └── {exp_name}/
│       ├── train.log
│       └── events.out.tfevents.*
├── .gitignore
└── README.md
```

---

## ⚙️ 扩散模型实现

本项目使用 **HuggingFace Diffusers** 库实现扩散模型，提供标准化、灵活的扩散训练和采样接口。

### 为什么使用 Diffusers？

1. **标准化实现**：避免自定义扩散代码的 bug 和维护成本
2. **丰富的调度器**：支持 DDPM、DDIM、PNDM 等多种采样策略
3. **灵活配置**：噪声调度、预测类型、采样步数等均可自由调整
4. **社区支持**：与主流扩散模型（Stable Diffusion 等）使用相同底层

### 训练 vs 推理

```python
# 训练时：DDPMScheduler（1000步完整扩散）
self.train_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample"  # 预测 x0
)

# 推理时：DDIMScheduler（10步快速采样）
self.inference_scheduler = DDIMScheduler(
    num_train_timesteps=1000,  # 与训练一致
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample"
)
# 设置推理步数
self.inference_scheduler.set_timesteps(num_sampling_steps=10)
```

### 关键配置说明

| 配置项 | 训练值 | 推理值 | 说明 |
|--------|--------|--------|------|
| `num_train_timesteps` | 1000 | 1000 | 扩散总步数（两者必须一致） |
| `beta_schedule` | squaredcos_cap_v2 | squaredcos_cap_v2 | 噪声调度（两者必须一致） |
| `prediction_type` | sample | sample | 预测目标：x0（两者必须一致） |
| 实际采样步数 | 1 步/iter | 10-50 步 | 训练随机采样，推理 DDIM 加速 |

---

## 🔧 训练细节

### 训练流程

1. **数据加载**：三个数据集分别加载，使用 `ConcatDataset` 组合
2. **随机采样**：每个 batch 随机从数据集中采样 (step_idx, 60-frame history, 5-frame target)
3. **文本编码**：文本 → T5/BGE → 512D/1024D 向量
4. **CFG 训练**：默认关闭（cfg_mask_prob=0.0），可设置为 0.1 启用（10% 概率置空文本）
5. **前向传播**：
   ```python
   history_tokens = token_mlp(history)                    # (B, 60, 38) -> (B, 60, 16)
   loss, pred = diffusion_head(
       target=target,                                     # (B, 5, 38)
       history_tokens=history_tokens,                     # (B, 60, 16)
       text_emb=feat_text                                 # (B, 512)
   )
   # 内部流程：
   # - 编码器融合 timestep + history_tokens + text -> memory
   # - 解码器通过交叉注意力生成去噪后的动作
   ```
6. **优化器**：AdamW (lr=1e-4, betas=(0.9, 0.99))
7. **学习率调度**：Warmup (10%) + Cosine Decay

### 检查点保存

- **每 10k 步**：覆盖保存 `latest.pth`
- **100k 步**：保存 `ckpt_100000.pth`
- **检查点内容**：
  ```python
  {
      "token_mlp": token_mlp.state_dict(),           # 动作 token MLP
      "action_diffusion": diffusion_head.state_dict(), # Transformer 编码器-解码器扩散头
      "optimizer": optimizer.state_dict(),           # 优化器状态
      "iter": current_iteration,                     # 当前迭代数
      "args": training_args                          # 训练参数
  }
  ```

**注意**：新架构的检查点与旧版本（包含独立 "trans" 键）不兼容。详见 `ARCHITECTURE_UPDATE.md`。

### 扩散训练细节

- **扩散框架**：HuggingFace Diffusers（标准化实现）
- **训练调度器**：DDPMScheduler
  - **噪声调度**：Squared Cosine Cap V2（训练更稳定）
  - **训练步数**：1000 步（训练时随机采样一步）
  - **预测类型**：sample（直接预测 x0，而非噪声 ε）
- **推理调度器**：DDIMScheduler（快速采样）
  - **推理步数**：10 步（默认，可配置 50 步提升质量）
  - **加速原理**：DDIM 跳步采样，10 步可达 1000 步约 90% 质量
- **损失函数**：MSE loss（预测 x0 vs 真实 x0）

---

## 💡 关键特性

### 1. 预加载缓存
- 所有 npz 文件启动时一次性加载到内存
- 归一化后存储，避免重复计算
- 大幅提升 DataLoader 速度

### 2. 随机窗口采样
- 每个 epoch 数据分布不同
- 避免过拟合特定时间段
- 充分利用长序列数据

### 3. 文本条件支持
- HumanML3D：多文本描述 + 时间段切片
- BABEL/HumanML3D Stream：动态 schedule 匹配
- CFG 训练：可选启用（cfg_mask_prob 可配置，默认 0.0）

### 4. 流式生成
- 滚动预测，支持任意长度
- 历史窗口自动滑动
- 适配在线/实时应用

### 5. 分布式训练
- Accelerate 多卡并行
- 自动梯度同步
- 主进程独占日志/保存

---

## 🔗 外部依赖

### 1. Python 包依赖

项目已通过 `environment.yaml` 配置所有依赖，包括：

**核心依赖：**
- **`diffusers==0.31.0`**：HuggingFace 扩散模型框架（DDPM/DDIM 调度器）
- **`accelerate`**：分布式训练框架
- **`torch`**：深度学习框架
- **`transformers`**：文本编码器（T5/BGE）

安装命令：
```bash
conda env create -f environment.yaml
conda activate mgpt
```

### 2. StableMoFusion 工具（必需）

用于处理 NPZ 动作文件。**任选一种方式**：

```bash
# 方式 A：放项目内（推荐，开箱即用）
cd MotionDiffusionCore/external
git clone [StableMoFusion-repo] StableMoFusion

# 方式 B：环境变量
export STABLE_UTILS_DIR=/path/to/StableMoFusion/utils
```

系统会自动搜索：`$STABLE_UTILS_DIR` → `external/` → 默认路径

### 3. 文本编码器（可选）

首次运行会自动下载。**可选加速**：

```bash
# 复制已有模型
cp -r /limx_embap/tos/user/Jensen/project/MotionStreamer/flan-t5-small \
      text_encoders/

# 否则首次运行时自动从 Hugging Face 下载
```

---

## 🔄 架构更新说明

本项目已从 **Transformer Decoder + MLP 扩散头** 架构升级为 **Transformer 编码器-解码器扩散** 架构。

**v2 最新改进（Token-level Text）**：
- ✅ **Token-level 文本编码**：T5 输出 60 个 token（vs 旧版 1 个池化向量）
- ✅ **Text padding mask**：短文本不会被 padding 稀释，增强文本控制力
- ✅ **零向量 CFG**：真正的无条件（vs 旧版空字符串非零嵌入）
- ✅ **权重衰减**：0.01 正则化，防止过拟合

**v1 主要变化**：
- ✅ 移除了独立的 LLaMAHF Transformer 编码器
- ✅ 扩散头使用统一的 Transformer 编码器-解码器
- ✅ 更强的条件融合和序列建模能力
- ⚠️ **旧检查点不兼容**，需要从头训练

**详细架构对比和迁移指南**：请参阅 [`ARCHITECTURE_UPDATE.md`](./ARCHITECTURE_UPDATE.md)

---

## 📊 实验配置

### 默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| History Length | 60 | 历史窗口（1.2秒 @ 50fps） |
| Prediction Length | 5 | 预测窗口（0.1秒） |
| Motion Dim | 38 | 动作维度 |
| Latent Dim | 16 | Token 维度 |
| Hidden Size | 512 | Token MLP 隐藏层 |
| Diffusion Width | 512 | 扩散 Transformer 维度 |
| Encoder Layers | 4 | Transformer 编码器层数 |
| Decoder Layers | 6 | Transformer 解码器层数 |
| Attention Heads | 8 | 注意力头数 |
| **Text Max Length** | **60** | **文本 token 最大长度（token-level 编码）** ✨ |
| Diffusion Steps (train) | 1000 | 训练时扩散步数（DDPM） |
| Diffusion Steps (infer) | 10 | 推理时采样步数（DDIM，可配置 50） |
| Beta Schedule | squaredcos_cap_v2 | 噪声调度类型 |
| Prediction Type | sample | 预测目标（x0） |
| CFG Mask Prob | 0.0 | CFG 训练零向量遮罩概率（关闭） |
| Weight Decay | 0.01 | 权重衰减（正则化，防止过拟合） |
| Batch Size | 256 | 批次大小 |
| Learning Rate | 1e-4 | 初始学习率 |
| Total Iterations | 200k | 总训练步数 |

### 模型大小

| 组件 | 参数量 | 说明 |
|------|--------|------|
| MotionTokenMLP | ~30K | 动作特征映射 |
| Encoder (4层) | ~4M | 条件融合 |
| Decoder (6层) | ~8M | 序列生成 |
| **总计** | **~12M** | 轻量级模型 |

### 硬件配置

- **推荐**：4x GPU (V100 / A100)
- **显存**：每卡约 16-20GB（batch_size=256）
- **训练时间**：200k 步约 24-48 小时（取决于 GPU）
- **单卡训练**：可行，但需要减小 batch_size（如 64）

---

## ❓ 常见问题

### Q1: 为什么历史窗口是 60 帧？
A: 60 帧 @ 50fps = 1.2 秒，足够捕捉短期运动模式（走路、转身等），同时避免序列过长导致计算负担。

### Q2: 为什么预测 5 帧而不是 1 帧？
A: 单帧预测误差会快速累积，5 帧一次性生成可以：
- 更好地建模短期轨迹
- 减少累积误差
- 提升生成稳定性

### Q3: 为什么 BABEL Stream 只从 B 段采样？
A: A 段是过渡动作（transition），质量和一致性不如 B 段主要动作，跳过可以提升训练效果。

### Q4: CFG 强度如何选择？
A: 
- **1.0**：关闭 CFG，标准生成
- **7.5**：**推荐值**，显著增强文本一致性
- **1.5-3.0**：轻度引导，适合需要保留一定随机性的场景
- **>10.0**：可能过度拟合文本，导致动作不自然

**注意**：默认训练时未启用 CFG 训练（cfg_mask_prob=0.0），但推理时仍可使用 CFG。如需更强效果，可在训练时设置 cfg_mask_prob=0.1。

### Q5: 训练多久可以收敛？
A: 经验值：
- 20k 步：初步学会动作模式
- 50k 步：文本-动作对齐较好
- 100k 步：充分收敛

### Q6: 推理采样步数如何选择？
A: 
- **10 步（默认）**：速度快，质量已达 90%，适合实时/交互应用
- **50 步**：质量最优，适合离线高质量生成
- **权衡**：10 步约 0.5 秒/次，50 步约 2.5 秒/次（单 GPU）

建议：
- 开发调试：10 步
- 最终演示/发布：50 步

### Q7: `/tmp` 空间不足怎么办？
A: 训练脚本已自动配置：
- 临时文件：`项目目录/.tmp/`（DataLoader multiprocessing）
- HuggingFace 缓存：`项目目录/.cache/huggingface/`
- PyTorch 缓存：`项目目录/.cache/torch/`

这些目录会自动创建，无需手动干预。如需清理缓存：
```bash
rm -rf .tmp/ .cache/
```

### Q8: 新旧架构有什么区别？
A: **重要架构更新**：

**旧架构（已废弃）**：
```
MotionTokenMLP → LLaMAHF Transformer → 条件向量 z → MLP 扩散头
```

**新架构（当前）**：
```
MotionTokenMLP → Transformer 编码器-解码器扩散头
                  ├─ 编码器：融合 timestep + history + text
                  └─ 解码器：生成去噪后的动作序列
```

**优势**：
- ✅ 更强的条件融合（编码器双向注意力）
- ✅ 更好的序列建模（解码器处理 5 帧序列）
- ✅ 统一架构，无需独立编码器

**注意**：旧检查点不兼容。详见 `ARCHITECTURE_UPDATE.md`。

### Q9: 何时需要启用 CFG 训练？
A: **默认关闭即可**（cfg_mask_prob=0.0），推理时仍可使用 CFG。

**启用 CFG 训练的情况**（设置 cfg_mask_prob=0.1）：
- ✅ 需要**更强的文本控制力**
- ✅ 希望推理时使用**高 CFG 值**（如 7.5-10.0）
- ✅ 生成的动作需要**严格遵循文本描述**

**不启用的优势**（默认）：
- ✅ 训练更快（无需额外的无条件分支）
- ✅ 遵循 Pi0 的设计理念
- ✅ 推理时仍可使用适度的 CFG（如 1.0-3.0）

**建议**：先用默认配置训练，如果发现文本控制不够强，再启用 CFG 训练。

### Q10: 如何加快数据加载速度？
A: **已实现自动缓存机制** 🚀
- **第一次训练**：预处理所有数据（2-5 分钟），并保存缓存到项目目录
- **后续训练**：直接加载缓存（5-10 秒），加速 **10-50 倍**！

缓存文件位置（在项目目录下，避免污染数据集目录）：
```
MotionDiffusionCore/
├── .cache/
│   └── datasets/
│       ├── humanml3d_a1b2c3d4_e5f6g7h8.pkl       # 自动生成
│       ├── babel_stream_i9j0k1l2_m3n4o5p6.pkl    # 自动生成
│       └── humanml3d_stream_q7r8s9t0_u1v2w3x4.pkl # 自动生成
```

缓存文件命名规则：`{dataset}_{data_path_hash}_{config_hash}.pkl`
- 数据路径变化 → 重新生成缓存
- 配置变化（history_len/pred_len等） → 重新生成缓存

清理缓存：
```bash
rm -rf .cache/datasets/
```

---

## 📝 注意事项

1. **路径配置（重要）**：
   - ✅ **核心代码无硬编码**：所有路径通过参数传入
   - ✅ 数据集路径、统计文件路径、工具目录路径均可配置
   - ✅ 默认值在 `configs/default.yaml` 中定义
   - ✅ 命令行参数优先级高于配置文件
   
2. **数据路径**：确保三个数据集路径正确，否则会加载失败

3. **统计文件**：`Mean.npy` 和 `Std.npy` 必须与训练数据一致

4. **文本编码器**：首次使用会自动下载模型（需网络）

5. **显存不足**：减少 `batch_size` 或使用混合精度 `--mixed_precision fp16`

6. **Zero-Padding**：短序列补 0 是在**归一化空间**，不影响训练

7. **路径迁移**：如需在不同机器/环境运行，只需修改配置文件或传入路径参数，无需修改代码

8. **⚠️ 扩散配置一致性（重要）**：
   - 推理时 `num_train_timesteps`、`beta_schedule`、`prediction_type` **必须与训练时一致**
   - 推理时 `n_decoder_layers`、`n_encoder_layers`、`n_heads` **必须与训练时一致**
   - 只有 `num_sampling_steps` 和 `cfg` 可以在推理时自由调整
   - 如果加载旧模型出错，检查 checkpoint 中保存的配置参数

9. **⚠️ 架构版本兼容性**：
   - 当前版本使用 Transformer 编码器-解码器架构
   - 旧版本检查点（包含 "trans" 键）不兼容
   - 详见 `ARCHITECTURE_UPDATE.md` 了解架构变更

---

## 📄 引用

如果本项目对您的研究有帮助，欢迎 star ⭐

---

## 📧 联系方式

如有问题或建议，欢迎通过 Issue 反馈。