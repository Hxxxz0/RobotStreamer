# 坐标对齐问题分析与修复

## 问题描述

训练时发现模型输出在90度之间跳动，loss虽然收敛但不稳定。

## 根本原因

`utils/robot_process.py` 中的坐标对齐逻辑有bug：

```python
# 第 66-77 行
if y_range > x_range:
    across = np.array([0, 1, 0])  # Y是横向
    # forward = Z × across = [-1, 0, 0] 或 [1, 0, 0]
else:
    across = np.array([1, 0, 0])  # X是横向  
    # forward = Z × across = [0, 1, 0]  ← 问题！对齐到Y而不是X
```

### 问题分析

1. **训练数据不一致**：
   - 某些样本初始朝向X → 对齐到 +X ✓
   - 某些样本初始朝向Y → 对齐到 +Y ✗（应该对齐到 +X）

2. **模型学到两种模式**：
   - 模型同时学习了 X朝向 和 Y朝向 的动作
   - 推理时在两种模式之间跳动（90度差异）

3. **Loss 波动**：
   - 不同batch包含不同朝向的样本
   - Loss在两种模式之间震荡

## 解决方案

### 方案1：使用 root_quat 计算前向向量（推荐）

直接从 root 四元数提取前向向量，更可靠：

```python
def get_forward_from_quat(quat_wxyz):
    """从四元数提取前向向量 (假设初始前向是+X)"""
    w, x, y, z = quat_wxyz
    # 旋转 [1, 0, 0] 向量
    forward_x = 1 - 2*(y*y + z*z)
    forward_y = 2*(x*y + w*z)
    forward_z = 2*(x*z - w*y)
    return np.array([forward_x, forward_y, forward_z])

# 在 process_robot_npz 中:
root_quat_init = body_quat_w[0, root_idx, :]  # (4,) wxyz
forward_init = get_forward_from_quat(root_quat_init)
forward_init[2] = 0  # 投影到XY平面
forward_init = forward_init / (np.linalg.norm(forward_init) + 1e-8)
```

### 方案2：简化启发式（次选）

无论哪种情况，都强制对齐到 +X：

```python
# 简化逻辑，直接检测主导方向
positions_init = body_pos_w[0]
y_range = positions_init[:, 1].max() - positions_init[:, 1].min()
x_range = positions_init[:, 0].max() - positions_init[:, 0].min()

if y_range > x_range:
    # 主要沿Y展开 → forward 在 X 方向
    forward_init = np.array([1, 0, 0])
else:
    # 主要沿X展开 → forward 在 Y 方向
    # 需要旋转到 X 方向（+90度）
    forward_init = np.array([0, 1, 0])
    # 最后都对齐到 +X

target = np.array([1, 0, 0])
rot_quat = qbetween_np(forward_init[np.newaxis, :], target[np.newaxis, :])[0]
```

## 修复后需要做的

1. **重新处理数据**：
   - 删除 `.cache/datasets/*.pkl`
   - 重新运行数据预处理

2. **从头训练**：
   - 旧的checkpoint已经学习了错误的对齐
   - 必须从头开始训练

3. **验证修复**：
   - 检查处理后的数据，确认所有样本都对齐到 +X
   - 训练时监控loss是否更稳定
