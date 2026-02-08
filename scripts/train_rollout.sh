#!/bin/bash

# Rollout Training Script for MotionDiffusionCore
# Stage 2: Scheduled Sampling (0% → 100% rollout)
# 从 Stage 1 (outputs/motion_diff/latest.pth) 继续训练

# Set project-local directories to avoid /tmp and /root space issues
PROJECT_DIR="/limx_embap/tos/user/Jensen/project/MotionDiffusionCore"
export TMPDIR="${PROJECT_DIR}/.tmp"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"

# Create temp directories if they don't exist
mkdir -p "$TMPDIR"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"
mkdir -p "logs/motion_diff_rollout"

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际情况修改（单卡改为 0，双卡改为 0,1）
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

# 配置文件
CONFIG="configs/rollout.yaml"

echo "Starting Rollout Training (Stage 2)..."
echo "Temporary files: $TMPDIR"
echo "HuggingFace cache: $HF_HOME"
echo "Checkpoint from: outputs/motion_diff/latest.pth"
echo "Logs: logs/motion_diff_rollout/train_rollout.log"
echo ""

# 训练参数（使用 accelerate 多卡训练，如果是单卡则移除 accelerate）
# 多卡训练（4 卡）
nohup accelerate launch \
  --mixed_precision no \
  --num_processes 4 \
  --num_machines 1 \
  --dynamo_backend no \
  --main_process_port 29501 \
  train/train_rollout.py \
  --config ${CONFIG} \
  > logs/motion_diff_rollout/train_rollout.log 2>&1 &

TRAIN_PID=$!
echo "Rollout training started with PID: $TRAIN_PID"
echo "View logs: tail -f logs/motion_diff_rollout/train_rollout.log"
echo "Kill training: kill $TRAIN_PID"
echo ""
echo "Next steps:"
echo "1. Monitor training: tail -f logs/motion_diff_rollout/train_rollout.log"
echo "2. TensorBoard: tensorboard --logdir logs/motion_diff_rollout"
echo "3. Verify improvement: python scripts/verify_exposure_bias.py --ckpt outputs/motion_diff_rollout/latest.pth ..."
echo "4. If needed, switch to Stage 3 by editing configs/rollout.yaml (stage: 3)"