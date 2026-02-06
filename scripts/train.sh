#!/bin/bash

# Train MotionDiffusionCore with 4 GPUs
# Copy the command below and run directly, or add nohup for background

# Set project-local directories to avoid /tmp and /root space issues
PROJECT_DIR="/limx_embap/tos/user/Jensen/project/MotionDiffusionCore"
export TMPDIR="${PROJECT_DIR}/.tmp"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"

# Create temp directories if they don't exist
mkdir -p "$TMPDIR"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

echo "Starting training..."
echo "Temporary files: $TMPDIR"
echo "HuggingFace cache: $HF_HOME"
echo "Logs: logs/motion_diff/train.log"

nohup accelerate launch \
  --mixed_precision no \
  --num_processes 4 \
  --num_machines 1 \
  --dynamo_backend no \
  train/train.py \
  --config configs/default.yaml \
  > logs/motion_diff/train.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "View logs: tail -f logs/motion_diff/train.log"
echo "Kill training: kill $TRAIN_PID"

# =====================================================================
# Run in background (copy and modify as needed):
# =====================================================================
#
# nohup accelerate launch \
#   --mixed_precision no \
#   --num_processes 4 \
#   --num_machines 1 \
#   --dynamo_backend no \
#   train/train_motiondiffusion.py \
#   --config configs/default.yaml \
#   > logs/motion_diff/train.log 2>&1 &
#
# Then check the PID and view logs:
#   ps aux | grep train_motiondiffusion
#   tail -f logs/motion_diff/train.log
# =====================================================================
