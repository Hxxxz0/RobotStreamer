#!/bin/bash
# Overfit training with correct hyperparameters (matching main training)

# Set project-local directories
PROJECT_DIR="/limx_embap/tos/user/Jensen/project/MotionDiffusionCore"
export TMPDIR="${PROJECT_DIR}/.tmp"
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TORCH_HOME="${PROJECT_DIR}/.cache/torch"

mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"
export TOKENIZERS_PARALLELISM=false

# Data paths
NPZ_PATH="/limx_embap/tos/user/Jensen/dataset/robot_humanml_data/npz/000000.npz"
TEXT_PATH="/limx_embap/tos/user/Jensen/dataset/robot_humanml_data/texts/000000.txt"
META_DIR="/limx_embap/tos/user/Jensen/project/dataset/statistics"

if [ ! -f "$NPZ_PATH" ] || [ ! -f "$TEXT_PATH" ]; then
    echo "Error: Data files not found"
    exit 1
fi

echo "Starting overfitting training (corrected hyperparameters)"
echo "Logs: logs/overfit_corrected/train.log"

# Use SAME hyperparameters as main training:
# - LR: 1e-4 (not 1e-3)
# - Weight decay: 0.01 (not 0.0)  
# - Root loss weight: 10.0 (not 1.0)
# - Larger batch size: 16 (not 4) for more stable gradients
nohup python train/train_overfit_single.py \
    --npz_path "$NPZ_PATH" \
    --text_path "$TEXT_PATH" \
    --meta_dir "$META_DIR" \
    --batch_size 16 \
    --total_iter 10000 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --root_loss_weight 10.0 \
    --log_every 50 \
    --exp_name overfit_corrected \
    --seed 123 \
    > logs/overfit_single/train.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Config: LR=1e-4, BS=16, WD=0.01, Root=10.0"
echo "View logs: tail -f logs/overfit_single/train.log"
echo "Kill training: kill $TRAIN_PID"
