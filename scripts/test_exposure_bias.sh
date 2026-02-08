#!/bin/bash

# 验证 Exposure Bias 的测试脚本

cd /limx_embap/tos/user/Jensen/project/MotionDiffusionCore
conda activate mgpt

python scripts/verify_exposure_bias.py \
  --ckpt outputs/motion_diff/ckpt_200000.pth \
  --data_path /limx_embap/tos/user/Jensen/dataset/robot_humanml_data \
  --sample_id 000000 \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text_encoder_type t5 \
  --text_max_length 60 \
  --num_sampling_steps 10 \
  --cfg 1.0 \
  --temperature 1.0 \
  --out_dir outputs/exposure_bias_test

# 测试多个样本（可选）
# for sample_id in 000001 000002 000003; do
#     python scripts/verify_exposure_bias.py \
#       --ckpt outputs/motion_diff/latest.pth \
#       --data_path /limx_embap/tos/user/Jensen/dataset/robot_humanml_data \
#       --sample_id $sample_id \
#       --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
#       --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
#       --text_max_length 60 \
#       --out_dir outputs/exposure_bias_test
# done
