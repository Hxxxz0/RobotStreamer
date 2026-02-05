cd /limx_embap/tos/user/Jensen/project/MotionDiffusionCore
conda activate mgpt

# Inference settings:
# - If trained with cfg_mask_prob=0.0 (default): use --cfg 1.0 (no CFG)
# - If trained with cfg_mask_prob>0.0 (e.g., 0.1): use --cfg 1.5~2.5 (enable CFG)
# - temperature: 1.0 for standard sampling, <1.0 for more deterministic results
python scripts/infer_stream.py \
  --ckpt outputs/motion_diff/latest.pth \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text "A person walks forward." \
  --max_motion_length 500 \
  --num_sampling_steps 10 \
  --cfg 1.0 \
  --temperature 1.0 \
  --out_dir outputs/infer_test


# --text "A person waves its left hand." \
# --text "A person walks forward slowly." \
# --text "A person jumps up and down." \
# --text "A person runs in place." \
# --text "A person raises both arms above head." \
# --text "A person sits down on a chair." \
# --text "A person stands up from sitting." \
# --text "A person kicks with right leg." \
# --text "A person claps hands together." \
# --text "A person turns around 360 degrees." \