cd /limx_embap/tos/user/Jensen/project/MotionDiffusionCore
conda activate mgpt


# Inference settings (v2 - token-level text + CFG zero-masking):
# - Now trained with cfg_mask_prob=0.1: use --cfg 2.0~5.0 for text control
# - CFG uses zero vectors internally (no empty string encoding)
# - temperature: 1.0 for standard sampling, <1.0 for more deterministic results
# - slide_step: 1=single-frame sliding, pred_len=jump by full window
# - num_iterations: total iteration count (overrides max_motion_length)
python scripts/infer_stream.py \
  --ckpt outputs/motion_diff/ckpt_200000.pth \
  --mean /limx_embap/tos/user/Jensen/project/dataset/statistics/Mean.npy \
  --std /limx_embap/tos/user/Jensen/project/dataset/statistics/Std.npy \
  --text "A person walks forward." \
  --num_iterations 100 \
  --slide_step 5 \
  --num_sampling_steps 10 \
  --cfg 1.0 \
  --temperature 1.0 \
  --history_len 60 \
  --pred_len 5 \
  --text_max_length 60 \
  --out_dir outputs/infer_test

# --ckpt outputs/motion_diff/latest.pth \
# --ckpt outputs/motion_diff_rollout/checkpoint_005000.pth \

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