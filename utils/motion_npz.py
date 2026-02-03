import os
import sys
import importlib.util

import numpy as np


def _load_robot_npz_utils():
    stable_mofusion_root = "/limx_embap/tos/user/Jensen/dataset/motion_data/StableMoFusion"

    prev_utils_modules = {k: v for k, v in sys.modules.items() if k == "utils" or k.startswith("utils.")}
    for k in list(prev_utils_modules.keys()):
        del sys.modules[k]

    try:
        if stable_mofusion_root not in sys.path:
            sys.path.insert(0, stable_mofusion_root)
        from utils.robot_npz_utils import reshape_generated_motion_38d
        return reshape_generated_motion_38d
    finally:
        sys.modules.update(prev_utils_modules)


def save_motion_npz(motion_38d, out_path, fps=50, smooth=False):
    reshape_func = _load_robot_npz_utils()
    npz_dict = reshape_func(
        motion_38d, fps=fps, smooth=smooth, adaptive_smooth=False, static_start=False
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **npz_dict)
