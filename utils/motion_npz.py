import os
import numpy as np


def save_motion_npz(motion_38d, out_path, fps=50, smooth=False):
    """Save 38D motion as robot NPZ format.
    
    Args:
        motion_38d: (T, 38) - [joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)]
        out_path: str - output NPZ file path
        fps: int - frames per second
        smooth: bool - whether to apply smoothing
    """
    # Import from project's utils directory
    from utils.robot_npz_utils import reshape_generated_motion_38d
    
    npz_dict = reshape_generated_motion_38d(
        motion_38d, fps=fps, smooth=smooth, adaptive_smooth=False, static_start=False
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **npz_dict)
