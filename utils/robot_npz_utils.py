"""Utilities for creating complete robot NPZ files from generated positions."""
import numpy as np
import torch
from scipy.signal import savgol_filter
from utils.rotation_utils import rot6d_to_quat_wxyz
from utils.robot_process import recover_root_xy_from_velocity


def compute_linear_velocity(positions, fps=50):
    """Compute linear velocity from positions using forward differences.
    
    Args:
        positions: (frames, 30, 3) numpy array
        fps: int - frame rate (default: 50)
    
    Returns:
        velocities: (frames, 30, 3) numpy array in m/s
    """
    dt = 1.0 / fps
    velocities = np.zeros_like(positions)
    # Forward difference for all frames except last, then divide by dt to get velocity
    velocities[:-1] = (positions[1:] - positions[:-1]) / dt
    # Copy last frame velocity from second-to-last
    velocities[-1] = velocities[-2] if len(velocities) > 1 else 0
    return velocities


def smooth_positions_savgol(positions, window_length=5, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑位置轨迹，消除高频噪声。
    
    Args:
        positions: (frames, N, 3) numpy array - 位置数据
        window_length: int - 窗口长度（必须为奇数），默认5帧（50FPS下约0.1秒）
        polyorder: int - 多项式阶数，默认3（保留加速度特征）
    
    Returns:
        smoothed_positions: (frames, N, 3) numpy array - 平滑后的位置
    """
    frames, N, dims = positions.shape
    
    # 确保窗口长度为奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 如果帧数太少，无法应用滤波器
    if frames < window_length:
        return positions
    
    # 确保 polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    smoothed = np.zeros_like(positions)
    
    # 对每个关节的每个维度分别应用Savitzky-Golay滤波
    for joint_idx in range(N):
        for dim_idx in range(dims):
            smoothed[:, joint_idx, dim_idx] = savgol_filter(
                positions[:, joint_idx, dim_idx],
                window_length=window_length,
                polyorder=polyorder,
                mode='interp'  # 使用插值处理边界
            )
    
    return smoothed


def adaptive_smooth_positions(positions, fps=50):
    """
    自适应平滑：对初始帧应用更强的平滑，解决初始速度过大问题。
    
    策略：
    1. 前15帧使用window=11的强平滑（消除初始跳变）
    2. 全局使用window=7的中度平滑（保持整体连贯性）
    
    Args:
        positions: (frames, N, 3) numpy array - 位置数据
        fps: int - 帧率
    
    Returns:
        smoothed_positions: (frames, N, 3) numpy array - 平滑后的位置
    """
    frames, N, dims = positions.shape
    smoothed = positions.copy()
    
    # 第一步：整体中度平滑 (window=7)
    if frames >= 7:
        for joint_idx in range(N):
            for dim_idx in range(dims):
                smoothed[:, joint_idx, dim_idx] = savgol_filter(
                    smoothed[:, joint_idx, dim_idx],
                    window_length=7,
                    polyorder=3,
                    mode='interp'
                )
    
    # 第二步：对前15帧再次强力平滑 (window=11)，解决初始跳变问题
    if frames >= 15:
        for joint_idx in range(N):
            for dim_idx in range(dims):
                smoothed[:15, joint_idx, dim_idx] = savgol_filter(
                    positions[:15, joint_idx, dim_idx],
                    window_length=11,
                    polyorder=3,
                    mode='interp'
                )
    
    return smoothed


def force_static_start(positions, static_frames=2, blend_frames=8):
    """
    强制静止起始：让动作从几乎静止状态开始，解决初始速度异常大的问题。
    
    策略：
    - 前static_frames帧保持第0帧位置不变（完全静止）
    - 接下来blend_frames帧线性插值到原始轨迹（平滑过渡）
    
    Args:
        positions: (frames, N, 3) numpy array - 位置数据
        static_frames: int - 完全静止的帧数，默认2帧
        blend_frames: int - 从静止到正常的过渡帧数，默认8帧
    
    Returns:
        positions: (frames, N, 3) numpy array - 修改后的位置
    """
    frames, N, dims = positions.shape
    
    if frames < static_frames + blend_frames:
        # 帧数太少，不做处理
        return positions
    
    # 保存原始第0帧位置
    start_pos = positions[0].copy()
    
    # 前static_frames帧保持不动
    for i in range(static_frames):
        positions[i] = start_pos
    
    # 接下来blend_frames帧线性插值到实际位置
    target_frame = static_frames + blend_frames
    target_pos = positions[target_frame].copy()
    
    for i in range(static_frames, target_frame):
        alpha = (i - static_frames) / blend_frames
        positions[i] = start_pos * (1 - alpha) + target_pos * alpha
    
    return positions


def create_complete_npz(positions, fps=50):
    """Create complete NPZ dictionary from generated positions.
    
    Args:
        positions: (frames, 30, 3) numpy array - generated positions
        fps: int - frame rate
    
    Returns:
        dict with keys: fps, joint_pos, joint_vel, body_pos_w, 
                       body_quat_w, body_lin_vel_w, body_ang_vel_w
    """
    frames, joints, _ = positions.shape
    assert joints == 30, f"Expected 30 joints, got {joints}"
    
    # Compute linear velocity (in m/s)
    lin_vel = compute_linear_velocity(positions, fps=fps)
    
    # Create NPZ dictionary
    npz_dict = {
        'fps': np.array([fps]),
        'body_pos_w': positions.astype(np.float32),
        'body_lin_vel_w': lin_vel.astype(np.float32),
        
        # Fields we cannot generate (fill with zeros/identity)
        'joint_pos': np.zeros((frames, 29), dtype=np.float32),
        'joint_vel': np.zeros((frames, 29), dtype=np.float32),
        'body_quat_w': np.zeros((frames, 30, 4), dtype=np.float32),
        'body_ang_vel_w': np.zeros((frames, 30, 3), dtype=np.float32),
    }
    
    # Initialize quaternions to identity (w=1, x=y=z=0)
    npz_dict['body_quat_w'][:, :, 0] = 1.0
    
    return npz_dict


def reshape_generated_motion(motion_flat, joints_num=30):
    """Reshape flattened motion (frames, 90) to (frames, 30, 3).
    
    Args:
        motion_flat: (frames, 90) numpy array or torch tensor
        joints_num: int - number of joints (default: 30)
    
    Returns:
        positions: (frames, joints_num, 3) numpy array
    """
    if isinstance(motion_flat, torch.Tensor):
        motion_flat = motion_flat.cpu().numpy()
    
    frames = motion_flat.shape[0]
    positions = motion_flat.reshape(frames, joints_num, 3)
    return positions


def save_robot_npz(output_path, positions, fps=50):
    """Save robot motion as complete NPZ file.
    
    Args:
        output_path: str - path to save NPZ file
        positions: (frames, 30, 3) numpy array - body positions
        fps: int - frame rate (default: 50)
    """
    npz_data = create_complete_npz(positions, fps=fps)
    np.savez(output_path, **npz_data)
    return npz_data


def compute_angular_velocity_from_quat(quaternions, fps=50):
    """
    从四元数差分计算角速度（近似）
    
    Args:
        quaternions: (frames, N, 4) - N个关节的四元数序列
        fps: 帧率
    
    Returns:
        ang_vel: (frames, N, 3) - 角速度
    """
    frames, N, _ = quaternions.shape
    ang_vel = np.zeros((frames, N, 3), dtype=np.float32)
    
    dt = 1.0 / fps
    
    for i in range(frames - 1):
        for j in range(N):
            q0 = quaternions[i, j]      # (4,)
            q1 = quaternions[i+1, j]    # (4,)
            
            # 四元数差分: dq = q1 * q0^{-1}
            q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])  # 共轭
            dq = quat_multiply(q1, q0_conj)  # 相对旋转
            
            # 转换为角速度: w ≈ 2 * [x, y, z] / dt
            ang_vel[i, j] = 2.0 * dq[1:4] / dt
    
    # 最后一帧复制
    ang_vel[-1] = ang_vel[-2] if frames > 1 else 0
    
    return ang_vel


def quat_multiply(q1, q2):
    """四元数乘法 [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def reshape_generated_motion_239d(motion_flat, fps=50, smooth=True, smooth_window=5, 
                                  adaptive_smooth=False, static_start=False,
                                  static_frames=2, blend_frames=8):
    """
    将239维展平数据重塑为完整NPZ格式（包含速度计算）
    
    Args:
        motion_flat: (frames, 239) numpy array
        fps: 帧率
        smooth: bool - 是否应用基础Savitzky-Golay平滑（默认True）
        smooth_window: int - 基础平滑窗口大小（必须为奇数，默认5）
        adaptive_smooth: bool - 是否使用自适应平滑（前15帧强力平滑，推荐True）
        static_start: bool - 是否强制静止起始（解决初始速度过大，推荐True）
        static_frames: int - 静止帧数（默认2）
        blend_frames: int - 过渡帧数（默认8）
        
    Returns:
        dict with all 7 NPZ fields
    """
    frames = motion_flat.shape[0]
    
    # 分割239维
    joint_pos = motion_flat[:, :29]                    # (frames, 29)
    body_pos_w_flat = motion_flat[:, 29:119]           # (frames, 90)
    body_quat_w_flat = motion_flat[:, 119:239]         # (frames, 120)
    
    # 重塑
    body_pos_w = body_pos_w_flat.reshape(frames, 30, 3)    # (frames, 30, 3)
    body_quat_w = body_quat_w_flat.reshape(frames, 30, 4)  # (frames, 30, 4)
    
    # 应用平滑策略
    if adaptive_smooth:
        # 自适应平滑：对初始帧应用更强的平滑
        body_pos_w = adaptive_smooth_positions(body_pos_w, fps=fps)
    elif smooth:
        # 基础平滑
        body_pos_w = smooth_positions_savgol(body_pos_w, window_length=smooth_window, polyorder=3)
    
    # 强制静止起始（在平滑之后应用，效果更自然）
    if static_start:
        body_pos_w = force_static_start(body_pos_w, static_frames=static_frames, blend_frames=blend_frames)
    
    # 归一化四元数（确保单位长度）
    quat_norm = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
    body_quat_w = body_quat_w / (quat_norm + 1e-8)
    
    # 计算速度（差分）
    # joint_vel: (frames, 29) - 直接差分
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = joint_pos[1:] - joint_pos[:-1]
    joint_vel[-1] = joint_vel[-2] if frames > 1 else 0
    
    body_lin_vel_w = compute_linear_velocity(body_pos_w, fps=fps)         # (frames, 30, 3)
    body_ang_vel_w = compute_angular_velocity_from_quat(body_quat_w, fps=fps)  # (frames, 30, 3)
    
    return {
        'fps': np.array([fps]),
        'joint_pos': joint_pos.astype(np.float32),
        'joint_vel': joint_vel.astype(np.float32),
        'body_pos_w': body_pos_w.astype(np.float32),
        'body_quat_w': body_quat_w.astype(np.float32),
        'body_lin_vel_w': body_lin_vel_w.astype(np.float32),
        'body_ang_vel_w': body_ang_vel_w.astype(np.float32),
    }


def reshape_generated_motion_38d(motion_38d, fps=50, smooth=False, smooth_window=5,
                                 adaptive_smooth=False, static_start=False,
                                 static_frames=2, blend_frames=8):
    """
    Reconstruct minimal NPZ from 38D velocity-based representation (for replay compatibility).
    
    NEW 38D format: [joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)]
    
    This function integrates root velocity to recover position, following the HumanML3D approach.
    
    Args:
        motion_38d: (T, 38) numpy array - generated motion in 38D format
        fps: int - frame rate (default: 50)
        smooth: bool - whether to apply basic smoothing (default: False)
        smooth_window: int - smoothing window size (default: 5)
        adaptive_smooth: bool - use adaptive smoothing (default: False)
        static_start: bool - force static start (default: False)
        static_frames: int - number of static frames at start (default: 2)
        blend_frames: int - number of blend frames (default: 8)
    
    Returns:
        dict with minimal NPZ fields for replay:
            - fps: (1,) - frame rate
            - joint_pos: (T, 29) - joint angles
            - root_pos: (T, 3) - root position
            - root_rot: (T, 4) - root rotation as quaternion (wxyz)
    """
    frames = motion_38d.shape[0]
    
    # Split 38D into components (NEW FORMAT)
    joint_pos = motion_38d[:, :29]         # (T, 29)
    root_vel_xy = motion_38d[:, 29:31]     # (T, 2) - XY velocity in local frame
    root_z = motion_38d[:, 31:32]          # (T, 1) - Z height
    root_rot_6d = motion_38d[:, 32:38]     # (T, 6)
    
    # Convert 6D rotation to quaternion (wxyz format)
    root_rot_quat = rot6d_to_quat_wxyz(root_rot_6d)  # (T, 4) - wxyz
    
    # Integrate velocity to recover root XY position
    root_xy_pos = recover_root_xy_from_velocity(root_vel_xy, root_rot_quat)  # (T, 2)
    
    # Combine XY position and Z height
    root_pos = np.concatenate([root_xy_pos, root_z], axis=1)  # (T, 3)
    
    # Apply smoothing if requested
    if smooth or adaptive_smooth or static_start:
        # 1. Smooth root position
        # Reshape root_pos for smoothing functions (expecting (T, N, 3))
        root_pos_reshaped = root_pos[:, np.newaxis, :]  # (T, 1, 3)
        
        if adaptive_smooth:
            root_pos_reshaped = adaptive_smooth_positions(root_pos_reshaped, fps=fps)
        elif smooth:
            root_pos_reshaped = smooth_positions_savgol(root_pos_reshaped, window_length=smooth_window, polyorder=3)
        
        if static_start:
            root_pos_reshaped = force_static_start(root_pos_reshaped, static_frames=static_frames, blend_frames=blend_frames)
        
        # Reshape back to (T, 3)
        root_pos = root_pos_reshaped[:, 0, :]
        
        # 2. Smooth joint positions
        # Reshape joint_pos to (T, 29, 1) for smoothing (treating each joint angle as a dimension)
        joint_pos_reshaped = joint_pos[:, :, np.newaxis] # (T, 29, 1)
        
        if adaptive_smooth:
            joint_pos_reshaped = adaptive_smooth_positions(joint_pos_reshaped, fps=fps)
        elif smooth:
            joint_pos_reshaped = smooth_positions_savgol(joint_pos_reshaped, window_length=smooth_window, polyorder=3)
            
        if static_start:
            joint_pos_reshaped = force_static_start(joint_pos_reshaped, static_frames=static_frames, blend_frames=blend_frames)
            
        joint_pos = joint_pos_reshaped[:, :, 0]
    
    # Create minimal NPZ dictionary (compatible with replay script)
    return {
        'fps': np.array([fps]),
        'joint_pos': joint_pos.astype(np.float32),
        'root_pos': root_pos.astype(np.float32),
        'root_rot': root_rot_quat.astype(np.float32),  # wxyz format
    }


if __name__ == '__main__':
    # Test the utilities
    print("Testing robot NPZ utilities...")
    
    # Create dummy data
    frames = 100
    joints = 30
    positions = np.random.randn(frames, joints, 3).astype(np.float32)
    
    print(f"Input positions shape: {positions.shape}")
    
    # Test velocity computation
    velocities = compute_linear_velocity(positions)
    print(f"Computed velocities shape: {velocities.shape}")
    
    # Test NPZ creation
    npz_dict = create_complete_npz(positions, fps=50)
    print(f"NPZ dictionary keys: {list(npz_dict.keys())}")
    for key, value in npz_dict.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # Test flattening and reshaping
    motion_flat = positions.reshape(frames, -1)  # (100, 90)
    print(f"Flattened motion shape: {motion_flat.shape}")
    
    positions_reshaped = reshape_generated_motion(motion_flat, joints_num=30)
    print(f"Reshaped positions shape: {positions_reshaped.shape}")
    
    # Verify reshaping is correct
    assert np.allclose(positions, positions_reshaped), "Reshaping failed!"
    print("✓ All tests passed!")

