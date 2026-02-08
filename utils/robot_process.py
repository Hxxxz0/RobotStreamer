"""
Robot motion data processing utilities.
Adapted from HumanML3D's motion_process.py for Z-up coordinate system.

Key differences from HumanML3D:
- Z-up coordinate system (X/Y ground plane, Z is height)
- Face direction unified to +X (forward)
- Uses existing body_lin_vel_w from NPZ files
"""
import numpy as np
from utils.quaternion import qbetween_np, qrot_np, qmul_np
from utils.rotation_utils import quat_wxyz_to_6d


def process_robot_npz(npz_data, root_idx=0):
    """
    Process robot NPZ data with root alignment and facing normalization.
    
    Pipeline (similar to HumanML3D's process_file):
    1. Put on floor: min(Z) -> 0
    2. Root XY at origin: root_pos[0, XY] -> (0, 0)
    3. Face +X direction: compute initial facing from hips/shoulders and rotate to +X
    4. Extract velocity: compute per-frame displacement from aligned root
       positions in the aligned global frame (faces +X)
    
    Args:
        npz_data: dict - loaded NPZ file with keys:
            - joint_pos: (T, 29)
            - body_pos_w: (T, 30, 3)
            - body_quat_w: (T, 30, 4) - wxyz format
        root_idx: int - index of root body (default: 0)
    
    Returns:
        features_38d: (T, 38) numpy array
            [joint_pos(29), root_vel_xy(2), root_z(1), root_rot_6d(6)]
            Note: root_vel_xy is per-frame displacement in aligned global frame.
    """
    joint_pos = npz_data['joint_pos']  # (T, 29)
    body_pos_w = npz_data['body_pos_w'].copy()  # (T, 30, 3)
    body_quat_w = npz_data['body_quat_w'].copy()  # (T, 30, 4)
    # body_lin_vel_w exists but is not used; we derive velocity after alignment
    
    T = body_pos_w.shape[0]
    
    # Step 1: Put on floor (Z-up: floor is XY plane)
    floor_height = body_pos_w[:, :, 2].min()  # Min Z across all bodies and time
    body_pos_w[:, :, 2] -= floor_height
    
    # Step 2: Root XY at origin
    root_pos_init = body_pos_w[0, root_idx, :].copy()  # Initial root position
    root_xy_init = root_pos_init[[0, 1]]  # XY components
    body_pos_w[:, :, 0] -= root_xy_init[0]
    body_pos_w[:, :, 1] -= root_xy_init[1]
    
    # Step 3: Align facing direction to +X
    # Extract forward direction from root quaternion (more robust than body spread)
    root_quat_init = body_quat_w[0, root_idx, :]  # (4,) wxyz format
    
    # Rotate [1, 0, 0] vector by quaternion to get forward direction
    # q * v * q^-1 where v = [0, 1, 0, 0] (pure quaternion)
    w, x, y, z = root_quat_init
    forward_x = 1 - 2 * (y * y + z * z)
    forward_y = 2 * (x * y + w * z)
    forward_z = 2 * (x * z - w * y)
    forward_init_3d = np.array([forward_x, forward_y, forward_z])
    
    # Project to XY plane (remove Z component) and normalize
    forward_init = forward_init_3d[:2]  # Take XY components
    forward_init = forward_init / (np.linalg.norm(forward_init) + 1e-8)
    
    # Add Z=0 component for qbetween_np (needs 3D vectors)
    forward_init = np.array([forward_init[0], forward_init[1], 0.0])
    
    # Target direction: +X (always align to positive X axis)
    target = np.array([1, 0, 0])
    
    # Compute rotation quaternion to align forward to target
    # This rotates around Z-axis to align horizontal direction
    root_quat_init = qbetween_np(forward_init[np.newaxis, :], target[np.newaxis, :])  # (1, 4)
    
    # Apply rotation to all body positions (vectorized for speed)
    root_quat_init_broadcast = np.repeat(root_quat_init, T, axis=0)  # (T, 4)
    # Vectorize: reshape to (T*30, 3) and (T*30, 4) for batch processing
    body_pos_w_flat = body_pos_w.reshape(-1, 3)  # (T*30, 3)
    root_quat_repeated = np.repeat(root_quat_init_broadcast, 30, axis=0)  # (T*30, 4)
    body_pos_w_flat = qrot_np(root_quat_repeated, body_pos_w_flat)  # (T*30, 3)
    body_pos_w = body_pos_w_flat.reshape(T, 30, 3)  # (T, 30, 3)
    
    # Apply rotation to root quaternions
    root_quat_original = body_quat_w[:, root_idx, :]  # (T, 4)
    root_quat_aligned = qmul_np(root_quat_init_broadcast, root_quat_original)  # (T, 4)
    
    # Compute per-frame displacement in aligned global frame
    root_pos_aligned = body_pos_w[:, root_idx, :]  # (T, 3)
    root_vel_xy_global = root_pos_aligned[1:, :2] - root_pos_aligned[:-1, :2]  # (T-1, 2)

    # Build per-frame velocity sequence (length T)
    # Convention: root_vel_xy[t] is displacement from t-1 to t
    root_vel_xy = np.zeros((T, 2), dtype=np.float32)
    if T > 1:
        root_vel_xy[1:] = root_vel_xy_global
    root_z = body_pos_w[:, root_idx, 2:3]  # (T, 1) - Z height
    
    # Convert root quaternion to 6D representation
    root_rot_6d = quat_wxyz_to_6d(root_quat_aligned)  # (T, 6)
    
    # Concatenate to 38D features
    features_38d = np.concatenate([
        joint_pos,      # (T, 29)
        root_vel_xy,    # (T, 2)
        root_z,         # (T, 1)
        root_rot_6d     # (T, 6)
    ], axis=1)  # (T, 38)
    
    return features_38d


def recover_root_xy_from_velocity(root_vel_xy, root_rot_quat):
    """
    Recover global XY position from per-frame displacement by integration.
    
    Adapted from HumanML3D's recover_root_rot_pos for Z-up coordinate system.
    
    Args:
        root_vel_xy: (T, 2) - displacement in aligned global XY (per frame)
        root_rot_quat: (T, 4) - root rotation quaternions (wxyz)
    
    Returns:
        root_xy_pos: (T, 2) - global XY positions (starts at origin)
    """
    T = root_vel_xy.shape[0]
    
    # Integrate velocity to get position
    # Position at frame t = sum of velocities from 0 to t-1
    root_xy_pos = np.zeros((T, 2), dtype=np.float32)
    root_xy_pos[0] = np.array([0, 0])  # Start at origin
    
    # Cumulative sum starting from frame 1
    for t in range(1, T):
        root_xy_pos[t] = root_xy_pos[t-1] + root_vel_xy[t]
    
    return root_xy_pos

# def recover_root_xy_from_velocity(root_vel_xy, root_rot_quat):
#     """
#     Correctly integrates local velocity into global position by applying rotation.
#     """
#     T = root_vel_xy.shape[0]
#     root_xy_pos = np.zeros((T, 2), dtype=np.float32)
    
#     # 1. 将四元数转换为旋转对象
#     # 注意：scipy 默认是 (x, y, z, w)，而你的注释说是 (w, x, y, z)
#     # 必须先调整顺序！
#     # 你的输入: wxyz -> scipy 需要: xyzw
#     r_obj = R.from_quat(root_rot_quat[:, [1, 2, 3, 0]])
    
#     # 2. 获取当前的朝向 (Heading / Yaw)
#     # 我们只关心围绕 Z 轴的旋转。
#     # 简单方法：直接用完整的旋转矩阵去旋转速度向量
#     # 或者提取欧拉角中的 yaw (更稳健，HumanML3D常用做法)
    
#     # 这里的实现取决于 root_vel_xy 到底有多 "Local"。
#     # 假设它是完全随身体旋转的局部速度：
    
#     # 将 2D 速度补全为 3D (Z=0) 以便进行 3D 旋转
#     vel_3d_local = np.zeros((T, 3))
#     vel_3d_local[:, :2] = root_vel_xy # 假设 XY 是局部的前/左速度
    
#     # 应用旋转: Global_Vel = Rotation * Local_Vel
#     vel_3d_global = r_obj.apply(vel_3d_local)
    
#     # 取回全局 XY 速度
#     global_vel_xy = vel_3d_global[:, :2]
    
#     # 3. 积分 (Cumulative Sum)
#     # 现在的 velocity 已经是世界坐标下的了，可以直接加
#     root_xy_pos[0] = np.array([0, 0])
#     for t in range(1, T):
#         root_xy_pos[t] = root_xy_pos[t-1] + global_vel_xy[t]
        
#     return root_xy_pos

if __name__ == "__main__":
    # Test the processing pipeline
    print("Testing robot data processing...")
    
    # Load a sample NPZ file
    import sys
    sys.path.insert(0, '..')
    
    npz_path = '../robot_humanml_data/npz/000000.npz'
    npz_data = np.load(npz_path)
    
    print(f"Loaded NPZ: {npz_path}")
    print(f"  joint_pos: {npz_data['joint_pos'].shape}")
    print(f"  body_pos_w: {npz_data['body_pos_w'].shape}")
    print(f"  body_quat_w: {npz_data['body_quat_w'].shape}")
    print(f"  body_lin_vel_w: {npz_data['body_lin_vel_w'].shape}")
    
    # Process the data
    features_38d = process_robot_npz(npz_data)
    
    print(f"\nProcessed features shape: {features_38d.shape}")
    print(f"Expected shape: (T, 38)")
    
    # Test velocity integration
    root_vel_xy = features_38d[:, 29:31]
    root_rot_6d = features_38d[:, 32:38]
    
    from utils.rotation_utils import rot6d_to_quat_wxyz
    root_rot_quat = rot6d_to_quat_wxyz(root_rot_6d)
    
    root_xy_recovered = recover_root_xy_from_velocity(root_vel_xy, root_rot_quat)
    
    print(f"\nRecovered root XY positions shape: {root_xy_recovered.shape}")
    print(f"Start position: {root_xy_recovered[0]}")
    print(f"End position: {root_xy_recovered[-1]}")
    
    print("\n✓ Test completed!")

