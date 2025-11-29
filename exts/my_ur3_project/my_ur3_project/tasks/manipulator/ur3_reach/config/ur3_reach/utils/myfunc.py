import torch
from omni.isaac.lab.utils.math import *

def world_to_env_coordinates(
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    env_root_pos: torch.Tensor,
    env_root_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将世界坐标系下的位置和姿态转换到各个环境坐标系。

    Args:
        world_pos: 世界坐标系中的位置，形状为 (N, 3)。
        world_quat: 世界坐标系中的四元数，形状为 (N, 4)。
        env_root_pos: 环境根坐标系在世界坐标系中的位置，形状为 (N, 3)。
        env_root_quat: 环境根坐标系在世界坐标系中的四元数，形状为 (N, 4)。

    Returns:
        env_pos: 转换到环境坐标系的位置，形状为 (N, 3)。
        env_quat: 转换到环境坐标系的四元数，形状为 (N, 4)。
    """
    # 计算环境根坐标系的逆四元数
    env_root_quat_inv = quat_inv(env_root_quat)

    # 转换位置
    relative_pos = world_pos - env_root_pos  # 世界位置减去根位置的平移向量
    env_pos = quat_apply(env_root_quat_inv, relative_pos)  # 旋转到环境坐标系

    # 转换四元数
    env_quat = quat_mul(env_root_quat_inv, world_quat)  # 使用逆四元数将世界方向转换到环境坐标系

    return env_pos, env_quat


def env_to_pipe_coordinates(
    env_points: torch.Tensor,
    pipe_pos: torch.Tensor,
    pipe_quat: torch.Tensor
) -> torch.Tensor:
    """
    将环境局部坐标系中的点转换到管道局部坐标系。

    Args:
        env_points: 环境局部坐标系中的点，形状为 (N, 3)。
        pipe_pos: 管道相对于环境局部坐标系的平移向量，形状为 (3,)。
        pipe_quat: 管道相对于环境局部坐标系的旋转四元数，形状为 (4，)。

    Returns:
        pipe_points: 转换到管道局部坐标系的点，形状为 (N, 3)。
    """
    # 1. 计算相对位置（平移）
    relative_pos = env_points - pipe_pos  # 环境点减去管道的平移向量

    # 2. 应用旋转（将相对位置旋转到管道坐标系）
    pipe_points = quat_apply(quat_inv(pipe_quat), relative_pos)

    return pipe_points

def quat_rotate_vector(quat, vec):
    """
    使用四元数旋转一个向量。
    quat: 四元数 (w, x, y, z), shape=(4,) 或 (N, 4)
    vec: 被旋转向量，shape=(3,) 或 (N, 3)
    """
    # 提取四元数分量
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # 计算旋转矩阵
    x2, y2, z2 = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    rot_matrix = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)
    ], dim=-1).reshape(-1, 3, 3)

    # 矩阵乘法
    rotated_vec = torch.matmul(rot_matrix, vec.unsqueeze(-1)).squeeze(-1)
    return rotated_vec
