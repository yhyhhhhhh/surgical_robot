import cv2
import torch
from omni.isaac.lab.utils.math import *
from omni.isaac.lab.utils.dict import *
import rospy

def euclidean_distance(src, tar, reduction='mean'):
    # B, (N), T, D
    diff = src - tar
    dist = torch.norm(diff, dim=-1)
    if reduction == 'mean':
        return dist.mean(dim=-1)
    elif reduction == 'none':
        return dist

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


def enhance_depth_image(depth_image):
    """
    针对 0 - 1m 的深度范围进行局部对比度增强，同时使用伪彩色映射优化视觉效果。
    :param depth_image: 输入深度图像 (numpy 数组, 可能包含 inf)
    :return: 增强后的彩色深度图
    """
    # 处理 inf 值：用最大有限深度替换 inf
    max_valid_depth = np.nanmax(depth_image[depth_image != np.inf])  
    depth_image = np.where(np.isinf(depth_image), max_valid_depth, depth_image)

    # **局部归一化：针对 0 - 1m 进行增强**
    depth_clipped = np.clip(depth_image, 0, 1)  # 只增强 0 - 1m 之间的区域
    normalized = (depth_clipped - np.min(depth_clipped)) / (np.max(depth_clipped) - np.min(depth_clipped) + 1e-8)

    # **Log 变换**：提升 0-1m 范围内的对比度
    log_transformed = np.log1p(normalized * 9) / np.log(10)  # log10 归一化

    # **转换为 8-bit 并应用 CLAHE（自适应直方图均衡化）**
    depth_uint8 = (log_transformed * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(depth_uint8)

    # **伪彩色映射**
    color_mapped = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)

    return color_mapped

def enhance_depth_images_batch(depth_images):
    """
    批量处理深度图像：输入形状 (n, 480, 640, 1)，输出形状 (n, 480, 640, 3)
    :param depth_images: 输入深度图像批量 (numpy 数组, 形状 n*480 * 640 * 1)
    :return: 增强后的彩色深度图批量 (numpy 数组, 形状 n*480 * 640 * 3)
    """
    enhanced_images = []
    # 遍历每个深度图像
    for i in range(depth_images.shape[0]):
        # 提取单张图像并去除单通道维度 (480, 640)
        depth_image = depth_images[i, ..., 0]
        
        # 处理 inf 值：用当前图像的最大有限深度替换 inf
        max_valid_depth = np.nanmax(depth_image[depth_image != np.inf])
        depth_image = np.where(np.isinf(depth_image), max_valid_depth, depth_image)
        
        # 局部归一化：针对 0 - 1m 进行增强
        depth_clipped = np.clip(depth_image, 0, 1)
        normalized = (depth_clipped - np.min(depth_clipped)) / (np.max(depth_clipped) - np.min(depth_clipped) + 1e-8)
        
        # Log 变换
        log_transformed = np.log1p(normalized * 9) / np.log(10)
        
        # 转换为 8-bit 并应用 CLAHE
        depth_uint8 = (log_transformed * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(depth_uint8)
        
        # 伪彩色映射
        color_mapped = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)
        enhanced_images.append(color_mapped)
    
    # 堆叠结果并返回 (n, 480, 640, 3)
    return np.stack(enhanced_images, axis=0)


def map_range(value, src_min, src_max, dst_min, dst_max):
    """
    将 value 从 [src_min, src_max] 映射到 [dst_min, dst_max] 的线性函数。
    """
    if src_max == src_min:
        return dst_min
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)

def map_range_1(value, src_min, src_max, dst_min, dst_max):
    """
    参数说明：
    value   : 输入的数值
    src_min : 源区间最小值，这里应为 -0.025
    src_max : 源区间最大值，这里应为 0.12
    dst_min : 目标区间最小值，这里应为 -0.25
    dst_max : 目标区间最大值，这里应为 0.0

    要求同时满足：x = 0.017 映射到 y = -0.062
    """
    # 定义断点
    breakpoint = 0.017
    # 断点对应的目标值
    break_dst = -0.12

    # 当 value 在第一段时：
    if value <= breakpoint:
        # 第一段：src: [src_min, breakpoint] -> dst: [dst_min, break_dst]
        src_range = breakpoint - src_min
        dst_range = break_dst - dst_min
        return dst_min + (value - src_min) / src_range * dst_range
    else:
        # 第二段：src: [breakpoint, src_max] -> dst: [break_dst, dst_max]
        src_range = src_max - breakpoint
        dst_range = dst_max - break_dst
        return break_dst + (value - breakpoint) / src_range * dst_range

def display_camera_image(camera, camera_index=0, window_name='Camera Feed'):
    """
    从 camera 数据中提取图像，并显示。
    该函数独立于具体环境，可用于其他应用场景。
    """
    single_cam_data = convert_dict_to_backend(
        {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
    )
    image = single_cam_data['rgb']
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image_bgr)
    cv2.waitKey(1)

def get_transform_tensors(tfBuffer, base_frame, link_frame, device):
    """
    从 tfBuffer 中查询 base_frame 到 link_frame 的变换，
    并将查询到的位置与旋转信息转换为 torch tensor。

    参数:
        tfBuffer: tf2_ros.Buffer 对象
        base_frame: 起始坐标系名称
        link_frame: 目标坐标系名称
        device: torch device（如 "cpu" 或 "cuda"）

    返回:
        若查询成功，返回一个二元组 (pos_tensor, orient_tensor)，其中：
            pos_tensor: 位置 tensor，shape (3,)
            orient_tensor: 旋转 tensor，shape (4,)
        若查询失败，返回 None
    """
    transform = tfBuffer.lookup_transform(base_frame, link_frame, rospy.Time(0), rospy.Duration(1.0))
    if transform is None:
        rospy.logwarn("未能获得 tf 变换，返回 None")
        return None

    # 提取位置和旋转信息，并转换为 torch tensor
    pos_list = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    ]
    orient_list = [
        transform.transform.rotation.w,
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
    ]
    pos_tensor = torch.tensor(pos_list, dtype=torch.float32, device=device)
    orient_tensor = torch.tensor(orient_list, dtype=torch.float32, device=device)
    return pos_tensor, orient_tensor



def bidirectional_sampler(strong, weak, obs_dict, prior, num_sample=10, beta=0.99, num_mode=3):
    """
    Sample an action that preserves coherence with a prior and contrast outputs from strong and weak policies.
    Args:
        strong: a strong policy to predict near-optimal sequences of actions
        weak: a weak policy to predict sub-optimal sequences of actions
        prior: the prediction made in the previous time step
        obs_dict: dictionary containing observations at the current time step
        num_sample (int, optional): number of samples to generate
        beta (float, optional): weight decay factor for backward coherence
        num_mode (int, optional): Factor to determine the number of top samples to consider

    Returns:
        dict: A dictionary of actions sampled using the contrastive approach.
    """    
    # pre-process
    B, OH, OD = obs_dict['obs'].shape
    obs_dict_batch = dict()
    for key in obs_dict.keys():
        if key == 'prior':
            continue        
        obs_dict_batch[key] = obs_dict[key].unsqueeze(1).repeat(1, num_sample, 1, 1).view(B * num_sample, OH, OD)

    # predict
    action_strong_batch = strong.predict_action(obs_dict_batch)

    # post-process
    AH, PH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action_pred'].shape[1], action_strong_batch['action_pred'].shape[2]

    action_strong_batch['action'] = action_strong_batch['action'].reshape(B, num_sample, AH, AD)
    action_strong_batch['action_pred'] = action_strong_batch['action_pred'].reshape(B, num_sample, PH, AD)
    if 'action_obs_pred' in action_strong_batch:
        action_strong_batch['action_obs_pred'] = action_strong_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
    if 'obs_pred' in action_strong_batch:
        action_strong_batch['obs_pred'] = action_strong_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    if weak:
        action_weak_batch = weak.predict_action(obs_dict_batch)
        action_weak_batch['action'] = action_weak_batch['action'].reshape(B, num_sample, AH, AD)
        action_weak_batch['action_pred'] = action_weak_batch['action_pred'].reshape(B, num_sample, PH, AD)
        if 'action_obs_pred' in action_weak_batch:
            action_weak_batch['action_obs_pred'] = action_weak_batch['action_obs_pred'].reshape(B, num_sample, AH, OD)
        if 'obs_pred' in action_weak_batch:
            action_weak_batch['obs_pred'] = action_weak_batch['obs_pred'].reshape(B, num_sample, PH, OD)

    # backward
    if prior is not None:
        # distance measure
        start_overlap = strong.n_obs_steps - 1
        end_overlap = prior.shape[1]
        num_sample = num_sample // num_mode
        dist_raw = euclidean_distance(action_strong_batch['action_pred'][:, :, start_overlap:end_overlap], prior.unsqueeze(1)[:, :, start_overlap:], reduction='none')

        weights = torch.tensor([beta**i for i in range(end_overlap-start_overlap)]).to(dist_raw.device)
        weights = weights / weights.sum()
        dist_weighted = dist_raw * weights.view(1, 1, end_overlap-start_overlap)
        dist_strong_sum = dist_weighted.sum(dim=2)
        _, cross_index = dist_strong_sum.sort(descending=False)
        index = cross_index[:, 0:num_sample]

        # slicing
        action_dict = dict()
        range_tensor = torch.arange(B, device=index.device)
        for key in action_strong_batch.keys():
            action_dict[key] = action_strong_batch[key][range_tensor.unsqueeze(1), index]
        action_strong_batch = action_dict
        dist_avg_prior = dist_strong_sum[range_tensor.unsqueeze(1), index]

        if weak:
            # sample selection
            dist_weak = euclidean_distance(action_weak_batch['action_pred'][:, :, start_overlap:end_overlap], prior.unsqueeze(1)[:, :, start_overlap:], reduction='none')
            dist_weighted = dist_weak * weights.view(1, 1, end_overlap-start_overlap)
            dist_weak_sum = dist_weighted.sum(dim=2)
            _, cross_index = dist_weak_sum.sort(descending=False)
            index = cross_index[:, 0:num_sample]

            # slicing
            action_dict = dict()
            range_tensor = torch.arange(B, device=index.device)
            for key in action_weak_batch.keys():
                action_dict[key] = action_weak_batch[key][range_tensor.unsqueeze(1), index]
            action_weak_batch = action_dict

        # balance between backward and forward
        ratio = (PH * beta) ** 2 / ((PH * beta) ** 2 + AH ** 2)
    else:
        dist_avg_prior = 0.0
        ratio = 0.0

    # positive samples
    # 对应于计算动作集与最优动作集之间的距离，这里动作集与最优动作集相同，都为strong policy中采样的动作，所以是自己与自己计算距离
    src_expand = action_strong_batch['action_pred'].unsqueeze(1)
    tar_expand =  action_strong_batch['action_pred'].unsqueeze(2)
    dist_pos = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

    # topk = num_sample
    # 这里对于每个动作，取前topk个最接近的动作之间的距离，然后取平均，作为距离的度量
    topk = num_sample // 2 + 1
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)      # skip the self-distance first element 

    if weak:
        # negative samples
        # 对应于计算动作集与弱动作集之间的距离，这里动作集与弱动作集不同，弱动作集为weak policy中采样的动作，所以是和弱动作集计算距离
        src_expand = action_strong_batch['action_pred'].unsqueeze(1)
        tar_expand = action_weak_batch['action_pred'].unsqueeze(2)
        dist_neg = euclidean_distance(src_expand, tar_expand).view(B, num_sample, num_sample)

        topk = num_sample // 2
        values, _ = torch.topk(dist_neg, k=topk, largest=False, dim=-1)
        dist_avg_neg = values[:, :, 0:].mean(dim=-1)
    else:
        dist_avg_neg = 0

    # sample selection
    dist_avg = dist_avg_prior * ratio + (dist_avg_pos - dist_avg_neg) * (1 - ratio)
    _, index = dist_avg.min(dim=-1)

    # slicing
    action_dict = dict()
    range_tensor = torch.arange(B, device=index.device)
    for key in action_strong_batch.keys():
        action_dict[key] = action_strong_batch[key][range_tensor, index]

    return action_dict
