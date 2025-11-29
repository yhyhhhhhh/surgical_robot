#!/usr/bin/env python
import cv2
import argparse
import rospy
import torch
import tf2_ros
import math
import numpy as np
from geometry_msgs.msg import TransformStamped
from omni.isaac.lab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--sensitivity", type=float, default=0.1, help="Sensitivity factor (0-1).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import *

# 全局变量：控制指令，初始维度为 (1,8)
actions = torch.zeros(8).unsqueeze(0)

def get_link_transform(tfBuffer, base_frame, link_frame):
    """
    利用 tfBuffer 查询 base_frame 到 link_frame 的变换
    """
    try:
        transform = tfBuffer.lookup_transform(base_frame, link_frame, rospy.Time(0), rospy.Duration(1.0))
        return transform
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn("tf 查询失败：%s" % e)
        return None

# ========== 1. 定义线性映射函数 ==========
def map_range(value, src_min, src_max, dst_min, dst_max):
    """
    将 value 从 [src_min, src_max] 映射到 [dst_min, dst_max] 的线性函数。
    如果需要超出范围后 clamp，可以在此额外处理，这里只做最简单的映射。
    """
    if src_max == src_min:
        return dst_min
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)

def quaternion_multiply(q1, q2):
    """
    四元数乘法，q = q1 * q2
    四元数格式均为 [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return [w, x, y, z]

# ========== 2. 定义 slerp 与增量式更新函数 ==========
def slerp(q0, q1, t):
    """
    对两个四元数 q0 和 q1 进行球面线性插值（slerp）
    输入：
      q0, q1: numpy 数组，形状 (4,)，格式为 [w, x, y, z]
      t: 插值因子，取值范围 [0,1]
    输出：
      插值后的四元数，numpy 数组格式 [w, x, y, z]
    """
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        result = result / np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)       # 初始角度
    theta = theta_0 * t            # 插值角度
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0) + (s1 * q1)

def incremental_pose_update(last_position, last_orientation, current_position, current_orientation, sensitivity):
    """
    计算增量式的位姿更新：
      - 平移部分：通过两次采样差值计算增量，并乘以灵敏度因子
      - 旋转部分：利用 slerp 插值平滑更新姿态

    输入：
      last_position: 上一次目标位置 (torch.Tensor, shape (3,))
      last_orientation: 上一次目标旋转 (torch.Tensor, shape (4,)) 格式 [w, x, y, z]
      current_position: 当前采样的位姿（经过必要映射后的绝对位置，torch.Tensor, shape (3,))
      current_orientation: 当前采样的旋转（四元数，torch.Tensor, shape (4,))
      sensitivity: 灵敏度因子（0-1），表示每个控制周期更新的比例
    输出：
      new_position: 更新后的目标位置 (torch.Tensor, shape (3,))
      new_orientation: 更新后的目标旋转 (torch.Tensor, shape (4,))
    """
    # 平移部分：计算位移增量
    delta_pos = (current_position - last_position) * sensitivity
    new_position = last_position + delta_pos

    # 旋转部分：使用 slerp 插值更新姿态
    q0 = last_orientation.cpu().numpy()
    q1 = current_orientation.cpu().numpy()
    t = max(0.0, min(1.0, sensitivity))
    new_q = slerp(q0, q1, t)
    new_orientation = torch.tensor(new_q, dtype=torch.float32, device=last_orientation.device)
    
    return new_position, new_orientation

def main():
    rospy.init_node('listener', anonymous=True)

    # 创建 tf2_ros Buffer 和 Listener（自动订阅 /tf 话题）
    tfBuffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tfBuffer)

    # 初始化 Isaac 仿真环境（UR3任务相关）
    env_cfg = parse_env_cfg(
        "My-Isaac-Ur3-Ik-Abs-Position-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make("My-Isaac-Ur3-Ik-Abs-Position-Direct-v0", cfg=env_cfg)
    camera = env.scene["Camera"]
    camera_index = 0
    env.reset()

    # 获取初始状态，假设从仿真环境中直接获取末端执行器位姿，格式为 (7,) 前三位平移，后四位四元数
    init_state = env.scene.articulations["left_robot"].data.body_state_w[0, -1, 0:7]
    last_position = init_state[0:3].clone().to(args_cli.device)
    last_orientation = init_state[3:7].clone().to(args_cli.device)

    rate = rospy.Rate(30)  # 30 Hz
    while not rospy.is_shutdown():
        transform = get_link_transform(tfBuffer, "base", "stylus")
        if transform is not None:
            # 提取当前 tf 位姿
            position = torch.tensor([
                -transform.transform.translation.x,
                -transform.transform.translation.y,
                transform.transform.translation.z
            ], dtype=torch.float32, device=args_cli.device)
            
            orientation = torch.tensor([
                transform.transform.rotation.w,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
            ], dtype=torch.float32, device=args_cli.device)
            
            # 对平移做映射（根据具体场景调整映射参数）
            mapped_position = torch.zeros(3, device=args_cli.device)
            mapped_position[0] = map_range(position[0].item(), -0.15, 0.15, 0.1, -0.1)
            mapped_position[1] = map_range(position[1].item(), 0.0, 0.20, -0.05, -0.3)
            mapped_position[2] = map_range(position[2].item(), 0.0, 0.26, -0.3, 0.3)
            
            # 使用增量式更新函数计算新的目标位姿
            new_position, new_orientation = incremental_pose_update(
                last_position, last_orientation, position, orientation, args_cli.sensitivity
            )
            
            # 更新 last_position 和 last_orientation 为下次更新做准备
            last_position = new_position.clone()
            last_orientation = new_orientation.clone()
            
            # 构造新的目标位姿张量 (1,7)
            base_tensor = torch.cat([new_position.unsqueeze(0), new_orientation.unsqueeze(0)], dim=1)
            actions = torch.cat([base_tensor, torch.zeros(1, 1, device=base_tensor.device)], dim=1)
        else:
            rospy.logwarn("未能获得 tf 变换，保持上一次的 actions")
        
        # 将 actions 传递给仿真环境
        env.step(actions)
        
        # 图像处理与显示
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )
        image = single_cam_data['rgb']
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Feed', image_bgr)
        cv2.waitKey(1)
        
        rate.sleep()
    
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    simulation_app.close()
