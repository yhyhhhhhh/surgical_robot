#!/usr/bin/env python
import cv2
import argparse
import rospy
import torch
import tf2_ros
from geometry_msgs.msg import TransformStamped
from omni.isaac.lab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--task", type=str, default="My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", help="Task name.")
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
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import *
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
### ========== 导入torch msg ===========
import sys
import os

os.environ['ROS_PACKAGE_PATH'] = '/home/yhy/code/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:' + os.environ.get('ROS_PACKAGE_PATH', '')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/yhy/code/touch_ws/devel/lib/python3/dist-packages')
# ======================================
from omni_msgs.msg import OmniState

actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
sign = False
def setup_viewport():
    """配置viewport函数"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)

# ========== 功能模块函数 ==========
def compute_action(tfBuffer, device, base_frame="base", link_frame="tip"):
    """
    计算并返回传递给仿真环境的 action 张量。
    该函数通过 tf 获取变换，再通过线性映射和 torch 版四元数乘法生成目标张量，
    其中位置与旋转信息均先转换为 tensor 以便复用。

    参数:
        tfBuffer: tf2_ros.Buffer 对象，用于查询变换
        device: torch device（如 "cpu" 或 "cuda"）
        base_frame: 起始坐标系名称（默认 "base"）
        link_frame: 目标坐标系名称（默认 "stylus"）

    返回:
        action 张量，形状为 (1, 8)，前 3 个元素为位置，后 4 个为姿态，最后一个元素为附加零；查询失败时返回 None。
    """
    global actions
    
    pos_tensor, orient_tensor =  get_transform_tensors(tfBuffer, base_frame, link_frame, device)

    # 定义旋转四元数（均为 torch tensor）
    q_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    # 使用 quat_mul 进行四元数连乘运算
    q_new = quat_mul(q_rot, orient_tensor)

    # 坐标映射（根据具体任务需求调整映射范围）
    mapped_x = map_range(pos_tensor[0].item(), -0.15, 0.15, 0.3, -0.3)
    mapped_y = map_range(pos_tensor[1].item(), 0.0, 0.20, -0.05, -0.3)
    mapped_z = map_range(pos_tensor[2].item(), 0.0, 0.26, -0.3, 0.4)

    # 构造 action 张量：前三个为位置，后四个为姿态，再附加一个零（兼容后续控制），得到 shape (1,8)
    base_tensor = torch.tensor([mapped_x, mapped_y, mapped_z] + q_new.tolist(),
                                 dtype=torch.float32,
                                 device=device).unsqueeze(0)
    actions[:,0:7] = base_tensor


def button_callback(msg):
    global actions,sign      # 添加全局变量      
    actions[:,7] = torch.tensor(msg.close_gripper, device='cuda').unsqueeze(0)
    sign = msg.locked
# ========== 主函数 ==========

def main():
    global actions

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/state", OmniState, button_callback)       
    
    # 初始化 tf2_ros Buffer 与 Listener（订阅 /tf）
    tfBuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfBuffer)

    # 初始化 Isaac 仿真环境
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    ### 注意要在导入环境后设置viewport
    setup_viewport()
    
    camera = env.scene["Camera"]  # type: ignore ##
    camera_index = 0
    env.reset()

    rate = rospy.Rate(30)  # 30 Hz 循环
    while not rospy.is_shutdown():
        # 计算 action
        compute_action(tfBuffer, args_cli.device, base_frame="base", link_frame="tip")
        
        env.step(actions)

        # 图像获取与显示
        display_camera_image(camera, camera_index, window_name='Camera Feed')

        rate.sleep()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    simulation_app.close()
