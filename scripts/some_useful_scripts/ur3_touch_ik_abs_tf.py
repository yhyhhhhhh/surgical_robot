

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
# 用于相对控制时保存参考
pos_ref = None           # 手柄参考位置
robot_ref_pos = None     # 机械臂末端执行器参考位置
robot_ref_quat = None    # 机械臂末端执行器参考姿态（四元数）
locked_once = False      # 标记是否已经完成“相对模式”的初次锁定

# 在相对控制时，你希望对 x, y, z 分别放缩的系数：
SCALE_X = -0.1
SCALE_Y = -0.1
SCALE_Z = 0.5

def setup_viewport():
    """配置viewport函数"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)

def button_callback(msg):
    global actions,sign      # 添加全局变量      
    actions[:,7] = torch.tensor(msg.close_gripper, device='cuda').unsqueeze(0)
    sign = msg.locked


# ========== 功能模块函数 ==========
def compute_action(tfBuffer, device, base_frame="base", link_frame="stylus"):
    """
    绝对方式计算 action。形状 (1, 8):
      - actions[:,0:3]: 位置
      - actions[:,3:7]: 姿态四元数
      - actions[:,7]:   抓手
    并返回 (new_action, pos_tensor, orient_tensor)，
    以便后续相对控制可用 pos_tensor, orient_tensor.
    """
    global actions
    pos_tensor, orient_tensor = get_transform_tensors(tfBuffer, base_frame, link_frame, device)

    # 这里示例：你原先的一些旋转修正
    q_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    q_y   = torch.tensor([0.707, 0.0, 0.707, 0.0], dtype=torch.float32, device=device)
    q_new = quat_mul(q_rot, orient_tensor)
    q_new = quat_mul(q_new, q_rot)
    q_new = quat_mul(q_new, q_y)

    # 位置映射示例
    mapped_x = map_range(pos_tensor[0].item(), -0.15, 0.15,  0.3,  -0.3)
    mapped_y = map_range(pos_tensor[1].item(),  0.0,  0.20, -0.05, -0.3)
    mapped_z = map_range(pos_tensor[2].item(),  0.0,  0.26, -0.25,   0.0)

    base_tensor = torch.tensor([
        mapped_x, mapped_y, mapped_z,
        q_new[0], q_new[1], q_new[2], q_new[3]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # 更新前7位 (index 0..6)
    actions[:, 0:7] = base_tensor
    return actions, pos_tensor, orient_tensor


def compute_relative_action(pos_tensor, device):
    """
    相对控制逻辑：
      - 仅对位置做相对控制
      - 姿态一律使用锁定的 robot_ref_quat
    """
    global actions, pos_ref, robot_ref_pos, robot_ref_quat

    # 计算手柄位置相对增量
    pos_delta = pos_tensor - pos_ref
    
    pos_scale = torch.tensor([SCALE_X, SCALE_Y, SCALE_Z], device=device, dtype=torch.float32)
    pos_delta = pos_delta * pos_scale
    
    # 将 pos_delta 叠加到 robot_ref_pos
    new_pos = robot_ref_pos + pos_delta

    # 姿态直接使用锁定时保存的 robot_ref_quat，不做相对运算
    new_quat = robot_ref_quat

    # 填入动作张量
    actions[:, 0] = new_pos[0]
    actions[:, 1] = new_pos[1]
    actions[:, 2] = new_pos[2]
    actions[:, 3] = new_quat[0]
    actions[:, 4] = new_quat[1]
    actions[:, 5] = new_quat[2]
    actions[:, 6] = new_quat[3]

    return actions
# ========== 主函数 ==========

def main():
    global pos_ref, robot_ref_pos, robot_ref_quat, locked_once

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
    robot = env.scene["left_robot"].data
    env.reset()

    rate = rospy.Rate(30)  # 30 Hz 循环
    
    while not rospy.is_shutdown():
        # 计算 action
        # 每帧先用绝对方式计算最新动作，以及当前的手柄 pos/orient
        new_action, pos_tensor, orient_tensor = compute_action(
            tfBuffer, args_cli.device,
            base_frame="base",
            link_frame="stylus"
        )
        if sign:
            # 若 sign == True，则采用相对控制
            if not locked_once:
                # 第一次进入相对模式时，保存参考
                pos_ref = pos_tensor.clone().detach()

                # 从 env.scene["left_robot"].data.body_state_w 取当前末端执行器姿态
                # body_state_w: [num_bodies, 13], 依次是
                #   (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z)

                ee_id = robot.body_names.index("scissors_tip")
                current_ee_pose = robot.body_state_w[:,ee_id, 0:7].clone().detach()
                # 取出位置(0..2)和四元数(3..6)
                robot_ref_pos  = current_ee_pose[0,0:3].to(args_cli.device)
                robot_ref_quat = current_ee_pose[0,3:7].to(args_cli.device)

                locked_once = True
                rospy.loginfo("已进入相对控制：位置做相对，姿态采用锁定姿态。")

            # 仅对位置做相对，姿态锁定
            new_action = compute_relative_action(pos_tensor, args_cli.device)
            
            env.step(new_action)

        else:
            # sign == False，退出相对控制，回到绝对控制逻辑
            locked_once = False
            env.step(new_action)

        # 在 while 循环内修改打印语句：
        phantom_str = np.array2string(
            new_action[:, 0:3].cpu().numpy().squeeze(),
            precision=4,  # 保留四位小数
            formatter={'float_kind': lambda x: f"{x:.4f}"}
        )
        robot_str = np.array2string(
            robot.body_pos_w[:,10, 0:3].cpu().numpy().squeeze(),
            precision=4,
            formatter={'float_kind': lambda x: f"{x:.4f}"}
        )

        print(f"Phantom控制位置: {phantom_str}")
        print(f"机器人实际位置: {robot_str}")
        # 图像获取与显示
        display_camera_image(camera, camera_index, window_name='Camera Feed')

        rate.sleep()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    simulation_app.close()
