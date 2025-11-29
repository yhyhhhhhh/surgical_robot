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
parser.add_argument(
    "--save",
    action="store_true",
    default=True,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
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
import h5py
### ========== 导入torch msg ===========
import sys
import os

os.environ['ROS_PACKAGE_PATH'] = '/home/yhy/code/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:' + os.environ.get('ROS_PACKAGE_PATH', '')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/yhy/code/touch_ws/devel/lib/python3/dist-packages')
# ======================================
from omni_msgs.msg import OmniState
import numpy as np  # 用于数组处理
import math
# 使用 pynput 监听键盘按键
from pynput import keyboard

# ----------------全局变量----------------
reset_requested = False
# yaw 使用 torch.tensor 存储（单位：弧度），初始为0
yaw = torch.zeros(1, device=args_cli.device)
# 其他全局变量
sign = True
fine_mode = False
pos_ref = None           # 手柄参考位置
robot_ref_pos = None     # 机械臂末端执行器参考位置
robot_ref_quat = None    # 机械臂末端执行器参考姿态（四元数）
locked_once = False      # 相对控制模式是否已完成初次锁定

actions = torch.zeros(8, device=args_cli.device).unsqueeze(0)  # shape (1,8)
new_action = torch.zeros(8, device=args_cli.device).unsqueeze(0)  # shape (1,8)
q_rot = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=args_cli.device)
base_tensor = torch.tensor([
    q_rot[0], q_rot[1], q_rot[2], q_rot[3]
], dtype=torch.float32, device=args_cli.device).unsqueeze(0)
new_action[0,3:7] = base_tensor

# ----------------------------------------

# 键盘监听回调函数
def on_press(key):
    global reset_requested, yaw, sign, fine_mode, locked_once
    try:
        if key.char == 'r':
            reset_requested = True
        elif key.char == 'a':
            # 向左旋转，增加 yaw（例如每次增加 0.1 弧度）
            delta = 0.0 if fine_mode else 0.004
            yaw = yaw + delta
            print(f"向左旋转，yaw = {yaw.item():.4f}")
        elif key.char == 'd':
            # 向右旋转，减少 yaw（例如每次减少 0.1 弧度）
            delta = 0.0 if fine_mode else 0.004
            yaw = yaw - delta
            print(f"向右旋转，yaw = {yaw.item():.4f}")
        elif key.char == 'q':
            # 切换精细控制模式
            fine_mode = not fine_mode
            locked_once = False  # 强制在下一次循环中重新计算参考值
            mode = "精细" if fine_mode else "粗控制"
            print(f"切换为 {mode} 模式的相对控制")
    except AttributeError:
        pass

def on_release(key):
    global reset_requested
    try:
        if key.char == 'r':
            reset_requested = False
    except AttributeError:
        pass


# 启动 pynput 键盘监听器（后台线程）
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def setup_viewport():
    """配置 viewport 函数"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)

def button_callback(msg):
    global actions
    actions[:, 7] = torch.tensor(msg.close_gripper, device='cuda').unsqueeze(0)


# ========== 功能模块函数 ==========
def compute_action(tfBuffer, device, base_frame="base", link_frame="stylus"):
    """
    绝对方式计算 action。形状 (1,8)：
      - actions[:,0:3]: 位置
      - actions[:,3:7]: 姿态四元数
      - actions[:,7]:   抓手
    并返回 (new_action, pos_tensor, orient_tensor) 以供后续相对控制使用。
    """
    global actions, yaw
    pos_tensor, orient_tensor = get_transform_tensors(tfBuffer, base_frame, link_frame, device)

    # 示例：原先的一些旋转修正
    q_rot = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=device)
    # 根据当前 yaw 计算绕 Z 轴旋转的四元数
    q_yaw = torch.tensor([
        torch.cos(yaw / 2),
        0.0,
        torch.sin(yaw / 2),
        0.0
    ], dtype=torch.float32, device=device)
    
    q_rot = quat_mul(q_rot, q_yaw)
    # 位置映射示例
    # mapped_x = map_range(pos_tensor[0].item(), -0.15, 0.15, 0.2, -0.2)
    # mapped_y = map_range(pos_tensor[1].item(), 0.07, 0.17, -0.15, -0.35)
    # mapped_z = map_range_1(pos_tensor[2].item(), -0.025, 0.12, -0.25, 0.0)
    mapped_x = map_range(pos_tensor[0].item(), -0.15, 0.15, 0.2, -0.2)
    mapped_y = map_range(pos_tensor[1].item(), 0.07, 0.17, -0.15, -0.35)
    mapped_z = map_range_1(pos_tensor[2].item(), -0.025, 0.12, -0.25, 0.0)
    base_tensor = torch.tensor([
        mapped_x, mapped_y, mapped_z,
        q_rot[0], q_rot[1], q_rot[2], q_rot[3]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    actions[:, 0:7] = base_tensor
    return actions, pos_tensor, orient_tensor


def compute_relative_action(pos_tensor, device):
    """
    相对控制逻辑：
      - 对位置做相对控制（根据 fine_mode 选择不同的缩放系数）
      - 对通过摄像头采集到的手柄位置增量进行坐标补偿，
        将其从摄像头坐标系转换到参考坐标系（补偿摄像头随末端旋转的影响）
      - 姿态在锁定的 robot_ref_quat 基础上叠加 global yaw 产生的旋转
    """
    global actions, pos_ref, robot_ref_pos, robot_ref_quat, fine_mode, yaw

    # 定义粗控制和精细控制的缩放系数
    coarse_scale = torch.tensor([-0.5, -0.5, 1.0], device=device, dtype=torch.float32)
    fine_scale   = torch.tensor([-0.1, -0.1, 1.0], device=device, dtype=torch.float32)
    scale = fine_scale if fine_mode else coarse_scale

    # 计算手柄位置增量（摄像头坐标系下的差值）
    pos_delta = pos_tensor - pos_ref
    pos_delta = pos_delta * scale

    # 计算新的目标位置：参考位置加上经过补偿后的位移
    new_pos = robot_ref_pos + pos_delta

    # 生成绕 Z 轴的旋转四元数，使用全局 yaw（yaw 为弧度）
    q_yaw = torch.tensor([
        torch.cos(yaw / 2),
        0.0,
        torch.sin(yaw / 2),
        0.0
    ], dtype=torch.float32, device=device)
    # 将锁定姿态与 yaw 旋转相乘得到新的姿态
    new_quat = quat_mul(robot_ref_quat, q_yaw)

    actions[:, 0] = new_pos[0]
    actions[:, 1] = new_pos[1]
    actions[:, 2] = new_pos[2]
    actions[:, 3] = new_quat[0]
    actions[:, 4] = new_quat[1]
    actions[:, 5] = new_quat[2]
    actions[:, 6] = new_quat[3]

    return actions

def compute_relative_action1(pos_tensor, device):
    """
    相对控制逻辑：
      - 平移部分：基于 touch 与初始 touch 位置之间的差（经过 scale），
        计算出 raw 的 pos_delta；再计算当前与上一次的差值，
        对该差值做旋转补偿（乘以旋转矩阵），并累加到 accumulated_delta 上，
        得到最终的平移命令（保证当 touch 不动时，只旋转不平移）。
      - 旋转部分：基于全局 yaw 生成旋转四元数，与锁定姿态相乘得到新的姿态。
      - 最后将计算得到的平移和平移后的旋转组合为动作，并映射到摄像头坐标系（如需要）。
    """
    global actions, pos_ref, robot_ref_pos, robot_ref_quat, fine_mode, yaw
    global prev_pos_delta # 新增全局变量

    # 如果是第一次调用，初始化 prev_pos_delta 与 accumulated_delta
    if 'prev_pos_delta' not in globals() or prev_pos_delta is None:
        prev_pos_delta = torch.zeros(3, device=device, dtype=torch.float32)
    
    # 定义缩放系数（这里用正值）
    coarse_scale = torch.tensor([-0.5, -0.5, 1.0], device=device, dtype=torch.float32)
    fine_scale   = torch.tensor([-0.1, -0.1, 1.0], device=device, dtype=torch.float32)
    scale = fine_scale if fine_mode else coarse_scale
    
    # 计算当前 raw 的平移差值（来自 touch 与初始位置的差）
    current_pos_delta = (pos_tensor - pos_ref) * scale
    
    # 计算本次与上一次的差值（增量）
    delta_incr = current_pos_delta - prev_pos_delta
    prev_pos_delta = current_pos_delta
    # 构造绕 Z 轴的旋转矩阵，用于对 delta_incr 做补偿，
    # 使得当 yaw 变化时，只有位移增量受到影响
    # 这里根据你的描述，我们用当前 yaw 对增量进行旋转
    comp_angle = -yaw.item()  # 使用正 yaw（如果你希望使用负值，可自行调整）
    R_comp = torch.tensor([
        [math.cos(comp_angle), -math.sin(comp_angle), 0],
        [math.sin(comp_angle),  math.cos(comp_angle), 0],
        [0,                    0,                   1]
    ], dtype=torch.float32, device=device)
    # 对增量进行旋转补偿
    compensated_incr = torch.matmul(R_comp, delta_incr)

    # 更新累积的平移命令
    current_pos_delta = prev_pos_delta + compensated_incr

    # 计算新的目标位置 = 锁定的参考位置 + 累积的平移命令
    new_pos = robot_ref_pos + current_pos_delta
    print((pos_tensor - pos_ref) * scale,current_pos_delta,compensated_incr)
    # 旋转部分：生成绕 Z 轴旋转的四元数（基于全局 yaw）
    q_yaw = torch.tensor([
        torch.cos(yaw / 2),
        0.0,
        torch.sin(yaw / 2),
        0.0
    ], dtype=torch.float32, device=device)
    # 新姿态 = 锁定姿态与 q_yaw 组合
    new_quat = quat_mul(robot_ref_quat, q_yaw)

    # 更新动作张量
    actions[:, 0] = new_pos[0]
    actions[:, 1] = new_pos[1]
    actions[:, 2] = new_pos[2]
    actions[:, 3] = new_quat[0]
    actions[:, 4] = new_quat[1]
    actions[:, 5] = new_quat[2]
    actions[:, 6] = new_quat[3]

    return actions


def check_object_condition(obj, device='cuda:0'):
    """
    检测 object 的位置是否满足：
      1. 纵坐标（假设为 y 分量） > -0.15；
      2. (x, y) 与目标点 (0, -0.29) 之间的距离 > 0.005。
      
    参数：
      obj: 包含属性 data.body_pos_w 的对象，
           其中 data.body_pos_w 是一个张量，其 shape 至少为 (N, 3)；
      device: 张量所在设备（默认为 'cuda:0'）。
      
    返回：
      success: 布尔值，True 表示两个条件均满足；
      pos: 检测到的 object 位置（以第一个采样为准）；
      dist: 计算得到的 (x,y) 平面距离。
    """
    # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
    pos_tensor = obj.data.body_pos_w[0][:, :3]  # 例如 shape 为 (num_bodies, 3)
    pos = pos_tensor[0]  # 取第一个 body's 坐标，形状 (3,)
   
    # 条件2：计算 (x, y) 平面上与目标点 (0, -0.29) 的欧氏距离是否大于 0.005
    xy = pos[:2]
    target_xy = torch.tensor([0.0, -0.29], device=device)
    dist = torch.norm(xy - target_xy)
    cond2 = dist > 0.01

    success = cond2
    return success

def reset_global():
    global success,pos_ref, robot_ref_pos, robot_ref_quat, locked_once, reset_requested, yaw, sign, actions, new_action
    # 清空所有相关全局变量，恢复初始状态
    reset_requested = False
    yaw = torch.zeros(1, device=args_cli.device)
    # 重新进入相对控制模式，并关闭精细模式（或根据需求自行调整）
    sign = True
    fine_mode = False
    pos_ref = None
    robot_ref_pos = None
    robot_ref_quat = None
    locked_once = False
    actions = torch.zeros(8, device=args_cli.device).unsqueeze(0)
    q_rot = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=args_cli.device)
    base_tensor = torch.tensor([
        q_rot[0], q_rot[1], q_rot[2], q_rot[3]
    ], dtype=torch.float32, device=args_cli.device).unsqueeze(0)
    new_action[0,3:7] = base_tensor
    success = False
# ========== 主函数 ==========
def main():
    global pos_ref, robot_ref_pos, robot_ref_quat, locked_once, reset_requested, yaw, sign, actions, new_action

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

    # 设置 viewport（注意要在环境初始化后调用）
    setup_viewport()
    
    camera = env.scene["Camera"]  # type: ignore ##
    camera_index = 0
    robot = env.scene["left_robot"].data    # type: ignore ##
    object = env.scene.rigid_objects["object"]  # type: ignore ##
    
    env.reset()

    rate = rospy.Rate(30)  # 30 Hz 循环
    new_action = torch.zeros(8, device=args_cli.device).unsqueeze(0)  # shape (1,8)

    episode_len = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    episode_idx = 35
    
     
    # load policy and stats
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250323_191249/'
    ckpt_path = ckpt_dir + 'policy_best.ckpt'
    policy_class = 'ACT'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    args = {
        'batch_size': 8, 
        'chunk_size': 100, 
        'ckpt_dir': '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250323_191249/', 
        'dim_feedforward': 3200, 
        'eval': False, 
        'hidden_dim': 512, 
        'kl_weight': 10, 
        'lr': 1e-05, 
        'num_epochs': 2000, 
        'onscreen_render': True, 
        'policy_class': 'ACT', 
        'seed': 0, 
        'task_name': 'sim_needle', 
        'temporal_agg': False
        }
    
    policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names':['top']
                        }
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    # 10-40%
    set_seed(0)
    query_frequency = 20
    temporal_agg =  True
    # temporal_agg =  False
    max_timesteps = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0
    while not rospy.is_shutdown():
        
        # ================================ 每个episode ======================================

        print(f"{episode_idx}开始")
        env.reset()               
        reset_global()
        ### 保存数据
        data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action/touch_raw': [],
                '/action/TipPose_raw': [],
                '/observations/images/top':[],
                '/observations/images/depth':[],
            }
        
        
        for step in range(episode_len):
            # 检测 reset 请求，若触发则重置环境并清空所有全局变量
            if reset_requested:
                rospy.loginfo("检测到 'r' 键，重置环境并清空全局变量！")
                break
            
            pos_tensor, orient_tensor = get_transform_tensors(tfBuffer, "base", "stylus", args_cli.device)
            # 若 sign 为 True，则采用相对控制模式
            if not locked_once:
                pos_ref = pos_tensor.clone().detach()
                # 获取当前末端执行器姿态（通过 body_state_w）
                ee_id = robot.body_names.index("scissors_tip")
                current_ee_pose = robot.body_state_w[:, ee_id, 0:7].clone().detach()
                robot_ref_pos  = current_ee_pose[0, 0:3].to(args_cli.device)
                robot_ref_quat = new_action[0,3:7]
                locked_once = True
                rospy.loginfo("进入相对控制模式：位置采用相对控制，姿态锁定。")
            
            new_action = compute_relative_action(pos_tensor, args_cli.device)
            if step <= episode_len-5:
               success = check_object_condition(object, device='cuda:0')

            env.step(new_action)

            ### 处理图像 包括深度图像和RGB图像
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            image = single_cam_data['rgb']
            depth_image = enhance_depth_image(single_cam_data['distance_to_image_plane'])
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera Feed', image_bgr)
            cv2.waitKey(1)

        print(f"{episode_idx}结束，状态{success}") 
 

        rate.sleep()

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    simulation_app.close()


