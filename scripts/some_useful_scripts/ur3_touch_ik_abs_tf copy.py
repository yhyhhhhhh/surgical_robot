#!/usr/bin/env python
import argparse
import os
import sys
import time
import math

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
from geometry_msgs.msg import Vector3
from omni.isaac.lab.app import AppLauncher

from pynput import keyboard

# --------------------------------------------------------------------------------------
# 命令行参数 & IsaacSim 启动
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperation with Isaac Lab + Geomagic Touch.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
    parser.add_argument("--task", type=str,
                        default="My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
                        help="Task name.")
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the data from camera.",
    )

    # 让 AppLauncher 自己注册需要的参数（device / headless 等）
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 统一配置
    args.enable_cameras = True
    args.seed = 422
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # 必须在导入 IsaacLab 之前创建

# --------------------------------------------------------------------------------------
# Isaac Lab / 其它依赖导入（注意：必须在 simulation_app 创建之后）
# --------------------------------------------------------------------------------------
import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import quat_mul
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: F401  (后面要用可以保留)
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf

from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import (
    convert_dict_to_backend,
    enhance_depth_image,
    get_transform_tensors,
    map_range,
    map_range_1,
)

# --------------------------------------------------------------------------------------
# ROS 相关路径（根据你自己的 workspace 调整）
# --------------------------------------------------------------------------------------
os.environ["ROS_PACKAGE_PATH"] = (
    "/home/yhy/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:"
    + os.environ.get("ROS_PACKAGE_PATH", "")
)
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/home/yhy/touch_ws/devel/lib/python3/dist-packages")
from omni_msgs.msg import OmniFeedback, OmniState

# --------------------------------------------------------------------------------------
# 常量 & 全局变量
# --------------------------------------------------------------------------------------

# 刚度/阻尼参数（单位：N/m 和 N·s/m）
K = np.diag([30.0, 30.0, 30.0])
B = np.diag([1.0, 1.0, 1.0])  # 目前没有用到，但以后可能会用到就先保留

# 力 / 位置记录（整个脚本生命周期共用）
data_log = {
    "pos_m": [],  # 主端位置
    "pos_p": [],  # 从端位置
    "vel_m": [],  # 主端速度
    "vel_p": [],  # 从端速度
    "force": [],  # 反馈力
    "time": [],   # 时间戳
}

# 遥操作全局状态
reset_requested = False
yaw = torch.zeros(1, device=args_cli.device)  # 弧度
sign = True
fine_mode = False

pos_ref = None          # 手柄参考位置（touch 初次锁定时的位姿）
robot_ref_pos = None    # 机械臂末端执行器参考位置
robot_ref_quat = None   # 机械臂末端执行器参考姿态
locked_once = False     # 是否已经完成一次锁定
prev_pos_delta = None   # compute_relative_action1 用到

# 当前 action（传给 Isaac Lab 的动作向量）
actions = torch.zeros(1, 8, device=args_cli.device)  # shape: (1, 8)

# --------------------------------------------------------------------------------------
# 小工具函数
# --------------------------------------------------------------------------------------


def yaw_to_quat(yaw_tensor: torch.Tensor, device) -> torch.Tensor:
    """根据 yaw（绕 Z 轴，单位弧度）生成四元数 [w, x, y, z]."""
    half = yaw_tensor / 2.0
    c = torch.cos(half)
    s = torch.sin(half)
    # c / s 的 shape 一般是 (1,) 或标量，用 view(-1) 保证返回 (4,)
    quat = torch.stack(
        (c, torch.zeros_like(c), s, torch.zeros_like(c)), dim=0
    ).view(4)
    return quat.to(device)


def setup_viewport():
    """配置 viewport 相机视角."""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)


# --------------------------------------------------------------------------------------
# 键盘 & ROS 回调
# --------------------------------------------------------------------------------------


def on_press(key):
    """键盘按下回调，用于 reset / yaw / 精细模式切换."""
    global reset_requested, yaw, sign, fine_mode, locked_once
    try:
        if key.char == "r":
            reset_requested = True
        elif key.char == "a":
            # 向左旋转
            delta = 0.0 if fine_mode else 0.004
            yaw = yaw + delta
            print(f"向左旋转，yaw = {yaw.item():.4f}")
        elif key.char == "d":
            # 向右旋转
            delta = 0.0 if fine_mode else 0.004
            yaw = yaw - delta
            print(f"向右旋转，yaw = {yaw.item():.4f}")
        elif key.char == "q":
            # 切换精细控制模式
            fine_mode = not fine_mode
            locked_once = False  # 下一个循环会重新锁定参考坐标
            mode = "精细" if fine_mode else "粗控制"
            print(f"切换为 {mode} 模式的相对控制")
    except AttributeError:
        # 功能键（Ctrl 等）会走到这里，直接忽略
        pass


def on_release(key):
    """键盘松开回调."""
    global reset_requested
    try:
        if key.char == "r":
            reset_requested = False
    except AttributeError:
        pass


# 启动键盘监听（后台线程即可）
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def button_callback(msg: OmniState):
    """来自 Geomagic Touch 的状态回调，这里只用到抓手开合."""
    global actions
    # close_gripper 通常是 0/1 或 bool，直接转换为 float
    actions[:, 7] = float(msg.close_gripper)


# --------------------------------------------------------------------------------------
# 动作计算（绝对 / 相对模式）
# --------------------------------------------------------------------------------------


def compute_action(tf_buffer, device, base_frame="base", link_frame="stylus"):
    """
    绝对控制：
      - actions[:, 0:3]  末端位置
      - actions[:, 3:7]  末端姿态四元数
      - actions[:, 7]    抓手开合
    返回 (actions, pos_tensor, orient_tensor) 方便后续切换到相对模式。
    """
    global actions, yaw

    pos_tensor, orient_tensor = get_transform_tensors(
        tf_buffer, base_frame, link_frame, device
    )

    # 固定旋转 + yaw
    base_rot = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=device)
    q_yaw = yaw_to_quat(yaw, device)
    q_rot = quat_mul(base_rot, q_yaw)

    # 手柄空间到机器人空间的简单线性映射
    mapped_x = map_range(pos_tensor[0].item(), -0.15, 0.15, 0.2, -0.2)
    mapped_y = map_range(pos_tensor[1].item(), 0.07, 0.17, -0.15, -0.35)
    mapped_z = map_range_1(pos_tensor[2].item(), -0.025, 0.12, -0.25, 0.0)

    base_tensor = torch.tensor(
        [mapped_x, mapped_y, mapped_z, q_rot[0], q_rot[1], q_rot[2], q_rot[3]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    actions[:, 0:7] = base_tensor
    return actions, pos_tensor, orient_tensor


def compute_relative_action(pos_tensor: torch.Tensor, device) -> torch.Tensor:
    """
    相对控制逻辑（更简单版本）：
      - 位置：touch 相对 pos_ref 的偏移 * 缩放系数 + robot_ref_pos
      - 姿态：锁定姿态 robot_ref_quat * q_yaw
    """
    global actions, pos_ref, robot_ref_pos, robot_ref_quat, fine_mode, yaw

    if pos_ref is None or robot_ref_pos is None or robot_ref_quat is None:
        # 安全保护，避免还没锁定就调用
        return actions

    # 粗 / 细 两种缩放
    coarse_scale = torch.tensor([-0.5, -0.5, 1.0], device=device, dtype=torch.float32)
    fine_scale = torch.tensor([-0.1, -0.1, 0.5], device=device, dtype=torch.float32)
    scale = fine_scale if fine_mode else coarse_scale

    pos_delta = (pos_tensor - pos_ref) * scale
    new_pos = robot_ref_pos + pos_delta

    q_yaw = yaw_to_quat(yaw, device)
    new_quat = quat_mul(robot_ref_quat, q_yaw)

    actions[0, 0:3] = new_pos
    actions[0, 3:7] = new_quat

    return actions


def compute_relative_action1(pos_tensor: torch.Tensor, device) -> torch.Tensor:
    """
    相对控制逻辑（带旋转补偿的版本，当前脚本没有在主循环中使用，
    可以作为实验用函数保留）：
      - 基于 touch 与初始 touch 的差得到 current_pos_delta
      - 对增量部分做旋转补偿，避免纯旋转导致平移漂移
      - 姿态同样使用 robot_ref_quat * q_yaw
    """
    global actions, pos_ref, robot_ref_pos, robot_ref_quat, fine_mode, yaw, prev_pos_delta

    if pos_ref is None or robot_ref_pos is None or robot_ref_quat is None:
        return actions

    if prev_pos_delta is None:
        prev_pos_delta = torch.zeros(3, device=device, dtype=torch.float32)

    coarse_scale = torch.tensor([-0.5, -0.5, 1.0], device=device, dtype=torch.float32)
    fine_scale = torch.tensor([-0.1, -0.1, 0.1], device=device, dtype=torch.float32)
    scale = fine_scale if fine_mode else coarse_scale

    current_pos_delta = (pos_tensor - pos_ref) * scale
    delta_incr = current_pos_delta - prev_pos_delta
    prev_pos_delta = current_pos_delta

    # 对增量做 yaw 方向的旋转补偿
    comp_angle = -yaw.item()
    R_comp = torch.tensor(
        [
            [math.cos(comp_angle), -math.sin(comp_angle), 0.0],
            [math.sin(comp_angle), math.cos(comp_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    compensated_incr = R_comp @ delta_incr

    current_pos_delta = prev_pos_delta + compensated_incr
    new_pos = robot_ref_pos + current_pos_delta

    q_yaw = yaw_to_quat(yaw, device)
    new_quat = quat_mul(robot_ref_quat, q_yaw)

    actions[0, 0:3] = new_pos
    actions[0, 3:7] = new_quat

    return actions


# --------------------------------------------------------------------------------------
# 任务相关小工具
# --------------------------------------------------------------------------------------


def check_object_condition(obj, device="cuda:0") -> bool:
    """
    根据 object 位置判断是否成功：
      目前逻辑：判断 (x, y) 到目标点 (0, -0.29) 的距离是否大于 0.01。
    如需附加高度等条件可以在这里扩展。
    """
    pos_tensor = obj.data.body_pos_w[0][:, :3]  # (num_bodies, 3)
    pos = pos_tensor[0]  # 只取第一个 body

    xy = pos[:2]
    target_xy = torch.tensor([0.0, -0.29], device=device)
    dist = torch.norm(xy - target_xy)

    return dist > 0.01


def reset_global_state():
    """重置遥操作相关的全局变量，在每个 episode 开始时调用."""
    global reset_requested, yaw, sign, fine_mode
    global pos_ref, robot_ref_pos, robot_ref_quat, locked_once, prev_pos_delta
    global actions

    reset_requested = False
    yaw = torch.zeros(1, device=args_cli.device)
    sign = True
    fine_mode = False

    pos_ref = None
    robot_ref_pos = None
    robot_ref_quat = None
    locked_once = False
    prev_pos_delta = None

    actions = torch.zeros(1, 8, device=args_cli.device)
    # 初始姿态（如果你希望锁死一个基准姿态，可以在这里写入）
    base_quat = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=torch.float32, device=args_cli.device)
    actions[0, 3:7] = base_quat


# --------------------------------------------------------------------------------------
# 主循环
# --------------------------------------------------------------------------------------


def main():
    global pos_ref, robot_ref_pos, robot_ref_quat, locked_once, reset_requested

    # ---------------- ROS 初始化 ----------------
    rospy.init_node("teleop_listener", anonymous=True)
    rospy.Subscriber("/phantom/phantom/state", OmniState, button_callback)
    force_pub = rospy.Publisher(
        "/phantom/phantom/force_feedback", OmniFeedback, queue_size=10
    )

    # TF 监听
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)

    # ---------------- Isaac 环境初始化 ----------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    setup_viewport()

    camera = env.scene["Camera"]
    camera_index = 0
    robot = env.scene["left_robot"].data
    target_object = env.scene.rigid_objects["object"]

    env.reset()

    ee_id = robot.body_names.index("scissors_tip")

    # episode 长度（step 数）
    episode_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    episode_idx = 0

    rate = rospy.Rate(30)  # 控制 episode 之间的频率

    # 主循环：每次 while 代表一个 episode
    while not rospy.is_shutdown():
        rospy.loginfo(f"Episode {episode_idx} 开始")
        env.reset()
        reset_global_state()

        # 轨迹数据（当前 episode）
        data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/action/touch_raw": [],
            "/action/TipPose_raw": [],
            "/observations/images/top": [],
            "/observations/images/depth": [],
            "/observations/object_pos": [],
        }

        prev_pos_m = robot.body_state_w[:, ee_id, 0:3].cpu().numpy()
        prev_pos_p = robot.body_state_w[:, ee_id, 0:3].cpu().numpy()

        success = False

        for step in range(episode_len - 5):
            # 外部重置请求（键盘 r）
            if reset_requested:
                rospy.loginfo("检测到 'r' 键，提前结束本 episode")
                break

            # 读取 touch 的位姿（相对 base）
            pos_tensor, orient_tensor = get_transform_tensors(
                tf_buffer, "base", "stylus", args_cli.device
            )

            # 首次进入时锁定参考位姿
            if not locked_once:
                pos_ref = pos_tensor.clone().detach()
                current_ee_pose = robot.body_state_w[:, ee_id, 0:7].clone().detach()
                robot_ref_pos = current_ee_pose[0, 0:3].to(args_cli.device)
                # 使用当前机器人姿态作为锁定基准
                robot_ref_quat = current_ee_pose[0, 3:7].to(args_cli.device)

                locked_once = True
                rospy.loginfo("进入相对控制模式：锁定 touch & 机械臂当前位姿为参考。")

            # 根据 touch 计算新的动作（相对控制）
            new_action = compute_relative_action(pos_tensor, args_cli.device)

            # 检查目标物体是否达到要求
            if step <= episode_len - 5:
                success = check_object_condition(target_object, device=args_cli.device)

            # 发送动作给 Isaac 环境
            env.step(new_action)

            # ---------------- 力反馈 & 数据记录 ----------------
            # 主端控制目标位置（通过 new_action）
            pos_m = new_action[0, 0:3].cpu().numpy()
            vel_m = (pos_m - prev_pos_m) * 100.0

            # 从端真实位置（UR3 末端）
            pos_p = robot.body_state_w[:, ee_id, 0:3].cpu().numpy()
            vel_p = (pos_p - prev_pos_p) * 100.0

            prev_pos_m = pos_m
            prev_pos_p = pos_p

            # 力反馈（目前只用于记录，不施加回 Touch，所以后面置零）
            dx = pos_m - pos_p
            dv = vel_m - vel_p  # noqa: F841  # 目前没用到阻尼项，如需可加 B @ dv
            F_h = -K @ dx[0]
            F_h = np.clip(F_h, -3.0, 3.0)

            feedback_msg = OmniFeedback()
            # 如果要启用力反馈，把下一行注释掉即可
            # feedback_msg.force = Vector3(x=F_h[0], y=F_h[1], z=F_h[2])
            feedback_msg.force = Vector3(x=0.0, y=0.0, z=0.0)
            feedback_msg.position = Vector3(
                x=float(pos_m[0]), y=float(pos_m[1]), z=float(pos_m[2])
            )
            force_pub.publish(feedback_msg)

            # 记录日志（全局）
            t_now = rospy.get_time()
            data_log["pos_m"].append(pos_m.tolist())
            data_log["pos_p"].append(pos_p[0].tolist())
            data_log["vel_m"].append(vel_m.tolist())
            data_log["vel_p"].append(vel_p[0].tolist())
            data_log["force"].append(F_h.tolist())
            data_log["time"].append(t_now)

            # 记录当前时刻的 touch / robot 末端 / 图像
            touch_raw = new_action.clone()
            tip_raw = robot.body_state_w[:, ee_id, 0:7].clone().detach()

            # 处理图像（RGB + depth）
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()},
                backend="numpy",
            )
            rgb = single_cam_data["rgb"]
            depth_image = enhance_depth_image(
                single_cam_data["distance_to_image_plane"]
            )

            # 显示一下相机画面
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Feed", image_bgr)
            cv2.waitKey(1)

            if args_cli.save:
                data_dict["/observations/qpos"].append(
                    robot.joint_pos.cpu().numpy().reshape(7)
                )
                data_dict["/observations/qvel"].append(
                    robot.joint_vel.cpu().numpy().reshape(7)
                )
                data_dict["/action/touch_raw"].append(
                    touch_raw.cpu().numpy().reshape(8)
                )
                data_dict["/action/TipPose_raw"].append(
                    tip_raw.cpu().numpy().reshape(7)
                )
                data_dict["/observations/images/top"].append(rgb)
                data_dict["/observations/images/depth"].append(depth_image)
                data_dict["/observations/object_pos"].append(
                    target_object.data.body_pos_w.cpu()
                    .numpy()
                    .reshape(3)
                )

        rospy.loginfo(f"Episode {episode_idx} 结束，状态 success={success}")

        # Episode 结束时置零反馈力，避免保持残余
        feedback_msg = OmniFeedback()
        feedback_msg.force = Vector3(x=0.0, y=0.0, z=0.0)
        feedback_msg.position = Vector3(
            x=float(pos_m[0]), y=float(pos_m[1]), z=float(pos_m[2])
        )
        force_pub.publish(feedback_msg)

        # 保存整条轨迹
        if args_cli.save:
            # 简单处理一下全局 data_log（丢掉第一帧可以避免 reset 带来的奇异值）
            for key in data_log:
                if len(data_log[key]) > 1:
                    data_log[key] = data_log[key][1:]

            task_name = args_cli.task.replace("/", "_")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"haptic_{task_name}_{timestamp}.npz"
            np.savez(filename, **data_log)

        if success:
            episode_idx += 1
            print("成功 episode:", episode_idx)

        rate.sleep()

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
