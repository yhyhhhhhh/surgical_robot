import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
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
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
from einops import rearrange
import numpy as np
from scipy.spatial.transform import Rotation as R
from omni.isaac.lab.devices import Se3Keyboard

import rospy
from geometry_msgs.msg import PoseStamped

actions = torch.zeros(8).unsqueeze(0)  # 添加batch维度 → shape (1,7)
def callback(data):
    global actions
    # 提取位置和姿态信息
    # 提取原始数据
    position = [data.pose.position.x*10, data.pose.position.y*10, data.pose.position.z*10]
    orientation = [data.pose.orientation.x, data.pose.orientation.y, 
                   data.pose.orientation.z, data.pose.orientation.w]
    
    # 创建基础张量
    base_tensor = torch.tensor(
        position + orientation, 
        dtype=torch.float32,
        device=args_cli.device  # 保持与仿真环境相同的设备
    ).unsqueeze(0)  # shape (1,7)

    # 追加0值
    actions = torch.cat([
        base_tensor, 
        torch.zeros(1, 1, device=base_tensor.device)
    ], dim=1)  # shape (1,8)

        
def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool, is_reach: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if is_reach:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)

def process_actions(teleop_interface, env, is_dual,is_reach) -> torch.Tensor:
    """Process actions for the environment."""
    if is_dual:
        # get keyboard command
        delta_pose_0, gripper_command_0, delta_pose_1, gripper_command_1 = teleop_interface.advance()
        delta_pose_0 = delta_pose_0.astype("float32")
        delta_pose_1 = delta_pose_1.astype("float32")
        # convert to torch
        delta_pose_0 = torch.tensor(delta_pose_0, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        delta_pose_1 = torch.tensor(delta_pose_1, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        # pre-process actions
        actions_0 = pre_process_actions(delta_pose_0, gripper_command_0,is_reach)
        actions_1 = pre_process_actions(delta_pose_1, gripper_command_1,is_reach)
        actions = torch.concat([actions_0, actions_1], dim=1)
    else:
        # get keyboard command
        delta_pose, gripper_command = teleop_interface.advance()
        delta_pose = delta_pose.astype("float32")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
        # pre-process actions
        actions = pre_process_actions(delta_pose, gripper_command,is_reach)
    return actions


def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, callback)
    
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)

    camera = env.scene["Camera"]
    camera_index = 0
    
    # teleop_interface = Se3Keyboard(
    #         pos_sensitivity=0.01 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
    #     )

    # teleop_interface.add_callback("L", env.reset)
    # print(teleop_interface)

    env.reset()
    # teleop_interface.reset()
    
    is_dual = False
    is_reach = False
    while not rospy.is_shutdown():
        
        # actions = process_actions(teleop_interface, env, is_dual,is_reach)
        print(actions)
        env.step(actions)
        ## 处理图像
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )
        image = single_cam_data['rgb']
        # 假设 image 是 RGB 格式的图像
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Feed', image_bgr)
        cv2.waitKey(1)
        # print(env.scene.articulations["left_robot"].data.body_state_w[0,-1, 0:3])
        
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()
    simulation_app.close()