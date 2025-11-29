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
import rospy
from geometry_msgs.msg import PoseStamped

actions = torch.zeros(7).unsqueeze(0)  # 添加batch维度 → shape (1,7)
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

    
def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, callback)
    
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-Ik-Abs-Direct-v0", cfg=env_cfg)

    camera = env.scene["Camera"]
    camera_index = 0

    env.reset()

    while not rospy.is_shutdown():
        

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