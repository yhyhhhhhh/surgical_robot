import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False
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


def main():
    num_envs = 2
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-PipeRelCam-Ik-RL-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )

    env = gym.make("My-Isaac-Ur3-PipeRelCam-Ik-RL-Direct-v0", cfg=env_cfg)

    env.reset()
    camera = env.scene["Camera"]
    camera_index = 0
    # ---------------------------------------------------------
    # 关键修改部分：准备动作张量
    # ---------------------------------------------------------
    # 定义单个机器人的动作 (假设维度是 5)
    # 比如: [x, y, z, rot_cmd, gripper]
    single_action = torch.tensor([-0.0619, -0.1895, -0.1179, 0, -0.1], device=env.device)
    
    # 将动作扩展到所有环境 -> 形状变为 (num_envs, 5)
    # repeat(num_envs, 1) 表示在第0维复制 num_envs 次，第1维保持不变
    actions = single_action.repeat(num_envs, 1)
    
    print(f"[Info] Action shape: {actions.shape}") # 应该是 torch.Size([2, 5])

    while simulation_app.is_running():
        # 执行动作
        # env.step 返回 (obs, rew, terminated, truncated, info)
        ret = env.step(actions)
        
        # 获取图像观察 (假设你的环境返回的 obs 中包含图像，或者你有自定义接口)
        # 注意：通常 env.step 返回的是 dict，或者你需要从 env.unwrapped.scene 中获取数据
        # 这里保留你的原始逻辑，但加了 try-catch 防止报错
        try:
            # 假设 get_image_observation 返回的是 list 或 tensor
            # 注意：如果是在 GPU 上运行，图像数据通常也在 GPU
            obs_images = env.unwrapped.get_image_observation() 
            
            # 取第 2 个环境的图像 (索引 1) 进行显示
            # 确保转换到 CPU 并转为 numpy
            if isinstance(obs_images, list) or isinstance(obs_images, tuple):
                 # 如果返回的是列表，取其中一个
                image_tensor = obs_images[1]
            else:
                # 如果直接是 tensor (num_envs, H, W, C)
                image_tensor = obs_images[1] 

            if isinstance(image_tensor, torch.Tensor):
                image = image_tensor.cpu().numpy()
            else:
                image = image_tensor
            
            # 显示图像
            if image is not None and image.shape[0] > 0:
                cv2.imshow('Camera Feed (Env 1)', image)
                cv2.waitKey(1)
                
        except Exception as e:
            # 如果获取图像失败，打印一次错误后跳过，避免刷屏
            pass

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()