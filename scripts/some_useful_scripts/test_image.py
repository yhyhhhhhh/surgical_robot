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
    # camera = env.scene["Camera"]
    camera_index = 0
    # ---------------------------------------------------------
    # 关键修改部分：准备动作张量
    # ---------------------------------------------------------
    # 定义单个机器人的动作 (假设维度是 5)
    # 比如: [x, y, z, rot_cmd, gripper]
    single_action = torch.tensor([0, 0, 0, 0, -0.1], device=env.device)
    
    # 将动作扩展到所有环境 -> 形状变为 (num_envs, 5)
    # repeat(num_envs, 1) 表示在第0维复制 num_envs 次，第1维保持不变
    actions = single_action.repeat(num_envs, 1)
    
    print(f"[Info] Action shape: {actions.shape}") # 应该是 torch.Size([2, 5])
    while simulation_app.is_running():
        # 执行动作
        ret = env.step(actions)
        
        # try:
        #     # 获取图像观察
        #     # 假设 obs_images 是 (num_envs, H, W, C) 的 Tensor
        #     obs_images = env.unwrapped.get_image_observation() 
            
        #     # 1. 提取特定环境的图像 (例如第 2 个环境，索引 1)
        #     if isinstance(obs_images, (list, tuple)):
        #         image_tensor = obs_images[1]
        #     else:
        #         image_tensor = obs_images[1] 

        #     image_np = image_tensor.clone().detach().cpu().numpy()

        #     # --- 处理颜色空间 (RGB/RGBA -> BGR) ---
        #     # OpenCV 默认使用 BGR，而 Isaac Lab 通常输出 RGB 或 RGBA
        #     if len(image_np.shape) == 3:
        #         channels = image_np.shape[2]
        #         if channels == 3:
        #             # RGB 转 BGR
        #             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #         elif channels == 4:
        #             # RGBA 转 BGR (丢弃透明通道)
        #             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

        #     # 4. 显示图像
        #     cv2.imshow('Camera Feed (Env 1)', image_np)
            
        #     # 必须调用 waitKey 才能刷新窗口，1ms 表示非阻塞
        #     cv2.waitKey(1)
                
        # except Exception as e:
        #     # 打印简短错误信息以便调试 (但防止刷屏)
        #     # print(f"Image Error: {e}") 
        #     pass
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()