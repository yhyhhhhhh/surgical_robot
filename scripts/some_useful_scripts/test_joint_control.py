import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
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
def main():

    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-ACT-Joint-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-ACT-Joint-Direct-v0", cfg=env_cfg)

    env.reset()
    camera = env.scene["Camera"]
    camera_index = 0

    while simulation_app.is_running():

        actions = env.scene.articulations["left_robot"].data.default_joint_pos
        env.step(actions)
        ### 处理图像
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )
        image = single_cam_data['rgb']
        # 假设 image 是 RGB 格式的图像
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Feed', image_bgr)
        cv2.waitKey(1)
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()