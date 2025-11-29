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
import numpy as np
from scipy.spatial.transform import Rotation as R
def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, and yaw angles to a quaternion.

    Args:
        roll (float): Rotation around X-axis in radians.
        pitch (float): Rotation around Y-axis in radians.
        yaw (float): Rotation around Z-axis in radians.

    Returns:
        tuple: Quaternion (w, x, y, z).
    """
    # Create a Rotation object using RPY angles
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

    # Convert to quaternion
    # scipy returns quaternion in the order (x, y, z, w)
    quaternion = rotation.as_quat()
    return (quaternion[3], quaternion[0], quaternion[1], quaternion[2])  # Reorder to (w, x, y, z)


def main():

    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Ik-Abs-Position-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-Ik-Abs-Position-Direct-v0", cfg=env_cfg)

    env.reset()
    camera = env.scene["Camera"]
    camera_index = 0
    ee_goals = [
        [-0.2, -0.1, 0.1, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))],
    ]
    while simulation_app.is_running():

        # actions = torch.tensor([[-0.0619, -0.1895, -0.1179]])
        env.step(env.scene.rigid_objects["needle"].data.body_state_w[0,:,0:7])
        # env.step(actions)
        ## 处理图像
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )
        curr_image = rearrange(single_cam_data["rgb"], 'h w c -> c h w')
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)
        image = single_cam_data['rgb']
        print(env.scene.articulations["left_robot"].data.body_state_w[0,-1, 0:3])
        cv2.imshow('Camera Feed', image)
        cv2.waitKey(1)
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()