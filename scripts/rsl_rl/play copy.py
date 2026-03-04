import cv2
import argparse
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0", help="Name of the task.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.core.utils.extensions import enable_extension
import my_ur3_project.tasks  # noqa: F401
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

def main():

    # 1) build env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # 2) load exported JIT policy
    jit_path = "/home/yhy/log/2026-03-01_20-18-40/exported/policy.pt"  # 改成你的实际路径
    device = env.unwrapped.device
    print(f"[INFO]: Loading JIT policy from: {jit_path}")
    policy = torch.jit.load(jit_path, map_location=device)
    policy.eval()

    # 3) rolloutf
    obs, _ = env.get_observations()
    timestep = 0
    # 在循环外定义噪声的幅度（标准差），你可以根据需要调整这个值
    noise_scale = 10.0

    while simulation_app.is_running():
        with torch.inference_mode():
            # 确保 dtype/device 对齐
            if hasattr(obs, "to"):
                obs_t = obs.to(device).float()
            else:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)

            # 1. 获取策略输出的原始动作
            actions = policy(obs_t)
            
            # 2. 生成并添加高斯噪声
            # torch.randn_like 生成均值为0，方差为1的标准正态分布张量
            noise = torch.randn_like(actions) * noise_scale
            noisy_actions = actions + noise

            print("Noisy Actions:", noisy_actions)
            
            # 4. 将带有噪声的动作传入环境
            obs, _, _, _ = env.step(noisy_actions)
            
        timestep += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
