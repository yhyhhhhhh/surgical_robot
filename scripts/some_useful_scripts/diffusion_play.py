import cv2
import argparse
from omni.isaac.lab.app import AppLauncher
import sys
sys.path.append("/home/yhy/IsaacLabExtensionTemplate/scripts/rsl_rl")  # 替换成你的实际路径
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
import torch
import numpy as np
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# 注意：请确保将你训练代码里的 ConditionalUnet1D 等网络类定义放在这个文件里，或者 import 进来
from diffusion import ConditionalUnet1D

def main():
    # =========================================================================
    # 0. 参数配置 (请根据你的实际训练参数修改)
    # =========================================================================
    obs_dim = 30         # 替换为你的状态维度
    action_dim = 5       # 替换为你的动作维度
    obs_horizon = 5     # 观测历史长度
    pred_horizon = 16   # 预测轨迹长度
    action_horizon = 2  # 实际采纳执行的步数
    num_diffusion_iters = 100 # 去噪步数
    
    ckpt_path = "/home/yhy/IsaacLabExtensionTemplate/checkpoints/best_ema.ckpt" # 你的模型权重
    stats_path = "/home/yhy/IsaacLabExtensionTemplate/checkpoints/dataset_stats.npy"                # 你的归一化统计数据

    # =========================================================================
    # 1. 构建环境 (保持原样)
    # =========================================================================
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    
    device = env.unwrapped.device
    num_envs = args_cli.num_envs

    # =========================================================================
    # 2. 加载 Diffusion 策略与统计数据
    # =========================================================================
    print(f"[INFO]: Loading Diffusion policy from: {ckpt_path}")
    
    # 初始化 UNet
    policy = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()

    # 初始化调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # 加载 stats 并转换为 GPU Tensors，以便进行极速的并行运算
    stats_np = np.load(stats_path, allow_pickle=True).item()
    stats_t = {
        'obs_min': torch.tensor(stats_np['obs']['min'], device=device, dtype=torch.float32),
        'obs_max': torch.tensor(stats_np['obs']['max'], device=device, dtype=torch.float32),
        'action_min': torch.tensor(stats_np['action']['min'], device=device, dtype=torch.float32),
        'action_max': torch.tensor(stats_np['action']['max'], device=device, dtype=torch.float32),
    }

    def normalize(x, s_min, s_max):
        return (x - s_min) / (s_max - s_min + 1e-8) * 2 - 1

    def unnormalize(x, s_min, s_max):
        return (x + 1) / 2 * (s_max - s_min + 1e-8) + s_min

    # =========================================================================
    # 3. 初始化观测历史缓冲 (Observation Buffer)
    # =========================================================================
    # Isaac 环境通常返回元组，我们取第一个元素作为 obs
    obs_raw = env.get_observations()
    obs = obs_raw[0] if isinstance(obs_raw, tuple) else obs_raw
    obs_t = obs.to(device).float()

    # 创建一个 Tensor 缓冲区来存储历史观测: shape (num_envs, obs_horizon, obs_dim)
    # 初始状态下，用第一帧画面把历史填满
    obs_history = obs_t.unsqueeze(1).repeat(1, obs_horizon, 1)

    timestep = 0
    # 动作执行队列
    action_queue = collections.deque()

    # =========================================================================
    # 4. 主循环 Rollout
    # =========================================================================
    while simulation_app.is_running():
        with torch.inference_mode():
            # ---------- A. 如果动作队列为空，则触发 Diffusion 推理 ----------
            if len(action_queue) == 0:
                # 1. 归一化观测历史
                nobs = normalize(obs_history, stats_t['obs_min'], stats_t['obs_max'])
                
                # 2. 展平为条件特征: (num_envs, obs_horizon * obs_dim)
                obs_cond = nobs.flatten(start_dim=1)

                # 3. 采样初始随机噪声轨迹: (num_envs, pred_horizon, action_dim)
                naction = torch.randn((num_envs, pred_horizon, action_dim), device=device)
                
                # 4. 去噪循环 (Denoising Loop)
                noise_scheduler.set_timesteps(num_diffusion_iters)
                for k in noise_scheduler.timesteps:
                    noise_pred = policy(sample=naction, timestep=k, global_cond=obs_cond)
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
                
                # 5. 反归一化，得到真实的物理动作轨迹
                action_pred = unnormalize(naction, stats_t['action_min'], stats_t['action_max'])

                # 6. 将需要执行的动作推入队列 (只取 start 到 end 之间的动作)
                start_idx = obs_horizon - 1
                end_idx = start_idx + action_horizon
                for i in range(start_idx, end_idx):
                    # 把每一步的动作按顺序放入队列
                    action_queue.append(action_pred[:, i, :])

            # ---------- B. 从队列中取出一个动作并执行 ----------
            current_actions = action_queue.popleft()
            
            # 由于你的原代码使用了 noisy_actions，这里直接传给 env.step
            # 注意：IsaacGym/Lab 的 step 返回值较多，使用 *rest 捕获以防报错
            step_returns = env.step(current_actions)
            
            # 解析返回值 (不同版本的 Wrapper 返回结构可能不同)
            next_obs_raw = step_returns[0]
            # 如果有 done 信号 (通常在第2或第3个返回值)，需要处理环境自动重置
            # dones = step_returns[2] 
            
            next_obs = next_obs_raw[0] if isinstance(next_obs_raw, tuple) else next_obs_raw
            next_obs_t = next_obs.to(device).float()

            # ---------- C. 更新观测历史缓冲 ----------
            # 把历史记录往左平移一位 (扔掉最老的，腾出最新位置)
            obs_history = torch.roll(obs_history, shifts=-1, dims=1)
            # 把最新的观测放入最后一位
            obs_history[:, -1, :] = next_obs_t

            # [进阶逻辑]: 如果环境触发了 Reset (Done)，那么该环境的历史应该被清空
            # 如果你能从 step_returns 提取出 dones (bool tensor)，加上下面这行代码会更严谨：
            # obs_history[dones] = next_obs_t[dones].unsqueeze(1).repeat(1, obs_horizon, 1)

        timestep += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
