import time
import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
args_cli.headless = True
device = args_cli.device

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import torch
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image,enhance_depth_images_batch
import h5py
from omni.isaac.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
from viztracer import VizTracer
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import random
import os
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnvCfg
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import (
    enhance_depth_image,
    enhance_depth_images_batch,
)
from policy import ACTPolicy
### ====================================== ACT算法 ========================================
from policy import ACTPolicy 
import os
import pickle
from einops import rearrange
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

from Residual_Policy.residual_act import ResidualACTPolicy
#-----------------------------------------------------------------------------
@torch.no_grad()
def calc_advantage(vals, rews, dones, T, gamma, lam):
    adv = torch.zeros_like(rews)
    gae = 0.0
    for t in reversed(range(T-1)):
        nonterm = 1.0 - dones[t + 1].float()
        nxt_v   = vals[t + 1]

        delta = rews[t] + gamma * nxt_v * nonterm - vals[t]
        gae   = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    ret = adv + vals
    return adv, ret

# 其它工具函数
def setup_viewport():
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_policy(policy_class: str, policy_config: dict) -> torch.nn.Module:
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    else:
        raise NotImplementedError


def make_optimizer(policy_class: str, policy: torch.nn.Module):
    if policy_class == 'ACT':
        return policy.configure_optimizers()
    else:
        raise NotImplementedError


def main():
    num_envs = args_cli.num_envs
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=device,
        num_envs=num_envs,
    )
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    env.env.init_robot_ik()
    setup_viewport()
    env.reset()

    camera = env.scene["Camera"]

    # 加载 policy 和统计数据
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/'
    ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_2300_seed_0.ckpt')
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    args = {
        'batch_size': 8,
        'chunk_size': 200,
        'ckpt_dir': ckpt_dir,
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
        'temporal_agg': False,
    }
    policy_config = {
        'lr': args['lr'],
        'num_queries': args['chunk_size'],
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': ['top'],
    }
    policy = make_policy(args['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    print(f'Loaded policy from {ckpt_path}')

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    set_seed(42)
    query_frequency = 100
    temporal_agg = True
    max_timesteps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    # 封装 policy 逻辑
    policy_wrapper = ResidualACTPolicy(
        policy=policy,
        stats=stats,
        device=device,
        num_envs=num_envs,
        max_timesteps=max_timesteps,
        state_dim=state_dim,
        query_frequency=query_frequency,
        temporal_agg=temporal_agg,
    )

    robot = env.unwrapped.scene["left_robot"]

    t = 0
    residual_policy = policy_wrapper.Residual_Policy
    ts = env.reset()
    obs = ts[0]['policy']

    # ---------------- 轨迹缓存 ----------------
    obs_buf  = torch.zeros((max_timesteps, args_cli.num_envs, policy_wrapper.obs_dim+policy_wrapper.action_dim),         device=device)
    act_buf  = torch.zeros((max_timesteps, args_cli.num_envs, policy_wrapper.action_dim),       device=device)
    logp_buf = torch.zeros((max_timesteps, args_cli.num_envs),                                 device=device)
    rew_buf  = torch.zeros((max_timesteps, args_cli.num_envs),                                 device=device)
    done_buf = torch.zeros((max_timesteps, args_cli.num_envs),                                 device=device)
    val_buf  = torch.zeros((max_timesteps, args_cli.num_envs),                                 device=device)
    # ------------------ PPO 超参数具体赋值 ------------------
    gamma                           = 0.99      # 折扣因子
    gae_lambda                      = 0.95      # GAE λ
    update_epochs                   = 4         # 每条轨迹重复学习的 epoch 数
    minibatch_size                  = 64        # 每次小批量大小
    clip_coef                       = 0.2       # PPO 裁剪系数 ε
    ent_coef                        = 0.01      # 熵奖励系数
    vf_coef                         = 0.5       # 价值函数 loss 权重
    max_grad_norm                   = 0.5       # 梯度裁剪阈值
    norm_adv                        = True      # 是否对优势值做归一化
    clip_vloss                      = True      # 是否对 value loss 做裁剪
    target_kl                       = 0.03      # KL 早停阈值（None 表示不早停）
    residual_l1                     = 1e-3      # 残差动作 L1 正则系数
    residual_l2                     = 1e-3      # 残差动作 L2 正则系数
    n_iterations_train_only_value   = 10        # 前多少次只训练 value

    # 学习率
    lr_actor   = 3e-4  # Actor 网络学习率
    lr_critic  = 1e-3  # Critic 网络学习率

    # ------------------ 优化器 ------------------
    optimizer_actor  = torch.optim.Adam(
        residual_policy.actor_mean.parameters(), lr=lr_actor
    )
    optimizer_critic = torch.optim.Adam(
        residual_policy.critic.parameters(),    lr=lr_critic
    )
    while simulation_app.is_running():
        
        with torch.inference_mode():
            obs = env.reset()[0]['policy']
            # ---------------------------------------------------------------------------------
            # PPO采集轨迹
            # ---------------------------------------------------------------------------------
            for step in range(0, max_timesteps):
                # 读取关节位置
                qpos_numpy = robot.data.joint_pos.cpu().numpy().reshape(num_envs, 7)

                # 获取相机图像
                multi_cam_data = convert_dict_to_backend(
                    {k: v[:] for k, v in camera.data.output.items()}, backend="numpy"
                )
                # for i, img_rgb in enumerate(multi_cam_data['rgb']):
                #     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                #     cv2.imshow(f"env_{i}", img_bgr)
                # cv2.waitKey(1)
                
                depth_image = enhance_depth_images_batch(multi_cam_data['distance_to_image_plane'])
                depth_image = rearrange(depth_image, 'n h w c -> n c h w')
                curr_image = torch.from_numpy(depth_image / 255.0).float().to(device).unsqueeze(1)

                # 获取动作并与环境交互
                raw_action = policy_wrapper.get_action(qpos_numpy, curr_image, env.env.episode_length_buf)
                
                next_residual_nobs = torch.cat([obs, raw_action], dim=-1)
                
                obs_buf[step]  = next_residual_nobs
                
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )
                
                residual_naction = residual_naction_samp
                naction = raw_action + residual_naction * residual_policy.action_scale
                # print(residual_naction)
                ts = env.step(naction)
                obs = ts[0]['policy']
                reward = ts[1]
                done, time_out = ts[2], ts[3]
                
                
                done_buf[step] = done
                act_buf[step]  = residual_naction
                logp_buf[step] = logprob
                val_buf[step]  = value.squeeze(-1)
                rew_buf[step]  = reward # type: ignore
                
                if temporal_agg:
                    policy_wrapper.all_time_actions[done] = 0
                    policy_wrapper.all_time_actions[time_out] = 0

            # ------------------ 在 while simulation_app.is_running(): 内，轨迹采集 for-step 循环结束后，插入以下训练代码 ------------------

        # 1) 计算 GAE 优势和 returns
        with torch.no_grad():
            # 最后一步的 value
            last_val = residual_policy.get_value(next_residual_nobs.to(device)).view(-1)  # [N]
            T, N = max_timesteps, num_envs
            advantages = torch.zeros_like(rew_buf)  # [T, N]
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                mask = 1.0 - done_buf[t]               # done=1 -> mask=0
                next_val = last_val if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + gamma * next_val * mask - val_buf[t]
                advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

            returns = advantages + val_buf  # [T, N]

        # 2) 展平到 (T*N, ...)
        b_obs        = obs_buf.reshape(T * N, -1)
        b_actions    = act_buf.reshape(T * N, -1)
        b_logprobs   = logp_buf.reshape(T * N)
        b_advantages = advantages.reshape(T * N)
        b_returns    = returns.reshape(T * N)
        b_values     = val_buf.reshape(T * N)

        # 3) PPO 更新
        batch_inds = np.arange(T * N)
        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, T * N, minibatch_size):
                mb_inds = batch_inds[start : start + minibatch_size]

                # 切片 & to(device)
                mb_obs        = b_obs[mb_inds].to(device)
                mb_actions    = b_actions[mb_inds].to(device)
                mb_logprobs   = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns    = b_returns[mb_inds].to(device)
                mb_values     = b_values[mb_inds].to(device)

                # 前向：新 logprob, 熵, 新 value, 动作均值
                _, newlogprob, entropy, newvalue, action_mean = \
                    residual_policy.get_action_and_value(mb_obs, mb_actions)

                # 比例 & KL
                logratio = newlogprob - mb_logprobs
                ratio    = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                # (可选) 归一化优势
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 计算 PPO policy loss（剪切版）
                pg1 = -mb_advantages * ratio
                pg2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # 价值函数 loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped   = mb_values + torch.clamp(newvalue - mb_values, -clip_coef, clip_coef)
                    v_loss      = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # 熵正则
                entropy_loss = entropy.mean() * ent_coef

                # 残差动作 L1/L2 正则
                l1_reg = torch.mean(torch.abs(action_mean))
                l2_reg = torch.mean(action_mean.square())

                # 总 policy+aux loss
                policy_loss = pg_loss - entropy_loss

                # 最终 loss
                loss = policy_loss + v_loss * vf_coef

                # 梯度 & 更新
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual_policy.parameters(), max_grad_norm)
                optimizer_actor.step()
                optimizer_critic.step()

                # Early stop on KL
                if target_kl is not None and approx_kl > target_kl:
                    print(f"Early stopping at epoch {epoch}, KL={approx_kl:.4f}")
                    break
            else:
                continue
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
