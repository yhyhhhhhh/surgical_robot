import csv
from pathlib import Path
import time
import cv2
import argparse
import numpy as np
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import torch
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image
import h5py
from omni.isaac.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import random
actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
prev_actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
sign = False

### ====================================== ACT算法 ========================================
from policy import ACTPolicy 
import os
import pickle
from einops import rearrange
# =========================================================================================
# 使用 pynput 监听键盘按键
from pynput import keyboard
reset_requested = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图

from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import (
    Ur3LiftNeedleEnv,
)
# from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_pipe_ik_env_cfg import (
#     Ur3LiftNeedleEnvCfg,
# )
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import (
    enhance_depth_image,
)
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from policy import ACTPolicy

# Omni debug draw（如果不需要画线可以去掉）
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw  # noqa: E402


# ==========================
# 全局变量 & 键盘控制
# ==========================
reset_requested = False


def on_press(key):
    global reset_requested
    try:
        if key.char == "r":
            reset_requested = True
    except AttributeError:
        pass


def on_release(key):
    global reset_requested
    try:
        if key.char == "r":
            reset_requested = False
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# ==========================
# 工具函数
# ==========================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_hdf5_file(file_path):
    """简单的 hdf5 解析示例（当前脚本里没真正用到动作内容，只保留接口）"""
    with h5py.File(file_path, "r") as f:
        sim = f.attrs.get("sim", None)
        action_group = f["action"]
        touch_raw = action_group["touch_raw"][:]
        TipPose_raw = action_group["TipPose_raw"][:]
        return {"touch_raw": touch_raw, "TipPose_raw": TipPose_raw}


def setup_viewport():
    """配置 viewport 视角"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    raise NotImplementedError


def check_object_condition(obj, device="cuda:0"):
    """
    判断物体是否离开鼻腔一定距离（简单的成功判定）。
    """
    pos_tensor = obj.data.body_pos_w[0][:, :3]  # (num_bodies,3)
    pos = pos_tensor[0]
    xy = pos[:2]
    target_xy = torch.tensor([0.0, -0.29], device=device)
    dist = torch.norm(xy - target_xy)
    success = dist > 0.01
    return success


# ==========================
# 主流程
# ==========================
def main():
    global reset_requested

    # ------------------ 创建环境 ------------------
    num_envs = args_cli.num_envs
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    setup_viewport()
    env.reset()

    # 奖励可视化需要的 key（对应 compute_validation_reward 里的名字）
    val_reward_keys = [
        "val/total",
        "val/reach_pipe",
        "val/grasp_reach_3d",
        "val/entry",
        "val/center_pipe",
        "val/pre_grasp_open",
        "val/grasp_close",
        "val/lift",
        "val/move_out",
        "val/success",
        "val/ee_obj_dist",
        "val/pipe_s_ee",
        "val/pipe_r_ee",
        "val/pipe_s_obj",
        "val/pipe_r_obj",
        "val/r_norm",
        "val/object_lift_h",
        "val/dist_xy",
        "val/grip_norm_mean",
        "val/near_for_grasp",
        "val/middle_near",
        "val/in_pipe_ratio",
        "val/outside_ratio",
        "val/success_rate",
              
    ]
         
    # ------------------ 相机 & 数据 ------------------
    # 如果你需要用 hdf5 里的数据，这里先解析（目前没直接用）
    # data = parse_hdf5_file('/path/to/episode_xx.hdf5')

    camera = env.scene["Camera"]
    camera_index = 0

    # ------------------ 加载策略 & 归一化参数 ------------------
    ckpt_dir = "/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/"
    ckpt_path = ckpt_dir + "policy_epoch_2300_seed_0.ckpt"

    policy_class = "ACT"
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = "resnet18"

    args = {
        "batch_size": 8,
        "chunk_size": 200,
        "dim_feedforward": 3200,
        "eval": False,
        "hidden_dim": 512,
        "kl_weight": 10,
        "lr": 1e-5,
        "num_epochs": 2000,
        "onscreen_render": True,
        "policy_class": "ACT",
        "seed": 0,
        "task_name": "sim_needle",
        "temporal_agg": False,
    }

    policy_config = {
        "lr": args["lr"],
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": lr_backbone,
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "camera_names": ["top"],
    }

    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded policy from: {ckpt_path}")

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    action_std = torch.as_tensor(stats["action_std"], device=args_cli.device)
    action_mean = torch.as_tensor(stats["action_mean"], device=args_cli.device)

    # ------------------ 评估设置 ------------------
    set_seed(42)
    query_frequency = 100
    temporal_agg = True

    max_timesteps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0

    # 保存奖励图的目录
    save_dir = "trajectory_logs_rl_tem"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------ 主循环 ------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            robot = env.unwrapped.scene["left_robot"]
            needle = env.unwrapped.scene.rigid_objects["object"]

            env.reset()
            t = 0
            total_cnt += 1
            print(f"===== Reset episode {total_cnt} =====")

            # 本 episode 验证奖励轨迹（用于画图）
            episode_val_rewards = {k: [] for k in val_reward_keys}

            frames = []  # 如果以后要保存视频帧，可以用
            draw = omni_debug_draw.acquire_debug_draw_interface()

            if temporal_agg:
                all_time_actions = torch.zeros(
                    [max_timesteps, max_timesteps + query_frequency, state_dim], device="cuda"
                )

            # ==========================
            # episode 里的 step 循环
            # ==========================
            while t <= max_timesteps - 1:
                # -------- 构造观测（关节+图像） --------
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(num_envs, 7))
                qpos_norm = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)  # (1, 1, 7) after later expand

                # 相机观测
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()},
                    backend="numpy",
                )
                rgb = single_cam_data["rgb"]
                depth = single_cam_data["distance_to_image_plane"]

                # 画出相机画面（RGB）
                image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera Feed", image_bgr)
                cv2.waitKey(1)
                frames.append(image_bgr.copy())

                # 深度图预处理作为 policy 输入
                depth_image = enhance_depth_image(depth)
                depth_image = rearrange(depth_image, "h w c -> c h w")
                depth_tensor = torch.from_numpy(depth_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)
                curr_image = depth_tensor

                obs_dict = {"qpos": qpos, "curr_image": curr_image}

                # -------- 策略查询 (temporal_agg) --------
                if temporal_agg:
                    B = obs_dict["qpos"].shape[0]
                    obs_dict_batch = {}
                    for key, value in obs_dict.items():
                        obs_dict_batch[key] = (
                            value.unsqueeze(1)
                            .repeat(1, 1, *([1] * (value.dim() - 1)))
                            .view(B * 1, *value.shape[1:])
                        )
                    qpos_batch = obs_dict_batch["qpos"][0]
                    img_batch = obs_dict_batch["curr_image"]

                    single_action = policy(qpos_batch, img_batch)  # (T, state_dim) 类似
                    all_actions = single_action[0].unsqueeze(0)
                    all_time_actions[[t], t : t + query_frequency] = all_actions[:, :query_frequency]

                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)

                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # 非 temporal_agg 的情况（你之前的逻辑）
                    if t % query_frequency == 0:
                        B = obs_dict["qpos"].shape[0]
                        obs_dict_batch = {}
                        for key, value in obs_dict.items():
                            obs_dict_batch[key] = (
                                value.unsqueeze(1)
                                .repeat(1, 1, *([1] * (value.dim() - 1)))
                                .view(B * 1, *value.shape[1:])
                            )
                        qpos_batch = obs_dict_batch["qpos"][0]
                        img_batch = obs_dict_batch["curr_image"]
                        single_action = policy(qpos_batch, img_batch)
                        all_actions = single_action[0].unsqueeze(0)
                    raw_action = all_actions[:, t % query_frequency]

                # -------- 后处理动作 & step 环境 --------
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = torch.tensor(action, device=args_cli.device).unsqueeze(0)

                # step
                env.step(action.to(torch.float32))

                # -------- 计算验证奖励 & 记录各分量 --------
                _ = env.unwrapped.compute_validation_reward()
                log_dict = env.unwrapped.extras.get("val_reward_log", {})

                for k in val_reward_keys:
                    if k in log_dict:
                        episode_val_rewards[k].append(log_dict[k].item())
                    else:
                        episode_val_rewards[k].append(0.0)

                # -------- 成功检测 / 手动 reset --------
                success = check_object_condition(needle, device=args_cli.device)
                if success:
                    success_cnt += 1
                    print(f"Episode {total_cnt}: SUCCESS")
                    break

                t += 1

                if reset_requested:
                    print("检测到 'r' 键，提前结束 episode")
                    break

            # ==========================
            # episode 结束：打印成功率 & 画奖励曲线
            # ==========================
            success_rate = success_cnt / total_cnt
            print(f"当前成功率：{success_rate:.2%}（{success_cnt}/{total_cnt}）")

            # ---- 画 reward 曲线 ----
            timesteps = range(len(next(iter(episode_val_rewards.values()))))
            # 你可以按需要修改想看的 key 列表
            plot_keys = [
                "val/total",
                "val/reach_pipe",
                "val/grasp_reach_3d",
                "val/entry",
                "val/center_pipe",
                "val/pre_grasp_open",
                "val/grasp_close",
                "val/lift",
                "val/move_out",
                "val/success",
                "val/ee_obj_dist",
                "val/pipe_s_ee",
                "val/pipe_r_ee",
                "val/pipe_s_obj",
                "val/pipe_r_obj",
                "val/r_norm",
                "val/object_lift_h",
                "val/dist_xy",
                "val/grip_norm_mean",
                "val/near_for_grasp",
                "val/middle_near",
                "val/in_pipe_ratio",
                "val/outside_ratio",
                "val/success_rate",
            ]
            n_plots = len(plot_keys)

            if n_plots > 0 and len(timesteps) > 0:
                fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.5 * n_plots), sharex=True)
                if n_plots == 1:
                    axes = [axes]

                for i, k in enumerate(plot_keys):
                    axes[i].plot(timesteps, episode_val_rewards[k], label=k)
                    axes[i].set_ylabel(k)
                    axes[i].legend(loc="upper right")

                axes[-1].set_xlabel("timestep")
                fig.suptitle(f"Validation rewards (episode {total_cnt})")
                fig.tight_layout()
                plt.savefig(os.path.join(save_dir, f"val_rewards_ep_{total_cnt:04d}.png"))
                plt.close(fig)

            # 如果需要，你可以在这里顺便保存 frames 做视频，这里就先不展开

    # ------------------ 清理 ------------------
    env.close()
    simulation_app.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
