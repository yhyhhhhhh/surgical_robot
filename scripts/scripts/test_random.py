import cv2
import argparse
from omni.isaac.lab.app import AppLauncher
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--episodes", type=int, default=20, help="number of random episodes")
parser.add_argument("--max-steps", type=int, default=400, help="max steps per episode")
parser.add_argument("--bins", type=int, default=40, help="bins for coverage histogram")
parser.add_argument("--plot-path", type=str, default="random_action_coverage.png", help="path to save coverage plot")
parser.add_argument("--show-plot", action="store_true", help="show matplotlib window")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--pipe-pos", type=float, nargs=3, default=(0.0, -0.29, -0.25), help="pipe center position (x y z)")
parser.add_argument("--pipe-size", type=float, nargs=3, default=(0.0149, 0.0150, 0.0320), help="pipe size (sx sy sz)")
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
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
from einops import rearrange
import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
import ur3_lite


def _to_bool_done(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, np.ndarray):
        return bool(np.any(x))
    if torch.is_tensor(x):
        return bool(torch.any(x).item())
    return bool(x)


def _plot_pipe_cylinder(ax, pipe_pos, pipe_size):
    cx, cy, cz = [float(v) for v in pipe_pos]
    sx, sy, sz = [float(v) for v in pipe_size]

    radius = 0.25 * (sx + sy)
    z_min = cz - 0.5 * sz
    z_max = cz + 0.5 * sz

    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    z_lin = np.linspace(z_min, z_max, 28)
    theta_grid, z_grid = np.meshgrid(theta, z_lin)

    x_grid = cx + radius * np.cos(theta_grid)
    y_grid = cy + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.22, linewidth=0.0, color="tab:blue", shade=False)

    x_top = cx + radius * np.cos(theta)
    y_top = cy + radius * np.sin(theta)
    ax.plot(x_top, y_top, np.full_like(theta, z_max), color="tab:blue", linewidth=1.2)
    ax.plot(x_top, y_top, np.full_like(theta, z_min), color="tab:blue", linewidth=1.2)


def _visualize_coverage(ee_world_xyz, pipe_srt, bins: int, save_path: str, show_plot: bool, pipe_pos, pipe_size):
    if len(ee_world_xyz) == 0:
        print("No samples collected, skip plotting.")
        return

    xyz = np.asarray(ee_world_xyz, dtype=np.float32)
    has_pipe = len(pipe_srt) > 0
    srt = np.asarray(pipe_srt, dtype=np.float32) if has_pipe else None

    if has_pipe:
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    else:
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(1, 1, 1, projection="3d")

    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=4, alpha=0.45)
    _plot_pipe_cylinder(ax1, pipe_pos=pipe_pos, pipe_size=pipe_size)
    ax1.set_title("EE Position in World")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_box_aspect([1, 1, 1])

    if has_pipe:
        s = srt[:, 0]
        r = srt[:, 1]
        th = srt[:, 2]

        ax2 = fig.add_subplot(1, 3, 2)
        sc = ax2.scatter(s, r, c=th, s=5, alpha=0.6, cmap="turbo")
        ax2.set_title("Pipe Coord Samples (s-r)")
        ax2.set_xlabel("s")
        ax2.set_ylabel("r")
        plt.colorbar(sc, ax=ax2, label="theta")

        ax3 = fig.add_subplot(1, 3, 3)
        h, s_edges, r_edges, im = ax3.hist2d(s, r, bins=bins, cmap="magma")
        ax3.set_title("Coverage Heatmap (s-r)")
        ax3.set_xlabel("s")
        ax3.set_ylabel("r")
        plt.colorbar(im, ax=ax3, label="visit count")

        occupied = float(np.sum(h > 0))
        total = float(h.size)
        occupancy_ratio = occupied / max(total, 1.0)
        th_bins = min(bins, 72)
        th_hist, _ = np.histogram(th, bins=th_bins, range=(-np.pi, np.pi))
        th_coverage = float(np.sum(th_hist > 0)) / float(th_bins)
        print(f"[Coverage] s-r occupancy: {occupancy_ratio:.4f} ({int(occupied)}/{int(total)})")
        print(f"[Coverage] theta occupancy: {th_coverage:.4f} ({int(np.sum(th_hist > 0))}/{th_bins})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"Saved coverage figure to: {save_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    num_envs = 1
    env_cfg = parse_env_cfg(
        "My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )

    env = gym.make("My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0", cfg=env_cfg)
    robot_env = env.env
    env.action_space.seed(args_cli.seed)

    ee_world_xyz = []
    pipe_srt = []

    # ---------------------------------------------------------
    # 随机探索并记录轨迹
    # ---------------------------------------------------------
    for ep in range(args_cli.episodes):
        if not simulation_app.is_running():
            break

        obs, info = env.reset()

        done = False
        step = 0
        while not done and step < args_cli.max_steps and simulation_app.is_running():
            action = env.action_space.sample()
            action = torch.as_tensor(action, dtype=torch.float32, device=args_cli.device)
            if action.ndim == 1:
                action = action.unsqueeze(0)

            obs, reward, terminated, truncated, info = env.step(action)

            done = _to_bool_done(terminated) or _to_bool_done(truncated)
            step += 1

            ee_pos_w = robot_env.get_ee_pos_w()
            ee_world_xyz.append(ee_pos_w[0, :3].detach().cpu().numpy())

            if hasattr(robot_env, "_world_to_pipe_coords"):
                s_cur, r_cur, th_cur, _, _ = robot_env._world_to_pipe_coords(ee_pos_w)
                pipe_srt.append(
                    np.array(
                        [
                            float(s_cur[0].item()),
                            float(r_cur[0].item()),
                            float(th_cur[0].item()),
                        ],
                        dtype=np.float32,
                    )
                )

        print(f"Episode {ep + 1}/{args_cli.episodes}, steps={step}, done={done}")

    _visualize_coverage(
        ee_world_xyz=ee_world_xyz,
        pipe_srt=pipe_srt,
        bins=args_cli.bins,
        save_path=args_cli.plot_path,
        show_plot=args_cli.show_plot,
        pipe_pos=args_cli.pipe_pos,
        pipe_size=args_cli.pipe_size,
    )

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
