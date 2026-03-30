import argparse
import math
import torch
import gymnasium as gym

from omni.isaac.lab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Open-loop test for UR3 pipe env with OSC backend.")
parser.add_argument("--task", type=str, default="Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0", help="Your registered task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=2000, help="Number of sim steps.")
parser.add_argument("--print_every", type=int, default=20, help="Print interval.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
from omni.isaac.lab_tasks.utils import parse_env_cfg
import ur3_lite

# -----------------------------------------------------------------------------
# Action generator
# -----------------------------------------------------------------------------
def make_test_action(step: int, num_envs: int, device: torch.device) -> torch.Tensor:
    """
    5D action:
        [Δs, Δr, Δθ, Δyaw, gripper]
    Keep it conservative. This is for controller sanity-check, not task completion.
    """
    a = torch.zeros(num_envs, 5, device=device)

    # 900-step cycle
    phase = step % 900

    # 0-99: gripper open, hold
    if phase < 100:
        a[:, 4] = -1.0

    # 100-279: slowly insert along pipe axis
    elif phase < 280:
        a[:, 0] = 0.020   # small positive Δs

    # 280-399: small radial move to one side
    elif phase < 400:
        a[:, 1] = 0.080

    # 400-519: radial move back
    elif phase < 520:
        a[:, 1] = -0.080

    # 520-639: small in-plane rotation
    elif phase < 640:
        a[:, 2] = 0.050

    # 640-739: very small yaw
    elif phase < 740:
        a[:, 3] = 0.010

    # 740-819: close gripper
    elif phase < 820:
        a[:, 4] = 1.0

    # 820-899: retract
    else:
        a[:, 0] = -0.020

    return a


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_contact_peak(sensor) -> torch.Tensor:
    """Return per-env peak contact norm."""
    if sensor is None:
        return None

    if hasattr(sensor.data, "net_forces_w_history") and sensor.data.net_forces_w_history is not None:
        f = sensor.data.net_forces_w_history
    elif hasattr(sensor.data, "net_forces_w") and sensor.data.net_forces_w is not None:
        f = sensor.data.net_forces_w.unsqueeze(1)
    else:
        return None

    # reshape to [N, *, 3]
    f = f.reshape(f.shape[0], -1, 3)
    return torch.linalg.vector_norm(f, dim=-1).max(dim=1).values


def quat_angle_error_rad(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Small helper for orientation error magnitude."""
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.arccos(dot)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # optional: if your env supports it, disable image obs for faster debug
    if hasattr(env_cfg, "use_image_obs"):
        env_cfg.use_image_obs = False
    if hasattr(env_cfg, "debug_vis"):
        env_cfg.debug_vis = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

    obs, info = env.reset()

    print("[INFO] Env created.")
    print(f"[INFO] num_envs = {raw_env.num_envs}")
    print(f"[INFO] device   = {raw_env.device}")

    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:
        actions = make_test_action(step, raw_env.num_envs, raw_env.device)

        obs, rew, terminated, truncated, info = env.step(actions)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        if step % args_cli.print_every == 0:
            ee_state = raw_env._robot.data.body_state_w[:, raw_env.ee_id]
            ee_pos_w = ee_state[:, 0:3]
            ee_quat_w = ee_state[:, 3:7]

            tgt_pos_w = raw_env.ee_target_pos_w
            tgt_quat_w = raw_env.ee_target_quat_w

            pos_err = torch.linalg.vector_norm(ee_pos_w - tgt_pos_w, dim=-1)
            rot_err = quat_angle_error_rad(ee_quat_w, tgt_quat_w)

            # pipe coordinates
            s, r, th, _, _ = raw_env._world_to_pipe_coords(ee_pos_w)
            s = s.squeeze(-1)
            r = r.squeeze(-1)
            th = th.squeeze(-1)

            # effort
            if hasattr(raw_env, "arm_effort_cmd"):
                effort_norm = torch.linalg.vector_norm(raw_env.arm_effort_cmd, dim=-1)
            else:
                effort_norm = None

            # contacts
            pipe_peak = get_contact_peak(getattr(raw_env, "_pipe_contact", None))
            bottom_peak = get_contact_peak(getattr(raw_env, "_bottom_contact", None))

            # only print env0 for readability
            msg = [
                f"[step {step:04d}]",
                f"pos_err={pos_err[0].item():.4f} m",
                f"rot_err={rot_err[0].item():.4f} rad",
                f"s={s[0].item():.4f}",
                f"r={r[0].item():.4f}",
                f"th={th[0].item():.4f}",
                f"grip={raw_env.gripper_cmd[0,0].item():.4f}",
            ]
            if effort_norm is not None:
                msg.append(f"eff={effort_norm[0].item():.4f}")
            if pipe_peak is not None:
                msg.append(f"pipeF={pipe_peak[0].item():.4f}")
            if bottom_peak is not None:
                msg.append(f"bottomF={bottom_peak[0].item():.4f}")

            print(" | ".join(msg))

        # ------------------------------------------------------------------
        # Auto-reset notice
        # ------------------------------------------------------------------
        if torch.is_tensor(terminated):
            done = torch.logical_or(terminated, truncated)
            if done.any():
                done_ids = torch.nonzero(done).squeeze(-1).tolist()
                print(f"[INFO] auto reset env ids: {done_ids}")

        step += 1

    env.close()
    print("[INFO] Test finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()