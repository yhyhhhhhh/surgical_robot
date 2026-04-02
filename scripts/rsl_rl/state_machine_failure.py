import argparse
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Union

import numpy as np
import torch

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip

# =========================
# 0) CLI
# =========================
parser = argparse.ArgumentParser(description="Collect dangerous trajectories with an FSM.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Ur3Lite-PipeRelCamFinalGoalForce-Ik-RL-Direct-v0",
    help="Name of the task.",
)
parser.add_argument("--save_dir", type=str, default="./danger_dataset_npz", help="Directory to save npz episodes.")
parser.add_argument("--max_trajs", type=int, default=500, help="Stop after collecting this many sub-trajectories.")
parser.add_argument("--print_every", type=int, default=100, help="Print status every N env steps.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for dangerous mode sampling.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================
# 1) Imports after app
# =========================
import gymnasium as gym

import my_ur3_project.tasks  # noqa: F401
import ur3_lite  # noqa: F401
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

enable_extension("omni.isaac.debug_draw")


# =========================
# 2) Dangerous modes / FSM states
# =========================
class DangerMode(IntEnum):
    RADIAL_WALL = 0        # 径向逼壁
    DEEP_INSERT = 1        # 过深插入
    YAW_SCRAPE = 2         # yaw 失配推进
    THETA_SWEEP_WALL = 3   # 近壁时绕截面扫动


class CollectState(IntEnum):
    PREPARE = 0
    SETUP = 1
    APPROACH_BOUNDARY = 2
    HOLD_BOUNDARY = 3
    PUSH_VIOLATE = 4
    HOLD_VIOLATE = 5
    RECOVER = 6


# =========================
# 3) Config
# =========================
@dataclass
class DangerFSMCfg:
    # ---- 继承你安全代码的核心阈值风格 ----
    s_prepare: float = -0.004
    s_recover: float = -0.006
    s_max_insert: float = 0.036

    r_center_prepare: float = 0.0015
    r_center_strict: float = 0.0010

    margin_soft: float = 0.0015
    margin_hard: float = 0.0008

    # ---- 危险采集专用阈值 ----
    boundary_margin_hi: float = 0.00135   # near-boundary 上界
    boundary_margin_lo: float = 0.00095   # near-boundary 下界
    margin_violate: float = 0.00060       # 认为明显越界/危险

    s_setup_min: float = 0.004
    s_setup_max: float = 0.012

    s_boundary_deep: float = 0.028        # deep insert 的 near-boundary
    s_violate_deep: float = 0.036         # deep insert 的 violation

    setup_tol_s: float = 0.0015
    setup_tol_r: float = 0.0012
    setup_tol_th: float = 0.20

    prepare_tol_s: float = 0.0010
    prepare_tol_r: float = 0.0015

    hold_boundary_steps: int = 6
    hold_violate_steps: int = 5
    max_recover_steps: int = 30
    yaw_prestep_steps: int = 6

    # ---- 动作限幅 ----
    action_max_outside = (0.30, 0.25, 0.12, 0.08)
    action_max_probe = (0.18, 0.16, 0.10, 0.08)
    action_max_recover = (0.20, 0.15, 0.06, 0.05)

    # ---- mode sampling weights ----
    mode_probs = (0.35, 0.30, 0.20, 0.15)

    # ---- 是否在恢复期打开夹爪 ----
    open_gripper_in_recover: bool = True


# =========================
# 4) observation layout
# 你安全代码里是: policy_dim = 24 + 2 * num_joints
# =========================
@dataclass
class ObsLayout:
    num_joints: int

    @staticmethod
    def infer_from_policy_dim(policy_dim: int) -> "ObsLayout":
        if (policy_dim - 24) % 2 != 0:
            raise ValueError(
                f"Cannot infer num_joints from policy_dim={policy_dim}. "
                f"Expected policy_dim = 24 + 2 * num_joints."
            )
        return ObsLayout(num_joints=(policy_dim - 24) // 2)


# =========================
# 5) Small writer
# =========================
class DangerTrajWriter:
    def __init__(self, save_dir: str, num_envs: int):
        self.save_dir = save_dir
        self.num_envs = num_envs
        os.makedirs(save_dir, exist_ok=True)
        self.buffers = [self._new_buffer() for _ in range(num_envs)]
        self.total_saved = 0

    def _new_buffer(self):
        return {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "done": [],
            "fsm_state": [],
            "danger_mode": [],
            "traj_id": [],
            "region_label": [],
        }

    def append_batch(
        self,
        obs_np: np.ndarray,
        act_np: np.ndarray,
        rew_np: np.ndarray,
        next_obs_np: np.ndarray,
        done_np: np.ndarray,
        state_np: np.ndarray,
        mode_np: np.ndarray,
        traj_id_np: np.ndarray,
        region_np: np.ndarray,
    ):
        for i in range(self.num_envs):
            buf = self.buffers[i]
            buf["obs"].append(obs_np[i].copy())
            buf["action"].append(act_np[i].copy())
            buf["reward"].append(np.array(rew_np[i], dtype=np.float32))
            buf["next_obs"].append(next_obs_np[i].copy())
            buf["done"].append(np.array(done_np[i], dtype=np.bool_))
            buf["fsm_state"].append(np.array(state_np[i], dtype=np.int64))
            buf["danger_mode"].append(np.array(mode_np[i], dtype=np.int64))
            buf["traj_id"].append(np.array(traj_id_np[i], dtype=np.int64))
            buf["region_label"].append(np.array(region_np[i], dtype=np.int64))

    def finalize_mask(self, mask: np.ndarray, reason: str):
        idxs = np.where(mask)[0].tolist()
        for i in idxs:
            buf = self.buffers[i]
            if len(buf["obs"]) == 0:
                self.buffers[i] = self._new_buffer()
                continue

            traj_id = int(buf["traj_id"][-1])
            mode_id = int(buf["danger_mode"][-1])
            mode_name = DangerMode(mode_id).name.lower()
            stamp = int(time.time() * 1000)

            path = os.path.join(
                self.save_dir,
                f"traj_{traj_id:06d}_{mode_name}_{reason}_{stamp}.npz",
            )
            np.savez_compressed(
                path,
                obs=np.asarray(buf["obs"], dtype=np.float32),
                action=np.asarray(buf["action"], dtype=np.float32),
                reward=np.asarray(buf["reward"], dtype=np.float32),
                next_obs=np.asarray(buf["next_obs"], dtype=np.float32),
                done=np.asarray(buf["done"], dtype=np.bool_),
                fsm_state=np.asarray(buf["fsm_state"], dtype=np.int64),
                danger_mode=np.asarray(buf["danger_mode"], dtype=np.int64),
                traj_id=np.asarray(buf["traj_id"], dtype=np.int64),
                region_label=np.asarray(buf["region_label"], dtype=np.int64),
            )
            self.total_saved += 1
            self.buffers[i] = self._new_buffer()


# =========================
# 6) FSM
# =========================
class DangerousCollectorFSM:
    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device],
        obs_example,
        cfg: DangerFSMCfg = DangerFSMCfg(),
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg

        policy = self._get_policy_tensor(obs_example)
        self.layout = ObsLayout.infer_from_policy_dim(policy.shape[-1])

        self.state = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.traj_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.setup_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.boundary_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.violate_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.recover_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.target_s_setup = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.target_theta = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.yaw_sign = torch.ones(num_envs, dtype=torch.float32, device=self.device)
        self.theta_sign = torch.ones(num_envs, dtype=torch.float32, device=self.device)

        self.just_finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self.max_outside = torch.tensor(cfg.action_max_outside, device=self.device, dtype=torch.float32)
        self.max_probe = torch.tensor(cfg.action_max_probe, device=self.device, dtype=torch.float32)
        self.max_recover = torch.tensor(cfg.action_max_recover, device=self.device, dtype=torch.float32)

        self.reset()

    # ---------- public ----------
    def reset(self, done_mask: torch.Tensor = None):
        if done_mask is None:
            done_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            done_mask = done_mask.to(self.device).bool().view(-1)
        self._resample(done_mask)

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        self.just_finished.zero_()
        o = self._parse_obs(obs)
        self._transition(o)
        actions = self._compute_actions(o)
        return actions.clamp(-1.0, 1.0)

    def pop_finished_mask(self) -> torch.Tensor:
        return self.just_finished.clone()

    def state_name(self, env_id: int = 0) -> str:
        return CollectState(int(self.state[env_id].item())).name

    def mode_name(self, env_id: int = 0) -> str:
        return DangerMode(int(self.mode[env_id].item())).name

    # ---------- internals ----------
    def _resample(self, mask: torch.Tensor):
        if not mask.any():
            return

        n = int(mask.sum().item())
        idx = torch.where(mask)[0]

        probs = torch.tensor(self.cfg.mode_probs, dtype=torch.float32, device=self.device)
        probs = probs / probs.sum()
        sampled = torch.multinomial(probs, n, replacement=True)

        self.mode[idx] = sampled.long()
        self.state[idx] = int(CollectState.PREPARE)

        self.setup_hold[idx] = 0
        self.boundary_hold[idx] = 0
        self.violate_hold[idx] = 0
        self.recover_hold[idx] = 0

        self.traj_id[idx] += 1

        self.target_s_setup[idx] = self._rand_uniform(
            n, self.cfg.s_setup_min, self.cfg.s_setup_max
        )
        self.target_theta[idx] = self._rand_uniform(n, -np.pi, np.pi)
        self.yaw_sign[idx] = torch.where(
            torch.rand(n, device=self.device) > 0.5,
            torch.ones(n, device=self.device),
            -torch.ones(n, device=self.device),
        )
        self.theta_sign[idx] = torch.where(
            torch.rand(n, device=self.device) > 0.5,
            torch.ones(n, device=self.device),
            -torch.ones(n, device=self.device),
        )

    def _rand_uniform(self, n: int, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, device=self.device)

    def _get_policy_tensor(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _parse_obs(self, obs) -> Dict[str, torch.Tensor]:
        x = self._get_policy_tensor(obs).to(self.device)
        J = self.layout.num_joints

        s_e = x[:, 0]
        r_e = x[:, 1]
        cos_th_e = x[:, 2]
        sin_th_e = x[:, 3]
        theta_e = torch.atan2(sin_th_e, cos_th_e)

        grip_norm = x[:, 4 + 2 * J]

        ds = x[:, 5 + 2 * J]
        dr = x[:, 6 + 2 * J]
        cos_dth = x[:, 7 + 2 * J]
        sin_dth = x[:, 8 + 2 * J]
        cos_dyaw = x[:, 9 + 2 * J]
        sin_dyaw = x[:, 10 + 2 * J]

        object_lift = x[:, 11 + 2 * J]
        g_lat = x[:, 12 + 2 * J]
        g_close = x[:, 13 + 2 * J]
        is_captured = x[:, 14 + 2 * J] > 0.5

        goal_vec_obj = x[:, 15 + 2 * J : 18 + 2 * J]
        goal_vec_ee = x[:, 18 + 2 * J : 21 + 2 * J]
        obj_goal_dist = x[:, 21 + 2 * J]
        ee_goal_dist = x[:, 22 + 2 * J]
        margin_to_wall = x[:, 23 + 2 * J]

        dth = torch.atan2(sin_dth, cos_dth)
        dyaw = torch.atan2(sin_dyaw, cos_dyaw)

        return {
            "s_e": s_e,
            "r_e": r_e,
            "theta_e": theta_e,
            "grip_norm": grip_norm,
            "ds": ds,
            "dr": dr,
            "dth": dth,
            "dyaw": dyaw,
            "object_lift": object_lift,
            "g_lat": g_lat,
            "g_close": g_close,
            "is_captured": is_captured,
            "goal_vec_obj": goal_vec_obj,
            "goal_vec_ee": goal_vec_ee,
            "obj_goal_dist": obj_goal_dist,
            "ee_goal_dist": ee_goal_dist,
            "margin_to_wall": margin_to_wall,
        }

    def region_label_from_obs(self, obs) -> torch.Tensor:
        o = self._parse_obs(obs)
        return self._region_label(o)

    def _region_label(self, o: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 0 = safe, 1 = near-boundary, 2 = violation
        safe = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        near = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        viol = 2 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        is_violate = self._is_violation(o)
        is_boundary = self._is_boundary(o)

        out = safe
        out = torch.where(is_boundary, near, out)
        out = torch.where(is_violate, viol, out)
        return out

    def _is_boundary(self, o: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.mode
        is_deep = mode == int(DangerMode.DEEP_INSERT)
        by_margin = (
            (o["margin_to_wall"] <= self.cfg.boundary_margin_hi)
            & (o["margin_to_wall"] >= self.cfg.boundary_margin_lo)
        )
        by_deep = o["s_e"] >= self.cfg.s_boundary_deep
        return torch.where(is_deep, by_deep, by_margin)

    def _is_violation(self, o: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.mode
        is_deep = mode == int(DangerMode.DEEP_INSERT)
        by_margin = o["margin_to_wall"] <= self.cfg.margin_violate
        by_deep = o["s_e"] >= self.cfg.s_violate_deep
        return torch.where(is_deep, by_deep, by_margin | (o["s_e"] > self.cfg.s_max_insert))

    @staticmethod
    def _scaled(err: torch.Tensor, tol: float, max_abs: float) -> torch.Tensor:
        tol = max(tol, 1e-6)
        return torch.clamp(err / tol * max_abs, -max_abs, max_abs)

    @staticmethod
    def _angle_diff(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(target - current), torch.cos(target - current))

    def _set_state(self, mask: torch.Tensor, new_state: CollectState):
        self.state[mask] = int(new_state)

    def _transition(self, o: Dict[str, torch.Tensor]):
        cfg = self.cfg
        state = self.state

        # PREPARE -> SETUP
        m = state == int(CollectState.PREPARE)
        ready = (
            (torch.abs(o["s_e"] - cfg.s_prepare) < cfg.prepare_tol_s)
            & (o["r_e"] < cfg.r_center_prepare)
        )
        self._set_state(m & ready, CollectState.SETUP)

        # SETUP
        m = self.state == int(CollectState.SETUP)
        self.setup_hold = torch.where(m, self.setup_hold + 1, torch.zeros_like(self.setup_hold))

        mode = self.mode
        common_ready = (
            (torch.abs(o["s_e"] - self.target_s_setup) < cfg.setup_tol_s)
            & (o["r_e"] < cfg.r_center_strict)
        )

        theta_ready = torch.abs(self._angle_diff(self.target_theta, o["theta_e"])) < cfg.setup_tol_th

        ready_radial = common_ready
        ready_deep = common_ready
        ready_yaw = common_ready & (self.setup_hold >= cfg.yaw_prestep_steps)
        ready_theta = common_ready & theta_ready

        setup_ready = torch.zeros_like(common_ready)
        setup_ready = torch.where(mode == int(DangerMode.RADIAL_WALL), ready_radial, setup_ready)
        setup_ready = torch.where(mode == int(DangerMode.DEEP_INSERT), ready_deep, setup_ready)
        setup_ready = torch.where(mode == int(DangerMode.YAW_SCRAPE), ready_yaw, setup_ready)
        setup_ready = torch.where(mode == int(DangerMode.THETA_SWEEP_WALL), ready_theta, setup_ready)

        self._set_state(m & setup_ready, CollectState.APPROACH_BOUNDARY)

        # APPROACH_BOUNDARY -> HOLD_BOUNDARY
        m = self.state == int(CollectState.APPROACH_BOUNDARY)
        reach_boundary = self._is_boundary(o)
        self._set_state(m & reach_boundary, CollectState.HOLD_BOUNDARY)

        # 如果过头了，直接进入 HOLD_VIOLATE，保证还能采到 violation 窗口
        overshoot = self._is_violation(o)
        self._set_state(m & overshoot, CollectState.HOLD_VIOLATE)

        # HOLD_BOUNDARY -> PUSH_VIOLATE
        m = self.state == int(CollectState.HOLD_BOUNDARY)
        self.boundary_hold = torch.where(
            m, self.boundary_hold + 1, torch.zeros_like(self.boundary_hold)
        )
        self._set_state(m & (self.boundary_hold >= cfg.hold_boundary_steps), CollectState.PUSH_VIOLATE)

        # PUSH_VIOLATE -> HOLD_VIOLATE
        m = self.state == int(CollectState.PUSH_VIOLATE)
        reach_violate = self._is_violation(o)
        self._set_state(m & reach_violate, CollectState.HOLD_VIOLATE)

        # 极端情况：太深/太靠壁，直接也进入 HOLD_VIOLATE
        extreme = (o["margin_to_wall"] < cfg.margin_hard) | (o["s_e"] > cfg.s_max_insert)
        self._set_state(m & extreme, CollectState.HOLD_VIOLATE)

        # HOLD_VIOLATE -> RECOVER
        m = self.state == int(CollectState.HOLD_VIOLATE)
        self.violate_hold = torch.where(
            m, self.violate_hold + 1, torch.zeros_like(self.violate_hold)
        )
        self._set_state(m & (self.violate_hold >= cfg.hold_violate_steps), CollectState.RECOVER)

        # RECOVER -> new trajectory
        m = self.state == int(CollectState.RECOVER)
        self.recover_hold = torch.where(
            m, self.recover_hold + 1, torch.zeros_like(self.recover_hold)
        )

        recover_ok = (
            (o["s_e"] < cfg.s_prepare + cfg.prepare_tol_s)
            & (o["r_e"] < cfg.r_center_prepare)
            & (o["margin_to_wall"] > cfg.margin_soft)
        )

        finished = m & (recover_ok | (self.recover_hold >= cfg.max_recover_steps))
        if finished.any():
            self.just_finished[finished] = True
            self._resample(finished)

    def _compute_actions(self, o: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        a = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float32)

        mode = self.mode

        # ----------------------------------------------------------
        # PREPARE: 回到安全起点
        # ----------------------------------------------------------
        m = self.state == int(CollectState.PREPARE)
        if m.any():
            a[m, 0] = self._scaled(cfg.s_prepare - o["s_e"][m], 0.004, float(self.max_outside[0]))
            a[m, 1] = self._scaled(-o["r_e"][m], 0.002, float(self.max_outside[1]))
            a[m, 4] = -1.0

        # ----------------------------------------------------------
        # SETUP
        # ----------------------------------------------------------
        m = self.state == int(CollectState.SETUP)
        if m.any():
            # 通用：先到一个危险采样前的 setup 位形
            a[m, 0] = self._scaled(
                self.target_s_setup[m] - o["s_e"][m],
                cfg.setup_tol_s,
                float(self.max_probe[0]),
            )
            a[m, 1] = self._scaled(-o["r_e"][m], cfg.setup_tol_r, float(self.max_probe[1]))
            a[m, 4] = -1.0

            # YAW_SCRAPE: SETUP 阶段先打 yaw 偏置
            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 3] = 0.55 * self.yaw_sign[my] * float(self.max_probe[3])

            # THETA_SWEEP_WALL: 先把截面角走到某个目标方位
            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                th_err = self._angle_diff(self.target_theta[mt], o["theta_e"][mt])
                a[mt, 2] = self._scaled(th_err, 0.25, float(self.max_probe[2]))

        # ----------------------------------------------------------
        # APPROACH_BOUNDARY
        # ----------------------------------------------------------
        m = self.state == int(CollectState.APPROACH_BOUNDARY)
        if m.any():
            # RADIAL_WALL: 保持 s，向壁面推
            mr = m & (mode == int(DangerMode.RADIAL_WALL))
            if mr.any():
                a[mr, 0] = self._scaled(
                    self.target_s_setup[mr] - o["s_e"][mr], 0.004, 0.6 * float(self.max_probe[0])
                )
                a[mr, 1] = +0.85 * float(self.max_probe[1])
                a[mr, 4] = -1.0

            # DEEP_INSERT: 居中继续深插
            md = m & (mode == int(DangerMode.DEEP_INSERT))
            if md.any():
                a[md, 0] = +0.85 * float(self.max_probe[0])
                a[md, 1] = self._scaled(-o["r_e"][md], 0.002, 0.6 * float(self.max_probe[1]))
                a[md, 4] = -1.0

            # YAW_SCRAPE: 保持 yaw 偏置，同时前插+略微外推
            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 0] = +0.70 * float(self.max_probe[0])
                a[my, 1] = +0.45 * float(self.max_probe[1])
                a[my, 3] = +0.80 * self.yaw_sign[my] * float(self.max_probe[3])
                a[my, 4] = -1.0

            # THETA_SWEEP_WALL: 轻微前插 + 外推 + 截面扫动
            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                a[mt, 0] = +0.35 * float(self.max_probe[0])
                a[mt, 1] = +0.60 * float(self.max_probe[1])
                a[mt, 2] = +0.80 * self.theta_sign[mt] * float(self.max_probe[2])
                a[mt, 4] = -1.0

        # ----------------------------------------------------------
        # HOLD_BOUNDARY
        # ----------------------------------------------------------
        m = self.state == int(CollectState.HOLD_BOUNDARY)
        if m.any():
            a[m, 4] = -1.0

            # 在边界附近保持，但不同 mode 仍保留一点特征动作
            mr = m & (mode == int(DangerMode.RADIAL_WALL))
            if mr.any():
                a[mr, 0] = self._scaled(
                    self.target_s_setup[mr] - o["s_e"][mr], 0.004, 0.3 * float(self.max_probe[0])
                )
                a[mr, 1] = +0.08 * float(self.max_probe[1])

            md = m & (mode == int(DangerMode.DEEP_INSERT))
            if md.any():
                a[md, 1] = self._scaled(-o["r_e"][md], 0.002, 0.25 * float(self.max_probe[1]))

            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 0] = +0.10 * float(self.max_probe[0])
                a[my, 1] = +0.08 * float(self.max_probe[1])
                a[my, 3] = +0.60 * self.yaw_sign[my] * float(self.max_probe[3])

            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                a[mt, 2] = +0.50 * self.theta_sign[mt] * float(self.max_probe[2])
                a[mt, 1] = +0.06 * float(self.max_probe[1])

        # ----------------------------------------------------------
        # PUSH_VIOLATE
        # ----------------------------------------------------------
        m = self.state == int(CollectState.PUSH_VIOLATE)
        if m.any():
            a[m, 4] = -1.0

            mr = m & (mode == int(DangerMode.RADIAL_WALL))
            if mr.any():
                a[mr, 1] = +1.00 * float(self.max_probe[1])

            md = m & (mode == int(DangerMode.DEEP_INSERT))
            if md.any():
                a[md, 0] = +1.00 * float(self.max_probe[0])

            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 0] = +0.90 * float(self.max_probe[0])
                a[my, 1] = +0.75 * float(self.max_probe[1])
                a[my, 3] = +1.00 * self.yaw_sign[my] * float(self.max_probe[3])

            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                a[mt, 1] = +0.70 * float(self.max_probe[1])
                a[mt, 2] = +1.00 * self.theta_sign[mt] * float(self.max_probe[2])

        # ----------------------------------------------------------
        # HOLD_VIOLATE
        # ----------------------------------------------------------
        m = self.state == int(CollectState.HOLD_VIOLATE)
        if m.any():
            a[m, 4] = -1.0

            # 保留很小动作，不要立刻松掉
            mr = m & (mode == int(DangerMode.RADIAL_WALL))
            if mr.any():
                a[mr, 1] = +0.08 * float(self.max_probe[1])

            md = m & (mode == int(DangerMode.DEEP_INSERT))
            if md.any():
                a[md, 0] = +0.08 * float(self.max_probe[0])

            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 3] = +0.35 * self.yaw_sign[my] * float(self.max_probe[3])

            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                a[mt, 2] = +0.25 * self.theta_sign[mt] * float(self.max_probe[2])

        # ----------------------------------------------------------
        # RECOVER
        # ----------------------------------------------------------
        m = self.state == int(CollectState.RECOVER)
        if m.any():
            a[m, 0] = self._scaled(cfg.s_recover - o["s_e"][m], 0.008, float(self.max_recover[0]))
            a[m, 1] = self._scaled(-o["r_e"][m], 0.002, float(self.max_recover[1]))

            # theta / yaw 轻微往回收
            mt = m & (mode == int(DangerMode.THETA_SWEEP_WALL))
            if mt.any():
                th_err = self._angle_diff(torch.zeros_like(o["theta_e"][mt]), o["theta_e"][mt])
                a[mt, 2] = self._scaled(th_err, 0.30, 0.5 * float(self.max_recover[2]))

            my = m & (mode == int(DangerMode.YAW_SCRAPE))
            if my.any():
                a[my, 3] = -0.60 * self.yaw_sign[my] * float(self.max_recover[3])

            a[m, 4] = -1.0 if cfg.open_gripper_in_recover else 0.0

        return a


# =========================
# 7) Main
# =========================
def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    device = env.unwrapped.device
    obs, _ = env.get_observations()

    fsm = DangerousCollectorFSM(
        num_envs=args_cli.num_envs,
        device=device,
        obs_example=obs,
        cfg=DangerFSMCfg(),
    )
    writer = DangerTrajWriter(args_cli.save_dir, args_cli.num_envs)

    timestep = 0
    print(f"[INFO] save_dir = {args_cli.save_dir}")
    print("[INFO] Start collecting dangerous trajectories...")

    while simulation_app.is_running() and writer.total_saved < args_cli.max_trajs:
        with torch.inference_mode():
            obs_before = fsm._get_policy_tensor(obs).detach().cpu().numpy()
            state_before = fsm.state.detach().cpu().numpy()
            mode_before = fsm.mode.detach().cpu().numpy()
            traj_before = fsm.traj_id.detach().cpu().numpy()
            region_before = fsm.region_label_from_obs(obs).detach().cpu().numpy()

            actions = fsm.act(obs)

            step_out = env.step(actions)
            if len(step_out) == 4:
                obs, rew, dones, infos = step_out
            else:
                raise RuntimeError("Unexpected env.step(...) output format.")

            next_obs_np = fsm._get_policy_tensor(obs).detach().cpu().numpy()
            act_np = actions.detach().cpu().numpy()
            rew_np = rew.detach().cpu().numpy() if isinstance(rew, torch.Tensor) else np.asarray(rew)
            done_np = dones.detach().cpu().numpy() if isinstance(dones, torch.Tensor) else np.asarray(dones)

            writer.append_batch(
                obs_np=obs_before,
                act_np=act_np,
                rew_np=rew_np,
                next_obs_np=next_obs_np,
                done_np=done_np,
                state_np=state_before,
                mode_np=mode_before,
                traj_id_np=traj_before,
                region_np=region_before,
            )

            # 1) FSM 内部完成一条危险子轨迹
            finished_mask = fsm.pop_finished_mask().detach().cpu().numpy().astype(bool)
            if finished_mask.any():
                writer.finalize_mask(finished_mask, reason="fsm_done")

            # 2) 环境自身 done
            if isinstance(dones, torch.Tensor):
                done_mask = dones.detach().cpu().numpy().astype(bool)
                if done_mask.any():
                    writer.finalize_mask(done_mask, reason="env_done")
                    fsm.reset(dones)

            if timestep % args_cli.print_every == 0:
                print(
                    f"[t={timestep}] saved={writer.total_saved} "
                    f"state[0]={fsm.state_name(0)} mode[0]={fsm.mode_name(0)}"
                )

            timestep += 1

    env.close()
    print(f"[INFO] Finished. Total saved sub-trajectories = {writer.total_saved}")


if __name__ == "__main__":
    main()
    simulation_app.close()