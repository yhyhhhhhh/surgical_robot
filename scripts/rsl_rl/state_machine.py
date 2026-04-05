import argparse

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric afnd use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0", help="Name of the task.")
parser.add_argument("--success_lift_thr", type=float, default=0.02, help="Lift threshold (m) to count success and reset.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import my_ur3_project.tasks  # noqa: F401
import carb
import omni.appwindow
from omni.isaac.core.utils.extensions import enable_extension

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

enable_extension("omni.isaac.debug_draw")

import ur3_lite  # noqa: F401

import gymnasium as gym
import torch
from omni.isaac.core.utils.extensions import enable_extension

from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

enable_extension("omni.isaac.debug_draw")

import ur3_lite  # noqa: F401

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Union

import torch


# =========================
# 1) 状态定义
# =========================
class FSMState(IntEnum):
    PREPARE = 0          # 回到管口外安全起点，夹爪张开
    APPROACH = 1         # 管口处对中
    INSERT = 2           # 安全入管到 pregrasp
    COARSE_ALIGN = 3     # 粗对齐
    FINE_ALIGN = 4       # 精对齐
    CLOSE = 5            # 静止闭合夹爪
    VERIFY = 6           # 小幅验证是否抓住
    RETRACT = 7          # 安全回撤出管
    TRANSPORT = 8        # 搬运（这里只给占位骨架）
    RELEASE = 9          # 释放
    RECOVER = 10         # 恢复：回撤 + 回中心


# =========================
# 2) 可调参数
# =========================
@dataclass
class FSMCfg:
    # ---- 全局速度倍率 ----
    # >1.0 整体加速，<1.0 整体减速
    global_motion_speed_scale: float = 2.5

    # ---- 位置/阈值 ----
    s_prepare: float = 0.0255        # 比初始位轻退 1.6 mm，作为局部安全准备位
    s_approach: float = 0.0264       # 比初始位轻退 0.7 mm，作为进入对齐区的近端位
    s_recover: float = 0.0238        # 恢复时退 3.3 mm，足够脱离危险但不过分
    s_retract_done: float = 0.0105   # 抓住后进一步回撤，提升“提起来”幅度
    s_retract_radial_enable: float = 0.0080  # RETRACT 先退到该深度，再开启径向归中

    pregrasp_offset: float = 0.0025  # 预抓取时与物体保留 2.5 mm 轴向缓冲
    s_max_insert: float = 0.0335     # 最大安全插入深度，给底部留 6.5 mm 余量

    r_center_prepare: float = 0.0015
    r_center_strict: float = 0.0010

    margin_soft: float = 0.0015        # 靠壁软阈值
    margin_hard: float = 0.0008        # 靠壁硬阈值

    ds_insert_tol: float = 0.0015
    ds_coarse_tol: float = 0.0025
    dr_coarse_tol: float = 0.0010
    dyaw_coarse_tol: float = 0.050
    ds_fine_tol: float = 0.0015
    dr_fine_tol: float = 0.0010
    dyaw_fine_tol: float = 0.15        # rad

    g_lat_coarse: float = 0.75
    g_close_fine: float = 0.70

    lift_verify_thr: float = 0.0015
    goal_thr: float = 0.005

    # ---- 持续计数 ----
    fine_stable_steps: int = 6
    close_steps: int = 300
    verify_steps: int = 8
    release_steps: int = 6
    max_recover_steps: int = 40
    retract_lift_stall_delta_tol: float = 0.00005
    retract_lift_stall_steps: int = 10

    # ---- 动作限幅（action ∈ [-1, 1]）----
    # 顺序: [a0(Δs), a1(Δr), a2(Δθ), a3(Δyaw)]
    action_max_outside = (0.30, 0.25, 0.12, 0.08)
    action_max_insert  = (0.20, 0.15, 0.12, 0.08)
    action_max_coarse  = (0.16, 0.8, 0.8, 0.08)
    action_max_fine    = (0.10, 0.10, 0.08, 0.05)
    action_max_retract = (0.45, 0.15, 0.05, 0.03)
    retract_radial_boost_cmd: float = 1.00   # 抓取后提起阶段，给更强的向中心补偿（a1 下限）
    transport_radial_boost_cmd: float = 0.55
    # ===== TRANSPORT / HOLD =====
    s_transport_target: float = -0.015   # 抓取后提到固定高度；可先试 -1.5cm
    transport_r_tol: float = 0.0020
    transport_s_tol: float = 0.0015
    transport_lift_min: float = 0.0025
    transport_hold_steps: int = 8

# =========================
# 3) observation layout
# policy dim = 24 + 2 * num_joints
# =========================
@dataclass
class ObsLayout:
    num_joints: int

    @staticmethod
    def infer_from_policy_dim(policy_dim: int) -> "ObsLayout":
        # 来自你当前 obs 拼接结构
        # fixed = 23, variable = 2 * num_joints
        if (policy_dim - 23) % 2 != 0:
            raise ValueError(
                f"Cannot infer num_joints from policy_dim={policy_dim}. "
                f"Expected policy_dim = 23 + 2 * num_joints."
            )
        return ObsLayout(num_joints=(policy_dim - 23) // 2)


# =========================
# 4) FSM 主体
# =========================
class SafeGraspFSM:
    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device],
        obs_example,
        cfg: FSMCfg = FSMCfg(),
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg

        policy = self._get_policy_tensor(obs_example)
        self.layout = ObsLayout.infer_from_policy_dim(policy.shape[-1])

        self.state = torch.full(
            (num_envs,),
            int(FSMState.PREPARE),
            dtype=torch.long,
            device=self.device,
        )

        self.fine_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.close_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.verify_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.release_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.recover_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.retract_stall_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.prev_object_lift = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        self.max_outside = torch.tensor(cfg.action_max_outside, device=self.device, dtype=torch.float32)
        self.max_insert = torch.tensor(cfg.action_max_insert, device=self.device, dtype=torch.float32)
        self.max_coarse = torch.tensor(cfg.action_max_coarse, device=self.device, dtype=torch.float32)
        self.max_fine = torch.tensor(cfg.action_max_fine, device=self.device, dtype=torch.float32)
        self.max_retract = torch.tensor(cfg.action_max_retract, device=self.device, dtype=torch.float32)
        self.transport_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    # ---------- 公共接口 ----------
    def reset(self, done_mask: torch.Tensor = None):
        if done_mask is None:
            done_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            done_mask = done_mask.to(self.device).bool().view(-1)

        self.state[done_mask] = int(FSMState.PREPARE)
        self.fine_hold[done_mask] = 0
        self.close_hold[done_mask] = 0
        self.verify_hold[done_mask] = 0
        self.release_hold[done_mask] = 0
        self.recover_hold[done_mask] = 0
        self.retract_stall_hold[done_mask] = 0
        self.prev_object_lift[done_mask] = 0.0
        self.transport_hold[done_mask] = 0

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        o = self._parse_obs(obs)
        self._transition(o)
        actions = self._compute_actions(o)
        # Apply one global speed knob on Cartesian motion channels [a0..a3].
        speed = float(self.cfg.global_motion_speed_scale)
        actions[:, 0:4] = actions[:, 0:4] * speed
        return actions.clamp(-1.0, 1.0)

    def state_name(self, env_id: int = 0) -> str:
        return FSMState(int(self.state[env_id].item())).name

    # ---------- 内部工具 ----------
    def _get_policy_tensor(self, obs):
        # 兼容：
        # 1) obs = dict(policy=...)
        # 2) obs = tensor
        # 3) obs = (obs, extras)
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _parse_obs(self, obs) -> Dict[str, torch.Tensor]:
        x = self._get_policy_tensor(obs).to(self.device)
        J = self.layout.num_joints

        # 索引按你当前 _get_observations 的拼接顺序写
        s_e = x[:, 0]
        r_e = x[:, 1]

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
        # margin_to_wall = x[:, 23 + 2 * J]

        dth = torch.atan2(sin_dth, cos_dth)
        dyaw_raw = torch.atan2(sin_dyaw, cos_dyaw)
        # Square object: treat yaw with 4-way symmetry (k * 90 deg are equivalent).
        dyaw = torch.atan2(torch.sin(4.0 * dyaw_raw), torch.cos(4.0 * dyaw_raw)) / 4.0

        return {
            "s_e": s_e,
            "r_e": r_e,
            "grip_norm": grip_norm,
            "ds": ds,
            "dr": dr,
            "dth": dth,
            "dyaw": dyaw,
            "dyaw_raw": dyaw_raw,
            "object_lift": object_lift,
            "g_lat": g_lat,
            "g_close": g_close,
            "is_captured": is_captured,
            "goal_vec_obj": goal_vec_obj,
            "goal_vec_ee": goal_vec_ee,
            "obj_goal_dist": obj_goal_dist,
            "ee_goal_dist": ee_goal_dist,
            # "margin_to_wall": margin_to_wall,
        }

    @staticmethod
    def _scaled(err: torch.Tensor, tol: float, max_abs: float) -> torch.Tensor:
        tol = max(tol, 1e-6)
        return torch.clamp(err / tol * max_abs, -max_abs, max_abs)

    def _set_state(self, mask: torch.Tensor, new_state: FSMState):
        self.state[mask] = int(new_state)

    def _transition(self, o: Dict[str, torch.Tensor]):
        cfg = self.cfg
        state = self.state

        # ===== 全局进入 RECOVER 的条件 =====
        # hard_recover = (
        #     (o["margin_to_wall"] < cfg.margin_hard)
        #     | (o["s_e"] > cfg.s_max_insert)
        # )
        # self._set_state(hard_recover, FSMState.RECOVER)

        # ===== PREPARE -> APPROACH =====
        m = state == int(FSMState.PREPARE)
        ready = (
            (torch.abs(o["s_e"] - cfg.s_prepare) < 0.001)
            & (o["r_e"] < cfg.r_center_prepare)
        )
        self._set_state(m & ready, FSMState.APPROACH)

        # ===== APPROACH -> INSERT =====
        m = self.state == int(FSMState.APPROACH)
        ready = o["r_e"] < cfg.r_center_strict
        
        self._set_state(m & ready, FSMState.INSERT)

        # ===== INSERT -> COARSE_ALIGN =====
        m = self.state == int(FSMState.INSERT)
        ready = (
            (torch.abs(o["ds"] - cfg.pregrasp_offset) < cfg.ds_insert_tol)
        )
        self._set_state(m & ready, FSMState.COARSE_ALIGN)

        # ===== COARSE_ALIGN -> FINE_ALIGN =====
        m = self.state == int(FSMState.COARSE_ALIGN)
        ready = (
            (torch.abs(o["dr"]) < cfg.dr_coarse_tol)
            & (torch.abs(o["dth"]) < cfg.dyaw_coarse_tol)
        )
        self._set_state(m & ready, FSMState.FINE_ALIGN)

        # ===== FINE_ALIGN -> CLOSE （连续稳定若干步）=====
        m = self.state == int(FSMState.FINE_ALIGN)
        fine_ready = (
            (torch.abs(o["ds"]) < cfg.ds_fine_tol)
            & (torch.abs(o["dr"]) < cfg.dr_fine_tol)
            # & (torch.abs(o["dyaw"]) < cfg.dyaw_fine_tol)
            # & (o["g_close"] > cfg.g_close_fine)
            # & (o["margin_to_wall"] > cfg.margin_soft)
        )
        self.fine_hold = torch.where(
            m,
            torch.where(fine_ready, self.fine_hold + 1, torch.zeros_like(self.fine_hold)),
            torch.zeros_like(self.fine_hold),
        )
        self._set_state(m & (self.fine_hold >= cfg.fine_stable_steps), FSMState.CLOSE)

        # ===== CLOSE -> VERIFY =====
        m = self.state == int(FSMState.CLOSE)
        self.close_hold = torch.where(m, self.close_hold + 1, torch.zeros_like(self.close_hold))
        self._set_state(m & (self.close_hold >= cfg.close_steps), FSMState.VERIFY)
        

        # ===== VERIFY -> RETRACT / RECOVER =====
        m = self.state == int(FSMState.VERIFY)
        self.verify_hold = torch.where(m, self.verify_hold + 1, torch.zeros_like(self.verify_hold))

        grasp_ok = o["is_captured"] | (o["object_lift"] > cfg.lift_verify_thr)
        self._set_state(m & grasp_ok, FSMState.RETRACT)

        verify_fail = m & (~grasp_ok) & (self.verify_hold >= cfg.verify_steps)
        self._set_state(verify_fail, FSMState.RECOVER)

        # ===== RETRACT -> TRANSPORT =====
        m = self.state == int(FSMState.RETRACT)

        # If object lift stalls for too long during retract, go back to FINE_ALIGN.
        lift_delta = torch.abs(o["object_lift"] - self.prev_object_lift)
        lift_stalled = lift_delta < cfg.retract_lift_stall_delta_tol
        self.retract_stall_hold = torch.where(
            m,
            torch.where(lift_stalled, self.retract_stall_hold + 1, torch.zeros_like(self.retract_stall_hold)),
            torch.zeros_like(self.retract_stall_hold),
        )
        stalled_to_realign = m & (self.retract_stall_hold >= cfg.retract_lift_stall_steps)
        self._set_state(stalled_to_realign, FSMState.COARSE_ALIGN)

        # Recompute mask after potential state change above.
        m = self.state == int(FSMState.RETRACT)
        ready = (
            (o["s_e"] < cfg.s_retract_done)
            & (o["object_lift"] > cfg.lift_verify_thr)
        )
        self._set_state(m & ready, FSMState.TRANSPORT)

        # ===== TRANSPORT -> RELEASE(HOLD) =====
        # 现在 RELEASE 不再是真释放，而是“提起到固定高度后的保持状态”
        m = self.state == int(FSMState.TRANSPORT)

        transport_ready = (
            (o["s_e"] <= (cfg.s_transport_target + cfg.transport_s_tol))
            & (o["r_e"] < cfg.transport_r_tol)
            & (o["object_lift"] > cfg.transport_lift_min)
        )

        self.transport_hold = torch.where(
            m,
            torch.where(transport_ready, self.transport_hold + 1, torch.zeros_like(self.transport_hold)),
            torch.zeros_like(self.transport_hold),
        )

        self._set_state(
            m & (self.transport_hold >= cfg.transport_hold_steps),
            FSMState.RELEASE,
        )

        # # 运输过程中如果掉了，回 RECOVER
        # transport_fail = m & (
        #     (~o["is_captured"]) | (o["object_lift"] < 0.5 * cfg.lift_verify_thr)
        # )
        # self._set_state(transport_fail, FSMState.RECOVEfR)

        # ===== RELEASE -> PREPARE =====
        # 通常环境会在成功后 reset；这里给一个占位回环
        m = self.state == int(FSMState.RELEASE)
        self.release_hold = torch.where(m, self.release_hold + 1, torch.zeros_like(self.release_hold))
        self._set_state(m & (self.release_hold >= cfg.release_steps), FSMState.PREPARE)

        # ===== RECOVER -> PREPARE =====
        m = self.state == int(FSMState.RECOVER)
        self.recover_hold = torch.where(m, self.recover_hold + 1, torch.zeros_like(self.recover_hold))

        recover_ok = (
            (o["s_e"] < cfg.s_prepare)
            & (o["r_e"] < cfg.r_center_prepare)
        )
        self._set_state(m & recover_ok, FSMState.PREPARE)

        # 恢复太久也回 PREPARE，避免卡死
        self._set_state(m & (self.recover_hold >= cfg.max_recover_steps), FSMState.PREPARE)

        # Store previous lift for next-step stall detection.
        self.prev_object_lift = o["object_lift"].clone()

    def _compute_actions(self, o: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        actions = torch.zeros(self.num_envs, 5, device=self.device, dtype=torch.float32)

        # ==========================================================
        # PREPARE: 回到管口外安全起点，夹爪张开
        # a0 控 s_e，a1 控 r_e，夹爪保持开
        # ==========================================================
        m = self.state == int(FSMState.PREPARE)
        if m.any():
            actions[m, 0] = self._scaled(cfg.s_prepare - o["s_e"][m], tol=0.004, max_abs=float(self.max_outside[0]))
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_outside[1]))
            actions[m, 4] = -1.0

        # ==========================================================
        # APPROACH: 管口处对中，仍保持张开
        # ==========================================================
        m = self.state == int(FSMState.APPROACH)
        if m.any():
            actions[m, 0] = self._scaled(cfg.s_approach - o["s_e"][m], tol=0.003, max_abs=float(self.max_outside[0]))
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_outside[1]))
            actions[m, 4] = -1.0

        # ==========================================================
        # INSERT: 安全入管到 pregrasp
        # ds = s_obj - s_ee
        # 想要 ds -> pregrasp_offset，因此 a0 ∝ (ds - offset)
        # dr = r_obj - r_ee
        # 想要 dr -> 0，因此 a1 ∝ dr
        # dth 想要 -> 0，因此 a2 ∝ dth
        # dyaw 想要 -> 0；按你当前定义 dyaw = yaw_ee - yaw_obj，因此 a3 ∝ -dyaw
        # ==========================================================
        m = self.state == int(FSMState.INSERT)
        if m.any():
            a0 = self._scaled(o["ds"][m] - cfg.pregrasp_offset, tol=0.004, max_abs=float(self.max_insert[0]))
            # 靠壁时禁止继续前插
            # a0 = torch.where(o["margin_to_wall"][m] < cfg.margin_soft, torch.minimum(a0, torch.zeros_like(a0)), a0)

            actions[m, 0] = a0
            actions[m, 1] = self._scaled(o["dr"][m], tol=0.002, max_abs=float(self.max_insert[1]))
            actions[m, 2] = self._scaled(o["dth"][m], tol=0.25, max_abs=float(self.max_insert[2]))
            actions[m, 3] = self._scaled(-o["dyaw"][m], tol=0.30, max_abs=float(self.max_insert[3]))
            actions[m, 4] = -1.0

        # ==========================================================
        # COARSE_ALIGN
        # ==================================f========================
        m = self.state == int(FSMState.COARSE_ALIGN)
        if m.any():
            actions[m, 0] = self._scaled(o["ds"][m], tol=0.000, max_abs=float(0.0005))
            actions[m, 1] = self._scaled(o["dr"][m], tol=0.002, max_abs=float(self.max_coarse[1]))
            actions[m, 2] = self._scaled(o["dth"][m], tol=0.001, max_abs=float(self.max_coarse[2]))
            actions[m, 3] = self._scaled(-o["dyaw"][m], tol=0.25, max_abs=float(self.max_coarse[3]))
            actions[m, 4] = -1.0
            print("COARSE_ALIGN: ds={:.4f}, dr={:.4f}, dth={:.3f}, dyaw={:.3f}, g_lat={:.3f}".format(
                o["ds"][m].mean().item(),
                o["dr"][m].mean().item(),
                o["dth"][m].mean().item(),
                o["dyaw"][m].mean().item(),
                o["g_lat"][m].mean().item(),
            ))

        # ==========================================================
        # FINE_ALIGN
        # ==========================================================
        m = self.state == int(FSMState.FINE_ALIGN)
        if m.any():
            actions[m, 0] = self._scaled(o["ds"][m], tol=0.0015, max_abs=float(self.max_fine[0]))
            actions[m, 1] = self._scaled(o["dr"][m], tol=0.0012, max_abs=float(self.max_fine[1]))
            actions[m, 2] = self._scaled(o["dth"][m], tol=0.10, max_abs=float(self.max_fine[2]))
            actions[m, 3] = self._scaled(-o["dyaw"][m], tol=0.12, max_abs=float(self.max_fine[3]))
            actions[m, 4] = -1.0

        # ==========================================================
        # CLOSE: 位姿基本不动，只闭夹爪
        # ==========================================================
        m = self.state == int(FSMState.CLOSE)
        if m.any():
            actions[m, 4] = +1.0

        # ==========================================================
        # VERIFY: 小幅回撤 + 保持闭合，验证是否抓住
        # ==========================================================
        m = self.state == int(FSMState.VERIFY)
        if m.any():
            actions[m, 0] = -0.12
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_retract[1]))
            actions[m, 4] = +1.0

        # ==========================================================
        # RETRACT: 安全回撤出管
        # ==========================================================
        m = self.state == int(FSMState.RETRACT)
        if m.any():
            actions[m, 0] = self._scaled(cfg.s_retract_done - o["s_e"][m], tol=0.004, max_abs=float(self.max_retract[0]))
            # Two-stage retract:
            # 1) axial-only retreat while still deep in pipe
            # 2) enable radial centering after enough axial clearance
            radial_enable = o["s_e"][m] >= cfg.s_retract_radial_enable
            a1_raw = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_retract[1]))
            captured = o["is_captured"][m] | (o["object_lift"][m] > cfg.lift_verify_thr)
            a1_boost = -torch.full_like(a1_raw, cfg.retract_radial_boost_cmd)
            a1_cmd = torch.where(captured, torch.minimum(a1_raw, a1_boost), a1_raw)
            a1 = torch.where(radial_enable, a1_cmd, torch.zeros_like(a1_cmd))
            actions[m, 1] = a1
            actions[m, 4] = +1.0

            print("RETRACT: s_e={:.4f}, r_e={:.4f}, a1={:.4f}, radial_on={:.2f}, captured={:.2f}, object_lift={:.4f}".format(
                o["s_e"][m].mean().item(),
                o["r_e"][m].mean().item(),
                a1.mean().item(),
                radial_enable.float().mean().item(),
                captured.float().mean().item(),
                o["object_lift"][m].mean().item(),
            ))

        # ==========================================================
        # TRANSPORT: 抓取后沿 -s 提到固定高度，并向中心收
        # 只允许继续“提起”，不允许再向管内推进
        # ==========================================================
        m = self.state == int(FSMState.TRANSPORT)
        if m.any():
            # 1) 轴向：朝固定抬起高度移动
            a0 = self._scaled(
                cfg.s_transport_target - o["s_e"][m],
                tol=0.010,
                max_abs=float(self.max_retract[0]),
            )

            # 安全限制：TRANSPORT 只允许向外提/保持，不允许再往管内送
            a0 = torch.minimum(a0, torch.zeros_like(a0))

            # 2) 径向：继续往中心收，避免带物体时擦壁
            a1_raw = self._scaled(
                -o["r_e"][m],
                tol=0.003,
                max_abs=float(self.max_retract[1]),
            )
            captured = o["is_captured"][m] | (o["object_lift"][m] > cfg.lift_verify_thr)
            a1_boost = -torch.full_like(a1_raw, cfg.transport_radial_boost_cmd)
            a1 = torch.where(captured, torch.minimum(a1_raw, a1_boost), a1_raw)

            # 3) 角度和 yaw：尽量静止，避免晃动
            a2 = torch.zeros_like(a0)
            a3 = torch.zeros_like(a0)

            actions[m, 0] = a0
            actions[m, 1] = a1
            actions[m, 2] = a2
            actions[m, 3] = a3
            actions[m, 4] = +1.0

        # ==========================================================
        # RELEASE: 这里先当 HOLD 用——到达固定抬起高度后保持夹紧静止
        # ==========================================================
        m = self.state == int(FSMState.RELEASE)
        if m.any():
            actions[m, 0] = 0.0
            actions[m, 1] = 0.0
            actions[m, 2] = 0.0
            actions[m, 3] = 0.0
            actions[m, 4] = +1.0
        # ==========================================================
        # RECOVER: 回撤 + 回中心
        # 如果已经抓住则保持闭合，否则打开夹爪
        # ==========================================================
        m = self.state == int(FSMState.RECOVER)
        if m.any():
            actions[m, 0] = self._scaled(cfg.s_recover - o["s_e"][m], tol=0.008, max_abs=float(self.max_retract[0]))
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_retract[1]))
            keep_close = o["is_captured"][m] | (o["object_lift"][m] > 0.0)
            actions[m, 4] = torch.where(keep_close, torch.ones_like(actions[m, 4]), -torch.ones_like(actions[m, 4]))

        return actions

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
    # 3) rolloutf
    obs, _ = env.get_observations()
    timestep = 0
    # 3) FSM
    fsm = SafeGraspFSM(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        obs_example=obs,
        cfg=FSMCfg(),
    )
    fsm.reset()

    timestep = 0
    reset_requested = False
    success_count_total = 0

    input_iface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None

    def _on_keyboard_event(event, *args, **kwargs):
        nonlocal reset_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.R:
            reset_requested = True
            print("[keyboard] R pressed -> request env reset")
        return True

    keyboard_sub = None
    if keyboard is not None:
        keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if reset_requested:
                    reset_out = env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    fsm.reset()
                    reset_requested = False
                    print("[env] manual reset done")

                # 确保 dtype/device 对齐
                actions = fsm.act(obs)

                # 这里 step 的返回格式可能因 wrapper 略有不同
                step_out = env.step(actions)

                obs, rew, dones, infos = step_out

                # Success condition: object lifted above threshold.
                lift_now = fsm._parse_obs(obs)["object_lift"]
                success_mask = lift_now > float(args_cli.success_lift_thr)
                if bool(success_mask.any().item()):
                    success_num = int(success_mask.sum().item())
                    success_count_total += success_num
                    print(
                        f"[success] +{success_num} (thr={args_cli.success_lift_thr:.4f} m), total={success_count_total}"
                    )
                    reset_out = env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    fsm.reset()
                    continue

                # 如果 dones 是 [num_envs] bool tensor，reset 对应 FSM 状态
                if isinstance(dones, torch.Tensor):
                    fsm.reset(dones)

                if timestep % 10 == 0:
                    print(f"[t={timestep}] FSM state[0] = {fsm.state_name(0)}")
                
            timestep += 1
    finally:
        if keyboard is not None and keyboard_sub is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
