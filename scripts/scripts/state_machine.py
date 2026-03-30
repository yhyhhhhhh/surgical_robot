import argparse

from omni.isaac.lab.app import AppLauncher

import cli_args  # isort: skip
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric afnd use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Ur3Lite-PipeRelCamFinalGoalForce-Ik-RL-Direct-v0", help="Name of the task.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import my_ur3_project.tasks  # noqa: F401
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
    # ---- 位置/阈值 ----
    s_prepare: float = -0.004          # 安全起点（管口外）
    s_approach: float = -0.001         # 管口附近
    s_recover: float = -0.006          # 恢复时回撤目标
    s_retract_done: float = -0.002     # 认为已经安全退出管口

    pregrasp_offset: float = 0.004     # INSERT 时希望 ds -> 4mm
    s_max_insert: float = 0.036        # 太深就进入 RECOVER

    r_center_prepare: float = 0.0015
    r_center_strict: float = 0.0010

    margin_soft: float = 0.0015        # 靠壁软阈值
    margin_hard: float = 0.0008        # 靠壁硬阈值

    ds_insert_tol: float = 0.0015
    ds_coarse_tol: float = 0.0025
    ds_fine_tol: float = 0.0010
    dr_fine_tol: float = 0.0010
    dyaw_fine_tol: float = 0.15        # rad

    g_lat_coarse: float = 0.75
    g_close_fine: float = 0.70

    lift_verify_thr: float = 0.0015
    goal_thr: float = 0.005

    # ---- 持续计数 ----
    fine_stable_steps: int = 6
    close_steps: int = 8
    verify_steps: int = 8
    release_steps: int = 6
    max_recover_steps: int = 40

    # ---- 动作限幅（action ∈ [-1, 1]）----
    # 顺序: [a0(Δs), a1(Δr), a2(Δθ), a3(Δyaw)]
    action_max_outside = (0.30, 0.25, 0.12, 0.08)
    action_max_insert  = (0.20, 0.15, 0.12, 0.08)
    action_max_coarse  = (0.16, 0.12, 0.10, 0.08)
    action_max_fine    = (0.10, 0.10, 0.08, 0.05)
    action_max_retract = (0.20, 0.15, 0.05, 0.03)


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
        # fixed = 24, variable = 2 * num_joints
        if (policy_dim - 24) % 2 != 0:
            raise ValueError(
                f"Cannot infer num_joints from policy_dim={policy_dim}. "
                f"Expected policy_dim = 24 + 2 * num_joints."
            )
        return ObsLayout(num_joints=(policy_dim - 24) // 2)


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

        self.max_outside = torch.tensor(cfg.action_max_outside, device=self.device, dtype=torch.float32)
        self.max_insert = torch.tensor(cfg.action_max_insert, device=self.device, dtype=torch.float32)
        self.max_coarse = torch.tensor(cfg.action_max_coarse, device=self.device, dtype=torch.float32)
        self.max_fine = torch.tensor(cfg.action_max_fine, device=self.device, dtype=torch.float32)
        self.max_retract = torch.tensor(cfg.action_max_retract, device=self.device, dtype=torch.float32)

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

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        o = self._parse_obs(obs)
        self._transition(o)
        actions = self._compute_actions(o)
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
        margin_to_wall = x[:, 23 + 2 * J]

        dth = torch.atan2(sin_dth, cos_dth)
        dyaw = torch.atan2(sin_dyaw, cos_dyaw)

        return {
            "s_e": s_e,
            "r_e": r_e,
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
        hard_recover = (
            (o["margin_to_wall"] < cfg.margin_hard)
            | (o["s_e"] > cfg.s_max_insert)
        )
        self._set_state(hard_recover, FSMState.RECOVER)

        # ===== PREPARE -> APPROACH =====
        m = state == int(FSMState.PREPARE)
        ready = (
            (torch.abs(o["s_e"] - cfg.s_prepare) < 0.001)
            & (o["r_e"] < cfg.r_center_prepare)
        )
        self._set_state(m & ready, FSMState.APPROACH)

        # ===== APPROACH -> INSERT =====
        m = self.state == int(FSMState.APPROACH)
        ready = (
            (o["r_e"] < cfg.r_center_strict)
            & (o["margin_to_wall"] > cfg.margin_soft)
        )
        self._set_state(m & ready, FSMState.INSERT)

        # ===== INSERT -> COARSE_ALIGN =====
        m = self.state == int(FSMState.INSERT)
        ready = (
            (torch.abs(o["ds"] - cfg.pregrasp_offset) < cfg.ds_insert_tol)
            & (o["g_lat"] > 0.50)
        )
        self._set_state(m & ready, FSMState.COARSE_ALIGN)

        # ===== COARSE_ALIGN -> FINE_ALIGN =====
        m = self.state == int(FSMState.COARSE_ALIGN)
        ready = (
            (o["g_lat"] > cfg.g_lat_coarse)
            & (torch.abs(o["ds"]) < cfg.ds_coarse_tol)
        )
        self._set_state(m & ready, FSMState.FINE_ALIGN)

        # ===== FINE_ALIGN -> CLOSE （连续稳定若干步）=====
        m = self.state == int(FSMState.FINE_ALIGN)
        fine_ready = (
            (torch.abs(o["ds"]) < cfg.ds_fine_tol)
            & (torch.abs(o["dr"]) < cfg.dr_fine_tol)
            & (torch.abs(o["dyaw"]) < cfg.dyaw_fine_tol)
            & (o["g_close"] > cfg.g_close_fine)
            & (o["margin_to_wall"] > cfg.margin_soft)
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
        ready = (
            (o["s_e"] < cfg.s_retract_done)
            & (o["object_lift"] > cfg.lift_verify_thr)
        )
        self._set_state(m & ready, FSMState.TRANSPORT)

        # ===== TRANSPORT -> RELEASE =====
        m = self.state == int(FSMState.TRANSPORT)
        ready = o["obj_goal_dist"] < cfg.goal_thr
        self._set_state(m & ready, FSMState.RELEASE)

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
            a0 = torch.where(o["margin_to_wall"][m] < cfg.margin_soft, torch.minimum(a0, torch.zeros_like(a0)), a0)

            actions[m, 0] = a0
            actions[m, 1] = self._scaled(o["dr"][m], tol=0.002, max_abs=float(self.max_insert[1]))
            actions[m, 2] = self._scaled(o["dth"][m], tol=0.25, max_abs=float(self.max_insert[2]))
            actions[m, 3] = self._scaled(-o["dyaw"][m], tol=0.30, max_abs=float(self.max_insert[3]))
            actions[m, 4] = -1.0

        # ==========================================================
        # COARSE_ALIGN
        # ==========================================================
        m = self.state == int(FSMState.COARSE_ALIGN)
        if m.any():
            actions[m, 0] = self._scaled(o["ds"][m], tol=0.003, max_abs=float(self.max_coarse[0]))
            actions[m, 1] = self._scaled(o["dr"][m], tol=0.002, max_abs=float(self.max_coarse[1]))
            actions[m, 2] = self._scaled(o["dth"][m], tol=0.20, max_abs=float(self.max_coarse[2]))
            actions[m, 3] = self._scaled(-o["dyaw"][m], tol=0.25, max_abs=float(self.max_coarse[3]))
            actions[m, 4] = -1.0

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
            actions[m, 0] = -0.08
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_retract[1]))
            actions[m, 4] = +1.0

        # ==========================================================
        # RETRACT: 安全回撤出管
        # ==========================================================
        m = self.state == int(FSMState.RETRACT)
        if m.any():
            actions[m, 0] = self._scaled(cfg.s_retract_done - o["s_e"][m], tol=0.006, max_abs=float(self.max_retract[0]))
            actions[m, 1] = self._scaled(-o["r_e"][m], tol=0.002, max_abs=float(self.max_retract[1]))
            actions[m, 4] = +1.0

        # ==========================================================
        # TRANSPORT: 这里只给占位
        # 你当前动作空间是管道坐标动作，真正搬运更适合切笛卡尔控制。
        # 先保守保持抓取，避免在 skeleton 里乱映射。
        # ==========================================================
        m = self.state == int(FSMState.TRANSPORT)
        if m.any():
            actions[m, 4] = +1.0
            # 这里先留空：后续换成 Cartesian / pose servo 更合理

        # ==========================================================
        # RELEASE
        # ==========================================================
        m = self.state == int(FSMState.RELEASE)
        if m.any():
            actions[m, 4] = -1.0

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

    # 2) load exported JIT policy
    jit_path = "/home/yhy/log/2026-03-25_22-49-56/exported/policy.pt"  # 改成你的实际路径
    device = env.unwrapped.device
    print(f"[INFO]: Loading JIT policy from: {jit_path}")
    policy = torch.jit.load(jit_path, map_location=device)
    policy.eval()

    # 3) rolloutf
    obs, _ = env.get_observations()
    timestep = 0
    # 在循环外定义噪声的幅度（标准差），你可以根据需要调整这个值
    noise_scale = 10.0

    # 3) FSM
    fsm = SafeGraspFSM(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        obs_example=obs,
        cfg=FSMCfg(),
    )
    fsm.reset()

    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # 确保 dtype/device 对齐
            actions = fsm.act(obs)

            # 这里 step 的返回格式可能因 wrapper 略有不同
            step_out = env.step(actions)

            # 常见情况：obs, rew, dones, infos
            if len(step_out) == 4:
                obs, rew, dones, infos = step_out
                # 如果 dones 是 [num_envs] bool tensor，reset 对应 FSM 状态
                if isinstance(dones, torch.Tensor):
                    fsm.reset(dones)
            else:
                # 兼容别的封装
                obs = step_out[0]

            if timestep % 100 == 0:
                print(f"[t={timestep}] FSM state[0] = {fsm.state_name(0)}")
            
        timestep += 1
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
