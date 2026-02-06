# ur3_lift_needle_env.py
from __future__ import annotations

import math
import weakref
from typing import Optional

import omni.kit.app
import omni.usd
import torch
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from pxr import UsdUtils



# ---------------------------------------------------------------------------
# Manager-based (MDP) 环境薄壳
#   - 不再在这里实现 _pre_physics_step/_apply_action/_get_rewards/_get_observations/_reset_idx
#   - 这些逻辑应迁移到 mdp/actions.py, mdp/observations.py, mdp/rewards.py, mdp/events.py, mdp/terminations.py
#   - 这里保留：通用几何工具 + debug 可视化 + 一些跨 term 共享的只读常量/缓存
# ---------------------------------------------------------------------------

class Ur3LiftNeedleEnv(ManagerBasedRLEnv):
    """UR3 鼻腔取物（管内精细抓取）— Manager-based MDP 薄壳环境。"""

    def __init__(self, cfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- 常用别名 ---
        self.num_envs = self.scene.num_envs
        self.device = self.scene.device

        # --- step_dt（给 reward term 做 dt 补偿时会用到）---
        # ManagerBasedRLEnv 通常也有 env.step_dt，但这里留一个显式字段更稳妥
        sim_dt = getattr(getattr(self.cfg, "sim", None), "dt", None)
        decimation = getattr(self.cfg, "decimation", None)
        if sim_dt is not None and decimation is not None:
            self.step_dt = float(sim_dt) * int(decimation)
        else:
            # 兜底：如果你的 cfg 里已经提供 step_dt/dt
            self.step_dt = float(getattr(self.cfg, "step_dt", getattr(self.cfg, "dt", 0.0)))

        # -----------------------
        # 管道/几何只读参数（建议也放到 cfg 里，term 只读 env.xxx）
        # -----------------------
        self.pipe_radius: float = float(getattr(self.cfg, "pipe_radius", 0.0075))
        self.pipe_length: float = float(getattr(self.cfg, "pipe_length", 0.032))
        self.pipe_safety_margin: float = float(getattr(self.cfg, "pipe_safety_margin", 0.0))

        # pipe 顶部偏移（你原来写死 0.04）
        self.pipe_top_height: float = float(getattr(self.cfg, "pipe_top_height", 0.04))

        # 轴向（你的原始定义：z-）
        self.u_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # workspace（安全限制，可选给 term 使用）
        self.workspace_min = torch.as_tensor(
            getattr(self.cfg, "workspace_min", [-0.03, -0.34, -0.24]),
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        self.workspace_max = torch.as_tensor(
            getattr(self.cfg, "workspace_max", [0.03, -0.24, -0.18]),
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )

        # -----------------------
        # 资产名（建议与你 cfg 的 SceneCfg/SceneEntityCfg 一致）
        # -----------------------
        self.robot_name: str = getattr(self.cfg, "robot_name", "left_robot")
        self.object_name: str = getattr(self.cfg, "object_name", "object")
        self.ee_body_name: str = getattr(self.cfg, "ee_body_name", "scissors_tip")

        # 获取 robot/object（如果你的 scene 配置里名字不同，改 cfg 里的 robot_name/object_name）
        self._robot = None
        self._object = None
        try:
            self._robot = self.scene.articulations[self.robot_name]
        except Exception:
            pass
        try:
            self._object = self.scene.rigid_objects[self.object_name]
        except Exception:
            pass

        # 末端 body id（仅用于 debug 可视化/工具函数；动作 term 自己也会算一份更独立）
        self.ee_id: int = -1
        if self._robot is not None and self._robot.is_initialized:
            if self.ee_body_name in self._robot.data.body_names:
                self.ee_id = self._robot.data.body_names.index(self.ee_body_name)

        # -----------------------
        # debug 可视化需要的“目标位姿占位”
        #   ActionTerm 可以每步调用 env.set_debug_target_pose(...) 来更新
        # -----------------------
        self.ee_target_pos_w = None
        self.ee_target_quat_w = None

        # debug vis
        self._debug_vis_handle = None
        self.set_debug_vis(bool(getattr(self.cfg, "debug_vis", False)))

        # 可选：灯光（如果 scene cfg 里没配也可以在这加）
        if bool(getattr(self.cfg, "spawn_dome_light", False)):
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # 给 ActionTerm 更新 debug 目标位姿用（可选）
    # ------------------------------------------------------------------
    def set_debug_target_pose(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        """供 actions term 调用：更新期望末端位姿，用于 marker 可视化。"""
        self.ee_target_pos_w = pos_w
        self.ee_target_quat_w = quat_w

    # ------------------------------------------------------------------
    # 管口位姿（复用你原来的 cfg.pipe_pos / cfg.pipe_quat 逻辑）
    # ------------------------------------------------------------------
    def get_pipe_top_pose(self):
        """
        计算每个并行环境下 pipe 顶部的世界坐标位置与姿态。
        依赖 cfg 提供：
          - cfg.pipe_pos: (3,) pipe 的局部位置（相对 env origin）
          - cfg.pipe_quat: (4,) pipe 的局部四元数
        """
        pipe_local_pos = torch.as_tensor(
            getattr(self.cfg, "pipe_pos", [0.0, 0.0, 0.0]),
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        pipe_quat_single = torch.as_tensor(
            getattr(self.cfg, "pipe_quat", [1.0, 0.0, 0.0, 0.0]),
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        pipe_quat = pipe_quat_single.expand(self.num_envs, -1)

        pipe_world_pos = self.scene.env_origins + pipe_local_pos
        pipe_top_pos = pipe_world_pos + torch.as_tensor(
            [0.0, 0.0, self.pipe_top_height],
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        return pipe_top_pos, pipe_quat

    # ------------------------------------------------------------------
    # 世界坐标 -> 管道坐标（复用你原来的定义）
    # ------------------------------------------------------------------
    def world_to_pipe_coords(self, pos_w: torch.Tensor):
        """
        把世界系位置 pos_w (N,3) 转到“以管口为原点、u_axis 为轴向”的管道坐标系：
            s:   轴向深度（>0 在管内）
            r:   径向距离
            th:  截面内的极角
        """
        pipe_top_pos, _ = self.get_pipe_top_pose()
        delta = pos_w - pipe_top_pos  # (N,3)

        s = torch.sum(delta * self.u_axis, dim=-1, keepdim=True)  # (N,1)

        radial = delta - s * self.u_axis
        x_r = radial[..., 0:1]
        y_r = radial[..., 1:2]

        r = torch.sqrt(x_r * x_r + y_r * y_r + 1e-8)
        th = torch.atan2(y_r, x_r)

        return s, r, th, x_r, y_r

    def get_axial_depth(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        """末端沿管轴向深度（>0 在管内，<0 在管外）。"""
        pipe_top_pos, _ = self.get_pipe_top_pose()
        delta = ee_pos_w - pipe_top_pos
        d_axial = torch.sum(delta * self.u_axis, dim=1)
        return d_axial

    # ------------------------------------------------------------------
    # debug 可视化（保留你原有逻辑：goal/current/object范围球）
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                if hasattr(self.cfg, "goal_pose_visualizer_cfg"):
                    self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                else:
                    self.goal_pose_visualizer = None

                if hasattr(self.cfg, "current_pose_visualizer_cfg"):
                    self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                else:
                    self.current_pose_visualizer = None

            if getattr(self, "goal_pose_visualizer", None) is not None:
                self.goal_pose_visualizer.set_visibility(True)
            if getattr(self, "current_pose_visualizer", None) is not None:
                self.current_pose_visualizer.set_visibility(True)

            # object 5mm 范围球（cfg.range_vis）
            if not hasattr(self, "range_visualizer"):
                if hasattr(self.cfg, "range_vis"):
                    self.range_visualizer = VisualizationMarkers(self.cfg.range_vis)
                else:
                    self.range_visualizer = None

            if getattr(self, "range_visualizer", None) is not None:
                self.range_visualizer.set_visibility(True)

        else:
            if getattr(self, "goal_pose_visualizer", None) is not None:
                self.goal_pose_visualizer.set_visibility(False)
            if getattr(self, "current_pose_visualizer", None) is not None:
                self.current_pose_visualizer.set_visibility(False)
            if getattr(self, "range_visualizer", None) is not None:
                self.range_visualizer.set_visibility(False)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        self._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        return True

    def _debug_vis_callback(self, event):
        # 资产可能还没初始化完
        if self._robot is None or (not self._robot.is_initialized):
            return
        if self._object is None or (not self._object.is_initialized):
            return
        if self.ee_id < 0:
            return

        # 目标位姿（由 ActionTerm 调用 set_debug_target_pose 更新）
        if getattr(self, "goal_pose_visualizer", None) is not None:
            if self.ee_target_pos_w is not None and self.ee_target_quat_w is not None:
                self.goal_pose_visualizer.visualize(self.ee_target_pos_w, self.ee_target_quat_w)

        # 当前末端位姿
        if getattr(self, "current_pose_visualizer", None) is not None:
            body_pose_w = self._robot.data.body_state_w[:, self.ee_id]  # (N, 13) usually
            self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

        # object 5mm 范围球
        if getattr(self, "range_visualizer", None) is not None:
            obj_state_w = self._object.data.body_state_w[:, 0]
            obj_pos_w = obj_state_w[:, 0:3]
            obj_quat_w = obj_state_w[:, 3:7]
            idx = torch.zeros((obj_pos_w.shape[0],), dtype=torch.int64, device=obj_pos_w.device)
            self.range_visualizer.visualize(obj_pos_w, obj_quat_w, marker_indices=idx)
