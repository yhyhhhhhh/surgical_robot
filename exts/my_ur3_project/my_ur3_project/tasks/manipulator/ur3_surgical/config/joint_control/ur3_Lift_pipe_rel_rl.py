from __future__ import annotations

import math
import time
import weakref
from typing import Sequence

import omni.kit.app
import omni.usd
import torch
from omni.isaac.core.prims import XFormPrimView
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sensors.camera import Camera
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import UsdUtils
import omni.isaac.lab.sim as sim_utils

# 自己的模块
from .utils.myfunc import *
from .ur3_lift_pipe_ik_env_cfg import Ur3LiftPipeEnvCfg
from .utils.robot_ik_fun import DifferentialInverseKinematicsAction


# ---------------------------------------------------------------------------
# 小工具：保存当前 stage
# ---------------------------------------------------------------------------

def save_current_stage(output_usd_path: str, flatten: bool = True):
    """
    保存当前 Stage 到一个 USD 文件。
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开的 Stage，确认已经完成 _setup_scene() 或环境初始化。")

    if flatten:
        print(f"[save_current_stage] Flatten stage to: {output_usd_path}")
        flat_layer = UsdUtils.FlattenLayerStack(stage)
        flat_layer.Export(output_usd_path)
    else:
        print(f"[save_current_stage] Export stage to: {output_usd_path}")
        stage.Export(output_usd_path)

    print("[save_current_stage] Done.")


# ---------------------------------------------------------------------------
# 纯强化学习环境
#   - 动作空间: a ∈ [-1,1]^5
#       a[0] → Δs    沿管轴向前进/后退
#       a[1] → Δr    朝管中心/管壁
#       a[2] → Δθ    在截面内绕中心旋转
#       a[3] → Δyaw  绕自身轴的旋转（末端局部 y 轴竖直）
#       a[4] → 抓手速度因子
#   - 使用 Differential IK 把末端目标 pose 转为关节目标
# ---------------------------------------------------------------------------

class Ur3LiftNeedleEnv(DirectRLEnv):
    """
    纯 RL 版本的 UR3 鼻腔取物环境（管内精细抓取版本）。
    """
    cfg: Ur3LiftPipeEnvCfg

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self, cfg: Ur3LiftPipeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 末端 link 的 id
        self.ee_id = self._robot.data.body_names.index("scissors_tip")

        # 关节软限位
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # 仍然保留一个世界系 workspace 盒子（作为安全限制，可按需用）
        self.workspace_min = torch.tensor([-0.03, -0.34, -0.24], device=self.device)
        self.workspace_max = torch.tensor([ 0.03, -0.24, -0.18], device=self.device)

        # 鼻腔顶部位置 & 轴向（z-）
        self.pipe_top_pos, _ = self.get_pipe_top_pose()
        self.u_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # === 管道 / 动作相关参数 ===
        # 管道半径与长度（根据实际模型稍微调一下）
        self.pipe_radius = 0.0075         # m
        self.pipe_safety_margin = 0.000   # 离管壁的安全间距
        self.pipe_length = 0.032           # m，可用的管长

        # 允许的轴向深度范围（s>0 在管内）
        self.s_min = -0.01
        self.s_max = self.pipe_length

        # 动作空间: [Δs, Δr, Δθ, Δyaw, grip_speed_factor]
        # 管外：大步长
        self.step_outside = torch.tensor(
            [[0.01, 0.003, 0.30, 0.6]],   # Δs, Δr, Δθ, Δyaw
            device=self.device,
        )
        # # 管内：细步长
        # self.step_inside = torch.tensor(
        #     [[0.004, 0.001, 0.08, 0.04]],
        #     device=self.device,
        # )
        # 管内：细步长
        self.step_inside = torch.tensor(
            [[0.002, 0.0005, 0.04, 0.04]],
            device=self.device,
        )
        # 抓手阻抗控制相关
        self.gripper_min = torch.tensor([-0.28], device=self.device)  # 完全闭合
        self.gripper_max = torch.tensor([-0.10], device=self.device)  # 完全张开
        self.gripper_cmd = torch.full((self.num_envs, 1), -0.10, device=self.device)  # 初始略张开
        self.gripper_speed = 0.7  # rad/s

        # 末端目标 pose（用于动作平滑）
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        self.ee_target_pos_w = ee_state[:, 0:3].clone()
        self.ee_target_quat_w = ee_state[:, 3:7].clone()
        self.ee_target_yaw = torch.zeros(self.num_envs, device=self.device)

        # 局部 y 轴旋到世界 z 轴的固定旋转
        yaw0 = torch.tensor(-math.pi / 2.0, device=self.device)
        pitch0 = torch.tensor(0.0, device=self.device)
        roll_align = torch.tensor(math.pi / 2.0, device=self.device)  # 绕 X 轴 +90°
        self.q_align_y_to_z = quat_from_euler_xyz(roll_align, pitch0, yaw0)

        # reset 时间统计（可选）
        self.last_reset_t = torch.full(
            (self.num_envs,), float("nan"),
            device=self.device, dtype=torch.float64
        )
        self.reset_interval = torch.zeros_like(self.last_reset_t)

        # debug 可视化
        self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b = torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        self.prev_ee_to_obj_dist = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # 场景搭建
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 地面
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
        )
        # 机器人桌子
        self.cfg.table_robot.spawn.func(
            self.cfg.table_robot.prim_path,
            self.cfg.table_robot.spawn,
            translation=self.cfg.table_robot.init_state.pos,
            orientation=self.cfg.table_robot.init_state.rot,
        )
        # 手术台
        self.cfg.table_operate.spawn.func(
            self.cfg.table_operate.prim_path,
            self.cfg.table_operate.spawn,
            translation=self.cfg.table_operate.init_state.pos,
            orientation=self.cfg.table_operate.init_state.rot,
        )

        # 机械臂
        self._robot = Articulation(self.cfg.left_robot)

        # # 摄像头
        # self._camera = Camera(cfg=self.cfg.camera)
        # self.scene.sensors["Camera"] = self._camera

        # 管道
        self.cfg.pipe.spawn.func(
            self.cfg.pipe.prim_path,
            self.cfg.pipe.spawn,
            translation=self.cfg.pipe.init_state.pos,
            orientation=self.cfg.pipe.init_state.rot,
        )

        # 小物体
        self._object = RigidObject(cfg=self.cfg.object)

        # 额外视图
        self.scene.extras["pipe"] = XFormPrimView(self.cfg.pipe.prim_path, reset_xform_properties=False)
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table_robot.prim_path, reset_xform_properties=False)
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)

        # 注册到 scene
        self.scene.articulations["left_robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # 保存展开后的场景（可选）
        save_current_stage(
            "/home/yhy/DVRK/scenes/ur3_surgery_scene_flat.usd",
            flatten=True,
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    # ------------------------------------------------------------------
    # 采样小物体初始位置/姿态（相对鼻腔）
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        """
        在以 (0, -0.29, z) 为圆心、半径 max_r 的圆内随机生成 object 的位置。
        """
        device = self.device
        n = len(env_ids)
        max_r = 0.004
        center_x = 0.0
        center_y = -0.29

        u = torch.rand(n, device=device)
        r = max_r * torch.sqrt(u)
        theta = torch.rand(n, device=device) * 2.0 * math.pi

        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)
        z = torch.full((n,), self.cfg.object.init_state.pos[2], device=device)

        new_pos = torch.stack([x, y, z], dim=1) + self.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, :3] = new_pos

        # 随机绕 z 轴旋转
        theta_rot = torch.rand(n, device=device) * 2.0 * math.pi
        q = torch.stack(
            [
                torch.cos(theta_rot / 2.0),  # w
                torch.zeros_like(theta_rot),
                torch.zeros_like(theta_rot),
                torch.sin(theta_rot / 2.0),  # z
            ],
            dim=1,
        )
        self.pose_command_w[env_ids, 3:] = q

    # ------------------------------------------------------------------
    # 世界坐标 → 管道坐标
    # ------------------------------------------------------------------
    def _world_to_pipe_coords(self, pos_w: torch.Tensor):
        """
        把世界系下的位置 pos_w (N,3) 转到“以管口为原点、u_axis 为轴向”的管道坐标系：
            s:   轴向深度（>0 在管内）
            r:   径向距离
            th:  截面内的极角
        """
        delta = pos_w - self.pipe_top_pos       # (N,3)
        s = torch.sum(delta * self.u_axis, dim=-1, keepdim=True)  # (N,1)

        radial = delta - s * self.u_axis        # (N,3)
        x_r = radial[..., 0:1]
        y_r = radial[..., 1:2]

        r = torch.sqrt(x_r * x_r + y_r * y_r + 1e-8)
        th = torch.atan2(y_r, x_r)

        return s, r, th, x_r, y_r

    # ------------------------------------------------------------------
    # pre-physics: 纯 RL 动作处理 (a ∈ [-1,1]^5)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        RL 动作:
            actions[:, 0] → Δs    沿管轴向前进/后退
            actions[:, 1] → Δr    朝管中心/管壁
            actions[:, 2] → Δθ    在截面内绕中心旋转
            actions[:, 3] → Δyaw  绕自身轴的旋转（末端局部 y 轴竖直）
            actions[:, 4] → 抓手速度因子（阻抗式控制）
        """
        # actions = torch.clamp(actions, -1.0, 1.0)
        # DEBUG
        # actions = torch.clamp(actions, -0.0, 0.0)

        # 1) 当前末端世界位姿
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]

        # 转到管道坐标系：s, r, θ
        s_cur, r_cur, th_cur, x_r, y_r = self._world_to_pipe_coords(ee_pos_w)
        s_cur_s = s_cur.squeeze(-1)   # (N,)
        r_cur_s = r_cur.squeeze(-1)
        th_cur_s = th_cur.squeeze(-1)

        # 是否在管内（用于步长插值）
        in_pipe = (s_cur_s > 0.0) & (s_cur_s < (self.pipe_length - self.pipe_safety_margin))

        # 2) 动作缩放：管外粗、管内细
        step_scale = torch.where(
            in_pipe.unsqueeze(-1),      # (N,1)
            self.step_inside,          # (1,4) → (N,4)
            self.step_outside,         # (1,4) → (N,4)
        )
        delta_pipe = actions[:, 0:4] * step_scale  # (N,4)

        delta_s = delta_pipe[:, 0]
        delta_r = delta_pipe[:, 1]
        delta_th = delta_pipe[:, 2]
        delta_yaw = delta_pipe[:, 3]

        # 抓手速度因子
        grip_speed_factor = actions[:, 4:5]  # (N,1)

        # 3) 在管道坐标系中更新目标位置
        # s_tgt = torch.clamp(s_cur_s + delta_s, self.s_min, self.s_max)
        # r_tgt = torch.clamp(
        #     r_cur_s + delta_r,
        #     0.0,
        #     self.pipe_radius - self.pipe_safety_margin,
        # )
        s_tgt = s_cur_s + delta_s
        r_tgt = r_cur_s + delta_r
        th_tgt = th_cur_s + delta_th

        x_r_new = r_tgt * torch.cos(th_tgt)
        y_r_new = r_tgt * torch.sin(th_tgt)
        radial_new = torch.stack(
            [x_r_new, y_r_new, torch.zeros_like(x_r_new)],
            dim=-1,
        )  # (N,3)

        axial_new = s_tgt.unsqueeze(-1) * self.u_axis  # (N,3)
        self.ee_target_pos_w = self.pipe_top_pos + axial_new + radial_new  # (N,3)

        # 4) 姿态目标：局部 y 轴竖直 + yaw 控制
        self.ee_target_yaw = self.ee_target_yaw + delta_yaw  # (N,)
        zeros = torch.zeros_like(self.ee_target_yaw)
        q_yaw = quat_from_euler_xyz(zeros, zeros, self.ee_target_yaw)  # (N,4)

        q_align = self.q_align_y_to_z.to(self.device)
        if q_align.ndim == 1:
            q_align = q_align.unsqueeze(0)
        q_align = q_align.expand(q_yaw.shape[0], -1)  # (N,4)

        self.ee_target_quat_w = math_utils.quat_mul(q_yaw, q_align)
        self.ee_target_quat_w = torch.nn.functional.normalize(self.ee_target_quat_w, dim=-1)

        # 5) world → base，交给 IK
        root_pos_w = self._robot.data.root_state_w[:, :3]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        pos_base, quat_base = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            self.ee_target_pos_w, self.ee_target_quat_w,
        )
        pose_base = torch.cat([pos_base, quat_base], dim=-1)
        self._robot_ik.process_actions(pose_base)

        # 6) 抓手阻抗
        target = self.gripper_min + (actions[:, 4:5] + 1.0) * 0.5 * (self.gripper_max - self.gripper_min)
        self.gripper_cmd = target

    # ------------------------------------------------------------------
    # 把 IK 输出写进关节目标
    # ------------------------------------------------------------------
    def _apply_action(self):
        ik_action = self._robot_ik.apply_actions()  # (N, 6)
        robot_action = torch.cat([ik_action, self.gripper_cmd], dim=1)
        self._robot.set_joint_position_target(robot_action)

    # ------------------------------------------------------------------
    # 终止条件（保留你原来的逻辑）
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # === 用抬起高度作为唯一成功条件 ===
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        lift_success_thr = 0.01  # 你和 reward 里保持一致
        success = (object_lift > lift_success_thr)

        self.terminated = success

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """
        Alignment-first reward（截面对齐优先 + 轴向门控 + 夹爪门控 + 微量3D兜底）

        目标行为：
        1) 先在截面内对齐（横纵 / r-θ -> e_lat）
        2) e_lat 足够小后，再沿轴向对齐/逼近（e_ax）
        3) 在“对齐较好 + 轴向足够近”时，再引导夹爪闭合
        4) 保证毫米级精度（重点 <2mm）

        注意：
        - 这是“阶段验证 + 行为规范化”的 reward
        - 不是最终“抬出成功”的完整任务 reward
        """

        # ------------------- 末端 / 物体世界坐标 -------------------
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]

        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        # 3D 距离（只做微量兜底 + 日志）
        ee_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)
        self.ee_dist = ee_dist

        # ------------------- 管道坐标 -------------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)

        s_e = s_e.squeeze(-1)
        r_e = r_e.squeeze(-1)
        th_e = th_e.squeeze(-1)

        s_o = s_o.squeeze(-1)
        r_o = r_o.squeeze(-1)
        th_o = th_o.squeeze(-1)

        # 是否在管内
        is_in_pipe = (s_e > 0.0) & (r_e < self.pipe_radius)
        in_pipe_f = is_in_pipe.float()
        outside_f = (~is_in_pipe).float()

        # ------------------- 1) 计算“截面误差 e_lat” -------------------
        # 在截面平面内把 (r,theta) 转成 (x,y)
        x_e = r_e * torch.cos(th_e)
        y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o)
        y_o = r_o * torch.sin(th_o)

        e_lat = torch.sqrt((x_e - x_o) ** 2 + (y_e - y_o) ** 2 + 1e-8)

        # ------------------- 2) 计算“轴向误差 e_ax” -------------------
        e_ax = torch.abs(s_e - s_o)

        # ------------------- 3) 截面 -> 轴向 的平滑门控 g(e_lat) -------------------
        # e_lat 小于 ~3mm 时，轴向奖励逐渐“变得很值钱”
        e_lat_thr = 0.003    # 3mm：你可以以后调到 0.0025 或 0.004
        k_gate = 0.0008      # 平滑带宽
        g = torch.sigmoid((e_lat_thr - e_lat) / (k_gate + 1e-8))  # (N,)

        # ------------------- 4) 轻量 entry（避免纯对齐策略不愿进管） -------------------
        ee_to_pipe = torch.norm(ee_pos_w - self.pipe_top_pos, dim=1)
        entry_reward = torch.exp(-8.0 * torch.clamp(ee_to_pipe, 0.0, 0.2))
        outside_penalty = -1 * torch.clamp(ee_to_pipe - 0.15, min=0.0) * outside_f

        # ------------------- 5) 截面对齐奖励：多尺度 + 进步量 -------------------
        # 多尺度阈值（截面内）
        l1, l2, l3 = 0.007, 0.003, 0.001  # 1cm / 5mm / 2mm
        lat_1 = 1.0 - torch.clamp(e_lat / l1, 0.0, 1.0)
        lat_2 = 1.0 - torch.clamp(e_lat / l2, 0.0, 1.0)
        lat_3 = 1.0 - torch.clamp(e_lat / l3, 0.0, 1.0)

        lat_shape = 0.2 * lat_1 + 0.6 * lat_2 + 1.4 * lat_3  # 2mm 层权重大

        # 进步量（防站桩）
        if not hasattr(self, "prev_e_lat"):
            self.prev_e_lat = e_lat.detach()
        first_step = (self.episode_length_buf == 0)
        self.prev_e_lat = torch.where(first_step, e_lat, self.prev_e_lat)

        lat_improve = (self.prev_e_lat - e_lat).clamp(min=0.0)
        lat_progress = 800.0 * lat_improve  # 系数可调
        lat_gate = (0.2 + 0.8 * in_pipe_f)
        lat_shape = lat_shape * lat_gate
        lat_progress = lat_progress * lat_gate
        self.prev_e_lat = e_lat.detach()

        # ------------------- 6) 轴向对齐奖励：被 g 门控 -------------------
        a1, a2, a3 = 0.020, 0.007, 0.002  # 2cm / 7mm / 2mm
        ax_1 = 1.0 - torch.clamp(e_ax / a1, 0.0, 1.0)
        ax_2 = 1.0 - torch.clamp(e_ax / a2, 0.0, 1.0)
        ax_3 = 1.0 - torch.clamp(e_ax / a3, 0.0, 1.0)

        ax_shape_raw = 0.2 * ax_1 + 0.6 * ax_2 + 1.4 * ax_3

        if not hasattr(self, "prev_e_ax"):
            self.prev_e_ax = e_ax.detach()
        self.prev_e_ax = torch.where(first_step, e_ax, self.prev_e_ax)

        ax_improve = (self.prev_e_ax - e_ax).clamp(min=0.0)
        ax_progress_raw = 800.0 * ax_improve

        self.prev_e_ax = e_ax.detach()

        # 门控 + 管内加权
        ax_shape = g * ax_shape_raw * (0.3 + 0.7 * in_pipe_f)
        ax_progress = g * ax_progress_raw * (0.3 + 0.7 * in_pipe_f)

        # ------------------- 7) “过早轴向推进”轻惩罚（实现你想要的顺序） -------------------
        # 用 s_e 的变化估计“这一帧是不是在猛推轴向”
        if not hasattr(self, "prev_s_e"):
            self.prev_s_e = s_e.detach()
        self.prev_s_e = torch.where(first_step, s_e, self.prev_s_e)

        delta_s = (s_e - self.prev_s_e)
        self.prev_s_e = s_e.detach()

        premature_ax_penalty = -2.0 * (1.0 - g) * torch.abs(delta_s)

        # ------------------- 8) 夹爪门控奖励（只作为行为引导） -------------------

        # grip_norm: 0=闭, 1=开
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
        grip_norm = torch.clamp(grip_norm, 0.0, 1.0).squeeze(-1)

        ax_open_high = 0.015
        ax_close_thr = 0.002
        close_mask = (e_ax < ax_close_thr)
        # 连续目标：e_ax <= close_thr -> target=0
        #           e_ax >= open_high -> target=1
        # 中间线性过渡
        target = torch.clamp((e_ax - ax_close_thr) / (ax_open_high - ax_close_thr + 1e-6), 0.2, 1.0)
        base = 1.0 - torch.abs(grip_norm - target)

        # 轻微错误惩罚（帮续训摆脱“张开惯性”）
        wrong = close_mask.float() * grip_norm

        gripper_reward = 1.5*g * (base - 0.5 * wrong)
        


        # ------------------- 9) 管壁越界惩罚（保持安全） -------------------
        wall_violation = torch.clamp(r_e - self.pipe_radius, min=0.0)
        wall_penalty = -5.0 * wall_violation

        # ------------------- 10) 微量 3D 精度兜底（只保留里程碑） -------------------
        # 目的：防止分解误差下“看起来对齐了但 3D 还是差一点”的边角情况
        bonus_2mm = (ee_dist < 0.002).float()
        dist_fallback_bonus = 0.5 * bonus_2mm
        # ====== 物体抬起量 ======
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        if not hasattr(self, "prev_object_lift"):
            self.prev_object_lift = object_lift.detach()
        first_step = (self.episode_length_buf == 0)
        self.prev_object_lift = torch.where(first_step, object_lift, self.prev_object_lift)

        lift_improve = (object_lift - self.prev_object_lift).clamp(min=0.0)
        self.prev_object_lift = object_lift.detach()

        # ====== 抬起奖励（只在 grasp_phase 生效） ======
        lift_reward = (100.0 * object_lift + 800.0 * lift_improve)

        # ====== 成功条件 ======
        lift_success_thr = 0.01  # 你按任务改
        success = (object_lift > lift_success_thr)
        success_reward = success.float() * 20.0
        self.terminated = success
        # ------------------- 11) 总奖励 -------------------
        rewards = (
            # 轻量入管/靠近管口
            0.15 * entry_reward +

            # 截面对齐是主菜
            3.0 * lat_shape +
            1.0 * lat_progress +

            # 轴向在截面对齐好时变得很值钱
            2.5 * ax_shape +
            1.0 * ax_progress +

            # 夹爪行为引导
            1.0 * gripper_reward +

            # 顺序约束
            premature_ax_penalty +

            # 小惩罚
            outside_penalty +
            wall_penalty +

            # 微量 3D 精度兜底
            dist_fallback_bonus+
            lift_reward+
            1.0 * success_reward
        )

        # ------------------- 12) 日志 -------------------
        self.extras["log"] = {
            "reward/total":            rewards.mean(),
            "reward/entry":            (0.15 * entry_reward).mean(),
            "reward/lat_shape":        (3.0 * lat_shape).mean(),
            "reward/lat_progress":     lat_progress.mean(),
            "reward/ax_shape":         (2.5 * ax_shape).mean(),
            "reward/ax_progress":      ax_progress.mean(),
            "reward/gripper":          gripper_reward.mean(),
            "reward/premature_ax":     premature_ax_penalty.mean(),
            "reward/outside_pen":      outside_penalty.mean(),
            "reward/wall_pen":         wall_penalty.mean(),
            "reward/3d_fallback":      dist_fallback_bonus.mean(),
            "reward/lift_reward":      lift_reward.mean(),

            "metrics/e_lat":           e_lat.mean(),
            "metrics/lift":            object_lift.mean(),
            "metrics/lift_h":          object_lift.max(),
            "metrics/e_ax":            e_ax.mean(),
            "metrics/ee_dist":         ee_dist.mean(),
            "metrics/g_gate":          g.mean(),
            "metrics/in_pipe_ratio":   in_pipe_f.mean(),
            "metrics/bonus_2mm_rate":  bonus_2mm.mean(),
        }

        return rewards

    # ------------------------------------------------------------------
    # reset 环节
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            return
        env_ids = env_ids.to(self.device).long().view(-1)

        # 统计 reset 间隔（可选）
        now = torch.tensor(time.perf_counter(), device=self.device, dtype=torch.float64)
        prev = self.last_reset_t.index_select(0, env_ids)
        gap = torch.where(torch.isnan(prev), prev, now - prev)
        self.last_reset_t.index_fill_(0, env_ids, now)
        self.reset_interval.scatter_(0, env_ids, gap)

        # 初始化 IK（第一次 reset 时）
        if not hasattr(self, "_robot_ik"):
            self._robot_ik = DifferentialInverseKinematicsAction(self.cfg.left_robot_ik, self.scene)

        super()._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0

        # 机器人
        self._robot.reset(env_ids)
        self._robot_ik.reset(env_ids)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

        # 物体
        if not hasattr(self, "pose_command_w"):
            self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
            self.pose_command_w[:, 3] = 1.0  # identity quat

        self._resample_command(env_ids)
        self._object.write_root_pose_to_sim(self.pose_command_w[env_ids, :], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(
            torch.zeros_like(self._object.data.root_vel_w[env_ids, :]), env_ids=env_ids
        )

        # 末端目标 & 抓手命令重置
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        self.ee_target_pos_w[env_ids] = ee_state[env_ids, 0:3]
        self.ee_target_quat_w[env_ids] = ee_state[env_ids, 3:7]
        self.gripper_cmd[env_ids] = -0.10

        # yaw 从 0 开始
        self.ee_target_yaw[env_ids] = 0.0

        self.scene.write_data_to_sim()

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 末端世界系
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        # ee_pos_w = ee_state[:, 0:3]
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        # 物体世界系
        obj_state = self._object.data.body_state_w[:, 0]
        obj_pos_w = obj_state[:, 0:3]

        # 3D 距离（额外观测）
        ee_obj_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)
        ee_obj_dist = torch.clamp(ee_obj_dist, 0.0, 0.1)

        # 转到管道坐标
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)

        s_e = s_e.squeeze(-1)
        r_e = r_e.squeeze(-1)
        th_e = th_e.squeeze(-1)

        s_o = s_o.squeeze(-1)
        r_o = r_o.squeeze(-1)
        th_o = th_o.squeeze(-1)

        # 相对量
        ds = s_o - s_e
        dr = r_o - r_e
        dth = th_o - th_e

        sin_th_e = torch.sin(th_e)
        cos_th_e = torch.cos(th_e)
        sin_th_o = torch.sin(th_o)
        cos_th_o = torch.cos(th_o)
        sin_dth = torch.sin(dth)
        cos_dth = torch.cos(dth)

        # 是否在管内
        in_pipe = ((s_e > 0.0) & (s_e < (self.pipe_length - self.pipe_safety_margin))).float()

        # 距离管壁的余量
        margin_to_wall = (self.pipe_radius - r_e).unsqueeze(-1)

        # 归一化的 s / r
        s_e_norm = (s_e / (self.pipe_length + 1e-6)).unsqueeze(-1)
        r_e_norm = (r_e / (self.pipe_radius + 1e-6)).unsqueeze(-1)

        # 关节
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel

        # 抓手命令（归一化）
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)

        # （可选）归一化时间步
        phase = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)

        obs = torch.cat(
            (
                # 末端在管道系下
                s_e.unsqueeze(-1),
                r_e.unsqueeze(-1),
                cos_th_e.unsqueeze(-1),
                sin_th_e.unsqueeze(-1),

                # 归一化版本
                s_e_norm,
                r_e_norm,
                margin_to_wall,

                # 物体在管道系下
                s_o.unsqueeze(-1),
                r_o.unsqueeze(-1),
                cos_th_o.unsqueeze(-1),
                sin_th_o.unsqueeze(-1),

                # 相对位置
                ds.unsqueeze(-1),
                dr.unsqueeze(-1),
                cos_dth.unsqueeze(-1),
                sin_dth.unsqueeze(-1),

                # 3D 距离
                ee_obj_dist.unsqueeze(-1),

                # 是否在管内
                in_pipe.unsqueeze(-1),

                # 关节信息
                joint_pos,
                joint_vel,

                # 抓手命令
                grip_norm,

                # 可选：阶段
                phase,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    # ------------------------------------------------------------------
    # debug 可视化
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            # === 新增：5mm 范围球 visualizer ===
            if not hasattr(self, "range_visualizer"):
                # 你在 cfg 里写的是 range_vis = VisualizationMarkersCfg(...)
                self.range_visualizer = VisualizationMarkers(self.cfg.range_vis)
            self.range_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        if not self.has_debug_vis_implementation:
            return False
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
        if not self._robot.is_initialized:
            return
        pose, quat = self.get_pipe_top_pose()
        self.goal_pose_visualizer.visualize(self.ee_target_pos_w, self.ee_target_quat_w)
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

        # === 新增：object 5mm 范围球 ===
        if hasattr(self, "range_visualizer"):
            # 取 object 的世界位姿
            obj_state_w = self._object.data.body_state_w[:, 0]   # (num_envs, 13?) 取第0刚体
            obj_pos_w = obj_state_w[:, 0:3]
            obj_quat_w = obj_state_w[:, 3:7]

            # 若你的 range_vis cfg 里只有一个 marker prototype（sphere）
            # 强烈建议显式传 marker_indices
            idx = torch.zeros((obj_pos_w.shape[0],), dtype=torch.int64, device=obj_pos_w.device)

            self.range_visualizer.visualize(obj_pos_w, obj_quat_w, marker_indices=idx)
    # ------------------------------------------------------------------
    # 其它工具函数
    # ------------------------------------------------------------------
    def init_robot_ik(self):
        self._robot_ik = DifferentialInverseKinematicsAction(self.cfg.left_robot_ik, self.scene)

    def get_pipe_top_pose(self):
        # 将配置中的 tuple 转成 tensor（只在运行时做）
        pipe_local_pos = torch.as_tensor(
            self.cfg.pipe_pos,
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )

        pipe_quat_single = torch.as_tensor(
            self.cfg.pipe_quat,
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        pipe_quat = pipe_quat_single.expand(self.num_envs, -1)

        pipe_world_pos = self.scene.env_origins + pipe_local_pos

        pipe_height = torch.as_tensor(
            [0.0, 0.0, 0.04],
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        pipe_top_pos = pipe_world_pos + pipe_height

        return pipe_top_pos, pipe_quat

    def get_axial_depth(self, ee_pos: torch.Tensor) -> torch.Tensor:
        """
        末端沿管道轴线方向的深度（>0 在管内，<0 在管外）。
        """
        delta = ee_pos - self.pipe_top_pos
        d_axial = torch.sum(delta * self.u_axis, dim=1)
        return d_axial

