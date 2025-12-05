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
        # 管内：细步长
        self.step_inside = torch.tensor(
            [[0.004, 0.001, 0.08, 0.04]],
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
        actions = torch.clamp(actions, -1.0, 1.0)
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
        self.gripper_cmd = self.gripper_cmd + grip_speed_factor * self.gripper_speed * self.dt
        self.gripper_cmd = torch.clamp(self.gripper_cmd, self.gripper_min, self.gripper_max)

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
        robot_ee_pos = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        # 物体相对于鼻腔中心 (0,-0.29)
        xy = obj_pos_w[:, :2] - self.scene.env_origins[:, :2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs, 1)
        dist_xy = torch.norm(xy - target_xy, dim=1)

        # “移出鼻腔”条件
        object_outside = dist_xy > 0.1
        self.dist = dist_xy

        # ee 与物体距离
        self.ee_dist = torch.norm(robot_ee_pos - obj_pos_w, dim=1)

        # 成功：物体出了管 & ee 和物体很近（说明是夹出来的）
        self.terminated = torch.logical_and(object_outside, self.ee_dist < 0.01)

        # 超时截断
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        """
        改进版本 reward（重点：小距离更精细的引导）：
        1) entry：从管外引导到管口附近
        2) reach_pipe：在管坐标系下靠近物体 (s, r)
        3) grasp_reach_3d：3D 距离接近（尤其在小距离更敏感）
        4) pre_grasp_open / grasp_close：中距离张开、近距离闭合（改为平滑权重）
        5) lift：抬高物体（核心奖励），gating 用 ee_dist < 0.007
        6) success：任务完成大奖励
        7) outside_penalty：在管外跑远的惩罚
        """

        device = self.device

        # ------------------- 末端 / 物体世界坐标 -------------------
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        # 3D 距离
        ee_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)  # (N,)
        self.ee_dist = ee_dist

        # ------------------- 管道坐标 (s, r, θ) -------------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)

        s_e = s_e.squeeze(-1)
        r_e = r_e.squeeze(-1)
        th_e = th_e.squeeze(-1)

        s_o = s_o.squeeze(-1)
        r_o = r_o.squeeze(-1)
        th_o = th_o.squeeze(-1)

        # 相对 (s, r)
        ds = torch.abs(s_e - s_o)
        dr = torch.abs(r_e - r_o)
        d_pipe = torch.sqrt(ds * ds + dr * dr + 1e-8)

        # 是否在管内（只要 s>0 且 r<半径 就算在管内）
        is_in_pipe = (s_e > 0.0) & (r_e < self.pipe_radius)
        inpipe_base_reward = is_in_pipe.float()

        # ------------------- 1) entry 引导（从管外到管口） -------------------
        # 全局：末端到管口的距离（防止在外面跑太远）
        ee_to_pipe = torch.norm(ee_pos_w - self.pipe_top_pos, dim=1)

        outside = ~is_in_pipe
        k_s = 10.0
        k_r = 40.0
        entry_reward = torch.exp(-k_s * torch.abs(s_e)) * torch.exp(-k_r * r_e)

        # “跑太远”惩罚，只在管外生效
        outside_penalty = -0.5 * torch.clamp(ee_to_pipe - 0.20, min=0.0) * outside.float()

        # ------------------- 2) 接近奖励（管坐标 + 3D，小距离更精细） -------------------
        # 2.1 管道平面接近物体 (s, r)
        reach_scale = 80.0
        reach_pipe_reward = torch.exp(-reach_scale * d_pipe)

        # 2.2 3D 接近（特别强调小距离）
        # 设定几个距离阈值：
        near_thr = 0.007   # 你观测到“足够近”的半径
        mid_thr  = 0.02    # 4cm 内认为是中距离

        # 粗尺度：0 ~ 4cm 线性拉近
        d_far_norm = torch.clamp(ee_dist / mid_thr, 0.0, 1.0)
        reach_far  = 1.0 - d_far_norm                   # [0,1]

        # 细尺度：0 ~ 0.7cm 范围内，再给一层更陡的 shaping
        d_near_norm = torch.clamp(ee_dist / near_thr, 0.0, 1.0)
        reach_near  = 1.0 - d_near_norm                 # [0,1]，越小越接近 1

        # 合成 3D 接近奖励（只在管内起作用）
        grasp_reach_reward = (
            0.3 * reach_far +
            0.7 * reach_near
        ) * is_in_pipe.float()

        # ------------------- 3) 抓取行为：开 / 闭（小距离权重更细腻） -------------------
        # 夹爪归一化：0=闭合, 1=张开
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
        grip_norm = torch.clamp(grip_norm, 0.0, 1.0).squeeze(-1)

        # 三段距离：
        #   far: ee_dist >= mid_thr          -> 不特别管抓取姿态
        #   mid: near_thr <= ee_dist < mid_thr  -> 鼓励张开
        #   near: ee_dist < near_thr            -> 鼓励闭合，且越近权重越大
        far_mask  = (ee_dist >= mid_thr).float()
        mid_mask  = ((ee_dist >= near_thr) & (ee_dist < mid_thr)).float()
        near_mask = (ee_dist < near_thr).float()

        # 中距离：希望张开 -> grip_norm 越大越好
        # 距离越接近 near_thr，张开的奖励越大
        # t_mid 在 [0,1]，表示“离 near_thr 有多近”
        t_mid  = torch.zeros_like(ee_dist)
        denom = (mid_thr - near_thr + 1e-6)
        t_mid  = (mid_thr - ee_dist) / denom           # ee_dist=mid_thr -> 0, =near_thr ->1
        t_mid  = torch.clamp(t_mid, 0.0, 1.0)
        pre_grasp_open_reward = mid_mask * t_mid * grip_norm  # 越靠近 near_thr、越张开越好

        # 近距离：希望闭合 -> grip_norm 越小越好
        # 用 proximity 表示“在 near_thr 内接近物体的程度”
        proximity = torch.clamp((near_thr - ee_dist) / (near_thr + 1e-6), 0.0, 1.0)
        grasp_close_reward = near_mask * proximity * (1.0 - grip_norm)

        # ------------------- 4) 抬高物体 -------------------
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        # 只有在“足够近”的时候才启用 lift（用同一个 near_thr）
        close_for_lift = (ee_dist < 0.02).float()

        # 可以减去一个很小的 offset，避免小抖动也拿 lift 分

        lift_reward = object_lift * 5000.0 * close_for_lift

        # ------------------- 5) 掉落惩罚（可选，先关掉也行） -------------------
        # if not hasattr(self, "prev_object_lift"):
        #     self.prev_object_lift = torch.zeros_like(object_lift)
        #
        # lift_drop = torch.clamp(self.prev_object_lift - object_lift, min=0.0)
        # was_lifted = (self.prev_object_lift > 0.005).float()
        # drop_penalty = -10.0 * lift_drop * was_lifted
        #
        # self.prev_object_lift = object_lift.detach()

        # ------------------- 6) 成功奖励 -------------------
        # 这里你原来是用高度 > 0.04 表示“出了鼻腔”
        object_outside = object_lift > 0.04
        ee_close = ee_dist < 0.01
        success = torch.logical_and(object_outside, ee_close)
        self.terminated = success
        success_reward = success.float() * 40.0

        # ------------------- 7) 汇总总奖励（含权重） -------------------
        rewards = (
            0.5 * entry_reward +
            0.8 * reach_pipe_reward +
            1.0 * inpipe_base_reward +
            10 * grasp_reach_reward +      # 3D 接近（小距离有更强梯度）
            0.2 * pre_grasp_open_reward +   # 保留 grasp 开启
            0.5 * grasp_close_reward +      # 保留 grasp 关闭，权重适中
            1.0 * lift_reward +             # 核心：抬高物体
            1.0 * success_reward +
            outside_penalty                 # 负的
            # + drop_penalty                # 如要启用再加上
        )

        # ------------------- 8) 日志：方便可视化 -------------------
        self.extras["log"] = {
            "reward/total":             rewards.mean(),
            "reward/entry":             0.5 * entry_reward.mean(),
            "reward/reach_pipe":        0.8 * reach_pipe_reward.mean(),
            "reward/grasp_reach_3d":    0.8 * grasp_reach_reward.mean(),
            "reward/pre_grasp_open":    0.2 * pre_grasp_open_reward.mean(),
            "reward/grasp_close":       0.5 * grasp_close_reward.mean(),
            "reward/lift":              lift_reward.mean(),
            "reward/success":           success_reward.mean(),
            "reward/outside_pen":       outside_penalty.mean(),
            # "reward/drop_penalty":    drop_penalty.mean() if 'drop_penalty' in locals() else 0.0,

            "metrics/ee_obj_dist":      ee_dist.mean(),
            "metrics/object_lift_mean": object_lift.mean(),
            "metrics/object_lift_max":  object_lift.max(),
            "metrics/in_pipe_ratio":    is_in_pipe.float().mean(),
            "metrics/outside_ratio":    outside.float().mean(),
            "metrics/success_rate":     success.float().mean(),
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
        ee_pos_w = ee_state[:, 0:3]

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
