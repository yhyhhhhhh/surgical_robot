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
# 开放空间抓取环境（无管道）
#   - 动作空间: a ∈ [-1,1]^5
#       a[0] → Δx    世界系 X 方向平移
#       a[1] → Δy    世界系 Y 方向平移
#       a[2] → Δz    世界系 Z 方向平移
#       a[3] → Δyaw  绕自身轴旋转
#       a[4] → 抓手开合（-1=闭合, +1=张开）
#   - 使用 Differential IK 把末端目标 pose 转为关节目标
# ---------------------------------------------------------------------------

class Ur3LiftObjectEnv(DirectRLEnv):
    """
    开放空间抓取小物体（不再使用管道约束）的纯 RL 环境。
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

        # 世界系 workspace 盒子（包住物体上方区域，按需要微调）
        self.workspace_min = torch.tensor([-0.10, -0.40, 0.05], device=self.device)
        self.workspace_max = torch.tensor([ 0.10, -0.20, 0.25], device=self.device)

        # 动作步长
        self.xyz_step = torch.tensor([[0.02, 0.02, 0.02]], device=self.device)   # Δx, Δy, Δz
        self.yaw_step = 0.4  # rad/step

        # 抓手控制
        self.gripper_min = torch.tensor([-0.28], device=self.device)  # 完全闭合
        self.gripper_max = torch.tensor([-0.10], device=self.device)  # 完全张开
        self.gripper_cmd = torch.full((self.num_envs, 1), -0.10, device=self.device)  # 初始略张开
        self.gripper_speed = 0.7  # 目前只用作插值系数，可以不动

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

        # debug 可视化开关
        self.set_debug_vis(getattr(self.cfg, "debug_vis", False))
        self.command_visualizer_b = torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        # 奖励用的缓存
        self.prev_ee_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_object_lift = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # 场景搭建（不再生成管道）
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

        # 小物体
        self._object = RigidObject(cfg=self.cfg.object)

        # 额外视图（可选）
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table_robot.prim_path, reset_xform_properties=False)
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)

        # 注册到 scene
        self.scene.articulations["left_robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # 保存展开后的场景（可选）
        save_current_stage(
            "/home/yhy/DVRK/scenes/ur3_surgery_scene_flat_nopipe.usd",
            flatten=True,
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    # ------------------------------------------------------------------
    # 随机采样小物体初始位置/姿态（桌面上的一块区域）
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        device = self.device
        n = len(env_ids)

        # 桌面上的一个矩形区域（根据实际场景微调）
        x_min, x_max = -0.03, 0.03
        y_min, y_max = -0.33, -0.25
        z = torch.full((n,), self.cfg.object.init_state.pos[2], device=device)

        x = torch.rand(n, device=device) * (x_max - x_min) + x_min
        y = torch.rand(n, device=device) * (y_max - y_min) + y_min

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
    # pre-physics: 纯 RL 动作处理 (a ∈ [-1,1]^5)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        RL 动作（无管道版）:
            actions[:, 0] → Δx   世界系 X
            actions[:, 1] → Δy   世界系 Y
            actions[:, 2] → Δz   世界系 Z
            actions[:, 3] → Δyaw 绕自身轴旋转
            actions[:, 4] → 抓手开合（-1=闭合, +1=张开）
        """
        actions = torch.clamp(actions, -1.0, 1.0)

        # 当前末端世界位姿
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]

        # 1) 位置：世界系 XYZ
        delta_xyz = actions[:, 0:3] * self.xyz_step      # (N,3)
        self.ee_target_pos_w = ee_pos_w + delta_xyz
        # 限制在 workspace 里
        self.ee_target_pos_w = torch.max(self.ee_target_pos_w, self.workspace_min)
        self.ee_target_pos_w = torch.min(self.ee_target_pos_w, self.workspace_max)

        # 2) 姿态：固定“y 轴朝上”+ yaw 控制
        self.ee_target_yaw = self.ee_target_yaw + actions[:, 3] * self.yaw_step
        zeros = torch.zeros_like(self.ee_target_yaw)
        q_yaw = quat_from_euler_xyz(zeros, zeros, self.ee_target_yaw)

        q_align = self.q_align_y_to_z.to(self.device)
        if q_align.ndim == 1:
            q_align = q_align.unsqueeze(0)
        q_align = q_align.expand(q_yaw.shape[0], -1)

        self.ee_target_quat_w = math_utils.quat_mul(q_yaw, q_align)
        self.ee_target_quat_w = torch.nn.functional.normalize(self.ee_target_quat_w, dim=-1)

        # 3) world → base，交给 IK
        root_pos_w = self._robot.data.root_state_w[:, :3]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        pos_base, quat_base = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            self.ee_target_pos_w, self.ee_target_quat_w,
        )
        pose_base = torch.cat([pos_base, quat_base], dim=-1)
        self._robot_ik.process_actions(pose_base)

        # 4) 抓手：简单从动作插值
        # actions[:,4]∈[-1,1] → [gripper_min, gripper_max]
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
    # 终止条件
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        终止条件：
        - terminated：物体抬起超过阈值（成功）
        - truncated：步数到上限
        """
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        lift_success_thr = 0.01  # 和 _get_rewards 里的一致
        success = (object_lift > lift_success_thr)

        self.terminated = success
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return self.terminated, truncated

    # ------------------------------------------------------------------
    # 奖励
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """
        极简抓取奖励（不对抓手本身打奖励）：

        1) reaching：
            - 末端距离物体越近越好（形状奖励）
            - 距离比上一帧变小就给进步奖励

        2) lifting：
            - 物体被抬起的高度（相对初始）
            - 抬起高度比上一帧增加也给进步奖励

        3) success bonus：
            - 抬起超过一定高度给一次性 bonus

        4) time penalty：
            - 每一步一个小惩罚，鼓励尽快完成
        """

        # ------------------- 末端 / 物体世界坐标 -------------------
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]           # (N,3)
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]  # (N,3)

        # 3D 距离
        ee_obj_vec = obj_pos_w - ee_pos_w
        ee_dist = torch.norm(ee_obj_vec, dim=1)  # (N,)

        # ------------------- 1) reaching：靠近物体 -------------------
        # 1.1 形状奖励：距离越小越好，范围约在 (0,1)
        k_reach = 10.0
        reach_shape = 1.0 - torch.tanh(k_reach * ee_dist)

        # 1.2 进步奖励：比上一帧更近就奖励
        if not hasattr(self, "prev_ee_dist"):
            self.prev_ee_dist = ee_dist.detach()

        first_step = (self.episode_length_buf == 0)
        self.prev_ee_dist = torch.where(first_step, ee_dist, self.prev_ee_dist)

        dist_improve = (self.prev_ee_dist - ee_dist).clamp(min=0.0)  # 只奖励“变近”的部分
        reach_progress = 5.0 * dist_improve

        self.prev_ee_dist = ee_dist.detach()

        # ------------------- 2) lifting：物体抬起 -------------------
        # 计算相对初始高度
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)  # (N,)

        if not hasattr(self, "prev_object_lift"):
            self.prev_object_lift = object_lift.detach()
        self.prev_object_lift = torch.where(first_step, object_lift, self.prev_object_lift)

        lift_improve = (object_lift - self.prev_object_lift).clamp(min=0.0)
        self.prev_object_lift = object_lift.detach()

        # 抬得高 + 抬得有进步都给奖励
        lift_reward = 100.0 * object_lift + 20.0 * lift_improve

        # ------------------- 3) success bonus：抬起成功 -------------------
        lift_success_thr = 0.01  # 1 cm
        success = (object_lift > lift_success_thr)
        success_bonus = success.float() * 10.0

        # ------------------- 4) 时间惩罚 -------------------
        time_penalty = -0.001 * torch.ones_like(ee_dist)

        # ------------------- 总奖励 -------------------
        rewards = (
            1.0 * reach_shape +
            1.0 * reach_progress +
            # 1.0 * lift_reward +
            # 1.0 * success_bonus +
            time_penalty
        )

        # ------------------- 日志 -------------------
        self.extras["log"] = {
            "reward/total":            rewards.mean(),

            "reward/reach_shape":      (1.0 * reach_shape).mean(),
            "reward/reach_progress":   reach_progress.mean(),
            "reward/lift":             lift_reward.mean(),
            "reward/success_bonus":    success_bonus.mean(),
            "reward/time_penalty":     time_penalty.mean(),

            "metrics/ee_dist":         ee_dist.mean(),
            "metrics/object_lift":     object_lift.mean(),
            "metrics/object_lift_max": object_lift.max(),
            "metrics/success_rate":    success.float().mean(),
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
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_lin_vel = ee_state[:, 7:10]

        # 物体世界系
        obj_state = self._object.data.body_state_w[:, 0]
        obj_pos_w = obj_state[:, 0:3]

        # 相对量
        rel_pos = obj_pos_w - ee_pos_w          # (N,3)
        ee_obj_dist = torch.norm(rel_pos, dim=1, keepdim=True)
        ee_obj_dist = torch.clamp(ee_obj_dist, 0.0, 0.1)

        # 关节
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel

        # 抓手命令（归一化）
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)

        # 时间
        phase = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)

        obs = torch.cat(
            (
                ee_pos_w,                  # 3
                obj_pos_w,                 # 3
                rel_pos,                   # 3
                ee_lin_vel,                # 3
                ee_obj_dist,               # 1
                joint_pos,                 # ndof
                joint_vel,                 # ndof
                grip_norm,                 # 1
                phase,                     # 1
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
            # 物体附近范围球 visualizer（如果 cfg 里配了）
            if hasattr(self.cfg, "range_vis") and not hasattr(self, "range_visualizer"):
                self.range_visualizer = VisualizationMarkers(self.cfg.range_vis)
            if hasattr(self, "range_visualizer"):
                self.range_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "range_visualizer"):
                self.range_visualizer.set_visibility(False)

    def set_debug_vis(self, debug_vis: bool) -> bool:
        if not getattr(self, "has_debug_vis_implementation", True):
            return False
        self._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if getattr(self, "_debug_vis_handle", None) is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            if getattr(self, "_debug_vis_handle", None) is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        return True

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        # 目标末端 pose
        self.goal_pose_visualizer.visualize(self.ee_target_pos_w, self.ee_target_quat_w)
        # 当前末端 pose
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

        # 物体附近范围球（如果配置了）
        if hasattr(self, "range_visualizer"):
            obj_state_w = self._object.data.body_state_w[:, 0]
            obj_pos_w = obj_state_w[:, 0:3]
            obj_quat_w = obj_state_w[:, 3:7]
            idx = torch.zeros((obj_pos_w.shape[0],), dtype=torch.int64, device=obj_pos_w.device)
            self.range_visualizer.visualize(obj_pos_w, obj_quat_w, marker_indices=idx)

    # ------------------------------------------------------------------
    # 其它工具函数
    # ------------------------------------------------------------------
    def init_robot_ik(self):
        self._robot_ik = DifferentialInverseKinematicsAction(self.cfg.left_robot_ik, self.scene)
