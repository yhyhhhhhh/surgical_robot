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
# 小工具：保存当前 stage（你原来的函数，直接保留）
# ---------------------------------------------------------------------------

def save_current_stage(output_usd_path: str, flatten: bool = True):
    """
    保存当前 Stage 到一个 USD 文件。

    :param output_usd_path: 输出文件路径，比如 "/home/yhy/DVRK/scenes/ur3_scene_flat.usd"
    :param flatten: True=展平（所有引用烤进一个文件），False=简单导出
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
#   - 动作空间: a ∈ [-1,1]^4
#       a[0:3] → 末端 Δx, Δy, Δz (世界坐标, 每步限制在若干 mm)
#       a[3]   → 抓手速度因子 (正 = 闭合, 负 = 张开)
#   - 使用 Differential IK 把末端目标 pose 转为关节目标
# ---------------------------------------------------------------------------

class Ur3LiftNeedleEnv(DirectRLEnv):
    """
    纯 RL 版本的 UR3 鼻腔取物环境。
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

        # 末端工作空间（相对于 env_origin 的世界坐标）
        # 根据你的场景可以微调这两个盒子
        self.workspace_min = torch.tensor([-0.03, -0.34, -0.24], device=self.device)
        self.workspace_max = torch.tensor([ 0.03, -0.24, -0.18], device=self.device)

        # 单步最大位移（m），动作在 [-1,1] 时映射为 [-max_step, max_step]
        self.max_pos_step = torch.tensor([0.01, 0.01, 0.01], device=self.device)  # 2mm
        self.max_yaw_step = 0.05  # rad，约 3 度
        # 抓手（tip_joint）阻抗控制相关
        self.gripper_min = torch.tensor([-0.28], device=self.device)  # 完全闭合
        self.gripper_max = torch.tensor([-0.10], device=self.device)  # 完全张开
        self.gripper_cmd = torch.full((self.num_envs, 1), -0.10, device=self.device)  # 初始略张开
        self.gripper_speed = 0.7  # rad/s，动作的第 4 维乘以它再乘以 dt

        # 末端目标 pose（用于动作平滑）
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        self.ee_target_pos_w = ee_state[:, 0:3].clone()
        self.ee_target_quat_w = ee_state[:, 3:7].clone()

        # 在 __init__ 里
        yaw0 = torch.tensor(-math.pi / 2.0, device=self.device)
        pitch0 = torch.tensor(0.0, device=self.device)
        roll_align = torch.tensor(math.pi / 2.0, device=self.device)  # 绕 X 轴 +90°

        # 把局部 y 轴旋到世界 z 轴的固定旋转
        self.q_align_y_to_z = quat_from_euler_xyz(
            roll_align,      # X
            pitch0,          # Y
            yaw0,            # Z
        )   # 形状 (4,) 或 (1,4) 视你的实现而定

        # 鼻腔顶部位置 & 轴向（z-）
        self.pipe_top_pos, _ = self.get_pipe_top_pose()
        self.u_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

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
    # 场景搭建：基本保持不变
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 手术室
        # self.cfg.room.spawn.func(
        #     self.cfg.room.prim_path,
        #     self.cfg.room.spawn,
        #     translation=self.cfg.room.init_state.pos,
        #     orientation=self.cfg.room.init_state.rot,
        # )
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

        # 摄像头
        self._camera = Camera(cfg=self.cfg.camera)
        self.scene.sensors["Camera"] = self._camera

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

        # 保存展开后的场景（方便你用 Composer 看）
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
    # pre-physics: 纯 RL 动作处理 (a ∈ [-1,1]^4)
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        RL 动作:
            actions[:, 0:3] → Δx, Δy, Δz       （末端位置增量，世界系）
            actions[:, 3]   → Δyaw             （绕世界 z 轴的增量）
            actions[:, 4]   → 抓手速度因子（阻抗式控制）
        """
        debug = True

        # 1) 裁剪动作
        actions = torch.clamp(actions, -1.0, 1.0)

        # 位置增量
        delta_xyz = actions[:, 0:3] * self.max_pos_step          # (N, 3)

        # yaw 增量（只控制绕 z 轴旋转）
        delta_yaw = actions[:, 3] * self.max_yaw_step            # (N,)

        # 抓手速度因子
        grip_speed_factor = actions[:, 4:5]                      # (N, 1)

        # 2) 当前末端世界位姿（只用位置；姿态我们自己重建）
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]
        
        # --- 位置目标 ---
        # 若首次调用时 self.ee_target_pos_w 尚未设定，可以用当前 ee_pos_w 初始化（建议在 reset() 里做）
        self.ee_target_pos_w = self.ee_target_pos_w + delta_xyz

        # ---------------------
        # 2) 姿态目标：局部 y 轴竖直 + 只控制 yaw
        # ---------------------
        # 累加 yaw
        self.ee_target_yaw = self.ee_target_yaw + delta_yaw  # (N,)

        zeros = torch.zeros_like(self.ee_target_yaw)

        q_yaw = quat_from_euler_xyz(zeros, zeros, self.ee_target_yaw)  # (N,4)

        # 固定的“y→z 对齐”四元数，扩展到 batch
        q_align = self.q_align_y_to_z.to(self.device)
        if q_align.ndim == 1:         # (4,) → (1,4) → (N,4)
            q_align = q_align.unsqueeze(0)
        q_align = q_align.expand(q_yaw.shape[0], -1)  # (N,4)

        # 目标姿态 = yaw ⊗ 对齐
        self.ee_target_quat_w = math_utils.quat_mul(q_yaw, q_align)
        self.ee_target_quat_w = torch.nn.functional.normalize(self.ee_target_quat_w, dim=-1)

        # DEBUG: 打印姿态信息
        if debug:
            i = 0
            print("  ee_target_yaw[0] (after) :", float(self.ee_target_yaw[i]))
            print("  q_yaw[0]                 :", q_yaw[i].detach().cpu().numpy())
            print("  q_align[0]               :", q_align[i].detach().cpu().numpy())
            print("  ee_target_quat_w[0]      :", self.ee_target_quat_w[i].detach().cpu().numpy())

        # 3) world → base 坐标系，交给 IK 控制
        root_pos_w = self._robot.data.root_state_w[:, :3]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        pos_base, quat_base = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w,
            self.ee_target_pos_w, self.ee_target_quat_w,
        )

        pose_base = torch.cat([pos_base, quat_base], dim=-1)     # (N, 7)
        self._robot_ik.process_actions(pose_base)

        # 4) 抓手阻抗：cmd += v * dt，裁剪到 [min, max]
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
    # 终止条件
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

    # ------------------------------------------------------------------
    # reward shaping
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 1) 末端靠近 object 的奖励
        dist_scale = 40.0
        reach_reward = torch.exp(-self.ee_dist * dist_scale)

        # 2) 靠得非常近的 bonus
        close_bonus = 0.5 * (self.ee_dist < 0.01).float()

        # 3) 把物体抬起来的奖励（只在靠近时算）
        obj_z = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)
        lift_reward = object_lift * 5.0
        lift_reward *= (self.ee_dist < 0.02).float()

        # 4) 鼻腔外距离奖励
        xy = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :2] - self.scene.env_origins[:, :2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs, 1)
        dist_xy = torch.norm(xy - target_xy, dim=1)
        move_out_reward = torch.clamp(dist_xy - 0.03, min=0.0) * 3.0

        # 5) 成功大奖励
        success_reward = self.terminated.float() * 20.0

        rewards = reach_reward + close_bonus + lift_reward + move_out_reward + success_reward
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

        self.ee_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.scene.write_data_to_sim()

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 末端
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        robot_local_pos = ee_state[:, 0:3] - self.scene.env_origins
        robot_quat = ee_state[:, 3:7]

        # object
        obj_state = self._object.data.body_state_w[:, 0]
        object_local_pos = obj_state[:, 0:3] - self.scene.env_origins
        object_quat = obj_state[:, 3:7]

        # 关节
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel

        # 相对管口位置
        pos_ref = ee_state[:, 0:3] - self.pipe_top_pos

        obs = torch.cat(
            (
                robot_local_pos,
                robot_quat,
                object_local_pos,
                object_quat,
                joint_pos,
                joint_vel,
                pos_ref,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # debug 可视化（基本保持你原来的）
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
        # pipe_pos: (3,) -> tensor(3,)
        pipe_local_pos = torch.as_tensor(
            self.cfg.pipe_pos,
            device=self.device,
            dtype=self.scene.env_origins.dtype,   # 和 env_origins 保持一致
        )

        # pipe_quat: (4,) -> tensor(num_envs, 4)
        pipe_quat_single = torch.as_tensor(
            self.cfg.pipe_quat,
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        # 扩展到每个 env 一份，不分配新内存
        pipe_quat = pipe_quat_single.expand(self.num_envs, -1)

        # env_origins: (num_envs, 3)，pipe_local_pos: (3,) -> 自动广播成 (num_envs, 3)
        pipe_world_pos = self.scene.env_origins + pipe_local_pos

        # 高度偏移，同样先做成 (3,) 的 tensor
        pipe_height = torch.as_tensor(
            [0.0, 0.0, 0.04],
            device=self.device,
            dtype=self.scene.env_origins.dtype,
        )
        # 广播到 (num_envs, 3)
        pipe_top_pos = pipe_world_pos + pipe_height

        return pipe_top_pos, pipe_quat

    def get_axial_depth(self, ee_pos: torch.Tensor) -> torch.Tensor:
        """
        末端沿管道轴线方向的深度（>0 在管内，<0 在管外）。
        """
        delta = ee_pos - self.pipe_top_pos
        d_axial = torch.sum(delta * self.u_axis, dim=1)
        return d_axial
