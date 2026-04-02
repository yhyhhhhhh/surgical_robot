from __future__ import annotations

import gymnasium as gym  # 或 gymnasium as gym，跟你工程里一致
import math
import numpy as np
import time
import torch
import weakref
from typing import Sequence

import omni.kit.app
from omni.isaac.core.prims import XFormPrimView

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sensors import Camera, ContactSensor, ContactSensorCfg
from omni.isaac.lab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from omni.isaac.lab.utils import math as math_utils
from .ur3_lift_pipe_rl_osc_cfg import Ur3LiftPipeEnvCfg

# 自己的模块
from .utils.myfunc import *
from .utils.robot_ik_fun import DifferentialInverseKinematicsAction

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

    def _use_image_obs(self) -> bool:
        """Single switch for camera creation and image observations."""
        return bool(getattr(self.cfg, "use_image_obs", True))

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(
        self, cfg: Ur3LiftPipeEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 末端 body id
        self.ee_id = self._robot.data.body_names.index("scissors_tip")
        self.ee_fixed_id = self._robot.data.body_names.index("scrissor_fixed")
        self.ee_move_id = self._robot.data.body_names.index("scrissor_move")

        # 关节 id
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.arm_joint_ids = self._robot.find_joints(self.arm_joint_names)[0]
        self.tip_joint_ids = self._robot.find_joints(["tip_joint"])[0]
        self.num_arm_joints = len(self.arm_joint_ids)

        # fixed-base 机械臂在 jacobian 里 body index 通常要 -1
        self.ee_jacobi_body_idx = self.ee_id - 1

        # 关节软限位
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        # 管口位置/轴向
        self.pipe_top_pos, _ = self.get_pipe_top_pose()
        self.u_axis = (
            torch.tensor([0.0, 0.0, -1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # === 管道 / 动作相关参数 ===
        self.pipe_radius = 0.0075
        self.pipe_safety_margin = 0.000
        self.pipe_length = 0.04
        # Pipe valid z range in local env frame.
        # Your setup: approximately [-0.23, -0.21], outside this band is treated as out.
        self.pipe_z_min_local = -0.235
        self.pipe_z_max_local = -0.22

        self.step_outside = torch.tensor([[0.01, 0.003, 0.30, 0.08]], device=self.device)
        self.step_inside = torch.tensor([[0.002, 0.0005, 0.04, 0.01]], device=self.device)

        self.gripper_min = torch.tensor([-0.28], device=self.device)
        self.gripper_max = torch.tensor([-0.10], device=self.device)
        self.gripper_cmd = torch.full((self.num_envs, 1), -0.10, device=self.device)
        self.gripper_speed = 0.7

        # 当前末端 pose / 目标 pose
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        self.ee_target_pos_w = ee_state[:, 0:3].clone()
        self.ee_target_quat_w = ee_state[:, 3:7].clone()

        # 记录一个“名义姿态”，后面 yaw 在这个基础上叠加
        self.ee_nominal_quat_w = ee_state[:, 3:7].clone()
        self.ee_target_yaw = torch.zeros(self.num_envs, device=self.device)

        # OSC controller
        self.osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
            motion_stiffness_task=[250.0, 250.0, 250.0, 40.0, 40.0, 40.0],
            motion_damping_ratio_task=1.0,
            nullspace_control="none",
        )

        self._osc = OperationalSpaceController(
            self.osc_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # command / effort buffer
        self.osc_cmd = torch.zeros(self.num_envs, 7, device=self.device)  # pose_abs in base: xyz + quat
        self.arm_effort_cmd = torch.zeros(self.num_envs, self.num_arm_joints, device=self.device)

        # reset 统计
        self.last_reset_t = torch.full(
            (self.num_envs,), float("nan"), device=self.device, dtype=torch.float64
        )
        self.reset_interval = torch.zeros_like(self.last_reset_t)

        self.goal_pos_local = torch.tensor([0.00, -0.29, -0.225], device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.goal_reach_thr = 0.005
        self.goal_lift_thr = 0.003
        self.prev_obj_goal_dist = torch.zeros(self.num_envs, device=self.device)


        # self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b = torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        self.last_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.cur_actions = torch.zeros(self.num_envs, 5, device=self.device)
        # Lift-success hysteresis (dense reward + anti-jitter around threshold).
        self.success_on_thr = 0.005
        self.success_off_thr = 0.0035
        self.success_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # failure termination on excessive contact force (can be overridden from cfg)
        self.pipe_force_fail_thr = 2.0
        self.bottom_force_fail_thr = 2.0
        self.force_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._build_dreamer_observation_space()

        self._use_external_pose = False
        self._external_pose_buffer = None

    def _build_dreamer_observation_space(self):
        # num_envs（IsaacLab vectorized env）
        nenv = int(self.num_envs)

        # state 维度：用你当前拼接逻辑推出来
        # state = 19 + 2 * num_joints
        state_dim = int(self._get_observations()["policy"].shape[-1])

        # 和你实际返回对齐：
        # policy: float32 in [-5, 5]
        policy_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(nenv, state_dim), dtype=np.float32
        )

        # flags / failure：你返回的是 int32（torch.int32），这里也用 int32 更一致
        flag_space = gym.spaces.Box(0, 1, (), dtype=bool)

        obs_dict = {
            "policy": policy_space,
            "is_first": flag_space,
            "is_last": flag_space,
            "is_terminal": flag_space,
            "failure": flag_space,
        }
        if self._use_image_obs():
            # 图像分辨率：尽量从 cfg/camera 里取，不行就 fallback
            H = getattr(getattr(self.cfg, "camera", None), "height", 128)
            W = getattr(getattr(self.cfg, "camera", None), "width", 128)
            obs_dict["image"] = gym.spaces.Box(
                low=0, high=255, shape=(nenv, H, W, 3), dtype=np.uint8
            )

        self._observation_space = gym.spaces.Dict(obs_dict)

    # ------------------------------------------------------------------
    # 场景搭建
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 机械臂
        self._robot = Articulation(self.cfg.left_robot)

        # 摄像头（可开关）
        self._camera = None
        if self._use_image_obs():
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

        # 地面
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
        )

        self._pipe_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/pipe/pipe",      # 改成你USD里内壁prim真实名字
                update_period=0.0,
                history_length=4,
                debug_vis=False,
                # filter_prim_paths_expr=["/World/envs/env_.*/Left_Robot/ur3_robot/tip_link"],
            )
        )
        self.scene.sensors["pipe_contact"] = self._pipe_contact

        self._bottom_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/pipe/bottom",    # 改成真实名字
                update_period=0.0,
                history_length=4,
                debug_vis=False,
                # filter_prim_paths_expr=["/World/envs/env_.*/Left_Robot/ur3_robot/tip_link"],
            )
        )
        self.scene.sensors["bottom_contact"] = self._bottom_contact

        self._gripper_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Left_Robot/ur3_robot/tip_Link",    # 改成真实名字
                update_period=0.0,
                history_length=4,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_.*/object/object"],
            )
        )
        self.scene.sensors["gripper_contact"] = self._gripper_contact

        self._extension_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Left_Robot/ur3_robot/Extension_Link",    # 改成真实名字
                update_period=0.0,
                history_length=4,
                debug_vis=False,
                filter_prim_paths_expr=["/World/envs/env_.*/object/object"],
            )
        )
        self.scene.sensors["extension_contact"] = self._extension_contact
        
        
        self._object_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/object/object",    # 改成真实名字
                update_period=0.0,
                history_length=4,
                debug_vis=False,
            )
        )
        self.scene.sensors["object_contact"] = self._object_contact

        # 注册到 scene
        self.scene.articulations["left_robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
    def _compute_osc_states(self):
        """计算 OSC 所需的机器人状态，全部转到 base/root frame。"""
        robot = self._robot

        # Jacobian: (N, 6, dof)
        jacobian_w = robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_body_idx, :, self.arm_joint_ids
        ]

        # M(q): (N, dof, dof)
        mass_matrix = robot.root_physx_view.get_mass_matrices()[
            :, self.arm_joint_ids, :
        ][:, :, self.arm_joint_ids]

        # g(q): (N, dof)
        gravity = robot.root_physx_view.get_generalized_gravity_forces()[:, self.arm_joint_ids]

        # root pose
        root_state_w = robot.data.root_state_w
        root_pos_w = root_state_w[:, 0:3]
        root_quat_w = root_state_w[:, 3:7]

        # world Jacobian -> base Jacobian
        root_rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, 0:3, :] = torch.bmm(root_rot_inv, jacobian_w[:, 0:3, :])
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_inv, jacobian_w[:, 3:6, :])

        # ee pose in world
        ee_state_w = robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state_w[:, 0:3]
        ee_quat_w = ee_state_w[:, 3:7]

        # ee pose in base
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # ee vel in base
        ee_lin_vel_w = ee_state_w[:, 7:10]
        ee_ang_vel_w = ee_state_w[:, 10:13]
        root_lin_vel_w = root_state_w[:, 7:10]
        root_ang_vel_w = root_state_w[:, 10:13]

        rel_lin_vel_w = ee_lin_vel_w - root_lin_vel_w
        rel_ang_vel_w = ee_ang_vel_w - root_ang_vel_w

        ee_lin_vel_b = math_utils.quat_rotate_inverse(root_quat_w, rel_lin_vel_w)
        ee_ang_vel_b = math_utils.quat_rotate_inverse(root_quat_w, rel_ang_vel_w)
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.arm_joint_ids]

        return jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel
    def set_external_command(self, pose_command: torch.Tensor):
        """
        接收外部传入的 pose_command，并强制开启'外部数据模式'。

        Args:
            pose_command (Tensor): 形状必须是 (Num_Envs, 7)。
                                   包含所有环境对应的 Pose 目标。
        """
        # 1. 格式检查与存储
        if pose_command.shape[0] != self.num_envs:
            print(
                f"⚠️ 警告: 输入 Pose 数量 ({pose_command.shape[0]}) 与环境数 ({self.num_envs}) 不一致！"
            )

        # 存入 buffer (确保在正确的 device 上)
        self._external_pose_buffer = pose_command.to(self.device).clone()

        # 2. 开启标志位
        self._use_external_pose = True

        print(f"✅ 已接收外部 Pose Command，模式切换为 [固定轨迹]。")

    # ------------------------------------------------------------------
    # 采样小物体初始位置/姿态（相对鼻腔）
    # ------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        """
        根据标志位决定：是使用 set_external_command 存下的数据，还是随机生成。
        """
        device = self.device

        # ====================================================
        # 分支 A: 标志位为 True -> 使用外部存入的数据
        # ====================================================
        if self._use_external_pose and self._external_pose_buffer is not None:
            # 1. 索引：从大 Buffer 中取出当前需要 Reset 的那几个环境的数据
            # env_ids 是索引列表，例如 [0, 5]，我们取出第 0 和第 5 行
            targets = self._external_pose_buffer[env_ids]

            # 2. 解析 (假设格式: pos[3] + rot[4])
            pos_local = targets[:, :3]
            rot = targets[:, 3:]

            # 3. 坐标系转换 (局部 -> 世界)
            # 必须加上环境原点，因为 robot 是在各自的 grid 里跑的
            pos_w = pos_local + self.scene.env_origins[env_ids]

            # 4. 赋值给 IsaacLab 的 buffer
            self.pose_command_w[env_ids, :3] = pos_w
            self.pose_command_w[env_ids, 3:] = rot

            return  # 【直接返回，不执行下面的随机逻辑】

        # ====================================================
        # 分支 B: 标志位为 False -> 原有的随机逻辑
        # ====================================================
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

        theta_rot = torch.rand(n, device=device) * 2.0 * math.pi
        q = torch.stack(
            [
                torch.cos(theta_rot / 2.0),
                torch.zeros_like(theta_rot),
                torch.zeros_like(theta_rot),
                torch.sin(theta_rot / 2.0),
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
        delta = pos_w - self.pipe_top_pos  # (N,3)
        s = torch.sum(delta * self.u_axis, dim=-1, keepdim=True)  # (N,1)

        radial = delta - s * self.u_axis  # (N,3)
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
        actions[:, 0] -> Δs
        actions[:, 1] -> Δr
        actions[:, 2] -> Δθ
        actions[:, 3] -> Δyaw
        actions[:, 4] -> gripper
        """
        self.cur_actions = torch.clamp(actions, -1.0, 1.0)

        # 1) 当前末端世界位置
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]

        # 当前在管道坐标系下的位置
        s_cur, r_cur, th_cur, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_cur = s_cur.squeeze(-1)
        r_cur = r_cur.squeeze(-1)
        th_cur = th_cur.squeeze(-1)

        in_pipe = (s_cur > 0.0) & (s_cur < (self.pipe_length - self.pipe_safety_margin))

        # 2) 管外粗 / 管内细
        step_scale = torch.where(
            in_pipe.unsqueeze(-1),
            self.step_inside,
            self.step_outside,
        )
        delta_pipe = self.cur_actions[:, 0:4] * step_scale

        delta_s = delta_pipe[:, 0]
        delta_r = delta_pipe[:, 1]
        delta_th = delta_pipe[:, 2]
        delta_yaw = delta_pipe[:, 3]   # 注意：这里用缩放后的 yaw

        # 3) 更新目标位置
        s_tgt = torch.clamp(s_cur + delta_s, min=-0.01, max=self.pipe_length)
        r_tgt = torch.clamp(
            r_cur + delta_r,
            min=0.0,
            max=self.pipe_radius - self.pipe_safety_margin,
        )
        th_tgt = th_cur + delta_th

        x_r_new = r_tgt * torch.cos(th_tgt)
        y_r_new = r_tgt * torch.sin(th_tgt)

        radial_new = torch.stack(
            [x_r_new, y_r_new, torch.zeros_like(x_r_new)],
            dim=-1,
        )
        axial_new = s_tgt.unsqueeze(-1) * self.u_axis

        self.ee_target_pos_w = self.pipe_top_pos + axial_new + radial_new
        # Clamp target z into pipe band in local frame: [-0.235, -0.21].
        ee_target_z_local = self.ee_target_pos_w[:, 2] - self.scene.env_origins[:, 2]
        ee_target_z_local = torch.clamp(
            ee_target_z_local, min=self.pipe_z_min_local, max=self.pipe_z_max_local
        )
        self.ee_target_pos_w[:, 2] = ee_target_z_local + self.scene.env_origins[:, 2]
        # 4) 更新目标姿态
        # 这里用“初始名义姿态 + 围绕局部 y 轴的 yaw 增量”
        self.ee_target_yaw = self.ee_target_yaw + delta_yaw * 0 

        local_z = torch.zeros(self.num_envs, 3, device=self.device)
        local_z[:, 2] = 1.0
        q_yaw = math_utils.quat_from_angle_axis(self.ee_target_yaw, local_z)

        self.ee_target_quat_w = math_utils.quat_mul(self.ee_nominal_quat_w, q_yaw)
        self.ee_target_quat_w = torch.nn.functional.normalize(self.ee_target_quat_w, dim=-1)

        # 5) world -> base，生成 OSC 的 pose_abs command
        root_pos_w = self._robot.data.root_state_w[:, :3]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]

        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            self.ee_target_pos_w,
            self.ee_target_quat_w,
        )
        self.osc_cmd = torch.cat([pos_b, quat_b], dim=-1)

        # 6) gripper 保持你原来的连续映射 + 限速
        raw_action = torch.clamp(self.cur_actions[:, 4:5], -1.0, 1.0)
        desired = self.gripper_max + (raw_action + 1.0) * 0.5 * (
            self.gripper_min - self.gripper_max
        )

        max_step = self.gripper_speed * self.dt
        delta = torch.clamp(desired - self.gripper_cmd, -max_step, max_step)
        self.gripper_cmd = self.gripper_cmd + delta

    # ------------------------------------------------------------------
    # 把 IK 输出写进关节目标
    # ------------------------------------------------------------------
    def _apply_action(self):
        jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b, joint_pos, joint_vel = self._compute_osc_states()

        # 设置 task-space 目标
        self._osc.set_command(
            command=self.osc_cmd,
            current_ee_pose_b=ee_pose_b,
        )

        # 计算关节 effort
        self.arm_effort_cmd = self._osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=None,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=None,
        )

        # arm: effort control
        self._robot.set_joint_effort_target(
            self.arm_effort_cmd,
            joint_ids=self.arm_joint_ids,
        )

        # tip: 仍然位置控制
        self._robot.set_joint_position_target(
            self.gripper_cmd,
            joint_ids=self.tip_joint_ids,
        )
    # ------------------------------------------------------------------
    # 终止条件（保留你原来的逻辑）
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # timeout -> truncated
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # task success -> terminated
        obj_pos_w = self._object.data.body_pos_w[:, 0, :3]
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self._object.data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        obj_goal_dist = torch.norm(self.goal_pos_w - obj_pos_w, dim=1)
        goal_success = (obj_goal_dist < self.goal_reach_thr) & (
            object_lift > self.goal_lift_thr
        )

        # failure termination: excessive contact forces on pipe / bottom
        force_fail, _, _ = self._get_force_fail_mask()
        self.force_fail = force_fail

        self.terminated = goal_success
        return self.terminated, truncated

    def _get_rewards(self) -> torch.Tensor:

        # print("-------------------------------")
        # print("Received force matrix of: ", self.scene.sensors["pipe_contact"].data.force_matrix_w)
        # print("Received contact force of: ", self.scene.sensors["gripper_contact"].data.net_forces_w)
        # print("-------------------------------")
        # print("Received force matrix of: ", self.scene.sensors["bottom_contact"].data.force_matrix_w)
        # print("Received contact force of: ", self.scene.sensors["extension_contact"].data.net_forces_w)
        # print("-------------------------------")
        # print("Received force matrix of: ", self.scene.sensors["object_contact"].data.net_forces_w)
        # ------------------- 末端 / 物体世界坐标 -------------------
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

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

        # ------------------- 截面误差 e_lat & 轴向误差 e_ax -------------------
        x_e = r_e * torch.cos(th_e)
        y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o)
        y_o = r_o * torch.sin(th_o)
        e_lat = torch.sqrt((x_e - x_o) ** 2 + (y_e - y_o) ** 2 + 1e-8)
        e_ax = torch.abs(s_e - s_o)

        # ------------------- 1) 顺序门控 (修改：给轴向一点低保) -------------------
        # 原始 g_lat 可能会在远处变成 0，导致机器人不想进管子
        # 加上 0.1 的基线，保证即使没对齐，稍微靠近点轴向也是有分的
        g_lat_raw = torch.sigmoid((0.001 - e_lat) / (0.0007 + 1e-8))
        g_lat = 0.1 + 0.9 * g_lat_raw

        # ------------------- 2) 对齐奖励 (保留) -------------------
        r_lat = torch.exp(-((e_lat / 0.004) ** 2))
        r_lat_fine = torch.exp(-((e_lat / 0.001) ** 2))

        r_ax = torch.exp(-((e_ax / 0.010) ** 2))
        r_ax_fine = torch.exp(-((e_ax / 0.003) ** 2))

        # 轴向奖励被 g_lat 门控，防止乱插
        r_ax_all = g_lat * (0.6 * r_ax + 0.4 * r_ax_fine)

        align_reward = 1.6 * (0.6 * r_lat + 0.4 * r_lat_fine) + 1.2 * r_ax_all

        # 3D 兜底 (不受 g_lat 限制，作为全域引导)
        dist_reward = 0.2 * torch.exp(-((ee_dist / 0.01) ** 2))

        # ------------------- 2.5) Yaw (保留) -------------------
        ee_quat_w = self._robot.data.body_state_w[:, self.ee_id, 3:7]
        obj_quat_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 3:7]

        def quat_to_yaw(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(siny_cosp, cosy_cosp)

        dyaw = torch.atan2(
            torch.sin(quat_to_yaw(ee_quat_w) - quat_to_yaw(obj_quat_w)),
            torch.cos(quat_to_yaw(ee_quat_w) - quat_to_yaw(obj_quat_w)),
        )
        dyaw_abs = torch.abs(dyaw)

        # Cube-like object: 4 equivalent yaw orientations (period = pi/2).
        # Fold yaw error into [-pi/4, pi/4] so 0/90/180/270 deg are all valid.
        yaw_period = 0.5 * math.pi
        dyaw_sym = torch.remainder(dyaw + 0.5 * yaw_period, yaw_period) - 0.5 * yaw_period
        dyaw_sym_abs = torch.abs(dyaw_sym)

        # Yaw 门控：只有位置比较准了才在乎 Yaw
        g_yaw = torch.sigmoid((0.003 - e_lat) / 0.0005) * torch.sigmoid(
            (0.004 - e_ax) / 0.0015
        )
        r_yaw = 0.6 * torch.exp(-((dyaw_sym_abs / 0.35) ** 2)) + 0.4 * torch.exp(
            -((dyaw_sym_abs / 0.12) ** 2)
        )
        yaw_reward = 0.5 * g_yaw * r_yaw

        # ------------------- 5) 状态监测：是否抬起 -------------------
        # 把这个提到前面，因为夹爪逻辑需要它
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self._object.data.default_root_state[:, 2]  # 假设你能取到这个
        # 或者简单的：init_z = 0.04 (你的桌子高度) + ...
        # 这里为了稳健，可以用相对高度

        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        is_lifted = object_lift > 0.002  # 抬起 2mm 就算抬了

        # ------------------- 3) 夹爪奖励 (核心修改：规则引导 + 状态锁定) -------------------
        # A. 获取当前 RL 动作 (连续值归一化到 0~1, 0=闭, 1=开)
        # 假设 actions 范围是 -1 到 1
        grip_act_norm = (self.gripper_cmd - self.gripper_min) / (
            self.gripper_max - self.gripper_min + 1e-6
        )
        grip_act_norm = torch.clamp(grip_act_norm, 0.0, 1.0).squeeze(-1)

        # B. 计算“老师”的目标 (几何规则)
        # 这里模拟 _update_gripper_rule 的逻辑，纯计算
        # 两片刃尖位置 (你需要保证 self.ee_fixed_id 在 init 里定义了)
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]

        # 这是一个简单的几何判断函数，判断物体是否在两指中间
        # check_object_in_gripper 需要你在外部定义或在这里展开
        is_captured_rule = self.check_object_in_gripper(obj_pos_w, p_fix_w, p_mov_w)

        # C. 确定目标状态 target_grip (0=闭, 1=开)
        # 逻辑：
        #   1. 如果已经抬起来了 (is_lifted)，必须死死闭合 (0.0) -> 防止掉落
        #   2. 如果没抬起，但几何条件满足 (is_captured_rule)，老师建议闭合 (0.0)
        #   3. 否则建议张开 (1.0)
        target_grip = torch.where(
            is_lifted | is_captured_rule,
            torch.zeros_like(grip_act_norm),  # 闭
            torch.ones_like(grip_act_norm),  # 开
        )

        # D. 模仿惩罚 (Imitation Penalty)
        # 权重给大一点 (0.8)，让它从零开始时重视夹爪
        grip_error = torch.abs(grip_act_norm - target_grip)
        g_close = torch.sigmoid((0.002 - e_lat) / (0.0005 + 1e-8)) * torch.sigmoid(
            (0.002 - e_ax) / (0.001 + 1e-8)
        )
        # 2. ✅ 核心修改：动态权重
        # g_close 是一个 0~1 的值，指示有多接近抓取条件
        # 远的时候 g_close -> 0, 权重 -> 0.1 (轻微引导)
        # 近的时候 g_close -> 1, 权重 -> 2.1 (强力纠正)
        dynamic_weight = 0.1 + 2.0 * g_close

        gripper_reward = -1.0 * dynamic_weight * grip_error

        # ------------------- 4) 约束与惩罚 -------------------
        wall_violation = torch.relu(r_e - (self.pipe_radius - self.pipe_safety_margin))
        wall_penalty = -10.0 * (wall_violation**2)


        # Additional out check from local z range.
        ee_z_local = ee_pos_w[:, 2] - self.scene.env_origins[:, 2]
        out_z = torch.relu(ee_z_local - self.pipe_z_max_local) + torch.relu(
            self.pipe_z_min_local - ee_z_local
        )
        out_penalty = - 5.0 * out_z

        # smaller living cost to avoid overwhelming long-horizon credit assignment in Dreamer
        step_penalty = -0.002

        # ------------------- 6) 抬起与成功奖励 -------------------
        lift_success_thr = self.success_on_thr
        # 连续抬起奖励 (鼓励它越抬越高)
        lift_reward = 5.0 * torch.clamp(object_lift / lift_success_thr, 0.0, 1.0)

        # Hysteresis state:
        # on  when lift > 5.0mm, off when lift < 3.5mm, keep state in-between.
        success_on = object_lift > self.success_on_thr
        success_off = object_lift < self.success_off_thr
        success_active = torch.where(
            success_on, torch.ones_like(self.success_active), self.success_active
        )
        success_active = torch.where(
            success_off, torch.zeros_like(success_active), success_active
        )
        self.success_active = success_active
        success_reward = 4.0 * success_active.float()
        # ------------------- 动作惩罚 -------------------
        dact = self.cur_actions - self.last_actions
        smooth_pen = -0.05 * (dact[:, :4] ** 2).sum(dim=1) - 0.02 * (
            dact[:, 4] ** 2
        ).squeeze(-1)
        self.last_actions = self.cur_actions.detach()
        jv = self._robot.data.joint_vel
        vel_pen = -1e-3 * (jv**2).sum(dim=1)

        transport_gate = torch.sigmoid((object_lift - self.goal_lift_thr) / 0.0008)
        obj_goal_vec = self.goal_pos_w - obj_pos_w
        obj_goal_dist = torch.norm(obj_goal_vec, dim=1)

        # 搬运到目标点的 dense reward
        goal_reward_coarse = torch.exp(-((obj_goal_dist / 0.020) ** 2))
        goal_reward_fine = torch.exp(-((obj_goal_dist / 0.006) ** 2))
        goal_reward = transport_gate * (
            2.0 * goal_reward_coarse + 4.0 * goal_reward_fine
        )
        # 进度奖励：只要比上一步更接近 goal 就给正反馈
        goal_progress = transport_gate * 2.0 * (self.prev_obj_goal_dist - obj_goal_dist)
        self.prev_obj_goal_dist = obj_goal_dist.detach()

        # 维持抓取，防止搬运中掉落
        hold_bonus = transport_gate * 1.0 * is_captured_rule.float()

        # 如果进入搬运阶段且未夹稳，按阶段强度给软惩罚（比 hard-threshold 更平滑）
        drop_penalty = -2.0 * transport_gate * (1.0 - is_captured_rule.float())

        # 最终成功：物体到目标点附近，且仍被抬起/夹持
        goal_success = (obj_goal_dist < self.goal_reach_thr) & (
            object_lift > self.goal_lift_thr
        )
        goal_success_reward = 8.0 * goal_success.float()
        # ------------------- 总和 -------------------
        rewards = (
            align_reward
            + dist_reward
            + gripper_reward  # ✅ 现在的模仿奖励
            # + wall_penalty
            + out_penalty
            + lift_reward
            + success_reward
            # + yaw_reward
            + step_penalty
            # + smooth_pen
            # + vel_pen
            + goal_reward  # 新增
            + goal_progress  # 新增
            + hold_bonus  # 新增
            # + drop_penalty  # 新增
            + goal_success_reward  # 新增
        )
        # mild clipping for world-model stability
        rewards = torch.clamp(rewards, min=-8.0, max=8.0)
        # self._push_live_metrics(rewards-success_reward-lift_reward, lift_reward)
        # contact force diagnostics (for failure termination tuning)
        force_fail, pipe_peak_force, bottom_peak_force = self._get_force_fail_mask()
        self.extras["log"] = {
            "reward/total": rewards.mean(),
            "reward/align": align_reward.mean(),
            "reward/dist": dist_reward.mean(),
            "reward/gripper": gripper_reward.mean(),
            "reward/wall_pen": wall_penalty.mean(),
            "reward/out_pen": out_penalty.mean(),
            "reward/lift": lift_reward.mean(),
            "reward/success": success_reward.mean(),
            "reward/goal_success": goal_success_reward.mean(),
            "metrics/success_active_rate": self.success_active.float().mean(),
            "goal_reward": goal_reward.mean(),
            "metrics/e_lat": e_lat.mean(),
            "metrics/e_ax": e_ax.mean(),
            "metrics/ee_dist": ee_dist.mean(),
            "metrics/ee_z_local": ee_z_local.mean(),
            "metrics/out_z": out_z.mean(),
            "metrics/g_lat": g_lat.mean(),
            "metrics/g_close": g_close.mean(),
            "metrics/lift": object_lift.mean(),
            "reward/yaw": yaw_reward.mean(),
            "metrics/dyaw_abs": dyaw_abs.mean(),
            "metrics/dyaw_sym_abs": dyaw_sym_abs.mean(),
            "metrics/g_yaw": g_yaw.mean(),
            "reward/smooth_pen": smooth_pen.mean(),
            "reward/vel_pen": vel_pen.mean(),
            "metrics/pipe_force_peak": pipe_peak_force.mean(),
            "metrics/bottom_force_peak": bottom_peak_force.mean(),
            "metrics/force_fail_rate": force_fail.float().mean(),
        }
        self.extras["pose"] = {
            "metrics/pose_command": self.pose_command_w[0, :],
        }
        # Log (略，保持你原来的) ...
        return rewards

    # ------------------------------------------------------------------
    # reset 环节
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            return
        env_ids = env_ids.to(self.device).long().view(-1)

        now = torch.tensor(time.perf_counter(), device=self.device, dtype=torch.float64)
        prev = self.last_reset_t.index_select(0, env_ids)
        gap = torch.where(torch.isnan(prev), prev, now - prev)
        self.last_reset_t.index_fill_(0, env_ids, now)
        self.reset_interval.scatter_(0, env_ids, gap)

        super()._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0

        # robot reset
        self._robot.reset(env_ids)

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

        # object reset
        if not hasattr(self, "pose_command_w"):
            self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
            self.pose_command_w[:, 3] = 1.0

        self._resample_command(env_ids)
        self._object.write_root_pose_to_sim(self.pose_command_w[env_ids, :], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(
            torch.zeros_like(self._object.data.root_vel_w[env_ids, :]),
            env_ids=env_ids,
        )

        # 更新目标末端状态
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        self.ee_target_pos_w[env_ids] = ee_state[env_ids, 0:3]
        self.ee_target_quat_w[env_ids] = ee_state[env_ids, 3:7]
        self.ee_nominal_quat_w[env_ids] = ee_state[env_ids, 3:7]
        self.ee_target_yaw[env_ids] = 0.0

        # gripper reset
        self.gripper_cmd[env_ids] = -0.10
        self.last_actions[env_ids] = 0.0
        self.success_active[env_ids] = False
        self.force_fail[env_ids] = False

        # goal reset
        self.goal_pos_w[env_ids] = self.goal_pos_local.unsqueeze(0) + self.scene.env_origins[env_ids]
        obj_pos_w = self._object.data.body_pos_w[env_ids, 0, :3]
        self.prev_obj_goal_dist[env_ids] = torch.norm(
            obj_pos_w - self.goal_pos_w[env_ids], dim=1
        )

        # reset OSC internal state
        self._osc.reset()

        # 当前 ee pose 作为初始 command，避免 reset 后第一拍乱冲
        root_pos_w = self._robot.data.root_state_w[:, :3]
        root_quat_w = self._robot.data.root_state_w[:, 3:7]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            self.ee_target_pos_w,
            self.ee_target_quat_w,
        )
        self.osc_cmd[env_ids] = torch.cat([pos_b[env_ids], quat_b[env_ids]], dim=-1)

        # 清零 arm effort buffer
        self.arm_effort_cmd[env_ids] = 0.0

        self.scene.write_data_to_sim()
    def check_object_in_gripper(
        self,
        p_obj_w,
        p_fix_w,
        p_mov_w,
        margin=0.000,  # 1mm, 避开铰链根部和刀尖边缘
        radius_thr=0.004,
    ):  # 4mm, 允许的横向偏差半径
        """
        判断物体是否在夹爪的“捕获区域”内。

        参数:
        - margin: 纵向容差。防止物体太靠根部（夹不住）或太靠尖端（容易滑）。
        - radius_thr: 横向容差。物体离中心线多远算“在里面”。
                    通常设为：(最大张开宽度 / 2) 或者 (当前张开宽度 / 2)。
        """

        # --- 1. 构建夹爪中心轴线向量 ---
        v_gap = p_mov_w - p_fix_w
        gap_len = torch.norm(v_gap, dim=1, keepdim=True)  # 夹爪当前长度
        gap_dir = v_gap / (gap_len + 1e-8)  # 单位方向向量

        # --- 2. 纵向投影 (Projection) ---
        # 计算物体在轴线上的投影位置 t (物理单位: 米)
        v_obj = p_obj_w - p_fix_w
        t_proj = torch.sum(v_obj * gap_dir, dim=1, keepdim=True)

        # 判定 A：物体是否在纵向有效范围内
        # margin < 投影位置 < (总长 - margin)
        is_within_length = (t_proj > margin) & (t_proj < (gap_len - margin))

        # --- 3. 横向距离 (Perpendicular Distance) ---
        # 计算物体到中心线的垂直距离
        # 投影点坐标 = 起点 + 投影长度 * 方向
        p_proj = p_fix_w + t_proj * gap_dir
        dist_perp = torch.norm(p_obj_w - p_proj, dim=1, keepdim=True)

        # 判定 B：物体是否在横向半径内
        is_within_radius = dist_perp < radius_thr

        # --- 4. 最终结果 ---
        # 必须同时满足：既在长度范围内，又在宽度范围内
        is_inside = is_within_length & is_within_radius

        return is_inside.squeeze(-1)  # 返回布尔值 Tensor (N,)

    # 辅助函数：提取 yaw (为了复用，建议放在类里或者 utils 里，这里写在 obs 里也可以)
    def _get_yaw_diff(self, ee_q, obj_q):
        def quat_to_yaw(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(siny_cosp, cosy_cosp)

        y1 = quat_to_yaw(ee_q)
        y2 = quat_to_yaw(obj_q)
        dy = y1 - y2
        return torch.atan2(torch.sin(dy), torch.cos(dy))  # wrap to -pi, pi

    def _get_observations(self) -> dict:
        # ---------------- 基础数据获取 ----------------
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]
        ee_quat_w = ee_state[:, 3:7]  # ✅ 需要四元数算 Yaw

        obj_state = self._object.data.body_state_w[:, 0]
        obj_pos_w = obj_state[:, 0:3]
        obj_quat_w = obj_state[:, 3:7]  # ✅ 需要四元数

        # ---------------- 管道坐标转换 ----------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)

        s_e = s_e.squeeze(-1)
        r_e = r_e.squeeze(-1)
        th_e = th_e.squeeze(-1)
        s_o = s_o.squeeze(-1)
        r_o = r_o.squeeze(-1)
        th_o = th_o.squeeze(-1)

        # ---------------- 相对量计算 ----------------
        ds = s_o - s_e
        dr = r_o - r_e
        dth = th_o - th_e

        # ✅ 新增：Yaw 角度差 (必须加，否则 yaw_reward 没法学)
        dyaw = self._get_yaw_diff(ee_quat_w, obj_quat_w)

        # ---------------- 几何误差 (用于计算引导信号) ----------------
        x_e = r_e * torch.cos(th_e)
        y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o)
        y_o = r_o * torch.sin(th_o)
        e_lat = torch.sqrt((x_e - x_o) ** 2 + (y_e - y_o) ** 2 + 1e-8)
        e_ax = torch.abs(ds)

        # ✅ 新增：显式引导信号 (给 Critic 的“进度条”)
        # 即使这里参数和 Reward 不完全一致也没关系，主要是给一个“归一化的进度指示”
        # 1. 对齐进度 (0~1)
        g_lat_obs = torch.sigmoid((0.001 - e_lat) / 0.0007)
        # 2. 抓取条件进度 (0~1)
        g_close_obs = torch.sigmoid((0.002 - e_ax) / 0.001) * g_lat_obs

        # ✅ 新增：物体抬起高度 (结果反馈)
        init_z = self._object.data.default_root_state[:, 2]
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        object_lift = obj_z - init_z  # 不 clamp，允许看到负值（压入地下）

        # ---------------- 其他特征 ----------------
        # 三角函数编码
        sin_th_e = torch.sin(th_e)
        cos_th_e = torch.cos(th_e)
        sin_dth = torch.sin(dth)
        cos_dth = torch.cos(dth)
        sin_dyaw = torch.sin(dyaw)
        cos_dyaw = torch.cos(dyaw)  # ✅

        # 状态标志
        margin_to_wall = (self.pipe_radius - r_e).unsqueeze(-1)

        # 归一化位置

        # 关节与夹爪
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        grip_norm = (self.gripper_cmd - self.gripper_min) / (
            self.gripper_max - self.gripper_min + 1e-6
        )
        # 计算几何规则 (复用之前的逻辑)
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]
        # 这是一个布尔值 Tensor (N,)
        is_captured_bool = self.check_object_in_gripper(obj_pos_w, p_fix_w, p_mov_w)

        # 转为 float (0.0 或 1.0)
        is_captured_obs = is_captured_bool.float().unsqueeze(-1)
        goal_vec_obj = self.goal_pos_w - obj_pos_w  # (N,3)
        goal_vec_ee = self.goal_pos_w - ee_pos_w  # (N,3)
        obj_goal_dist = torch.norm(goal_vec_obj, dim=1, keepdim=True)
        ee_goal_dist = torch.norm(goal_vec_ee, dim=1, keepdim=True)
        # ---------------- 拼接 ----------------
        obs = torch.cat(
            (
                # 1. 自身状态
                s_e.unsqueeze(-1),
                r_e.unsqueeze(-1),
                cos_th_e.unsqueeze(-1),
                sin_th_e.unsqueeze(-1),
                joint_pos,
                joint_vel,
                grip_norm,
                # 2. 相对目标状态（原来的物体相对末端）
                ds.unsqueeze(-1),
                dr.unsqueeze(-1),
                cos_dth.unsqueeze(-1),
                sin_dth.unsqueeze(-1),
                cos_dyaw.unsqueeze(-1),
                sin_dyaw.unsqueeze(-1),
                # 3. 任务进度
                object_lift.unsqueeze(-1),
                g_lat_obs.unsqueeze(-1),
                g_close_obs.unsqueeze(-1),
                is_captured_obs,
                # 4. 新增：goal conditioning
                goal_vec_obj,  # 物体到目标点
                goal_vec_ee,  # 末端到目标点（可选，但通常有帮助）
                obj_goal_dist,
                ee_goal_dist,
                # 5. 环境约束
                # margin_to_wall,
            ),
            dim=-1,
        )
        state = torch.clamp(obs, -5.0, 5.0).to(torch.float32)

        is_first = (self.episode_length_buf == 0).to(torch.int32)
        zeros = torch.zeros_like(is_first)
        force_fail, _, _ = self._get_force_fail_mask()
        self.force_fail = force_fail

        # rgb = self.get_image_observation(data_type="rgb")[..., :3]  # [N,H,W,3]

        obs = {
            "policy": state,
            "is_first": is_first,
            "is_last": zeros,
            "is_terminal": zeros,
            "failure": force_fail.to(torch.int32),
        }
        if self._use_image_obs():
            obs["image"] = self.get_image_observation(data_type="rgb")[
                ..., :3
            ]  # [N,H,W,3]
        return obs

    def get_image_observation(
        self,
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        normalize: bool = True,
        # Dreamer 常用：回放存 uint8；训练时再转 float/归一化
        rgb_mode: str = "float-11",  # "uint8" | "float01" | "float-11"
        depth_mode: str = "float01",  # "float01" | "uint8"
        max_depth: float = 10.0,  # 你环境里合理的深度上限，按需调
        output_chw: bool = False,  # True -> (N,C,H,W); False -> (N,H,W,C)
    ) -> torch.Tensor:

        sensor = self.scene.sensors["Camera"]
        images = sensor.data.output[data_type]

        # depth image conversion
        if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(
                images, sensor.data.intrinsic_matrices
            )

        # ---------- RGB ----------
        if data_type == "rgb":
            img = images

            # 统一成 float32 便于处理
            if img.dtype != torch.float32:
                img = img.float()

            if normalize:
                # 兼容：有的相机给 0~255，有的给 0~1
                # 用 max 判断比硬除 255 更稳
                if img.max() > 1.5:
                    img = img / 255.0
                img = img.clamp(0.0, 1.0)

            # 输出模式（更适合 Dreamer）
            if rgb_mode == "uint8":
                out = (img * 255.0 + 0.5).to(torch.uint8)
            elif rgb_mode == "float01":
                out = img
            elif rgb_mode == "float-11":
                out = img * 2.0 - 1.0
            else:
                raise ValueError(f"Unknown rgb_mode: {rgb_mode}")

            # Dreamer 实现有的吃 CHW，有的吃 HWC；给你一个开关
            if output_chw and out.ndim == 4 and out.shape[-1] in (1, 3, 4):
                out = out.permute(0, 3, 1, 2).contiguous()

            return out.clone()

        # ---------- Depth / Distance ----------
        if ("distance_to" in data_type) or ("depth" in data_type):
            depth = images.clone()

            # 清理 inf/nan
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            if normalize:
                # 固定尺度归一化到 [0,1]（Dreamer 更喜欢稳定的输入分布）
                depth = depth.clamp(0.0, max_depth) / max_depth

            if depth_mode == "uint8":
                out = (depth * 255.0 + 0.5).to(torch.uint8)
            elif depth_mode == "float01":
                out = depth.float()
            else:
                raise ValueError(f"Unknown depth_mode: {depth_mode}")

            if output_chw and out.ndim == 4 and out.shape[-1] in (1, 3, 4):
                out = out.permute(0, 3, 1, 2).contiguous()

            return out.clone()

        # 其他类型：原样返回（或你自行加分支）
        return images.clone()

    # ------------------------------------------------------------------
    # debug 可视化
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
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
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(
                        event
                    )
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
        object_pose_w = self._object.data.body_state_w[:, 0]
        self.goal_pose_visualizer.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])

        # === 新增：object 5mm 范围球 ===
        if hasattr(self, "range_visualizer"):
            # 取 object 的世界位姿
            obj_state_w = self._object.data.body_state_w[
                :, 0
            ]  # (num_envs, 13?) 取第0刚体
            obj_pos_w = obj_state_w[:, 0:3]

            # 若你的 range_vis cfg 里只有一个 marker prototype（sphere）
            # 强烈建议显式传 marker_indices

            # self.range_visualizer.visualize(obj_pos_w, obj_quat_w, marker_indices=idx)

    # ------------------------------------------------------------------
    # 其它工具函数
    # ------------------------------------------------------------------
    def init_robot_ik(self):
        self._robot_ik = DifferentialInverseKinematicsAction(
            self.cfg.left_robot_ik, self.scene
        )

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

    def _get_contact_peak(self, sensor) -> torch.Tensor:
        """Return per-env peak norm from contact sensor history/current forces."""
        if sensor is None:
            return torch.zeros(self.num_envs, device=self.device)

        data = sensor.data
        if hasattr(data, "net_forces_w_history") and data.net_forces_w_history is not None:
            forces = data.net_forces_w_history
        elif hasattr(data, "net_forces_w") and data.net_forces_w is not None:
            forces = data.net_forces_w.unsqueeze(1)
        elif hasattr(data, "force_matrix_w") and data.force_matrix_w is not None:
            forces = data.force_matrix_w
        else:
            return torch.zeros(self.num_envs, device=self.device)

        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        forces = forces.reshape(forces.shape[0], -1, 3)
        return torch.linalg.vector_norm(forces, dim=-1).max(dim=1).values

    def _get_force_fail_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Failure mask based on excessive contact force on pipe/bottom."""
        pipe_peak = self._get_contact_peak(getattr(self, "_pipe_contact", None))
        bottom_peak = self._get_contact_peak(getattr(self, "_bottom_contact", None))
        force_fail = (pipe_peak > self.pipe_force_fail_thr) | (
            bottom_peak > self.bottom_force_fail_thr
        )
        return force_fail, pipe_peak, bottom_peak
