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
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
# 自己的模块
from .utils.myfunc import *
from .ur3_lift_pipe_ik_env_cfg import Ur3LiftPipeEnvCfg
from .utils.robot_ik_fun import DifferentialInverseKinematicsAction
import numpy as np
import gym  # 或 gymnasium as gym，跟你工程里一致


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
        self.ee_fixed_id = self._robot.data.body_names.index("scrissor_fixed")
        self.ee_move_id = self._robot.data.body_names.index("scrissor_move")
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

        self._build_dreamer_observation_space() 
        
        # debug 可视化
        self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b = torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        self.prev_ee_to_obj_dist = torch.zeros(self.num_envs, device=self.device)
        # __init__
        self.last_actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.cur_actions  = torch.zeros(self.num_envs, 5, device=self.device)

    def _build_dreamer_observation_space(self):
        # num_envs（IsaacLab vectorized env）
        nenv = int(self.num_envs)

        # state 维度：用你当前拼接逻辑推出来
        # state = 19 + 2 * num_joints
        num_joints = int(self._robot.data.joint_pos.shape[1])
        state_dim = int(self._get_observations()["policy"].shape[-1])

        # 图像分辨率：尽量从 cfg/camera 里取，不行就 fallback
        H = getattr(getattr(self.cfg, "camera", None), "height", 128)
        W = getattr(getattr(self.cfg, "camera", None), "width", 128)

        # 和你实际返回对齐：
        # policy: float32 in [-5, 5]
        policy_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(nenv, state_dim), dtype=np.float32
        )

        # image: uint8 [0,255]，shape=(nenv,H,W,3)
        image_space = gym.spaces.Box(
            low=0, high=255, shape=(nenv, H, W, 3), dtype=np.uint8
        )

        # flags / failure：你返回的是 int32（torch.int32），这里也用 int32 更一致
        flag_space = gym.spaces.Box(0, 1, (), dtype=bool)

        self._observation_space = gym.spaces.Dict({
            "policy": policy_space,
            "image": image_space,
            "is_first": flag_space,
            "is_last": flag_space,
            "is_terminal": flag_space,
            "failure": flag_space,
        })

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
        self.cur_actions = torch.clamp(actions, -1.0, 1.0)

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
        delta_yaw = actions[:, 3]
        self.ee_target_yaw = self.ee_target_yaw + delta_yaw

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

        # q_align = self.q_align_y_to_z.to(self.device)
        # if q_align.ndim == 1:
        #     q_align = q_align.unsqueeze(0)
        # q_align = q_align.expand(q_yaw.shape[0], -1)  # (N,4)

        # self.ee_target_quat_w = math_utils.quat_mul(q_yaw, q_align)
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
        # --- 原始代码 (即现在的做法) ---
        # close_flag = (actions[:, 4:5] > 0.0).float()
        # desired = close_flag * self.gripper_min + (1.0 - close_flag) * self.gripper_max

        # --- ✅ 优化后：连续映射控制 ---
        # 1. 将动作 [-1, 1] 线性映射到 [gripper_max, gripper_min]
        #    注意：通常定义 action=1 为闭合(min_width)，action=-1 为张开(max_width)
        #    公式：target = open + (action + 1)/2 * (close - open)
        raw_action = actions[:, 4:5]
        # 限制范围，防止 tanh 失效后越界（虽然通常不需要，但在 IK 里安全第一）
        raw_action = torch.clamp(raw_action, -1.0, 1.0) 

        # 线性插值
        desired = self.gripper_max + (raw_action + 1.0) * 0.5 * (self.gripper_min - self.gripper_max)

        # --- 2. 速度限制 (Rate Limiter) ---
        # 这部分你原来的逻辑很好，必须保留！用来模拟真实电机的速度，防止瞬移
        max_step = self.gripper_speed * self.dt
        delta = torch.clamp(desired - self.gripper_cmd, -max_step, max_step)
        self.gripper_cmd = self.gripper_cmd + delta

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
        # ------------------- 末端 / 物体世界坐标 -------------------
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        ee_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)
        self.ee_dist = ee_dist

        # ------------------- 管道坐标 -------------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)
        s_e = s_e.squeeze(-1); r_e = r_e.squeeze(-1); th_e = th_e.squeeze(-1)
        s_o = s_o.squeeze(-1); r_o = r_o.squeeze(-1); th_o = th_o.squeeze(-1)

        # ------------------- 截面误差 e_lat & 轴向误差 e_ax -------------------
        x_e = r_e * torch.cos(th_e); y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o); y_o = r_o * torch.sin(th_o)
        e_lat = torch.sqrt((x_e - x_o) ** 2 + (y_e - y_o) ** 2 + 1e-8)
        e_ax = torch.abs(s_e - s_o)

        # ------------------- 1) 顺序门控 (修改：给轴向一点低保) -------------------
        # 原始 g_lat 可能会在远处变成 0，导致机器人不想进管子
        # 加上 0.1 的基线，保证即使没对齐，稍微靠近点轴向也是有分的
        g_lat_raw = torch.sigmoid((0.001 - e_lat) / (0.0007 + 1e-8))
        g_lat = 0.1 + 0.9 * g_lat_raw 

        # ------------------- 2) 对齐奖励 (保留) -------------------
        r_lat = torch.exp(- (e_lat / 0.004) ** 2)
        r_lat_fine = torch.exp(- (e_lat / 0.001) ** 2)

        r_ax = torch.exp(- (e_ax / 0.010) ** 2)
        r_ax_fine = torch.exp(- (e_ax / 0.003) ** 2)
        
        # 轴向奖励被 g_lat 门控，防止乱插
        r_ax_all = g_lat * (0.6 * r_ax + 0.4 * r_ax_fine)

        align_reward = 1.6 * (0.6 * r_lat + 0.4 * r_lat_fine) + 1.2 * r_ax_all

        # 3D 兜底 (不受 g_lat 限制，作为全域引导)
        dist_reward = 0.2 * torch.exp(- (ee_dist / 0.01) ** 2)

        # ------------------- 2.5) Yaw (保留) -------------------
        ee_quat_w = self._robot.data.body_state_w[:, self.ee_id, 3:7]
        obj_quat_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 3:7]
        
        def quat_to_yaw(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(siny_cosp, cosy_cosp)

        dyaw_abs = torch.abs(torch.atan2(torch.sin(quat_to_yaw(ee_quat_w) - quat_to_yaw(obj_quat_w)), 
                                         torch.cos(quat_to_yaw(ee_quat_w) - quat_to_yaw(obj_quat_w))))
        
        # Yaw 门控：只有位置比较准了才在乎 Yaw
        g_yaw = torch.sigmoid((0.003 - e_lat) / 0.0005) * torch.sigmoid((0.004 - e_ax) / 0.0015)
        r_yaw = 0.6 * torch.exp(- (dyaw_abs / 0.35) ** 2) + 0.4 * torch.exp(- (dyaw_abs / 0.12) ** 2)
        yaw_reward = 0.5 * g_yaw * r_yaw

        # ------------------- 5) 状态监测：是否抬起 -------------------
        # 把这个提到前面，因为夹爪逻辑需要它
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self._object.data.default_root_state[:, 2] # 假设你能取到这个
        # 或者简单的：init_z = 0.04 (你的桌子高度) + ...
        # 这里为了稳健，可以用相对高度
        
        object_lift = torch.clamp(obj_z - init_z, min=0.0)
        is_lifted = object_lift > 0.002 # 抬起 2mm 就算抬了

        # ------------------- 3) 夹爪奖励 (核心修改：规则引导 + 状态锁定) -------------------
        # A. 获取当前 RL 动作 (连续值归一化到 0~1, 0=闭, 1=开)
        # 假设 actions 范围是 -1 到 1
        grip_act_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
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
            torch.zeros_like(grip_act_norm), # 闭
            torch.ones_like(grip_act_norm)   # 开
        )

        # D. 模仿惩罚 (Imitation Penalty)
        # 权重给大一点 (0.8)，让它从零开始时重视夹爪
        grip_error = torch.abs(grip_act_norm - target_grip)
        g_close = torch.sigmoid((0.002 - e_lat) / (0.0005 + 1e-8)) * torch.sigmoid((0.002 - e_ax) / (0.001 + 1e-8))
        # 2. ✅ 核心修改：动态权重
        # g_close 是一个 0~1 的值，指示有多接近抓取条件
        # 远的时候 g_close -> 0, 权重 -> 0.1 (轻微引导)
        # 近的时候 g_close -> 1, 权重 -> 2.1 (强力纠正)
        dynamic_weight = 0.1 + 2.0 * g_close 

        gripper_reward = -1.0 * dynamic_weight * grip_error

        # ------------------- 4) 约束与惩罚 -------------------
        wall_violation = torch.relu(r_e - (self.pipe_radius - self.pipe_safety_margin))
        wall_penalty = -10.0 * (wall_violation ** 2)

        out_ax = torch.relu(-s_e) + torch.relu(s_e - (self.pipe_length - self.pipe_safety_margin))
        out_penalty = -3.0 * out_ax

        step_penalty = -0.01

        # ------------------- 6) 抬起与成功奖励 -------------------
        lift_success_thr = 0.01
        # 连续抬起奖励 (鼓励它越抬越高)
        lift_reward = 5.0 * torch.clamp(object_lift / lift_success_thr, 0.0, 1.0)
        
        success = (object_lift > lift_success_thr)
        self.terminated = success
        success_reward = 10.0 * success.float()
        # ------------------- 动作惩罚 -------------------
        dact = self.cur_actions - self.last_actions
        smooth_pen = -0.05 * (dact[:, :4] ** 2).sum(dim=1) - 0.02 * (dact[:, 4] ** 2).squeeze(-1)
        self.last_actions = self.cur_actions.detach()
        jv = self._robot.data.joint_vel
        vel_pen = -1e-3 * (jv ** 2).sum(dim=1)
        # ------------------- 总和 -------------------
        rewards = (
            align_reward +
            dist_reward +
            gripper_reward +  # ✅ 现在的模仿奖励
            wall_penalty +
            out_penalty +
            lift_reward +
            success_reward +
            yaw_reward +
            step_penalty +
            smooth_pen +
            vel_pen
        ) * 0.05
        self.extras["log"] = {
            "reward/total": rewards.mean(),
            "reward/align": align_reward.mean(),
            "reward/dist": dist_reward.mean(),
            "reward/gripper": gripper_reward.mean(),
            "reward/wall_pen": wall_penalty.mean(),
            "reward/out_pen": out_penalty.mean(),
            "reward/lift": lift_reward.mean(),
            "reward/success": success_reward.mean(),
            "metrics/e_lat": e_lat.mean(),
            "metrics/e_ax": e_ax.mean(),
            "metrics/ee_dist": ee_dist.mean(),
            "metrics/g_lat": g_lat.mean(),
            "metrics/g_close": g_close.mean(),
            "metrics/lift": object_lift.mean(),
            "reward/yaw": yaw_reward.mean(),
            "metrics/dyaw_abs": dyaw_abs.mean(),
            "metrics/g_yaw": g_yaw.mean(),
            "reward/smooth_pen": smooth_pen.mean(),
            "reward/vel_pen": vel_pen.mean(),
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

    def check_object_in_gripper(self, p_obj_w, p_fix_w, p_mov_w, 
                                margin=0.000,       # 1mm, 避开铰链根部和刀尖边缘
                                radius_thr=0.004):  # 4mm, 允许的横向偏差半径
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
        gap_dir = v_gap / (gap_len + 1e-8)                # 单位方向向量

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

        return is_inside.squeeze(-1) # 返回布尔值 Tensor (N,)
    

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
        return torch.atan2(torch.sin(dy), torch.cos(dy)) # wrap to -pi, pi

    def _get_observations(self) -> dict:
        # ---------------- 基础数据获取 ----------------
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        ee_pos_w = ee_state[:, 0:3]
        ee_quat_w = ee_state[:, 3:7] # ✅ 需要四元数算 Yaw

        obj_state = self._object.data.body_state_w[:, 0]
        obj_pos_w = obj_state[:, 0:3]
        obj_quat_w = obj_state[:, 3:7] # ✅ 需要四元数

        # ---------------- 管道坐标转换 ----------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)

        s_e = s_e.squeeze(-1); r_e = r_e.squeeze(-1); th_e = th_e.squeeze(-1)
        s_o = s_o.squeeze(-1); r_o = r_o.squeeze(-1); th_o = th_o.squeeze(-1)

        # ---------------- 相对量计算 ----------------
        ds = s_o - s_e
        dr = r_o - r_e
        dth = th_o - th_e
        
        # ✅ 新增：Yaw 角度差 (必须加，否则 yaw_reward 没法学)
        dyaw = self._get_yaw_diff(ee_quat_w, obj_quat_w)

        # ---------------- 几何误差 (用于计算引导信号) ----------------
        x_e = r_e * torch.cos(th_e); y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o); y_o = r_o * torch.sin(th_o)
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
        object_lift = obj_z - init_z # 不 clamp，允许看到负值（压入地下）

        # ---------------- 其他特征 ----------------
        # 三角函数编码
        sin_th_e = torch.sin(th_e); cos_th_e = torch.cos(th_e)
        sin_dth = torch.sin(dth);   cos_dth = torch.cos(dth)
        sin_dyaw = torch.sin(dyaw); cos_dyaw = torch.cos(dyaw) # ✅

        # 状态标志
        in_pipe = ((s_e > 0.0) & (s_e < (self.pipe_length - self.pipe_safety_margin))).float()
        margin_to_wall = (self.pipe_radius - r_e).unsqueeze(-1)
        
        # 归一化位置
        s_e_norm = (s_e / (self.pipe_length + 1e-6)).unsqueeze(-1)
        r_e_norm = (r_e / (self.pipe_radius + 1e-6)).unsqueeze(-1)

        # 关节与夹爪
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
        # 计算几何规则 (复用之前的逻辑)
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]
        # 这是一个布尔值 Tensor (N,)
        is_captured_bool = self.check_object_in_gripper(obj_pos_w, p_fix_w, p_mov_w)
        
        # 转为 float (0.0 或 1.0)
        is_captured_obs = is_captured_bool.float().unsqueeze(-1)
        # ---------------- 拼接 ----------------
        obs = torch.cat(
            (
                # 1. 自身状态 (Proprioception)
                s_e.unsqueeze(-1), r_e.unsqueeze(-1),
                cos_th_e.unsqueeze(-1), sin_th_e.unsqueeze(-1),
                joint_pos, joint_vel,
                grip_norm,

                # 2. 目标相对状态 (Goal Relative)
                ds.unsqueeze(-1), dr.unsqueeze(-1),
                cos_dth.unsqueeze(-1), sin_dth.unsqueeze(-1),
                cos_dyaw.unsqueeze(-1), sin_dyaw.unsqueeze(-1), # ✅ 加了 Yaw
                
                # 3. 任务进度指示器 (Task Progress) - 关键！
                object_lift.unsqueeze(-1), # ✅ 告诉它提起来没
                g_lat_obs.unsqueeze(-1),   # ✅ 告诉它对齐没
                g_close_obs.unsqueeze(-1), # ✅ 告诉它能抓没
                is_captured_obs,

                # 4. 环境感知 (Environment)
                margin_to_wall,
                # in_pipe.unsqueeze(-1),
            ),
            dim=-1,
        )

        state = torch.clamp(obs, -5.0, 5.0).to(torch.float32)

        is_first = (self.episode_length_buf == 0).to(torch.int32)
        zeros = torch.zeros_like(is_first)

        rgb = self.get_image_observation(data_type="rgb")[..., :3]  # [N,H,W,3]
        return {
            "policy": state,
            "image": rgb,
            "is_first": is_first,
            "is_last": zeros,
            "is_terminal": zeros,
            "failure": zeros,
        }


    def get_image_observation(
        self,
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        normalize: bool = True,
        # Dreamer 常用：回放存 uint8；训练时再转 float/归一化
        rgb_mode: str = "float-11",     # "uint8" | "float01" | "float-11"
        depth_mode: str = "float01", # "float01" | "uint8"
        max_depth: float = 10.0,     # 你环境里合理的深度上限，按需调
        output_chw: bool = False,    # True -> (N,C,H,W); False -> (N,H,W,C)
    ) -> torch.Tensor:
        
        sensor = self.scene.sensors["Camera"]
        images = sensor.data.output[data_type]

        # depth image conversion
        if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

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
        object_pose_w = self._object.data.body_state_w[:, 0]
        self.goal_pose_visualizer.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])
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

            # self.range_visualizer.visualize(obj_pos_w, obj_quat_w, marker_indices=idx)
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



