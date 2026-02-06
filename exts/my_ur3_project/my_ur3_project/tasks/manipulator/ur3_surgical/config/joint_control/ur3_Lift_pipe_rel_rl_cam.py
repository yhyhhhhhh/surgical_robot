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

        # ------------------- 规则夹爪：模式 & 防抖 -------------------
        # 0=open, 1=close
        self.grip_mode = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.grip_hold_steps = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int32)

        # 最小保持时间（秒 -> steps），防止频繁开关
        self.grip_min_hold_sec = 0.15
        self.grip_min_hold = max(1, int(self.grip_min_hold_sec / self.dt))

        # 滞回阈值：进入夹缝程度 pre_score 的开/关阈值
        self.pre_close_thr = 0.65   # 满足则倾向闭合
        self.pre_open_thr  = 0.35   # 低于则倾向张开（滞回）

        # 距离中心线阈值（越小越“在夹缝中”）
        self.perp_close_thr = 0.0020
        self.perp_open_thr  = 0.0030

        # gap 关闭阈值（跟你 compute_scissor_scores 里的 gap_close_thr 保持一致或略放宽）
        self.gap_close_thr = 0.001

        # ------------------------------------------------------------------


    def _build_dreamer_observation_space(self):
        # num_envs（IsaacLab vectorized env）
        nenv = int(self.num_envs)

        # state 维度：用你当前拼接逻辑推出来
        # state = 19 + 2 * num_joints
        num_joints = int(self._robot.data.joint_pos.shape[1])
        state_dim = 19 + 2 * num_joints

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
    def _resample_command(self, env_ids: Sequence[int], p_use_default: float = 1.0,):
        device = self.device
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)
        n = env_ids.numel()

        # 默认 root state（world系）
        obj_default = self._object.data.default_root_state[env_ids].clone()
        obj_default[:, :3] += self.scene.env_origins[env_ids]

        # 先全写默认
        self.pose_command_w[env_ids, :7] = obj_default[:, :7]

        # 抽样：哪些 env 用随机（其余保留默认）
        use_random = (torch.rand(n, device=device) >= p_use_default)
        if not use_random.any():
            return

        rid = env_ids[use_random]
        m = rid.numel()

        # 随机位置（你的原逻辑）
        max_r = 0.004
        center_x = 0.0
        center_y = -0.29

        u = torch.rand(m, device=device)
        r = max_r * torch.sqrt(u)
        theta = torch.rand(m, device=device) * 2.0 * math.pi

        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)
        z = obj_default[use_random, 2]  # 用默认 z 更通用

        new_pos = torch.stack([x, y, z], dim=1) + self.scene.env_origins[rid]
        self.pose_command_w[rid, :3] = new_pos

        # 随机 yaw
        theta_rot = torch.rand(m, device=device) * 2.0 * math.pi
        q = torch.stack(
            [
                torch.cos(theta_rot / 2.0),  # w
                torch.zeros_like(theta_rot), # x
                torch.zeros_like(theta_rot), # y
                torch.sin(theta_rot / 2.0),  # z
            ],
            dim=1,
        )
        self.pose_command_w[rid, 3:7] = q


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
        self._update_gripper_rule()
        # 6) 抓手阻抗
        # target = self.gripper_min + (actions[:, 4:5] + 1.0) * 0.5 * (self.gripper_max - self.gripper_min)
        # self.gripper_cmd = target

        # --- 夹爪：开关目标（用 a4 的符号） ---
        # close_flag = (actions[:, 4:5] > 0.0).float()  # a4>0 关，否则开
        # desired = close_flag * self.gripper_min + (1.0 - close_flag) * self.gripper_max

        # # --- 速度限制（平滑追踪，防抖）---
        # max_step = self.gripper_speed * self.dt  # rad per step
        # delta = torch.clamp(desired - self.gripper_cmd, -max_step, max_step)
        # self.gripper_cmd = self.gripper_cmd + delta
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

        ee_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)  # 3D 距离（兜底用）
        self.ee_dist = ee_dist

        # ------------------- 管道坐标 -------------------
        s_e, r_e, th_e, _, _ = self._world_to_pipe_coords(ee_pos_w)
        s_o, r_o, th_o, _, _ = self._world_to_pipe_coords(obj_pos_w)
        s_e = s_e.squeeze(-1); r_e = r_e.squeeze(-1); th_e = th_e.squeeze(-1)
        s_o = s_o.squeeze(-1); r_o = r_o.squeeze(-1); th_o = th_o.squeeze(-1)

        # ------------------- 截面误差 e_lat（在截面里算 xy） -------------------
        x_e = r_e * torch.cos(th_e)
        y_e = r_e * torch.sin(th_e)
        x_o = r_o * torch.cos(th_o)
        y_o = r_o * torch.sin(th_o)
        e_lat = torch.sqrt((x_e - x_o) ** 2 + (y_e - y_o) ** 2 + 1e-8)

        # ------------------- 轴向误差 e_ax -------------------
        e_ax = torch.abs(s_e - s_o)

        # ------------------- 1) 顺序门控：截面对齐好，轴向才“值钱” -------------------
        # 3mm 左右开始打开轴向奖励（可调）
        g_lat = torch.sigmoid((0.002 - e_lat) / (0.0007 + 1e-8))  # in (0,1)
        g_lat = g_lat

        # ------------------- 2) 绝对型对齐奖励（多尺度，高精度更重） -------------------
        # 截面：粗(4mm) + 细(1.5mm)
        r_lat = torch.exp(- (e_lat / 0.004) ** 2)
        r_lat_fine = torch.exp(- (e_lat / 0.001) ** 2)

        # 轴向：粗(10mm) + 细(3mm)，并乘门控 g_lat
        r_ax = torch.exp(- (e_ax / 0.010) ** 2)
        r_ax_fine = torch.exp(- (e_ax / 0.003) ** 2)
        r_ax_all = g_lat * (0.6 * r_ax + 0.4 * r_ax_fine)

        # 合并：截面主导，轴向次之（可调权重）
        align_reward = 1.6 * (0.6 * r_lat + 0.6 * r_lat_fine) + 1.0 * r_ax_all

        # 3D 兜底：防止分解误差边角（权重小）
        dist_reward = 0.25 * torch.exp(- (ee_dist / 0.006) ** 2)

        # ------------------- 2.5) yaw 对齐奖励（绕 world z 轴） -------------------
        # 取末端/物体世界四元数 (w,x,y,z)
        ee_quat_w = self._robot.data.body_state_w[:, self.ee_id, 3:7]
        obj_quat_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 3:7]

        # 从 quat 提取 yaw（Z 轴旋转），采用常用 ZYX yaw 公式
        def quat_to_yaw(q):
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            return torch.atan2(siny_cosp, cosy_cosp)

        yaw_ee = quat_to_yaw(ee_quat_w)
        yaw_obj = quat_to_yaw(obj_quat_w)

        # wrap 到 [-pi, pi]
        dyaw = yaw_ee - yaw_obj
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        dyaw_abs = torch.abs(dyaw)

        # 只在“已经很接近”时才奖励 yaw（避免远处刷姿态）
        # 你可以用 g_close 或者单独门控，这里用单独门控更直观
        g_yaw = torch.sigmoid((0.003 - e_lat) / (0.0005 + 1e-8)) * torch.sigmoid((0.004 - e_ax) / (0.0015 + 1e-8))

        # 多尺度 yaw shaping（粗对齐 + 精对齐）
        r_yaw = 0.6 * torch.exp(- (dyaw_abs / 0.35) ** 2) + 0.4 * torch.exp(- (dyaw_abs / 0.12) ** 2)

        yaw_reward = 0.4 * g_yaw * r_yaw

        # ------------------- 3) 夹爪“跟随目标”奖励（绝对型） -------------------
        # grip_norm: 0=闭, 1=开
        # grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
        # grip_norm = torch.clamp(grip_norm, 0.0, 1.0).squeeze(-1)

        # # 当 (e_lat<~1.8mm 且 e_ax<~3mm) 时，开始希望闭合
        g_close = torch.sigmoid((0.0018 - e_lat) / (0.0005 + 1e-8)) * torch.sigmoid((0.002 - e_ax) / (0.001 + 1e-8))
        # target_grip = 1.0 - g_close  # 远 -> 1(开), 近 -> 0(闭)

        # # “跟随目标”的绝对奖励：1 - |grip - target|  (范围[0,1])
        # gripper_reward = 0.6 * (1.0 - torch.abs(grip_norm - target_grip))
        # grip_norm: 0=闭, 1=开
        grip_norm = (self.gripper_cmd - self.gripper_min) / (self.gripper_max - self.gripper_min + 1e-6)
        grip_norm = torch.clamp(grip_norm, 0.0, 1.0).squeeze(-1)

        # 希望闭合的门控（你已有 g_close）
        # target 开关：远=1(开)，近=0(闭)
        target_grip = (1.0 - g_close)

        # 用“接近开/闭两端”的 reward（避免卡在中间）
        gripper_reward = 0.4 * (1.0 - torch.abs(grip_norm - target_grip))

        # 两片刃尖位置（你已经有 id）
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]

        # 物体位置/速度
        p_obj_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]
        v_obj_w = self.scene.rigid_objects["object"].data.root_vel_w[:, 0:3]  # 有就传，没有就传 None

        # pregrasp, grasp_proxy, grasped, gap, d_perp, t = self.compute_scissor_grasp_flags(
        #     p_obj_w, p_fix_w, p_mov_w, v_obj_w=v_obj_w,
        #     gap_close_thr=0.0030,  # 你后面根据 gap 分布调
        #     perp_thr=0.0020,
        #     t_margin=0.0005,
        #     v_obj_thr=0.02,
        # )
        # ------------------- 4) 安全约束：越管壁/跑出管长 -------------------
        # 径向越界：平方惩罚更强、且平滑
        wall_violation = torch.relu(r_e - (self.pipe_radius - self.pipe_safety_margin))
        wall_penalty = -10.0 * (wall_violation ** 2)

        # 轴向越界：如果你希望“必须留在管内”，这里给明显惩罚
        # （起点已在管内，这项主要防 retreat 或冲出长度）
        out_ax = torch.relu(-s_e) + torch.relu(s_e - (self.pipe_length - self.pipe_safety_margin))
        out_penalty = -3.0 * out_ax

        # ------------------- 5) 抬起/成功（绝对型） -------------------
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)
        target_pos_w = self.scene.rigid_objects["object"].data.default_root_state[:, 0:3].clone()
        target_pos_w[:, 2] += 0.002  # 设置目标高度 Z

        # 3. 计算 3D 欧氏距离 (Euclidean Distance)
        dist_to_target = torch.norm(obj_pos_w - target_pos_w, dim=1)

        # 4. 将距离转化为奖励 (0~1)
        #    使用高斯核：距离越近，奖励越接近 1.0
        #    分母 0.005 (5mm) 控制灵敏度：如果偏离目标 5mm，奖励会衰减到约 0.36
        lift_reward_val = torch.exp(- (dist_to_target / 0.002) ** 2)

        # 5. 缩放权重 (配合 step_scale)
        #    注意：一定要乘 step_scale，否则这个 1.0 的奖励会淹没对齐奖励
        #    step_scale 是我在上一条回答建议的 0.02
        lift_reward =  5.0 * lift_reward_val
        lift_success_thr = 0.01
        lift_reward += 8.0 * torch.clamp(object_lift / lift_success_thr, 0.0, 1.0)

        success = (object_lift > lift_success_thr)
        self.terminated = success
        success_reward = 8.0 * success.float()

        # ------------------- 6) 轻微每步成本：防止“站桩拿小奖励” -------------------
        step_penalty = -0.01

        rewards = (
            0.6 * align_reward +
            # dist_reward +
            # gripper_reward +
            wall_penalty +
            out_penalty +
            lift_reward +
            success_reward +
            # yaw_reward +
            step_penalty
        )

        # ------------------- 日志 -------------------
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
        }

        return rewards
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
    def compute_scissor_scores(self, p_obj_w, p_fix_w, p_mov_w,
                           gap_close_thr=0.0030,  # 3mm
                           perp_thr=0.0020,       # 2mm
                           t_margin=0.0000,       # 0.5mm
                           k_t=0.0005,            # 平滑带宽
                           k_gap=0.0004):         # 平滑带宽
        """
        返回：
        gap, d_perp: 物理量
        pre_score:   物体进入夹缝的连续分数(0~1)
        close_score: 闭合程度连续分数(0~1)
        grasp_score: 抓取连续分数(0~1)
        grasp_proxy: bool（用于事件/门控）
        """
        v = p_mov_w - p_fix_w
        gap = torch.norm(v, dim=1)                           # (N,)
        o = v / (gap.unsqueeze(-1) + 1e-8)                   # (N,3)

        w = p_obj_w - p_fix_w
        t = torch.sum(w * o, dim=1)                          # (N,)

        # 物体到夹缝中心线的垂直距离
        p_proj = p_fix_w + t.unsqueeze(-1) * o
        d_perp = torch.norm(p_obj_w - p_proj, dim=1)         # (N,)

        # “between”的连续版本：在 (t_margin, gap - t_margin) 内得分高
        left  = torch.sigmoid((t - t_margin) / (k_t + 1e-8))
        right = torch.sigmoid(((gap - t_margin) - t) / (k_t + 1e-8))
        between_score = left * right                          # (N,) 0~1

        # “near”的连续版本：距离越小越接近1
        near_score = torch.exp(- (d_perp / (perp_thr + 1e-8)) ** 2)

        pre_score = between_score * near_score               # 进入夹缝程度

        # 闭合程度：gap 小于阈值越多越接近 1
        close_score = torch.sigmoid((gap_close_thr - gap) / (k_gap + 1e-8))

        grasp_score = pre_score * close_score

        # 布尔判定（给事件/门控用，不用于平滑奖励）
        grasp_proxy = (pre_score > 0.6) & (gap < gap_close_thr)

        return gap, d_perp, pre_score, close_score, grasp_score, grasp_proxy

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

        self._resample_command(env_ids,p_use_default=0.0)
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
        self.grip_mode[env_ids] = 0.0  # 默认张开
        self.grip_hold_steps[env_ids] = 0
    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:


        # 末端世界系
        ee_state = self._robot.data.body_state_w[:, self.ee_id]
        # ee_pos_w = ee_state[:, 0:3]
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        # 是否夹取物体
        # 两片刃尖位置
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]

        # 物体位置/速度
        p_obj_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]
        v_obj_w = self.scene.rigid_objects["object"].data.root_vel_w[:, 0:3]
        is_captured = self.check_object_in_gripper(p_obj_w, p_fix_w, p_mov_w)
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
                is_captured.unsqueeze(-1),
            ),
            dim=-1,
        )

        state = torch.clamp(obs, -5.0, 5.0).to(torch.float32)

        rgb = self.get_image_observation(data_type="rgb")[..., :3]  # [N,H,W,3]
        

        # reset 后 episode_length_buf 会变成 0，所以这里能得到 is_first
        is_first = (self.episode_length_buf == 0).to(torch.int32)
        zeros = torch.zeros_like(is_first)

        return {
            "policy": state,
            "image": rgb,
            "is_first": is_first,     # ✅ 可靠
            "is_last": zeros,         # 先占位（真正 last 在 step() 里做）
            "is_terminal": zeros,     # 先占位
            "failure": zeros,       # 先占位
        }


    def get_image_observation(self,data_type="rgb",convert_perspective_to_orthogonal=False) -> torch.Tensor:

        sensor = self.scene.sensors["Camera"]

        # obtain the input image
        images = sensor.data.output[data_type]

        # depth image conversion
        if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

        # rgb/depth image normalization
        if normalize:
            if data_type == "rgb":
                images = images.float() / 255.0
                mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
                images -= mean_tensor
            elif "distance_to" in data_type or "depth" in data_type:
                images[images == float("inf")] = 0

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
        object_pose_w = self._robot.data.body_state_w[:, self.ee_move_id]
        self.goal_pose_visualizer.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])
        body_pose_w = self._robot.data.body_state_w[:, self.ee_fixed_id]
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

    def _update_gripper_rule(self):
        """根据几何关系规则触发夹爪开/关（带滞回+最小保持时间+限速平滑）"""

        # 两片刃尖位置
        p_fix_w = self._robot.data.body_pos_w[:, self.ee_fixed_id, 0:3]
        p_mov_w = self._robot.data.body_pos_w[:, self.ee_move_id, 0:3]

        # 物体位置/速度
        p_obj_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]
        v_obj_w = self.scene.rigid_objects["object"].data.root_vel_w[:, 0:3]
        is_captured = self.check_object_in_gripper(p_obj_w, p_fix_w, p_mov_w)

        # ========== 规则触发条件（滞回） ==========
        # 关：物体明显在夹缝中（pre_score高）且离中心线足够近
        want_close = is_captured

        # 开：物体明显不在夹缝中（pre_score低）或离得太偏（d_perp大）
        want_open = ~is_captured

        # ========== 最小保持时间（hold）防抖 ==========
        # 每步递减 hold
        self.grip_hold_steps = torch.clamp(self.grip_hold_steps - 1, min=0)
        can_switch = (self.grip_hold_steps == 0)

        is_open = (self.grip_mode < 0.5)
        is_close = ~is_open

        # 开->关
        close_now = is_open & want_close.unsqueeze(-1) & can_switch
        # 关->开（但如果已抓住/已抬起，就不允许打开）
        open_now = is_close & want_open.unsqueeze(-1)  & can_switch

        # 应用切换
        self.grip_mode = torch.where(close_now, torch.ones_like(self.grip_mode), self.grip_mode)
        self.grip_mode = torch.where(open_now,  torch.zeros_like(self.grip_mode), self.grip_mode)

        switched = close_now | open_now
        self.grip_hold_steps = torch.where(
            switched,
            torch.full_like(self.grip_hold_steps, self.grip_min_hold),
            self.grip_hold_steps,
        )

        # ========== 把模式映射到目标开/关位置，并限速平滑 ==========
        desired = self.grip_mode * self.gripper_min + (1.0 - self.grip_mode) * self.gripper_max

        max_step = self.gripper_speed * self.dt
        delta = torch.clamp(desired - self.gripper_cmd, -max_step, max_step)
        self.gripper_cmd = self.gripper_cmd + delta

