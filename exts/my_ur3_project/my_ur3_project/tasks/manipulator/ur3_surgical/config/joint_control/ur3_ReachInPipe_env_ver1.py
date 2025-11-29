# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, AssetBase
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform,quat_inv,subtract_frame_transforms
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_error_magnitude, quat_mul, transform_points
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.core.prims import XFormPrimView
import numpy as np
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

import omni.kit.app
import weakref
# 自己编写的模块
from .utils.myfunc import *

@configclass
class Ur3ReachInPipeEnvCfg(DirectRLEnvCfg):
    
    # env
    episode_length_s = 5
    decimation = 5
    action_space = 6

    observation_space = 30
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
            usd_path=f"/home/yhy/DVRK/ur3_scissor/ur3_isaac_sdf.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 0.0,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )
    
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    
    # 计算pipe的参数信息
    roll = torch.tensor(0, device=torch.device('cuda'))   # Roll in radians
    pitch = torch.tensor(torch.pi*0.25, device=torch.device('cuda'))   # Pitch in radians
    yaw = torch.tensor(0, device=torch.device('cuda'))     # Yaw in radians
    
    pipe_inner_radius=0.014
    pipe_outer_radius=0.02
    pipe_height = 0.5
    pipe_pos = torch.tensor([0.41, -0.1, 0.51], device=torch.device('cuda'))
    pipe_quat = quat_from_euler_xyz(roll,pitch,yaw)
    pipe_quat_inv = quat_inv(pipe_quat)
    local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=torch.device('cuda'))
    pipe_axis = quat_rotate_vector(pipe_quat, local_z_axis)  # shape = (3,)
    pipe_axis = pipe_axis / torch.norm(pipe_axis, dim=-1, keepdim=True)  # 确保是单位向量
    
    pipe  = AssetBaseCfg(
        prim_path="/World/envs/env_.*/pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yhy/DVRK/ur3_scissor/pipe20_stl.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, -0.1, 0.5),rot = pipe_quat),
    )

    
    # 可视化部分
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/World/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    
    # 指令输出范围
    ranges = {
        "radius": (0, 0.01),
        "theta": (0, 3.14),
        "z_height": (0.08, 0.18),
        "roll": (-torch.pi*0.75, -torch.pi*0.75),
        "pitch": (0, 0),
        "yaw": (-torch.pi*0.5, -torch.pi*0.5),
    }
    
    make_quat_unique = True
    debug_vis = True
    
    action_scale = 1
    dof_velocity_scale = 0.1
    wall_threshold = 0.002
    
    position_tan_reward_scale = 3
    orientation_align_scale = 1
    inside_pipe_reward_scale = 1.0
    depth_reward_scale = 3.0
    incremental_depth_bonus = 5.0
    action_rate_reward_scale = 0.01
    joint_vel_reward_scale = 0.01
    out_of_pipe_penalty_scale = 2.0
    wall_distance_penalty_scale = 1.0
    stagnation_penalty_scale = 1.0

    std = 0.1
   


class Ur3ReachInPipeEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Ur3ReachInPipeEnvCfg

    def __init__(self, cfg: Ur3ReachInPipeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        self.ee_id = self._robot.data.body_names.index("scissors_tip")

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_action = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        # 初始化命令缓冲区
        self.radius_b = torch.empty(self.num_envs, device=self.device)
        self.theta_b = torch.empty(self.num_envs, device=self.device)
        self.z_height_b = torch.empty(self.num_envs, device=self.device)
        
        self.local_pipe_pos = torch.zeros(self.num_envs, 3, device=self.device) 
        self.env_pipe_pos = torch.zeros(self.num_envs, 3, device=self.device) 
        self.env_pipe_pos = transform_points(self.env_pipe_pos,self.cfg.pipe_pos ,self.cfg.pipe_quat)
        
        self.pipe_axis = self.cfg.pipe_axis.expand(self.num_envs, 3)  # 广播到每个环境
        self.pipe_quat = self.cfg.pipe_quat.expand(self.num_envs, 4)  # 广播到每个环境
        
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)  # 缓冲区存储 (x, y, z, qw, qx, qy, qz)
        self.pose_command_b[:, 3] = 1.0  # 设置四元数的实部为默认值 1
        self.pose_command_w = torch.zeros_like(self.pose_command_b)  # 初始化世界坐标系的命令缓冲区

        self.is_in_pipe = torch.zeros(self.num_envs, 1, device=self.device)
        self.ee2pipe_dis = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b =  torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # 桌面生成
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path,
            self.cfg.table.spawn,
            translation = self.cfg.table.init_state.pos,
            orientation = self.cfg.table.init_state.rot,
        )
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table.prim_path, reset_xform_properties=False)
        # 地面生成
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation = self.cfg.ground.init_state.pos,
        )
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)
        # 管道生成
        self.cfg.pipe.spawn.func(
            self.cfg.pipe.prim_path,
            self.cfg.pipe.spawn,
            translation = self.cfg.pipe.init_state.pos,
            orientation = self.cfg.pipe.init_state.rot,
        )
        self.scene.extras["pipe"] = XFormPrimView(self.cfg.pipe.prim_path, reset_xform_properties=False)
               
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定的环境重新采样位姿命令（位置和姿态）。"""
        self.radius_b.uniform_(*self.cfg.ranges["radius"])
        self.theta_b.uniform_(*self.cfg.ranges["theta"])
        self.z_height_b.uniform_(*self.cfg.ranges["z_height"])
        
        # 转换为笛卡尔坐标，获取pipe坐标系的采样点
        local_point = torch.stack([
            self.radius_b * torch.cos(self.theta_b),
            self.radius_b * torch.sin(self.theta_b),
            self.z_height_b
        ], dim=1)
        # 将pipe坐标系的采样点转换到每个env的局部坐标系
        self.pose_command_b[:,0:3] = transform_points(local_point,self.cfg.pipe_pos ,self.cfg.pipe_quat)
        
        # 采样姿态命令
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])  # 初始化欧拉角缓冲区
        euler_angles[:, 0].uniform_(*self.cfg.ranges["roll"])  # 在指定范围内采样 roll
        euler_angles[:, 1].uniform_(*self.cfg.ranges["pitch"])  # 在指定范围内采样 pitch
        euler_angles[:, 2].uniform_(*self.cfg.ranges["yaw"])  # 在指定范围内采样 yaw
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])  # 将欧拉角转换为四元数

        # 如果配置中要求四元数唯一性，则确保四元数的实部为正
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        # 将姿态指令转换到世界坐标系
        self.pose_command_w[env_ids, :3], self.pose_command_w[env_ids, 3:] = combine_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3],
            self._robot.data.root_state_w[env_ids, 3:7],
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:]
        )
    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        # self.actions = actions.clone()
        self.actions = actions.clone()
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
    
    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # # 从指令中获取期望位姿
        # des_pos_b = self.pose_command_b[:, :3]
        # des_pos_w, _ = combine_frame_transforms(self._robot.data.root_state_w[:, :3],
        #                                         self._robot.data.root_state_w[:, 3:7],
        #                                         des_pos_b)

        # # 当前末端姿态
        # curr_pos_w = self._robot.data.body_state_w[:, self.ee_id, :3]  

        # # 基本误差指标
        # distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        # terminated = torch.where(distance < 0.07, 
        #     torch.full_like(distance, True, dtype=torch.bool), 
        #     torch.full_like(distance, False, dtype=torch.bool)
        #     )
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps

        return self._compute_rewards(
            self.pose_command_b,
            self.cfg.std
    )
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # 更新末端空间姿态指令
        self._resample_command(env_ids)
        self.robot_dof_targets = joint_pos
        # 得到pipe管口的世界坐标
        self.pipe_pos_w, _ = combine_frame_transforms(self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self.env_pipe_pos)
        
        

    def _get_observations(self) -> dict:

        # 关节相对于初始角度的变化
        joint_pos = self._robot.data.joint_pos- self._robot.data.default_joint_pos
        # 关节相对于初始速度的变化
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        # 末端空间姿态指令
        PoseCommand = self.pose_command_b
        # 末端在环境中的位置、末端是否在管道中、末端相对于管口的位姿
        ee_pos, self.is_in_pipe , self.pipe_ee_pos,pipe_ee_quat, ee_radius = self.is_end_effector_in_pipe()
        
        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                PoseCommand,
                ee_pos,
                self.pipe_ee_pos,
                pipe_ee_quat,
                # ee_radius,
                self.is_in_pipe.unsqueeze(1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self):
        return
    def _compute_rewards(
        self,
        pose_command,
        std
    ):
        asset = self._robot
        
        # 从指令中获取期望位姿
        des_pos_b = pose_command[:, :3]
        des_quat_b = pose_command[:, 3:7]
        des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3],
                                                asset.data.root_state_w[:, 3:7],
                                                des_pos_b)
        des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

        # 当前末端姿态
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  
        curr_quat_w = asset.data.body_state_w[:, self.ee_id, 3:7]

        # 基本误差指标
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        guide = torch.norm(curr_pos_w - self.pipe_pos_w, dim=1)
        position_error_tanh_reward = 1.0 - torch.tanh(distance/0.1)+1.0 - torch.tanh(distance/0.05)
        guide_reward = 1.0 - torch.tanh(guide/0.03)
        guide_reward1 = 1.0 - torch.tanh(guide/0.1)
        
        orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        
        # 转换为对齐度[0,1]:越接近0误差，对齐度越高
        orientation_align = (1.0 - torch.tanh(orientation_error))
        local_neg_y_axis = torch.tensor([0.0, -1.0, 0.0], device=curr_quat_w.device)

        # 将负 y 轴旋转到全局坐标系
        neg_y_axis_global = quat_rotate_vector(curr_quat_w, local_neg_y_axis)  # shape = (num_envs, 3)

        cos_theta = torch.sum(neg_y_axis_global * self.pipe_axis, dim=1).clamp(-1.0, 1.0)  # 方向余弦值
        cos_error = torch.abs(cos_theta - torch.full_like(cos_theta, 1))
        # cos_reward = (1.0 - 0.5*torch.tanh(cos_error/0.05)-0.5*torch.tanh(cos_error/0.1))
        
        ee_pos, self.is_in_pipe , self.pipe_ee_pos,pipe_ee_quat,ee_radius = self.is_end_effector_in_pipe()
        z = self.pipe_ee_pos[:, 2]
        is_within_height = (z >= 0.0)
        
        # 是否在管内
        inside_pipe = self.is_in_pipe.float()

        guide_reward = torch.where(self.is_in_pipe, 
                    torch.full_like(guide, 1.5),
                    2.0 - torch.tanh(guide/0.03) - torch.tanh(guide/0.1), 
                    )
        position_reward = torch.where(self.is_in_pipe, 
                    distance*-1, 
                    torch.full_like(distance, -0.7)
                    )
                # 终端奖励
        tinminal_reward = torch.where(position_reward > -0.03, 
                    torch.full_like(distance, 10), 
                    torch.full_like(distance, 0)
                    )
        position_tan_reward = position_error_tanh_reward * inside_pipe
        outside_pipe = (~self.is_in_pipe).float()
        # 动作与关节速度惩罚
        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)*outside_pipe
        self.prev_action = self.actions
        joint_vel_l2_reward = torch.sum(torch.square(asset.data.joint_vel), dim=1)*outside_pipe
        ee_radius = torch.sqrt(ee_radius)
        ee2pipe_dis = torch.where(
            self.is_in_pipe,
            torch.abs(ee_radius - self.cfg.pipe_inner_radius),
            torch.abs(ee_radius - self.cfg.pipe_outer_radius)
        )
        # 管壁距离惩罚
        wall_distance_penalty = torch.where(
            is_within_height & (ee2pipe_dis < 0.002),
            torch.full_like(ee2pipe_dis, -2.0),
            torch.full_like(ee2pipe_dis, 0.0)
        )
        # 将各奖励项组合
        rewards = (
            position_tan_reward * 2
            + position_reward* 2
            + tinminal_reward
            + cos_error * -1
            + guide_reward
            + guide*-1
            + wall_distance_penalty
            # - self.cfg.action_rate_reward_scale * action_rate_l2_reward
            # - self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
        )
        self.reward_components = {
            "position_tan_reward": torch.mean(position_tan_reward*2),
            "position_reward": torch.mean(position_reward * 2 ),
            "tinminal_reward": torch.mean(tinminal_reward),
            "guide_reward": torch.mean(guide_reward),
            "cos_theta_scaled":  torch.mean(cos_error * -1) , 
            "cos_error": torch.mean(cos_error),
            "distance": torch.mean(distance),
            "guide": torch.mean(guide),
            "inside_pipe":self.is_in_pipe.sum(),
            "wall_distance_penalty":torch.mean(wall_distance_penalty)
            # "action_rate":torch.mean(self.cfg.action_rate_reward_scale * action_rate_l2_reward),
            # "joint_vel":torch.mean(self.cfg.joint_vel_reward_scale * joint_vel_l2_reward),

        }
        return rewards

    
    def _get_infos(self):
        infos = {}
        infos['episode'] = self.reward_components 
        return infos

    def is_end_effector_in_pipe(self):
        """
        判断机械臂末端是否在管道内部。

        Args:
            end_effector_pos: 机械臂末端在全局坐标系中的位置，形状为 (N, 3)。
            env_ids: 环境索引序列，用于处理多个环境。

        Returns:
            一个布尔张量，表示每个环境中机械臂末端是否在管道内。
            形状为 (N,)。
        """
        # 将世界坐标系中的位置变换到各个env的局部坐标系
        ee_pos_b, _ = world_to_env_coordinates(
            self._robot.data.body_state_w[:, self.ee_id, :3],
            self._robot.data.body_state_w[:, self.ee_id, 3:7],
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
        )
        
        # 将机械臂末端在env局部坐标系的位置变换到每个env的pipe的局部坐标系中的位置
        pipe_ee_pos, pipe_ee_quat = subtract_frame_transforms(
            self.cfg.pipe_pos,  # 管道在全局坐标系中的位置
            self.pipe_quat, # 管道在全局坐标系中的旋转
            ee_pos_b,
            self._robot.data.body_state_w[:, self.ee_id, 3:7],           
        )
        # 提取局部坐标系中的 r 和 z
        r_squared = pipe_ee_pos[:, 0]**2 + pipe_ee_pos[:, 1]**2
 
        # 获取管道的圆柱体限制范围
        radius_min = 0
        radius_max = self.cfg.pipe_inner_radius
        
        z_min, z_max = self.cfg.ranges["z_height"]  # 管道高度范围
        z_min = 0
        # 判断是否在管道内部
        is_within_radius = (r_squared >= radius_min**2) & (r_squared <= radius_max**2)
        is_within_height = pipe_ee_pos[:, 2] >= z_min
        is_in_pipe = is_within_radius & is_within_height

        return ee_pos_b, is_in_pipe, pipe_ee_pos, pipe_ee_quat,r_squared
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
    
    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        


