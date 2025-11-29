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
from omni.isaac.lab.utils.math import sample_uniform,quat_inv
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
    #TODO 待修改
    observation_space = 31
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
    pipe_height = 0.5
    pipe_pos = torch.tensor([0.4, 0.0, 0.5], device=torch.device('cuda'))
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.5),rot = pipe_quat),
    )

    
    # 可视化部分
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/World/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    # 指令输出范围
    ranges = {
        "radius": (0, 0.012),
        "theta": (0, 3.14),
        "z_height": (0.0, 0.4),
        "roll": (-torch.pi*0.75, -torch.pi*0.75),
        "pitch": (0, 0),
        "yaw": (-torch.pi*0.5, -torch.pi*0.5),
    }
    
    make_quat_unique = True
    debug_vis = True
    
    action_scale = 1
    dof_velocity_scale = 0.1
    wall_threshold = 0.002
    
    # rewards权重
    # position_reward_scale = -0.5
    # position_tan_reward_scale = 1
    # orientation_reward_scale = -0.01
    # action_rate_reward_scale = -0.0007
    # joint_vel_reward_scale = -0.0007
    # out_of_pipe_penalty_scale = -0.5
    # wall_distance_penalty_scale = -20
    # guide_scale = 0.2
    
    position_tan_reward_scale = 3
    orientation_align_scale = 1
    inside_pipe_reward_scale = 1.0
    depth_reward_scale = 3.0
    incremental_depth_bonus = 5.0
    action_rate_reward_scale = 0.05
    joint_vel_reward_scale = 0.05
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
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
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
        # 上一次的action
        actions = self.actions
        # 末端到管的深度
        vec_to_ee = (self._robot.data.body_state_w[:, self.ee_id,0:3] - self.pipe_pos_w)
        depth = torch.sum(vec_to_ee * self.pipe_axis, dim=1)  
        depth = torch.clamp(depth, min=0.0)  # 深度小于0无意义
        # 末端是否在管道中
        ee_pos, self.is_in_pipe , self.ee2pipe_dis = self.is_end_effector_in_pipe()
        
        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                PoseCommand,
                ee_pos,
                actions,
                depth.unsqueeze(1),
                self.is_in_pipe.unsqueeze(1),
                self.ee2pipe_dis.unsqueeze(1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

   
    def _compute_rewards1(
        self,
        pose_command,
        std
    ):
        asset = self._robot
        
        # 相对坐标系的姿态指令
        des_pos_b = pose_command[:, :3]
        # 转换到世界坐标系的姿态
        des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
        # 末端在世界坐标系的姿态
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  # type: ignore
        # 计算姿态指令和末端姿态的误差
        position_error_reward = torch.norm(curr_pos_w - des_pos_w, dim=1)
        
        # 引导前往管口
        des_pos_b = self.env_pipe_pos
        # 转换到世界坐标系的姿态
        des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
        # 末端在世界坐标系的姿态
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  # type: ignore
        # 计算姿态指令和末端姿态的误差
        guide_reward = torch.norm(curr_pos_w - des_pos_w, dim=1)
        
        # obtain the desired and current positions
        des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  # type: ignore
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        position_error_tanh_reward =  1 - torch.tanh(distance / std)
        
        # obtain the desired and current orientations
        des_quat_b = pose_command[:, 3:7]
        des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
        curr_quat_w = asset.data.body_state_w[:, self.ee_id, 3:7]  # type: ignore
        orientation_error_reward =  quat_error_magnitude(curr_quat_w, des_quat_w)
        
        """Penalize the rate of change of the actions using L2 squared kernel."""
        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)
        self.prev_action = self.actions
        
        """Penalize joint velocities on the articulation using L2 squared kernel.
        NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
        """
        # extract the used quantities (to enable type-hinting)

        joint_vel_l2_reward =  torch.sum(torch.square(asset.data.joint_vel), dim=1)
        
        # 对于不在pipe中的惩罚，和离管壁太近的惩罚
        out_of_pipe_penalty = (~self.is_in_pipe).float()
        wall_distance_penalty = torch.where(
            (self.ee2pipe_dis != -1) & (self.ee2pipe_dis < self.cfg.wall_threshold),  # 两个条件：距离有效且小于阈值
            (self.cfg.wall_threshold - self.ee2pipe_dis).clamp(min=0),  # 满足条件时计算惩罚
            torch.tensor(0.0, device=self.device)  # 不满足条件时惩罚为0
        )

        rewards = (
            self.cfg.position_reward_scale * position_error_reward
            + self.cfg.position_tan_reward_scale * position_error_tanh_reward
            + self.cfg.orientation_reward_scale * orientation_error_reward
            + self.cfg.action_rate_reward_scale * action_rate_l2_reward
            + self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
            + self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty
            + self.cfg.wall_distance_penalty_scale * wall_distance_penalty
        )

        return rewards
    
    
    def _compute_rewards2(
        self,
        pose_command,
        std
    ):
        asset = self._robot
        
        # --- 最终目标位姿（世界坐标系下） ---
        # 提取最终期望位置 (self._pose_command_b)
        # 这里假设 self._pose_command_b 是在机器人基坐标系下定义的相对位姿命令
        des_pos_b_final = pose_command[:, :3]  # 最终目标点的相对位置命令
        des_quat_b_final = pose_command[:, 3:7] # 最终目标的相对姿态命令
        
        # 将最终目标转换到世界坐标系
        des_pos_w_final, _ = combine_frame_transforms(
            asset.data.root_state_w[:, :3],
            asset.data.root_state_w[:, 3:7],
            des_pos_b_final
        )
        
        # 当前末端姿态
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  # 末端世界坐标位置
        curr_quat_w = asset.data.body_state_w[:, self.ee_id, 3:7]  # 末端世界坐标姿态

        # --- 管口相关 ---
        # 将管口位置（self.env_pipe_pos）转换到世界坐标系（如果该位置是相对坐标需要同样变换）
        # 如果self.env_pipe_pos本身已为世界坐标，则不需要转换，直接使用即可
        des_pos_b_pipe = self.env_pipe_pos  # 假设已在基坐标系下定义
        des_pos_w_pipe, _ = combine_frame_transforms(
            asset.data.root_state_w[:, :3],
            asset.data.root_state_w[:, 3:7],
            des_pos_b_pipe
        )

        # 距离管口的距离
        distance_to_pipe = torch.norm(curr_pos_w - des_pos_w_pipe, dim=1)
        # 距离最终目标点的距离
        distance_to_final = torch.norm(curr_pos_w - des_pos_w_final, dim=1)
        
        # 基于距离的奖励项（使用tanh压缩以获得平滑奖励）
        position_error_tanh_reward_final = 1 - torch.tanh(distance_to_final / std)
        
        # 指令姿态的方向误差
        des_quat_w_final = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b_final)
        orientation_error_reward = quat_error_magnitude(curr_quat_w, des_quat_w_final)

        # 动作变化率惩罚
        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)
        self.prev_action = self.actions

        # 关节速度惩罚
        joint_vel_l2_reward = torch.sum(torch.square(asset.data.joint_vel), dim=1)
        
        # 管道相关的惩罚（已给出）
        out_of_pipe_penalty = (~self.is_in_pipe).float()
        pipe_reached_mask = self.is_in_pipe.float()
        wall_distance_penalty = torch.where(
            (self.ee2pipe_dis != -1) & (self.ee2pipe_dis < self.cfg.wall_threshold),
            (self.cfg.wall_threshold - self.ee2pipe_dis).clamp(min=0),
            torch.tensor(0.0, device=self.device)
        )
        
        # 第一阶段奖励：鼓励靠近管口
        # 使用一个基于距离管口的奖励，如同上对final位置使用tanh压缩
        # 当还未达到管口时（pipe_reached_mask=0），奖励值更高；到达后不再依赖这个奖励
        position_error_tanh_reward_pipe = 1 - torch.tanh(distance_to_pipe / std)
        
        # 定义平滑过渡的权重函数:
        # 当机械臂与管口距离减小到一定范围内（pipe_reached_threshold）时，管口奖励权重逐渐降低，
        # 最终目标奖励权重逐渐增加，实现平滑过渡。
        pipe_reached_threshold = getattr(self.cfg, 'pipe_reached_threshold', 0.05)
        # 这里使用一个sigmoid函数实现平滑过渡:
        # 当distance_to_pipe < pipe_reached_threshold时，sigmoid输出接近1，更多权重在管口奖励上。
        # 随着distance_to_pipe更小甚至接近0，可以让权重慢慢减小或保持一部分，视情况而定。
        # 调整 0.01（陡峭度）让过渡更平滑或更急剧。
        transition_sharpness = 0.01
        # pipe_weight = torch.sigmoid(-(distance_to_pipe - pipe_reached_threshold)/transition_sharpness)

        # pipe_weight接近1时：强调管口奖励，淡化最终目标奖励
        # pipe_weight接近0时：强调最终目标奖励，淡化管口奖励
        # 为了避免在管口处形成局部最优点，可以让pipe_weight不完全到1或0，而是有一定缓冲
        # 比如可以使用稍大的transition_sharpness来实现更加平缓的曲线。
        
        # 综合两者的奖励
        guide_reward = position_error_tanh_reward_pipe * self.cfg.guide_scale
        final_reward = position_error_tanh_reward_final * self.cfg.position_tan_reward_scale * pipe_reached_mask

        
        # 将两个阶段的奖励与其他惩罚项相加
        rewards = (
            final_reward
            + guide_reward
            + self.cfg.orientation_reward_scale * orientation_error_reward
            + self.cfg.action_rate_reward_scale * action_rate_l2_reward
            + self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
            + self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty
            # + self.cfg.wall_distance_penalty_scale * wall_distance_penalty
        )
        
        self.reward_elements = {
            "final_reward": final_reward,
            "guide_reward": guide_reward,
            "orientation_error_reward": self.cfg.orientation_reward_scale * orientation_error_reward,
            "action_rate_reward": self.cfg.action_rate_reward_scale * action_rate_l2_reward,
            "joint_vel_reward": self.cfg.joint_vel_reward_scale * joint_vel_l2_reward,
            "out_of_pipe_penalty": self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty,
            "wall_distance_penalty": self.cfg.wall_distance_penalty_scale * wall_distance_penalty
        }
        return rewards

    def _compute_rewards3(
        self,
        pose_command,
        std
    ):
        asset = self._robot
        
        # =====================
        # 基本数据提取
        # =====================
        # 从指令中获取期望的相对姿态（位置+四元数）
        des_pos_b = pose_command[:, :3]
        des_quat_b = pose_command[:, 3:7]

        # 将期望姿态从基坐标系转换到世界坐标系
        des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3],
                                                asset.data.root_state_w[:, 3:7],
                                                des_pos_b)
        des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

        # 当前末端执行器姿态（位置+四元数）
        curr_pos_w = asset.data.body_state_w[:, self.ee_id, :3]  # type: ignore
        curr_quat_w = asset.data.body_state_w[:, self.ee_id, 3:7]  # type: ignore

        # =====================
        # 基本误差计算
        # =====================
        # 位置误差与缩放后Tanh奖励（位置越接近目标点奖励越高）
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        position_error_tanh_reward = 1 - torch.tanh(distance / std)  # [0,1]之间

        # 姿态误差转对齐奖励
        orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        orientation_align_reward = 1 - torch.tanh(orientation_error / (std * 0.1))  # 可调节分母以控制敏感度

        # =====================
        # 动作与关节速度惩罚
        # =====================
        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)
        self.prev_action = self.actions
        joint_vel_l2_reward = torch.sum(torch.square(asset.data.joint_vel), dim=1)

        # =====================
        # 管道相关奖励与惩罚
        # =====================
        # 当末端在管道外时的惩罚
        out_of_pipe_penalty = (~self.is_in_pipe).float()

        # 离管壁太近的惩罚
        wall_distance_penalty = torch.where(
            (self.ee2pipe_dis != -1) & (self.ee2pipe_dis < self.cfg.wall_threshold),
            (self.cfg.wall_threshold - self.ee2pipe_dis).clamp(min=0),
            torch.tensor(0.0, device=self.device)
        )

        # 当末端在管道内部时的额外正向奖励：鼓励深入管道
        inside_pipe_reward = self.is_in_pipe.float() * self.cfg.inside_pipe_reward_scale

        # =====================
        # 基于管道轴线的深度奖励(需要您在初始化中定义pipe_entry_point和pipe_axis)
        # 例如:
        # self.pipe_entry_point = ...
        # self.pipe_axis = ... # 已归一化的管道轴线方向向量
        # =====================
        # 计算末端执行器相对于管道入口点在轴线方向上的投影距离
        vec_to_ee = (curr_pos_w - self.pipe_entry_point)  # shape [N,3]
        depth = torch.sum(vec_to_ee * self.pipe_axis, dim=1)  # 沿轴线的投影距离标量
        # 仅在管道内部并且深度>0时给奖励
        depth_reward = torch.clamp(depth, min=0.0) * self.is_in_pipe.float() * self.cfg.depth_reward_scale

        # =====================
        # 将对齐奖励与深度奖励耦合
        # 只有在管道内，对齐奖励才显著计入，这样可避免在管道外错误对齐也获得奖励
        # =====================
        orientation_align_reward = orientation_align_reward * self.cfg.orientation_align_scale
        
        # =====================
        # 汇总最终奖励
        # =====================
        # 注意这里的position_error_tanh_reward是正向奖励（越近越接近1），如果希望距离越小奖励越高，
        # 可以保持该形式。如果更倾向将误差最小化作为目标，也可使用distance的负数或其他方式。
        # 下面假设position_error_tanh_reward本身是个正向度量（越接近1越好），可适当缩放。
        rewards = (
            self.cfg.position_tan_reward_scale * position_error_tanh_reward
            + orientation_align_reward
            - self.cfg.action_rate_reward_scale * action_rate_l2_reward
            - self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
            - self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty
            - self.cfg.wall_distance_penalty_scale * wall_distance_penalty
            + inside_pipe_reward
            + depth_reward
        )

        return rewards

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
        position_error_tanh_reward = 1.0 - torch.tanh(distance)
        position_reward = torch.norm(curr_pos_w - des_pos_w, dim=1) * -4
        guide_reward = 1.0 - torch.tanh(guide / std)
        
        orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        # # 转换为对齐度[0,1]:越接近0误差，对齐度越高
        orientation_align = (1.0 - torch.tanh(orientation_error / (std * 0.1)))
        local_neg_y_axis = torch.tensor([0.0, -1.0, 0.0], device=curr_quat_w.device)

        # 将负 y 轴旋转到全局坐标系
        neg_y_axis_global = quat_rotate_vector(curr_quat_w, local_neg_y_axis)  # shape = (num_envs, 3)

        cos_theta = torch.sum(neg_y_axis_global * self.pipe_axis, dim=1).clamp(-1.0, 1.0)  # 方向余弦值
        
        

        # 动作与关节速度惩罚
        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)
        self.prev_action = self.actions
        joint_vel_l2_reward = torch.sum(torch.square(asset.data.joint_vel), dim=1)

        # 是否在管内
        inside_pipe = self.is_in_pipe.float()

        # 计算深度（沿管道轴线方向的投影）
        vec_to_ee = (curr_pos_w - self.pipe_pos_w)
        depth = torch.sum(vec_to_ee * self.pipe_axis, dim=1)  
        depth = torch.clamp(depth, min=0.0)  # 深度小于0无意义

        # 深度奖励：越深越好
        depth_reward = depth * self.cfg.depth_reward_scale * inside_pipe

        # 在管内奖励：一旦进入管道内部就给予基础正向奖励
        inside_pipe_reward = inside_pipe * self.cfg.inside_pipe_reward_scale

        # 出管惩罚
        out_of_pipe_penalty = (~self.is_in_pipe).float()

        # 管壁距离惩罚
        wall_distance_penalty = torch.where(
            (self.ee2pipe_dis != -1) & (self.ee2pipe_dis < self.cfg.wall_threshold),
            (self.cfg.wall_threshold - self.ee2pipe_dis).clamp(min=0),
            torch.tensor(0.0, device=self.device)
        )


        # 对齐奖励仅在一定深度后才生效，这里假设深度>指定阈值才考虑对齐奖励
        depth_threshold_for_orientation = 0.05  # 根据实际情况调整
        # orientation_align_reward = orientation_align * (depth > depth_threshold_for_orientation).float() * self.cfg.orientation_align_scale
        orientation_align_reward = orientation_align * self.cfg.orientation_align_scale
        # 位置奖励在管外可降低其权重，防止在外部接近无意义
        # 在管内位置奖励更高，鼓励深入到正确的位置
        position_tan_reward = position_error_tanh_reward * inside_pipe * self.cfg.position_tan_reward_scale
        position_reward = torch.where(self.is_in_pipe, 
                                position_reward, 
                                torch.full_like(position_reward, -1))
        # 条件判断：inside_pipe 时 guide_reward 固定
        fixed_guide_reward = 1  # 固定值，可根据需要调整
        guide_reward = torch.where(self.is_in_pipe, 
                                torch.full_like(guide, fixed_guide_reward), 
                                1.0 - torch.tanh(guide / std))
        # 将各奖励项组合
        rewards = (
            position_tan_reward
            + position_reward 
            # + depth_reward * 10
            + orientation_align_reward
            + inside_pipe_reward
            + guide_reward
            # - self.cfg.action_rate_reward_scale * action_rate_l2_reward
            # - self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
            # - self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty
            # - self.cfg.wall_distance_penalty_scale * wall_distance_penalty
        )
        # 创建奖励分量字典
        # 创建奖励分量字典
        self.reward_components = {
            "position_tan_reward": torch.mean(position_tan_reward),
            "position_reward_scaled": torch.mean(position_reward ),
            # "depth_reward": torch.mean(depth_reward)* 100,
            "inside_pipe_reward": torch.mean(inside_pipe_reward),
            "guide_reward": torch.mean(guide_reward),
            # "orientation_align_reward": orientation_align_reward,  # 注释掉的分量
            "cos_theta_scaled":  torch.mean(guide_reward*cos_theta) ,               # 注释掉的分量
            # "action_rate_penalty": -self.cfg.action_rate_reward_scale * action_rate_l2_reward,  # 注释掉的分量
            # "joint_vel_penalty": -self.cfg.joint_vel_reward_scale * joint_vel_l2_reward,        # 注释掉的分量
            # "out_of_pipe_penalty": -self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty,  # 注释掉的分量
            # "wall_distance_penalty": -self.cfg.wall_distance_penalty_scale * wall_distance_penalty, # 注释掉的分量
        }
        return rewards

    def _compute_rewards4(
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

        # =====================
        # 计算夹角条件
        # =====================
        # 定义局部负 y 轴
        local_neg_y_axis = torch.tensor([0.0, -1.0, 0.0], device=curr_quat_w.device)

        # 将负 y 轴旋转到全局坐标系
        neg_y_axis_global = quat_rotate_vector(curr_quat_w, local_neg_y_axis)  # shape = (num_envs, 3)

        # 计算负 y 轴与管道轴线的余弦夹角
        pipe_axis_normalized = self.pipe_axis / torch.norm(self.pipe_axis, dim=1, keepdim=True)  # 确保单位向量
        cos_theta = torch.sum(neg_y_axis_global * pipe_axis_normalized, dim=1).clamp(-1.0, 1.0)  # 方向余弦值

        # 判断是否符合角度条件（小于 30° -> cos(30°) ≈ 0.866）
        is_angle_reasonable = cos_theta >= 0.866

        # =====================
        # 判断是否深入管道
        # =====================
        # 原始管道判断
        inside_pipe = self.is_in_pipe

        # 更新“深入管道”的定义
        deeply_in_pipe = (self.is_in_pipe & is_angle_reasonable).float()

        # =====================
        # 计算奖励
        # =====================
        distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
        guide = torch.norm(curr_pos_w - self.pipe_pos_w, dim=1)
        position_error_tanh_reward = 1.0 - torch.tanh(distance / std)
        guide_reward = 1.0 - torch.tanh(guide / std)

        orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
        orientation_align = (1.0 - torch.tanh(orientation_error / (std * 0.1)))

        action_rate_l2_reward = torch.sum(torch.square(self.actions - self.prev_action), dim=1)
        self.prev_action = self.actions
        joint_vel_l2_reward = torch.sum(torch.square(asset.data.joint_vel), dim=1)

        # 计算深度（沿管道轴线方向的投影）
        vec_to_ee = (curr_pos_w - self.pipe_pos_w)
        depth = torch.sum(vec_to_ee * pipe_axis_normalized, dim=1)
        depth = torch.clamp(depth, min=0.0)

        # 深度奖励：只有深入管道时才计算
        depth_reward = depth * self.cfg.depth_reward_scale * deeply_in_pipe

        # 在管内奖励：只有深入管道时才计算
        inside_pipe_reward = deeply_in_pipe * self.cfg.inside_pipe_reward_scale

        # 出管惩罚
        out_of_pipe_penalty = (~inside_pipe).float()

        orientation_align_reward = orientation_align * self.cfg.orientation_align_scale
        position_tan_reward = position_error_tanh_reward * deeply_in_pipe * self.cfg.position_tan_reward_scale

        rewards = (
            position_tan_reward
            + orientation_align_reward
            + inside_pipe_reward
            + depth_reward
            + guide_reward
            - self.cfg.action_rate_reward_scale * action_rate_l2_reward
            - self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
            - self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty
        )
        # 创建奖励分量字典
        self.reward_components = {
            "position_tan_reward": position_tan_reward,
            "orientation_align_reward": orientation_align_reward,
            "inside_pipe_reward": inside_pipe_reward,
            "depth_reward": depth_reward,
            "guide_reward": guide_reward,
            "action_rate_penalty": -self.cfg.action_rate_reward_scale * action_rate_l2_reward,
            "joint_vel_penalty": -self.cfg.joint_vel_reward_scale * joint_vel_l2_reward,
            "out_of_pipe_penalty": -self.cfg.out_of_pipe_penalty_scale * out_of_pipe_penalty,
        }
        
        return rewards

    
    def _get_infos(self):
        infos = {}
        infos['log'] = self.reward_components 
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
        pipe_ee_pos = env_to_pipe_coordinates(
            ee_pos_b,
            self.cfg.pipe_pos,  # 管道在全局坐标系中的位置
            self.cfg.pipe_quat  # 管道在全局坐标系中的旋转
        )
        
        # 提取局部坐标系中的 r 和 z
        x, y, z = pipe_ee_pos[:, 0], pipe_ee_pos[:, 1], pipe_ee_pos[:, 2]
        r_squared = x**2 + y**2  # 计算 r^2
 
        # 获取管道的圆柱体限制范围
        radius_min = 0
        radius_max = self.cfg.pipe_inner_radius
        
        z_min, z_max = self.cfg.ranges["z_height"]  # 管道高度范围
        z_min = 0
        # 判断是否在管道内部
        is_within_radius = (r_squared >= radius_min**2) & (r_squared <= radius_max**2)
        is_within_height = (z >= z_min) & (z <= z_max)
        is_in_pipe = is_within_radius & is_within_height
        

        ee2pipe_dis = torch.where(
            is_in_pipe,
            torch.abs(r_squared - self.cfg.pipe_inner_radius),
            torch.tensor(-1,device=self.device)
        )
        ee2pipe_dis = r_squared
        
        return ee_pos_b, is_in_pipe , ee2pipe_dis
    
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
        


