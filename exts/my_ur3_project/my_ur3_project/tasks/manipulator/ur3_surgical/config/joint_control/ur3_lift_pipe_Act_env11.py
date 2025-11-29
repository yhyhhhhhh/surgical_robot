from __future__ import annotations
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform,quat_inv
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_unique, quat_error_magnitude, quat_mul, transform_points
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.assets import RigidObjectCfg,RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
import omni.kit.app
import weakref
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sensors.camera import Camera
import omni.isaac.lab.utils.math as math_utils

# 自己编写的模块
from .utils.myfunc import *
from .ur3_lift_pipe_ik_env_cfg import Ur3LiftPipeEnvCfg
from .utils.robot_ik_fun import DifferentialInverseKinematicsAction

import math
import time
### ====================================== ACT算法 ========================================
from .ACT.policy import ACTPolicy 
import os
import pickle
from einops import rearrange

from .ACT.Residual_Policy.residual_act import ResidualACTPolicy
#-----------------------------------------------------------------------------
import h5py
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import random

def make_policy(policy_class: str, policy_config: dict) -> torch.nn.Module:
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    else:
        raise NotImplementedError

class Ur3LiftNeedleEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Ur3LiftPipeEnvCfg

    def __init__(self, cfg: Ur3LiftPipeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        
        self.ee_id = self._robot.data.body_names.index("scissors_tip")

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.prev_action = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)  # 缓冲区存储 (x, y, z, qw, qx, qy, qz)
        self.pose_command_b[:, 3] = 1.0  # 设置四元数的实部为默认值 1
        self.pose_command_w = torch.zeros_like(self.pose_command_b)  # 初始化世界坐标系的命令缓冲区
        
        self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b =  torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)
        
        # 初始化命令缓冲区
        self.object_x_b = torch.empty(self.num_envs, device=self.device)
        self.object_y_b = torch.empty(self.num_envs, device=self.device)
        
        
        self.pipe_top_pos, _ = self.get_pipe_state()  # (N,3)
        self.u_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.u_axis = self.u_axis.unsqueeze(0).repeat(self.num_envs, 1)  # (N,3)
        
        # ---- 初始化（一次）---------------------------------------------------------
        self.last_reset_t   = torch.full((self.num_envs,), float("nan"),
                                        device=self.device, dtype=torch.float64)
        self.reset_interval = torch.zeros_like(self.last_reset_t)  # 上次→本次间隔
        # --------------------------------------------------------------------------
        
        # 加载 policy 和统计数据
        ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250409_150236/'
        ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_2000_seed_0.ckpt')
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        state_dim = 8
        lr_backbone = 1e-5
        backbone = 'resnet18'
        args = {
            'batch_size': 8,
            'chunk_size': 200,
            'ckpt_dir': ckpt_dir,
            'dim_feedforward': 3200,
            'eval': False,
            'hidden_dim': 512,
            'kl_weight': 10,
            'lr': 1e-05,
            'num_epochs': 2000,
            'onscreen_render': True,
            'policy_class': 'ACT',
            'seed': 0,
            'task_name': 'sim_needle',
            'temporal_agg': False,
        }
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': ['top'],
        }
        policy = make_policy(args['policy_class'], policy_config)
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        print(f'Loaded policy from {ckpt_path}')

        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
        self.query_frequency = 100
        self.temporal_agg = True
        self.max_timesteps = 500
        self.action_scale = 0.02
        # 封装 policy 逻辑
        self.policy_wrapper = ResidualACTPolicy(
            policy=policy,
            stats=stats,
            device=self.device,
            num_envs=self.num_envs,
            max_timesteps=self.max_timesteps,
            state_dim=state_dim,
            query_frequency=self.query_frequency,
            temporal_agg=self.temporal_agg,
        )

        # 打开（或创建）HDF5 文件
        self.h5file = h5py.File("all_rewards_noclose.h5", "a")

        # 为每个 reward 分量创建可扩展 dataset
        self.datasets = {}
        for key in [
            'Outside Pipe Reward',
            'Direction Projection Reward',
            'Close Reward',
            'Height Reward',
            'Task Reward',
            'All Reward'
        ]:
            if key in self.h5file:
                ds = self.h5file[key]
            else:
                ds = self.h5file.create_dataset(
                    key,
                    shape=(0, self.max_episode_length),
                    maxshape=(None, self.max_episode_length),
                    dtype='float32',
                    chunks=True
                )
            self.datasets[key] = ds
            
        self.episode_counter = 0
        self.all_rewards = []
        self.reward_buffer = 0
        self.enable_draw = True
        if self.enable_draw:
            self.draw = omni_debug_draw.acquire_debug_draw_interface()
        
        # ------- 然后生成10组固定的半径r和角度theta -------
        u = torch.ones(1, device=self.device)  # 均匀分布
        self.fixed_r = 0.0025 * torch.sqrt(u)    # 半径均匀分布（面积均匀）
        self.fixed_theta = torch.ones(1, device=self.device) * 0.5 * math.pi  # 角度均匀分布 [0, 2pi)
        self.prev_ee_to_obj_dist = torch.zeros(self.num_envs, device=self.device)
    
    
    def _setup_scene(self):
        # 手术室生成
        self.cfg.room.spawn.func(
            self.cfg.room.prim_path,
            self.cfg.room.spawn,
            translation = self.cfg.room.init_state.pos,
            orientation = self.cfg.room.init_state.rot,
        )
        # 地面生成
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation = self.cfg.ground.init_state.pos,
        )
        # 手术机器人桌子
        self.cfg.table_robot.spawn.func(
            self.cfg.table_robot.prim_path,
            self.cfg.table_robot.spawn,
            translation = self.cfg.table_robot.init_state.pos,
            orientation = self.cfg.table_robot.init_state.rot,
        )
        # 手术台生成
        self.cfg.table_operate.spawn.func(
            self.cfg.table_operate.prim_path,
            self.cfg.table_operate.spawn,
            translation = self.cfg.table_operate.init_state.pos,
            orientation = self.cfg.table_operate.init_state.rot,
        )
        
        # 机械臂控制器配置
        self._robot = Articulation(self.cfg.left_robot)

        # 摄像头生成
        self._camera = Camera(cfg=self.cfg.camera)
        self.scene.sensors["Camera"] = self._camera
        
        # 管道生成
        self.cfg.pipe.spawn.func(
            self.cfg.pipe.prim_path,
            self.cfg.pipe.spawn,
            translation = self.cfg.pipe.init_state.pos,
            orientation = self.cfg.pipe.init_state.rot,
        )
        
        self._object = RigidObject(cfg=self.cfg.object)
        
        self.scene.extras["pipe"] = XFormPrimView(self.cfg.pipe.prim_path, reset_xform_properties=False)
        
        # 环境配置
        self.scene.articulations["left_robot"] = self._robot
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table_robot.prim_path, reset_xform_properties=False)
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)
        
        self.scene.rigid_objects["object"] = self._object
        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])


    def _resample_command1(self, env_ids: Sequence[int]):
        """
        在以 (-0.29, 0, z) 为圆心，半径为 0.004 的圆内随机生成 object 的位置。
        采样参数范围：
        "object_r": (0.0, 0.004)
        "object_theta": (0, 3.14)
        注意：
        为了实现均匀采样，半径使用 sqrt(u) * max_r 形式。
        这里假设 z 坐标取 self.cfg.object.init_state.pos[2]（可根据实际需要调整）。
        """
        device = self.device
        n = len(env_ids)
        max_r = 0.004  # 半径
        center_x = 0.0  # 圆心 x 坐标
        center_y = -0.29    # 圆心 y 坐标（可根据需要调整）

        # 均匀采样：u ~ Uniform(0,1)，再乘以 max_r
        u = torch.rand(n, device=device)
        # u = torch.ones(n, device=device)*0.53
        r = max_r * torch.sqrt(u)  # 均匀采样圆面积

        # theta 从 0 到 3.14 均匀采样（注意，这里只采样半圆，如果需要全圆，请改为 2*pi）
        theta = torch.rand(n, device=device) * 3.14 * 2
        # theta = torch.ones(n, device=device) * 3.14 * 2 * 0.63
        # 转换为笛卡尔坐标
        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)
        # z 坐标可以固定，也可以根据需求采样；这里使用初始状态中的 z
        z = torch.full((n,), self.cfg.object.init_state.pos[2], device=device)

        # 将生成的坐标更新到 pose_command_w 中（假设前 3 个元素为位置）
        new_pos = torch.stack([x, y, z], dim=1) + self.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, :3] = new_pos[:, :3]

        theta = torch.rand(len(env_ids), device=device) * 2 * math.pi

        # 生成绕 Z 轴旋转的四元数，公式：q = [cos(theta/2), 0, 0, sin(theta/2)]
        q = torch.stack([
            torch.cos(theta / 2),                # w
            torch.zeros_like(theta),             # x
            torch.zeros_like(theta),             # y
            torch.sin(theta / 2)                 # z
        ], dim=1)

        # 将姿态设置为上述随机生成的绕 Z 轴旋转
        self.pose_command_w[env_ids, 3:] = q
    def _resample_command(self, env_ids: Sequence[int]):
        """
        从固定的10组 (r, theta) 中随机选一组，生成object的位置和朝向。
        """
        device = self.device
        n = len(env_ids)

        idx = torch.randint(0, 1, (n,), device=device)

        r = self.fixed_r[idx]
        theta = self.fixed_theta[idx]

        center_x = 0.0
        center_y = -0.29
        x = center_x + r * torch.cos(theta)
        y = center_y + r * torch.sin(theta)

        z = torch.full((n,), self.cfg.object.init_state.pos[2], device=device)

        new_pos = torch.stack([x, y, z], dim=1) + self.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, :3] = new_pos

        q = torch.stack([
            torch.cos(theta / 2),
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            torch.sin(theta / 2)
        ], dim=1)
        self.pose_command_w[env_ids, 3:] = q
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        # actions = -actions
        self.actions = self.raw_action.clone()
        # 使用torch.where实现条件限制
        self.actions[:, 2] = torch.where(
            self.actions[:, 2] < -0.235,
            torch.full_like(self.actions[:, 2], -0.235),
            self.actions[:, 2]
        )
        # print(self._robot.data.body_pos_w[:,self.ee_id,2],self.actions[:,2])
        ee_pos_tensor = self._robot.data.body_pos_w[:,self.ee_id,:3] -self.scene.env_origins[:,:3] 
        xy = ee_pos_tensor[:,:2]
        z = ee_pos_tensor[:,2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs,1)
        dist = torch.norm(xy - target_xy, dim = 1)
        self.ee_in_pipe =  dist<0.008 and z< -0.215
        # if self.ee_in_pipe:
            # self.actions[:, 0:3] += actions[:,0:3] * self.action_scale
            # print(actions[:,-1])
            # self.actions[:,-1] = (actions[:,-1] >0).float()
            # self.actions[:,-1] += actions[:,-1] * 0.5

        self.policy_action = actions[:,0:3] * self.action_scale
        # self.actions[:, -1] += actions[:,-1]
        # print( actions[:,0:3] * self.action_scale )
        # 获取基座在世界坐标系中的位姿
        root_pos_w = self._robot.data.root_state_w[:, :3]  # (num_envs, 3)
        root_quat_w = self._robot.data.root_state_w[:, 3:7]  # (num_envs, 4)
        
        # 2. 将 action 从环境坐标系转换到世界坐标系
        # 假设 self.actions[:, :3] 为相对于环境原点的目标位置，形状 (num_envs, 3)
        # 假设 self.scene.env_origins 为每个环境的原点，形状 (num_envs, 3)
        action_world_pos = self.actions[:, :3] + self.scene.env_origins

        # 姿态部分不变（如果环境坐标系与世界坐标系没有旋转差异）
        action_world_quat = self.actions[:, 3:7]

        # 3. 将目标位姿从世界坐标系转换到机器人基座坐标系
        pos_base, quat_base = math_utils.subtract_frame_transforms(
            root_pos_w,             # 机器人基座在世界坐标系的位置
            root_quat_w,            # 机器人基座在世界坐标系的姿态
            action_world_pos,       # 目标在世界坐标系的位置
            action_world_quat       # 目标在世界坐标系的姿态
        )
        pose =  torch.cat([pos_base, quat_base], dim=1)
        self._robot_ik.process_actions(pose)
        
        self.gripper_action = torch.where(
            self.actions[:, -1].unsqueeze(1) > 0.5,
            torch.tensor([-0.28], device=self.device),
            torch.tensor([-0.18], device=self.device)
        )
        # self.gripper_action = torch.where(
        #     self.actions[:, -1].unsqueeze(1)==1,
        #     torch.tensor([-0.28], device=self.device),
        #     torch.tensor([-0.18], device=self.device)
        # )
    def _apply_action(self):
        ik_action = self._robot_ik.apply_actions()
        robot_action = torch.cat([ik_action, self.gripper_action], dim=1)
        self._robot.set_joint_position_target(robot_action)


    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_ee_pos =  self._robot.data.body_pos_w[:,self.ee_id,0:3]
        # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
        pos_tensor = self.scene.rigid_objects["object"].data.body_pos_w[:,0,:3]  # 例如 shape 为 (num_bodies, 3)

        # 条件2：计算 (x, y) 平面上与目标点 (0, -0.29) 的欧氏距离是否大于 0.005
        xy = pos_tensor[:,:2]-self.scene.env_origins[:,:2]
        
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs,1)
        dist = torch.norm(xy - target_xy, dim = 1)
        terminated = dist > 0.01
        self.dist = dist

        self.ee_dist =  torch.norm(robot_ee_pos - pos_tensor, dim = 1)
        # 条件1：机器人末端执行器与 object 的距离小于 0.005
        self.terminated = torch.logical_and(terminated, self.ee_dist < 0.01)
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.truncated = truncated
        if self.enable_draw:
             self.draw.clear_points()
        

        if self.enable_draw:
            vis_action_np = self.actions[:,0:3].detach().cpu().numpy()
            vis_action_act = self.raw_action[:,0:3].detach().cpu().numpy()
            point_list = [tuple(pt) for pt in vis_action_np.tolist()]
            point_list_act = [tuple(pt) for pt in vis_action_act.tolist()]
            self.draw.draw_points(point_list_act, [(0, 0, 1, 1)],[10])
            self.draw.draw_points(point_list, [(0, 1, 0, 1)],[3])
            #绘制物体和机械臂末端的位置
            # obj_pos = self._object.data.body_pos_w[:, 0,:3]
            # obj_pos[:,2] += 0.001
            # vis_obj = obj_pos.detach().cpu().numpy()
            # vis_ee = self._robot.data.body_pos_w[:,self.ee_id, :3].detach().cpu().numpy()
            # point_obj = [tuple(pt) for pt in vis_obj.tolist()]
            # point_ee = [tuple(pt) for pt in vis_ee.tolist()]
            # self.draw.draw_points(point_obj, [(0, 0.5, 0.5, 1)],[30])
            # self.draw.draw_points(point_ee, [(0.5, 0.5, 0, 1)],[30])
        return self.terminated, truncated

    def _get_rewards_11(self) -> torch.Tensor:
 
        
        # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
        ee_pos_tensor = self._robot.data.body_pos_w[:,self.ee_id,:3] -self.scene.env_origins[:,:3] # 例如 shape 为 (num_bodies, 3)
        obj_pos_tensor = self.scene.rigid_objects["object"].data.body_pos_w[:,0,:3] - self.scene.env_origins[:,:3]
        
        ## 在管道外给惩罚
        xy = ee_pos_tensor[:,:2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs,1)
        dist = torch.norm(xy - target_xy, dim = 1)
        
        z = ee_pos_tensor[:,2]
        
        ## 如果末端在管道内且超出了管道，给予惩罚
        outside_pipe_reward = -1.0 * (dist > 0.008).float() * (z < -0.22).float()

    
        # ----------------------------------------------------------
        # 1″. 方向投影奖励（xy 和 z 分权重）
        # ----------------------------------------------------------
        # 已知：
        #   p_base      (N,3)  ← 仅执行 raw_action 的末端位姿
        #   delta_res   (N,3)  ← residual Δpose  = self.policy_action
        #   obj_pos     (N,3)  ← 物体中心位置

        # 若你想把目标 z 再抬 1 mm，可在这里改：
        obj_pos_adj = obj_pos_tensor.clone()

        # 1. 到物体的单位向量  u
        obj_2_ee = obj_pos_adj - ee_pos_tensor
        u = torch.nn.functional.normalize(obj_pos_adj - ee_pos_tensor, dim=1)      # (N,3)

        # ------ 2. xy 分量 -------------------------------------------------
        # 2.1 取 xy 分量并归一化（防零除）
        u_xy_raw = u[:, :2]                           # (N,2)
        u_xy_norm = torch.linalg.norm(u_xy_raw, dim=1, keepdim=True).clamp(min=1e-8)
        u_xy = u_xy_raw / u_xy_norm                   # (N,2) ，仍保持行向量

        # 2.2 residual 的 xy 分量
        dx = self.policy_action[:, :2]                         # (N,2)

        # 2.3 投影长度：正值 = 朝目标，负值 = 背离
        proj_xy = torch.sum(dx * u_xy, dim=1)         # (N,)

        # 2.4 xy 奖励 / 惩罚
        alpha_xy_fwd  = 20.0        # xy 正向奖励
        alpha_xy_back = 20.0        # xy 反向惩罚

        r_xy_pos =  alpha_xy_fwd  * torch.clamp(proj_xy, min=0.0)
        r_xy_neg = -alpha_xy_back * torch.clamp(proj_xy, max=0.0).abs()
        r_xy = r_xy_pos + r_xy_neg                       # (N,)

        # ------ 3. z 分量 --------------------------------------------------
        # u_z = u[:, 2]                     # (N,)
        # dz  = self.policy_action[:, 2]             # (N,)
        # proj_z = dz * u_z                 # (N,)

        # alpha_z_fwd  = 5.0                # z 正向奖励（较小）
        # alpha_z_back = 5.0                # z 反向惩罚

        # r_z_pos =  alpha_z_fwd  * torch.clamp(proj_z, min=0.0)
        # r_z_neg = -alpha_z_back * torch.clamp(proj_z, max=0.0).abs()
        # r_z = r_z_pos + r_z_neg                           # (N,)

        # ------ 4. 合并 ----------------------------------------------------
        r_z = 0 
        r_proj = r_xy + r_z                               # (N,)
        # print(r_proj)
        # print(dist_reward)
        ## 机械臂末端在object范围内给予奖励
        close_reward = 0.5 * (self.ee_dist < 0.002).float()
        
        ## 提起object的奖励
        obj_height = self.scene.rigid_objects["object"].data.body_pos_w[:,0,2] - self.scene.env_origins[:,2]
        object_up_height = obj_height - self.scene.rigid_objects["object"].data.default_root_state[:,2]
        height_reward = object_up_height * 50.0  # 高度越高奖励越大
        height_reward_instant = (object_up_height > 0.01).float() * (self.ee_dist < 0.002).float() * 1 # 只要抬起就给奖励
        height_reward *= (self.ee_dist < 0.002).float()  # 仅在末端接近时生效       
        
        ## 完成任务的奖励
        task_reward = self.terminated.float() * 100
        # print(task_reward)
        rewards = task_reward + height_reward_instant + r_proj + close_reward + outside_pipe_reward + height_reward
        if self.ee_in_pipe == False:
            rewards = outside_pipe_reward
        t = self.episode_length_buf.item()
        self.reward_dict['Outside Pipe Reward'][t] = outside_pipe_reward[0].item()  # 使用t作为索引填充
        self.reward_dict['Direction Projection Reward'][t] = r_proj[0].item()
        self.reward_dict['Close Reward'][t] = close_reward[0].item()
        self.reward_dict['Height Reward'][t] = height_reward[0].item()
        self.reward_dict['Task Reward'][t] = task_reward[0].item()
        self.reward_dict['All Reward'][t] = rewards[0].item()
        # print(rewards)
        return rewards
    def _get_rewards(self) -> torch.Tensor:
        # ========== 末端 & 物体位置 ==========
        ee_pos = self._robot.data.body_pos_w[:, self.ee_id, :3] - self.scene.env_origins[:, :3]
        obj_pos = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3] - self.scene.env_origins[:, :3]

        # ------------------------------------------------------------------
        # A. 末端-物体距离（整体 + 分轴）
        # ------------------------------------------------------------------
        diff      = ee_pos - obj_pos - 0.0006                         # (N,3)
        dist_xyz  = torch.linalg.norm(diff, dim=1)            # ‖Δ‖
        dist_xy   = torch.linalg.norm(diff[:, :2], dim=1)     # xy-平面
        dist_z    = diff[:, 2].abs()                          # z

        # ----- (a) 全局势能（稠密、凸） -----
        k_dist = 40.0                                         # 全局斜率
        r_dense = -k_dist * dist_xyz                          # 越近越高

        # ----- (b) 精细收敛（0-3 mm 内平滑推向 1） -----
        d_fine = 0.003
        r_fine = torch.where(
            dist_xyz < d_fine,
            1.0 - (dist_xyz / d_fine) ** 2,                   # 0->1 平滑
            torch.zeros_like(dist_xyz),
        )

        # ----- (c) 分轴权重 -----
        w_xy, w_z = 1.0, 0.5
        r_axis = - (w_xy * dist_xy + w_z * dist_z) * 50.0     # 权重化惩罚

        # ----- (d) 成功奖励 -----
        d_success = 0.002                               # 0.6 mm
        r_success = (dist_xyz < d_success).float() * 8.0      # 到位 +8
        # 也可在此处把 self.terminated 置 True（如果你想让 env 立即重置）

        # ------------------------------------------------------------------
        # B. 你的其余奖励项（管道惩罚、抬高奖励等）
        # ------------------------------------------------------------------
        # >>>>>> 下面保持不变 / 只把变量名对应过来即可 <<<<<<
        xy        = ee_pos[:, :2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs, 1)
        dist_pipe = torch.norm(xy - target_xy, dim=1)
        z         = ee_pos[:, 2]
        outside_pipe_reward = -1.0 * (dist_pipe > 0.008).float() * (z < -0.22).float()

        # 抬高物体、终止奖励 ... （保持你的原实现）
        # ------------------------------------------------------------------
        obj_height       = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, 2] - self.scene.env_origins[:, 2]
        lift             = obj_height - self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        height_reward    = lift.clamp(min=0) * 50.0 * (dist_xyz < d_fine).float()
        height_instant   = (lift > 0.01).float() * (dist_xyz < d_fine).float() * 1.0
        task_reward      = self.terminated.float() * 100.0

        
        # ======================= 总奖励 =======================
        if self.ee_in_pipe == False:
            r_dense = r_dense*0
            r_fine = r_fine*0
            r_axis = r_axis*0
        rewards = (
            r_dense        # 稠密吸引
            + r_fine       # 精细收敛
            + r_axis       # 分轴纠正
            + r_success    # 到位奖
            + height_reward
            + height_instant
            + outside_pipe_reward
            + task_reward
        )

        print(rewards)
        # ---------- 记录到字典 ----------
        t = self.episode_length_buf.item()
        # self.reward_dict["Dense"][t]        = r_dense[0].item()
        # self.reward_dict["Fine"][t]         = r_fine[0].item()
        # self.reward_dict["Axis"][t]         = r_axis[0].item()
        # self.reward_dict["Success"][t]      = r_success[0].item()
        # self.reward_dict["Height"][t]       = height_reward[0].item()
        # self.reward_dict["Outside"][t]      = outside_pipe_reward[0].item()
        # self.reward_dict["Task"][t]         = task_reward[0].item()
        # self.reward_dict["All"][t]          = rewards[0].item()

        return rewards

    def _get_rewards2(self) -> torch.Tensor:
 
        
        # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
        ee_pos_tensor = self._robot.data.body_pos_w[:,self.ee_id,:3] -self.scene.env_origins[:,:3] # 例如 shape 为 (num_bodies, 3)
        obj_pos_tensor = self.scene.rigid_objects["object"].data.body_pos_w[:,0,:3] - self.scene.env_origins[:,:3]
        
        ## 在管道外给惩罚
        xy = ee_pos_tensor[:,:2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs,1)
        dist = torch.norm(xy - target_xy, dim = 1)
        
        z = ee_pos_tensor[:,2]
        
        ## 如果末端在管道内且超出了管道，给予惩罚
        outside_pipe_reward = -1.0 * (dist > 0.008).float() * (z < -0.22).float()

        # ----------------------------------------------------------
        # 1″. 方向投影奖励（xy 和 z 分权重）
        # ----------------------------------------------------------
        # 已知：
        #   p_base      (N,3)  ← 仅执行 raw_action 的末端位姿
        #   delta_res   (N,3)  ← residual Δpose  = self.policy_action
        #   obj_pos     (N,3)  ← 物体中心位置

        # 若你想把目标 z 再抬 1 mm，可在这里改：
        obj_pos_adj = obj_pos_tensor.clone()
        obj_pos_adj[:,2] +=0.0005
        # 1. 到物体的单位向量  u
        obj_2_ee = obj_pos_adj - ee_pos_tensor
        # print(obj_2_ee)
        u = torch.nn.functional.normalize(obj_pos_adj - ee_pos_tensor, dim=1)      # (N,3)
        # ------ 3. y 分量 --------------------------------------------------
        u_x = u[:, 0]                     # (N,)
        dx  = self.policy_action[:, 0]             # (N,)
        proj_x = dx * u_x                # (N,)

        alpha_x_fwd  = 10.0                # z 正向奖励（较小）
        alpha_x_back = 10.0                # z 反向惩罚

        r_x_pos =  alpha_x_fwd  * torch.clamp(proj_x, min=0.0)
        r_x_neg = -alpha_x_back * torch.clamp(proj_x, max=0.0).abs()
        r_x = r_x_pos + r_x_neg                           # (N,)
        
        # ------ 3. y 分量 --------------------------------------------------
        u_y = u[:, 1]                     # (N,)
        dy  = self.policy_action[:, 1]             # (N,)
        proj_y = dy * u_y                # (N,)

        alpha_y_fwd  = 10.0                # z 正向奖励（较小）
        alpha_y_back = 10.0                # z 反向惩罚

        r_y_pos =  alpha_y_fwd  * torch.clamp(proj_y, min=0.0)
        r_y_neg = -alpha_y_back * torch.clamp(proj_y, max=0.0).abs()
        r_y = r_y_pos + r_y_neg                           # (N,)

        # ------ 3. z 分量 --------------------------------------------------
        u_z = u[:, 2]                     # (N,)
        dz  = self.policy_action[:, 2]             # (N,)
        proj_z = dz * u_z                 # (N,)

        alpha_z_fwd  = 10.0                # z 正向奖励（较小）
        alpha_z_back = 10.0                # z 反向惩罚

        r_z_pos =  alpha_z_fwd  * torch.clamp(proj_z, min=0.0)
        r_z_neg = -alpha_z_back * torch.clamp(proj_z, max=0.0).abs()
        r_z = r_z_pos + r_z_neg                           # (N,)

        # ------ 4. 合并 ----------------------------------------------------
        r_proj = r_x + r_y + r_z                               # (N,)
        # print(r_proj)
        # print(dist_reward)
        ## 机械臂末端在object范围内给予奖励
        close_reward = 0.0 * (self.ee_dist < 0.002).float()
        
        ## 提起object的奖励
        obj_height = self.scene.rigid_objects["object"].data.body_pos_w[:,0,2] - self.scene.env_origins[:,2]
        object_up_height = obj_height - self.scene.rigid_objects["object"].data.default_root_state[:,2]
        height_reward = object_up_height * 50.0  # 高度越高奖励越大
        height_reward_instant = (object_up_height > 0.01).float() * (self.ee_dist < 0.002).float() * 1 # 只要抬起就给奖励
        height_reward *= (self.ee_dist < 0.002).float()  # 仅在末端接近时生效       
        
        ## 完成任务的奖励
        task_reward = self.terminated.float() * 500
        
        # 计算当前距离
        current_dist = torch.norm(ee_pos_tensor - obj_pos_tensor, dim=1)
        # 计算接近奖励
        approach_reward = torch.zeros_like(current_dist)
        if self.prev_ee_to_obj_dist is not None:
            delta = current_dist - self.prev_ee_to_obj_dist
            approach_reward = -delta * 10  # 距离减少时奖励为正
        self.prev_ee_to_obj_dist = current_dist.clone()  # 更新上一时刻的距离
        if self.ee_in_pipe == False:
            approach_reward = torch.zeros_like(approach_reward)
        # print(task_reward)
        rewards = task_reward + height_reward_instant + close_reward + outside_pipe_reward + r_proj
        
        t = self.time_buf
        self.reward_dict['Outside Pipe Reward'][t] = outside_pipe_reward[0].item()  # 使用t作为索引填充
        self.reward_dict['Direction Projection Reward'][t] = r_proj[0].item()
        self.reward_dict['Close Reward'][t] = close_reward[0].item()
        self.reward_dict['Height Reward'][t] = height_reward[0].item()
        self.reward_dict['Task Reward'][t] = task_reward[0].item()
        self.reward_dict['All Reward'][t] = rewards[0].item()
        self.time_buf = self.time_buf+1
        self.reward_buffer = self.reward_buffer+rewards
        print(rewards)
        return rewards        
    def _get_rewards11(self) -> torch.Tensor:
        """
        奖励设计：
        1) 位置势能 shaping（xy / z 分权重）
        2) 夹爪接近 & 抬升奖励
        3) 任务完成奖励
        4) 越界惩罚 + 动作幅值惩罚
        """
        device = self.device
        N      = self.num_envs

        # ====================== 几何信息 ======================
        ee_pos  = self._robot.data.body_pos_w[:, self.ee_id, :3] - self.scene.env_origins[:, :3]     # (N,3)
        obj_pos = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3] - self.scene.env_origins[:, :3]

        xy_ee   = ee_pos[:, :2]                                        # (N,2)
        target_xy = torch.tensor([0.0, -0.29], device=device).expand(N, -1)

        dist_xy = torch.linalg.norm(xy_ee - target_xy, dim=1)          # (N,)
        obj_dist_xy = torch.linalg.norm(xy_ee - obj_pos[:,:2], dim=1)          # (N,)
        dist_z  = (ee_pos[:, 2] - obj_pos[:, 2]).abs()                 # (N,)

        # ====================== 1. 位置势能 shaping ======================
        #   Φ(s) = clip(max_d - d, 0)           ⇒ r = γΦ(s') - Φ(s)
        # γ = 0.99  (近似一阶差分)
        gamma = 0.99
        max_d_xy, max_d_z = 0.05, 0.05         # 5 cm & 5 cm
        phi_xy = (max_d_xy - obj_dist_xy).clamp(min=0.0)     # [0, 0.05]
        phi_z  = (max_d_z  - dist_z ).clamp(min=0.0)

        # 保存上一步 φ 以计算差分；首次 reset 时初始化为 0
        if not hasattr(self, "_prev_phi_xy"):
            self._prev_phi_xy = torch.zeros_like(phi_xy)
            self._prev_phi_z  = torch.zeros_like(phi_z)

        r_xy =  5.0 * (gamma * phi_xy - self._prev_phi_xy)             # ≈ ±5
        r_z  =  2.0 * (gamma * phi_z  - self._prev_phi_z)              # ≈ ±2

        # 更新缓存
        self._prev_phi_xy.copy_(phi_xy)
        self._prev_phi_z.copy_(phi_z)

        # ====================== 2. 接近 & 抬升奖励 ======================
        ee_close  = (self.ee_dist < 0.003).float()                     # 3 mm 内
        close_reward   = 1.0 * ee_close                                # 0 / 1

        obj_height     = obj_pos[:, 2] - self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        lift_reward    = 5.0 * (obj_height - 0.0).clamp(min=0.0) * ee_close   # 抬 1 cm 得 0.5
        lift_trigger   = (obj_height > 0.01).float() * ee_close                # 抬离 1 cm 触发
        lift_reward += lift_trigger * 1.0                                     # 额外 +1

        # ====================== 3. 任务完成奖励 ======================
        task_reward = self.terminated.float() * 200.0                          # 0 / 200

        # ====================== 4. 负向奖励 ======================
        # 4.1 越界惩罚（并可在外部逻辑里 done=True）
        outside_mask = ((dist_xy > 0.008) & (ee_pos[:, 2] < -0.22)).float()
        outside_pen  = outside_mask * -50.0                                    # -50

        # 4.2 动作幅值惩罚（L2）
        action_pen   = -1.0 * self.policy_action.pow(2).sum(dim=1)             # ≈ -0~ -3

        # ====================== 合并 ======================
        rewards = (
            r_xy + r_z                # 势能 shaping
            + close_reward
            + lift_reward
            + task_reward
            + outside_pen
            + action_pen
        )

        # ====================== 日志 ======================
        # t = self.episode_length_buf.item()
        # rd = self.reward_dict
        # rd['Shaping_xy'][t]   = r_xy[0].item()
        # rd['Shaping_z'][t]    = r_z[0].item()
        # rd['Close'][t]        = close_reward[0].item()
        # rd['Lift'][t]         = lift_reward[0].item()
        # rd['Task'][t]         = task_reward[0].item()
        # rd['Outside'][t]      = outside_pen[0].item()
        # rd['ActionPen'][t]    = action_pen[0].item()
        # rd['Total'][t]        = rewards[0].item()

        return rewards

    def _get_rewards3(self) -> torch.Tensor:
        
        # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
        ee_pos_tensor = self._robot.data.body_pos_w[:,self.ee_id,:3] -self.scene.env_origins[:,:3] # 例如 shape 为 (num_bodies, 3)
        obj_pos_tensor = self.scene.rigid_objects["object"].data.body_pos_w[:,0,:3] - self.scene.env_origins[:,:3]
        
        ## 在管道外给惩罚
        xy = ee_pos_tensor[:,:2]
        target_xy = torch.tensor([0.0, -0.29], device=self.device).repeat(self.num_envs,1)
        dist = torch.norm(xy - target_xy, dim = 1)
        
        z = ee_pos_tensor[:,2]
        obj_pos_tensor2 = obj_pos_tensor.clone()
        obj_pos_tensor2[:,2] = obj_pos_tensor.clone()[:,2]+0.002
        ee2obj_dist = torch.norm(obj_pos_tensor - ee_pos_tensor, dim = 1)
        clipped_dist = torch.clamp(-ee2obj_dist*100, min=-1)  
        dist_reward = clipped_dist

        ## 如果末端在管道内且超出了管道，给予惩罚
        outside_pipe_reward = -1.0 * (dist > 0.008).float() * (z < -0.22).float()
                                                            
        # 
        # print(r_proj)
        # print(dist_reward)
        ## 机械臂末端在object范围内给予奖励
        close_reward = 0.0 * (self.ee_dist < 0.002).float()
        
        ## 提起object的奖励
        obj_height = self.scene.rigid_objects["object"].data.body_pos_w[:,0,2] - self.scene.env_origins[:,2]
        object_up_height = obj_height - self.scene.rigid_objects["object"].data.default_root_state[:,2]
        height_reward = object_up_height * 50.0  # 高度越高奖励越大
        height_reward_instant = (object_up_height > 0.01).float() * (self.ee_dist < 0.002).float() * 1 # 只要抬起就给奖励
        height_reward *= (self.ee_dist < 0.002).float()  # 仅在末端接近时生效       
        
        ## 完成任务的奖励
        task_reward = self.terminated.float() * 500
        # print(task_reward)
        rewards = task_reward + height_reward_instant + dist_reward + close_reward + outside_pipe_reward
        if self.ee_in_pipe == False:
            rewards = outside_pipe_reward
        t = self.time_buf
        self.reward_dict['Outside Pipe Reward'][t] = outside_pipe_reward[0].item()  # 使用t作为索引填充
        self.reward_dict['Direction Projection Reward'][t] = dist_reward[0].item()
        self.reward_dict['Close Reward'][t] = close_reward[0].item()
        self.reward_dict['Height Reward'][t] = height_reward[0].item()
        self.reward_dict['Task Reward'][t] = task_reward[0].item()
        self.reward_dict['All Reward'][t] = rewards[0].item()
        self.time_buf = self.time_buf+1
        self.reward_buffer = self.reward_buffer+rewards
        print(rewards)
        return rewards            
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        self.time_buf = 0
        # ---- 在开始新一集前，把上一集的 reward_dict 写入同一个 HDF5 文件 ----
        if hasattr(self, 'reward_dict'):
            for key, arr in self.reward_dict.items():
                ds = self.datasets[key]
                ds.resize((ds.shape[0] + 1, self.max_episode_length))
                ds[-1, :] = arr
            self.h5file.flush()
            self.episode_counter += 1
        if not hasattr(self, "_robot_ik"):
            self._robot_ik = DifferentialInverseKinematicsAction(
                self.cfg.left_robot_ik,
                self.scene,
            )
        # ---------- 统计时间间隔（仅当 env_ids 给出时） ----------
        # if env_ids is not None:
        #     env_ids = env_ids.to(self.device).long().view(-1)

        #     now  = torch.tensor(time.perf_counter(),
        #                         device=self.device, dtype=torch.float64)

        #     prev = self.last_reset_t.index_select(0, env_ids)
        #     gap  = torch.where(torch.isnan(prev), prev, now - prev)

        #     # 写回
        #     self.last_reset_t.index_fill_(0, env_ids, now) 
        #     self.reset_interval.scatter_(0, env_ids, gap)
        #     for eid, dt in zip(env_ids.cpu().tolist(),
                # self.reset_interval.index_select(0, env_ids).cpu().tolist()):
                # print(f"env {eid:4d} : {dt:.6f} s")
        # ---------------------------------------------------------
        super()._reset_idx(env_ids)
        self.episode_length_buf[env_ids] = 0
        
        self._robot.reset(env_ids)
        
        self._robot_ik.reset(env_ids)
        # 机器人状态reset
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

        # 重新生成needle的位置
        self._resample_command(env_ids)
        self._object.write_root_pose_to_sim(self.pose_command_w[env_ids,:],env_ids=env_ids)
        self._object.write_root_velocity_to_sim(torch.zeros_like(self._object.data.root_vel_w[env_ids,:]), env_ids=env_ids)
        self.robot_dof_targets = joint_pos
        self.scene.write_data_to_sim()

        self.policy_wrapper.reset(env_ids)
        print(self.reward_buffer)
        self.all_rewards.append(self.reward_buffer)
        
        ee_pos = self._robot.data.body_pos_w[:, self.ee_id, :3]
        obj_pos = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]
        initial_dist = torch.norm(ee_pos - obj_pos, dim=1)
        self.prev_ee_to_obj_dist[env_ids] = initial_dist[env_ids]
        
        # 修改后
        with open('all_rewards_192_2.txt', 'a') as f:
            f.write("%s\n" % self.reward_buffer)
        self.reward_buffer = 0
        # 初始化 reward_dict，每个奖励项是一个大小为 num_steps 的 numpy 数组
        self.reward_dict = {
            'Outside Pipe Reward': np.zeros(self.max_episode_length),
            'Direction Projection Reward': np.zeros(self.max_episode_length),
            'Close Reward': np.zeros(self.max_episode_length),
            'Height Reward': np.zeros(self.max_episode_length),
            'Task Reward': np.zeros(self.max_episode_length),
            'All Reward': np.zeros(self.max_episode_length)
        }

        
        if self.enable_draw:
             self.draw.clear_points()

    def _get_observations1(self) -> dict:

        # ACT动作
        raw_action = self.get_act_actions()
        self.raw_action = raw_action.clone()

        # 机械臂末端位置
        self.robot_local_pos =self._object.data.body_state_w[:,0,0:3] - self._robot.data.body_state_w[:,self.ee_id,0:3]
        # self.robot_local_pos =self._object.data.body_state_w[:,0,0:3] -  self.scene.env_origins
        self.robot_quat = self._robot.data.body_state_w[:,self.ee_id,3:7]
        
        # object位置
        self.object_local_pos = self._object.data.body_state_w[:,0,0:3] - self.scene.env_origins
        self.object_quat = self._object.data.body_state_w[:,0,3:7]
        
        # 机械臂关节角度
        joint_pos = self._robot.data.joint_pos- self._robot.data.default_joint_pos
        
        # 机械臂关节角速度
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        
        # 相对于管口位置
        pos_ref = self._robot.data.body_state_w[:,self.ee_id,0:3] - self.pipe_top_pos
        
        obs = torch.cat(
            (
                self.raw_action,
                self.robot_local_pos,
                self.robot_quat,
                self.object_local_pos,
                self.object_quat,
                joint_pos,
                joint_vel,
                pos_ref,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
    def _get_observations(self) -> dict:
        """
        只暴露精细操作所需的局部信息：
        • Δp_fine   ∈ [-1,1]  ← 末端相对物体 ±1 cm
        • Δq_fine   ∈ [-1,1]  ← 姿态误差 ±3°
        • ACT raw_action      ← 模仿策略输出 [-1,1]
        • prev_residual       ← 上一步残差 (若需要)
        • sin/cos(joint_pos), joint_vel_norm
        """
        device = self.device
        # ---------- 1. 模仿策略动作 ----------
        raw_action = self.get_act_actions()              # (N, action_dim)
        self.raw_action = raw_action.clone()

        # ---------- 2. 末端-物体相对位姿 ----------
        d_pos = (
            self._object.data.body_state_w[:, 0, :3]
            - self._robot.data.body_state_w[:, self.ee_id, :3]
        )                                                # (N,3), 单位 m

        fine_scale = 0.005                      # 5 mm
        mid_scale  = 0.02                       # 2 cm

        d_pos_fine = torch.clamp(d_pos, -fine_scale, fine_scale) / fine_scale
        d_pos_mid  = torch.clamp(d_pos, -mid_scale,  mid_scale)  / mid_scale
        self.object_local_pos = self._object.data.body_state_w[:,0,0:3] - self.scene.env_origins
        # ---------- 3. 关节状态 ----------
        joint_pos  = self._robot.data.joint_pos          # (N, J)
        joint_vel  = self._robot.data.joint_vel          # (N, J)

        joint_sin  = torch.sin(joint_pos)
        joint_cos  = torch.cos(joint_pos)

        vel_scale  = self._robot.data.soft_joint_vel_limits       # (J,) 物理上限
        joint_vel_norm = torch.clamp(
            joint_vel / vel_scale, -1.0, 1.0
        )                                                # (N, J)

        # ---------- 5. 拼接 ----------
        obs = torch.cat(
            (
                raw_action,          # imit-policy reference
                self.object_local_pos,
                d_pos_fine,         # 重点特征
                d_pos_mid,          # 辅助远距离
                joint_sin, joint_cos,
                joint_vel_norm,
            ),
            dim=-1,
        )

        # ---------- 6. 返回 ----------
        return {"policy": obs}        # 已在 [-1,1] 区间，无需再 clamp ±5
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
        pose,quat = self.get_pipe_state()
        # self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        self.goal_pose_visualizer.visualize(self.scene.rigid_objects["object"].data.body_pos_w[:,0,:3], self.scene.rigid_objects["object"].data.body_state_w[:,0,3:7])
        # self.goal_pose_visualizer.visualize(pose,quat)
        # -- current body pose
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        
    def init_robot_ik(self):
        self._robot_ik = DifferentialInverseKinematicsAction(self.cfg.left_robot_ik,self.scene)


    def get_pipe_state(self):
        pipe_local_pos = self.cfg.pipe_pos
        pipe_quat = self.cfg.pipe_quat.repeat(self.num_envs, 1)
        pipe_world_pos = pipe_local_pos + self.scene.env_origins
        pipe_height = torch.tensor([0,0,0.04], device=self.device).repeat(self.num_envs, 1)
        pipe_top_pos = pipe_world_pos + pipe_height
        return pipe_top_pos, pipe_quat
    
    def get_axial_depth(self,
                        ee_pos: torch.Tensor
                        ) -> torch.Tensor:
        """
        计算末端执行器沿管道轴线方向的深度（axial depth）。
        
        输入:
          ee_pos      — 末端执行器世界坐标，shape (N,3)
        输出:
          d_axial     — 轴向深度，shape (N,)
                         d>0 表示在管道内部，
                         d<0 表示在管道外，
                         d=0 即恰好在管口处
        """
        # 3) 计算末端到管口的偏移向量
        delta = ee_pos - self.pipe_top_pos  # (N,3)
        
        # 4) 点积得到轴向深度
        d_axial = torch.sum(delta * self.u_axis, dim=1)  # (N,)
        return d_axial
    
    def get_act_actions(self):
        
        with torch.no_grad():
            if self.temporal_agg:
                # 读取关节位置
                qpos_numpy = self._robot.data.joint_pos.cpu().numpy().reshape(self.num_envs, 7)

                # 获取相机图像
                multi_cam_data = convert_dict_to_backend(
                    {k: v[:] for k, v in self._camera.data.output.items()}, backend="numpy"
                )
                for i, img_rgb in enumerate(multi_cam_data['rgb']):
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"env_{i}", img_bgr)
                cv2.waitKey(1)
                depth_image = enhance_depth_images_batch(multi_cam_data['distance_to_image_plane'])
                depth_image = rearrange(depth_image, 'n h w c -> n c h w')
                depth_image_process = torch.from_numpy(depth_image / 255.0).float().to(self.device).unsqueeze(1)

                curr_image = depth_image_process

                # 获取动作并与环境交互
                raw_action = self.policy_wrapper.get_action(qpos_numpy, curr_image, self.episode_length_buf)
            else:
                                    # 获取相机图像
                multi_cam_data = convert_dict_to_backend(
                    {k: v[:] for k, v in self._camera.data.output.items()}, backend="numpy"
                )
                for i, img_rgb in enumerate(multi_cam_data['rgb']):
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"env_{i}", img_bgr)
                cv2.waitKey(1)
                if self.time_buf % self.query_frequency == 0:
                    
                    qpos_numpy = self._robot.data.joint_pos.cpu().numpy().reshape(self.num_envs, 7)


                    depth_image = enhance_depth_images_batch(multi_cam_data['distance_to_image_plane'])
                    depth_image = rearrange(depth_image, 'n h w c -> n c h w')
                    depth_image_process = torch.from_numpy(depth_image / 255.0).float().to(self.device).unsqueeze(1)

                    curr_image = depth_image_process
                    raw_action = self.policy_wrapper.get_action(qpos_numpy, curr_image, self.time_buf)
                else:
                    raw_action = self.policy_wrapper.get_chunked_action(self.time_buf)
        return raw_action