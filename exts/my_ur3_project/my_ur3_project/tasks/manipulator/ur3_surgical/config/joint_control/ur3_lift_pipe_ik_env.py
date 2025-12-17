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

import omni.usd
from pxr import UsdUtils
def save_current_stage(output_usd_path: str, flatten: bool = True):
    """
    保存当前 Stage 到一个 USD 文件。

    :param output_usd_path: 输出文件路径，比如 "/home/yhy/DVRK/scenes/ur3_scene_flat.usd"
    :param flatten: True=展平（所有引用烤进一个文件），False=简单导出
    """
    # 1) 拿到当前 stage
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("当前没有打开的 Stage，确认已经完成 _setup_scene() 或环境初始化。")

    if flatten:
        # 2) 展平 LayerStack，变成一个完全自包含的 layer
        print(f"[save_current_stage] Flatten stage to: {output_usd_path}")
        flat_layer = UsdUtils.FlattenLayerStack(stage)
        flat_layer.Export(output_usd_path)
    else:
        # 2) 简单导出：保留 reference / sublayer 结构
        print(f"[save_current_stage] Export stage to: {output_usd_path}")
        stage.Export(output_usd_path)

    print("[save_current_stage] Done.")

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
        
        # =========================================================
        # 在类的 __init__ 里加三行
        # =========================================================
        # ↙  事先用 generate_command_buffer.py 生成的文件
        self._cmd_buffer = torch.load('/home/yhy/DVRK/IsaacLabExtensionTemplate/command_buffer.pt', map_location=self.device)  # (N,7)
        self._cmd_size   = self._cmd_buffer.shape[0]
        self._cmd_ptr    = 0                     # 读指针
        # ------- 然后生成10组固定的半径r和角度theta -------
        u = torch.ones(1, device=self.device)  # 均匀分布
        self.fixed_r = 0.0025 * torch.sqrt(u)    # 半径均匀分布（面积均匀）
        self.fixed_theta = torch.ones(1, device=self.device) * 0.5 * math.pi  # 角度均匀分布 [0, 2pi)
        self.prev_ee_to_obj_dist = torch.zeros(self.num_envs, device=self.device)
    def _resample_command1(self, env_ids: Sequence[int]):
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
    

    def _resample_command2(self, env_ids: Sequence[int]):
        """
        顺序取用预生成的 object pose：
            • 位置 = _cmd_buffer[:,:3] + env_origin
            • 姿态 = _cmd_buffer[:,3:]
        若序列读到末尾则自动从头循环，但 **不会** 写入新随机数据。
        """
        n = len(env_ids)

        # ------- 取出 n 条 command（循环缓冲区） -------
        idxs = (torch.arange(n, device=self.device) + self._cmd_ptr) % self._cmd_size
        cmds = self._cmd_buffer[idxs]                                   # (n,7)

        # 更新指针
        self._cmd_ptr = (self._cmd_ptr + n) % self._cmd_size

        # ------- 写入 pose_command_w -----------
        pos  = cmds[:, :3] + self.scene.env_origins[env_ids]            # 加上各环境偏移
        quat = cmds[:, 3:]

        self.pose_command_w[env_ids, :3] = pos
        self.pose_command_w[env_ids, 3:] = quat

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
        save_current_stage(
            "/home/yhy/DVRK/scenes/ur3_surgery_scene_flat.usd",
            flatten=True,   # 想要“一整个场景都在一个 usd 里看”就设 True
        )
        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])


    def _resample_command(self, env_ids: Sequence[int]):
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
        r = max_r * torch.sqrt(u)  # 均匀采样圆面积

        # theta 从 0 到 3.14 均匀采样（注意，这里只采样半圆，如果需要全圆，请改为 2*pi）
        theta = torch.rand(n, device=device) * 3.14 * 2

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
    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # 使用torch.where实现条件限制
        self.actions[:, 2] = torch.where(
            self.actions[:, 2] < -0.235,
            torch.full_like(self.actions[:, 2], -0.235),
            self.actions[:, 2]
        )

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
            torch.tensor([-0.1], device=self.device)
        )
    
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
        terminated = dist > 0.1
        self.dist = dist

        self.ee_dist =  torch.norm(robot_ee_pos - pos_tensor, dim = 1)
        # 条件1：机器人末端执行器与 object 的距离小于 0.005
        self.terminated = torch.logical_and(terminated, self.ee_dist < 0.01)
        
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return self.terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        ## 机械臂末端到object的距离,距离越近越大
        dist_scale = 2
        dist_reward = torch.exp(-self.ee_dist * dist_scale)
        
        ## 机械臂末端在object范围内给予奖励
        close_reward = 0.5 * (self.ee_dist < 0.01).float()
        
        ## 提起object的奖励
        pos_tensor = self.scene.rigid_objects["object"].data.body_pos_w[:,0,2] - self.scene.env_origins[:,2]
        object_up_height = pos_tensor - self.scene.rigid_objects["object"].data.default_root_state[:,2]
        height_reward = object_up_height * 5.0  # 高度越高奖励越大
        height_reward *= (self.ee_dist < 0.01).float()  # 仅在末端接近时生效       
        
        ## 完成任务的奖励
        task_reward = self.terminated.float() * 10.0
        rewards = dist_reward + close_reward + height_reward + task_reward
        return task_reward
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # ---------- 统计时间间隔（仅当 env_ids 给出时） ----------
        if env_ids is not None:
            env_ids = env_ids.to(self.device).long().view(-1)

            now  = torch.tensor(time.perf_counter(),
                                device=self.device, dtype=torch.float64)

            prev = self.last_reset_t.index_select(0, env_ids)
            gap  = torch.where(torch.isnan(prev), prev, now - prev)

            # 写回
            self.last_reset_t.index_fill_(0, env_ids, now) 
            self.reset_interval.scatter_(0, env_ids, gap)
            for eid, dt in zip(env_ids.cpu().tolist(),
                self.reset_interval.index_select(0, env_ids).cpu().tolist()):
                print(f"env {eid:4d} : {dt:.6f} s")
        if not hasattr(self, "_robot_ik"):
            self._robot_ik = DifferentialInverseKinematicsAction(
                self.cfg.left_robot_ik,
                self.scene,
            )
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




    def _get_observations(self) -> dict:

        # 机械臂末端位置
        self.robot_local_pos = self._robot.data.body_state_w[:,self.ee_id,0:3] - self.scene.env_origins 
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
    def action_go_pipe_mouth(self):
        # 管口(世界) + 管姿态
        pipe_top_pos_w, pipe_quat_w = self.get_pipe_state()   # (N,3), (N,4)

        actions = torch.zeros((self.num_envs, 8), device=self.device)
        # actions 里位置是 env frame，所以要减去 env_origin
        actions[:, :3]  = pipe_top_pos_w - self.scene.env_origins
        actions[:, 3:7] = pipe_quat_w
        actions[:, 7]   = 0.0   # <0.5 => 张开

        return actions
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
        self.goal_pose_visualizer.visualize(pose,quat)
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

    def _world_to_pipe_coords(self, pos_w: torch.Tensor):
        """世界坐标 → 管道坐标 (s, r, theta)。

        Args:
            pos_w: (N,3) 世界坐标下的位置

        Returns:
            s:   (N,1) 轴向深度（>0 在管内）
            r:   (N,1) 径向距离
            th:  (N,1) 截面内极角
            x_r: (N,1) 径向向量在 x 轴上的 x 分量
            y_r: (N,1) 径向向量在 y 轴上的 y 分量
        """
        # self.pipe_top_pos: (N,3)
        # self.u_axis:      (N,3)
        delta = pos_w - self.pipe_top_pos  # (N,3)

        # 轴向深度：沿管道轴的投影
        s = torch.sum(delta * self.u_axis, dim=-1, keepdim=True)  # (N,1)

        # 去掉轴向分量，得到截面平面的径向向量
        radial = delta - s * self.u_axis  # (N,3)
        x_r = radial[..., 0:1]
        y_r = radial[..., 1:2]

        r = torch.sqrt(x_r * x_r + y_r * y_r + 1e-8)
        th = torch.atan2(y_r, x_r)

        return s, r, th, x_r, y_r

    @torch.no_grad()
    def compute_validation_reward(self) -> torch.Tensor:
        """使用强化学习版本的奖励设计，对当前状态打分并记录每个分量，方便可视化。

        返回:
            rewards: (num_envs,) 每个环境当前时刻的验证奖励（总和）。
        """
        device = self.device

        # ===== 和管道有关的常数（按 RL 环境的设定来） =====
        pipe_radius = 0.006          # m
        pipe_safety_margin = 0.001   # m
        pipe_length = 0.04           # m

        # ===== 末端 & 物体的世界坐标 =====
        ee_pos_w = self._robot.data.body_pos_w[:, self.ee_id, 0:3]
        obj_pos_w = self.scene.rigid_objects["object"].data.body_pos_w[:, 0, :3]

        # 3D 距离
        ee_dist = torch.norm(ee_pos_w - obj_pos_w, dim=1)  # (N,)

        # ===== 管道坐标系下的位置 =====
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

        # ==========================
        # 1) 管道平面接近奖励（末端 vs 物体）
        # ==========================
        reach_scale = 80.0
        reach_reward = torch.exp(-reach_scale * d_pipe)

        # ==========================
        # 2) 管口引导（从管外靠近管口）
        # ==========================
        outside = (s_e < 0.0)  # 管外区域
        k_s = 10.0
        k_r = 40.0
        entry_reward = torch.exp(-k_s * torch.abs(s_e)) * torch.exp(-k_r * r_e)
        entry_reward = entry_reward * outside.float()

        # ==========================
        # 3) 管内居中 / 避免贴壁
        # ==========================
        in_pipe = (s_e > 0.0) & (s_e < (pipe_length - pipe_safety_margin))
        r_norm = r_e / (pipe_radius + 1e-6)
        center_reward = torch.exp(-4.0 * (r_norm ** 2)) * in_pipe.float()

        # ==========================
        # 4) 抓取相关奖励
        # ==========================
        # 4.1 3D 接近物体（只在管内生效）
        is_in_pipe = (s_e > 0.0) & (r_e < pipe_radius)

        grasp_reach_scale = 40.0
        # 注意：这里直接用 ee_dist，避免 self.ee_dist 没有定义的问题
        raw_grasp_reward = torch.exp(-grasp_reach_scale * ee_dist)
        grasp_reach_reward = raw_grasp_reward * is_in_pipe.float()

        # 4.2 张开 / 闭合状态
        if hasattr(self, "gripper_action"):
            grip_cmd = self.gripper_action  # (N,1)
        else:
            # 兜底：取最后一个关节位置当夹爪
            grip_cmd = self._robot.data.joint_pos[:, -1:].clone()

        gripper_min = -0.28
        gripper_max = -0.10
        grip_norm = (grip_cmd - gripper_min) / (gripper_max - gripper_min + 1e-6)
        grip_norm = torch.clamp(grip_norm, 0.0, 1.0).squeeze(-1)  # 0=闭合,1=张开

        # 真正贴近物体（准备夹）
        near_for_grasp = (ee_dist < 0.01).float()
        # 稍远一点，用来对准
        middle_near = ((ee_dist > 0.01) & (ee_dist < 0.03)).float()

        # 靠得很近时，鼓励闭合
        grasp_close_reward = near_for_grasp * (1.0 - grip_norm)
        # 稍远时，鼓励张开
        pre_grasp_open_reward = middle_near * grip_norm

        # ==========================
        # 5) 抬起物体的奖励
        # ==========================
        obj_z = obj_pos_w[:, 2] - self.scene.env_origins[:, 2]
        init_z = self.scene.rigid_objects["object"].data.default_root_state[:, 2]
        object_lift = torch.clamp(obj_z - init_z, min=0.0)

        close_for_lift = (ee_dist < 0.02).float()
        lift_reward = object_lift * 100.0 * close_for_lift

        # ==========================
        # 6) 鼻腔外距离奖励（把物体往外拖）
        # ==========================
        xy = obj_pos_w[:, :2] - self.scene.env_origins[:, :2]
        target_xy = torch.tensor([0.0, -0.29], device=device).repeat(self.num_envs, 1)
        dist_xy = torch.norm(xy - target_xy, dim=1)

        move_out_reward = torch.clamp(dist_xy - 0.03, min=0.0) * 3.0

        # ==========================
        # 7) 成功条件 + 大奖励
        # ==========================
        object_outside = dist_xy > 0.1
        ee_close = ee_dist < 0.01
        success = torch.logical_and(object_outside, ee_close)
        success_reward = success.float() * 40.0

        # ==========================
        # 8) 汇总总奖励
        # ==========================
        rewards = (
            0.5 * entry_reward +
            0.8 * reach_reward +
            0.5 * center_reward +
            1.2 * grasp_reach_reward +
            0.3 * pre_grasp_open_reward +
            0.7 * grasp_close_reward +
            1.0 * lift_reward +
            1.0 * move_out_reward +
            1.0 * success_reward
        )

        # ==========================
        # 9) 写入 extras，记录奖励各分量 & 关键几何信息
        # ==========================
        # 注意：这里只记录均值，方便在 TensorBoard 或你自己的脚本中画曲线
        self.extras["val_reward_log"] = {
            # 总奖励
            "val/total":              rewards.mean(),

            # 各项奖励（未再乘 batch 大小）
            "val/reach_pipe":         reach_reward.mean(),
            "val/entry":              entry_reward.mean(),
            "val/center_pipe":        center_reward.mean(),
            "val/grasp_reach_3d":     grasp_reach_reward.mean(),
            "val/pre_grasp_open":     pre_grasp_open_reward.mean(),
            "val/grasp_close":        grasp_close_reward.mean(),
            "val/lift":               lift_reward.mean(),
            "val/move_out":           move_out_reward.mean(),
            "val/success":            success_reward.mean(),

            # 距离 / 几何信息
            "val/ee_obj_dist":        ee_dist.mean(),
            "val/d_pipe":             d_pipe.mean(),
            "val/pipe_s_ee":          s_e.mean(),
            "val/pipe_r_ee":          r_e.mean(),
            "val/pipe_s_obj":         s_o.mean(),
            "val/pipe_r_obj":         r_o.mean(),
            "val/r_norm":             r_norm.mean(),
            "val/object_lift_h":      object_lift.mean(),
            "val/dist_xy":            dist_xy.mean(),

            # 夹爪 & 区域占比
            "val/grip_norm_mean":     grip_norm.mean(),
            "val/near_for_grasp":     near_for_grasp.mean(),
            "val/middle_near":        middle_near.mean(),
            "val/in_pipe_ratio":      in_pipe.float().mean(),
            "val/outside_ratio":      outside.float().mean(),
            "val/success_rate":       success.float().mean(),
        }

        return rewards

