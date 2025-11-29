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
from .ur3_lift_needle_ik_env_cfg import Ur3LiftNeedleEnvCfg
from .utils.robot_ik_fun import DifferentialInverseKinematicsAction
class Ur3LiftNeedleEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Ur3LiftNeedleEnvCfg

    def __init__(self, cfg: Ur3LiftNeedleEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.needle_x_b = torch.empty(self.num_envs, device=self.device)
        self.needle_y_b = torch.empty(self.num_envs, device=self.device)

        
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
        # 左侧ur3机械臂
        self.cfg.left_robot.spawn.func(
            self.cfg.left_robot.prim_path,
            self.cfg.left_robot.spawn,
            translation = self.cfg.left_robot.init_state.pos,
            orientation = self.cfg.left_robot.init_state.rot,
        )
        # 右侧ur3机械臂
        self.cfg.right_robot.spawn.func(
            self.cfg.right_robot.prim_path,
            self.cfg.right_robot.spawn,
            translation = self.cfg.right_robot.init_state.pos,
            orientation = self.cfg.right_robot.init_state.rot,
        )
        # 内窥镜
        self.cfg.ecm.spawn.func(
            self.cfg.ecm.prim_path,
            self.cfg.ecm.spawn,
            translation = self.cfg.ecm.init_state.pos,
            orientation = self.cfg.ecm.init_state.rot,
        )
        # 手术台生成
        self.cfg.table_operate.spawn.func(
            self.cfg.table_operate.prim_path,
            self.cfg.table_operate.spawn,
            translation = self.cfg.table_operate.init_state.pos,
            orientation = self.cfg.table_operate.init_state.rot,
        )
        # 模拟组织平台生成
        self.cfg.plantom.spawn.func(
            self.cfg.plantom.prim_path,
            self.cfg.plantom.spawn,
            translation = self.cfg.plantom.init_state.pos,
            orientation = self.cfg.plantom.init_state.rot,
        )
        # 手术针生成
        self._needle = RigidObject(cfg=self.cfg.needle)
        
        # 机械臂控制器配置
        self._robot = Articulation(self.cfg.left_robot)
        self._right_robot = Articulation(self.cfg.right_robot)
        self._ecm = Articulation(self.cfg.ecm)
        
        # 摄像头生成
        self._camera = Camera(cfg=self.cfg.camera)
        self.scene.sensors["Camera"] = self._camera
        
        # 环境配置
        self.scene.articulations["left_robot"] = self._robot
        self.scene.articulations["right_robot"] = self._right_robot
        self.scene.articulations["ecm"] = self._ecm
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table_robot.prim_path, reset_xform_properties=False)
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)
        self.scene.rigid_objects["needle"] = self._needle
        # 并行复制环境
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        


    def _resample_command(self, env_ids: Sequence[int]):
        """随机生成needle位置"""
        self.needle_x_b.uniform_(*self.cfg.ranges["needle_x"])
        self.needle_y_b.uniform_(*self.cfg.ranges["needle_y"])

        # 转换为笛卡尔坐标，获取pipe坐标系的采样点
        local_point = torch.stack([
            self.needle_x_b,
            self.needle_y_b,
            torch.tensor([-0.2],device=self.device)
        ], dim=1)
        
        # 将姿态指令转换到世界坐标系
        self.pose_command_w[env_ids, :3], self.pose_command_w[env_ids, 3:] = combine_frame_transforms(
            self.scene.env_origins[env_ids, :3],
            self._robot.data.root_state_w[env_ids, 3:7],
            local_point[env_ids, :3]
        )


    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # 获取基座在世界坐标系中的位姿
        root_pos_w = self._robot.data.root_state_w[:, :3]  # (num_envs, 3)
        root_quat_w = self._robot.data.root_state_w[:, 3:7]  # (num_envs, 4)
        
        # 将目标姿态转换到基座坐标系
        pos_base, quat_base = math_utils.subtract_frame_transforms(
            root_pos_w, 
            root_quat_w,
            self.actions[:,0:3],
            self.actions[:,3:7],
        )
        pose =  torch.cat([pos_base, quat_base], dim=1)
        self._robot_ik.process_actions(pos_base)
        
        # self._robot_ik.process_actions(self.actions[:,0:6]) 
        # if self.actions[:,7]:
        if self.actions[:,-1]>0.5:
            self.gripper_action = torch.tensor([-0.22],device=self.device).unsqueeze(0)
        else:
            self.gripper_action = torch.tensor([0.2],device=self.device).unsqueeze(0)

    
    def _apply_action(self):
        ik_action = self._robot_ik.apply_actions()
        robot_action = torch.cat([ik_action, self.gripper_action], dim=1)
        self._robot.set_joint_position_target(robot_action)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps

        return torch.tensor(0.0)
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._robot_ik = DifferentialInverseKinematicsAction(self.cfg.left_robot_ik,self.scene)
        # 机器人状态reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        ecm_joint_pos = self._ecm.data.default_joint_pos[env_ids]
        ecm_joint_vel = torch.zeros_like(ecm_joint_pos)
        self._ecm.set_joint_position_target(ecm_joint_pos, env_ids=env_ids)
        self._ecm.write_joint_state_to_sim(ecm_joint_pos, ecm_joint_vel, env_ids=env_ids)
        right_joint_pos = self._right_robot.data.default_joint_pos[env_ids]
        self._right_robot.set_joint_position_target(right_joint_pos, env_ids=env_ids)
        self._right_robot.write_joint_state_to_sim(right_joint_pos, ecm_joint_vel, env_ids=env_ids)
        # 逆运动学控制器
        self._robot_ik.reset()
        # 重新生成needle的位置
        self._resample_command(env_ids)
        self._needle.write_root_pose_to_sim(self.pose_command_w)
        self.robot_dof_targets = joint_pos
        self.scene.write_data_to_sim()
        

    def _get_observations(self) -> dict:

        # 关节相对于初始角度的变化
        joint_pos = self._robot.data.joint_pos- self._robot.data.default_joint_pos
        # 关节相对于初始速度的变化
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        # 末端空间姿态指令
        PoseCommand = self.pose_command_b
        # 上一次的action
        actions = self.actions
        # 末端是否在管道中
        
        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                PoseCommand,
                actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
    
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
        


