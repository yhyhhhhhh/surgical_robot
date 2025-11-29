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
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_error_magnitude, quat_mul
from omni.isaac.core.prims import XFormPrimView
import numpy as np
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

import omni.kit.app
import weakref

@configclass
class Ur3ReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333
    decimation = 2
    action_space = 6

    observation_space = 25
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
            usd_path=f"/home/yhy/DVRK/ur3_scissor/ur3_isaac.usd",
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
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0,
            },
            # joint_pos={
            #     "shoulder_pan_joint": 3.14,
            #     "shoulder_lift_joint": 0,
            #     "elbow_joint": -1.57,
            #     "wrist_1_joint": 0.0,
            #     "wrist_2_joint": 0.0,
            #     "wrist_3_joint": -1.57,
            # },
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
    
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/World/Visuals/Command/body_pose"
    )

    command_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/command_range",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.3, 0.4, 0.4),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity = 0.1),
            ),
        },
    )
    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    # 指令输出范围
    ranges = {
        "pos_x": (0.35, 0.5),
        "pos_y": (-0.2, 0.2),
        "pos_z": (0.25, 0.55),
        "roll": (0.3, 0.3),
        "pitch": (-1.57, 1.57),
        "yaw": (1.3, 3),
    }
    
    make_quat_unique = True
    debug_vis = True
    
    action_scale = 7.5
    dof_velocity_scale = 0.1
    
    # rewards权重
    position_reward_scale = -0.2
    position_tan_reward_scale = 0.1
    orientation_reward_scale = -0.1
    action_rate_reward_scale = -0.0007
    joint_vel_reward_scale = -0.0007
    
    std = 0.1
   


class Ur3ReachEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Ur3ReachEnvCfg

    def __init__(self, cfg: Ur3ReachEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)  # 缓冲区存储 (x, y, z, qw, qx, qy, qz)
        self.pose_command_b[:, 3] = 1.0  # 设置四元数的实部为默认值 1
        self.pose_command_w = torch.zeros_like(self.pose_command_b)  # 初始化世界坐标系的命令缓冲区

        # 初始化误差度量
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)  # 存储位置误差
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)  # 存储姿态误差
    
        self.set_debug_vis(self.cfg.debug_vis)
        self.command_visualizer_b =  torch.tensor([[0.4, 0, 0.35]] * self.num_envs, device=self.device)

        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # self.table = XFormPrimView(self.cfg.table.prim_path, reset_xform_properties=False)
        self.scene.articulations["robot"] = self._robot
        # self.scene.extras["Table"] =  self.table
        self.cfg.table.spawn.func(
            self.cfg.table.prim_path,
            self.cfg.table.spawn,
            translation = self.cfg.table.init_state.pos,
            orientation = self.cfg.table.init_state.rot,
        )
        self.scene.extras["Table"] = XFormPrimView(self.cfg.table.prim_path, reset_xform_properties=False)
        
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation = self.cfg.ground.init_state.pos,
        )
        self.scene.extras["ground"] = XFormPrimView(self.cfg.ground.prim_path, reset_xform_properties=False)
               
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定的环境重新采样位姿命令（位置和姿态）。"""
        # 采样位置命令
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges["pos_x"])  # 在指定范围内采样 x 坐标
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges["pos_y"])  # 在指定范围内采样 y 坐标
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges["pos_z"])  # 在指定范围内采样 z 坐标

        # 采样姿态命令
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])  # 初始化欧拉角缓冲区
        euler_angles[:, 0].uniform_(*self.cfg.ranges["roll"])  # 在指定范围内采样 roll
        euler_angles[:, 1].uniform_(*self.cfg.ranges["pitch"])  # 在指定范围内采样 pitch
        euler_angles[:, 2].uniform_(*self.cfg.ranges["yaw"])  # 在指定范围内采样 yaw
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])  # 将欧拉角转换为四元数

        # 如果配置中要求四元数唯一性，则确保四元数的实部为正
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        # 将姿态指令转换到世界坐标系
        self.pose_command_w[env_ids, :3], self.pose_command_w[env_ids, 3:] = combine_frame_transforms(self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7],self.pose_command_b[env_ids, :3] ,self.pose_command_b[env_ids, 3:])
    
    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self._robot.data.body_state_w[:, self.ee_id, :3],
            self._robot.data.body_state_w[:, self.ee_id, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
    
    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        #self.actions = actions.clone().clamp(-1.0, 1.0)
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
        

    def _get_observations(self) -> dict:
        # 关节相对于初始角度的变化
        joint_pos = self._robot.data.joint_pos- self._robot.data.default_joint_pos
        # 关节相对于初始速度的变化
        joint_vel = self._robot.data.joint_vel - self._robot.data.default_joint_vel
        # 末端空间姿态指令
        PoseCommand = self.pose_command_b
        # 上一次的action
        actions = self.actions
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

   
    def _compute_rewards(
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
        
        rewards = (
            self.cfg.position_reward_scale * position_error_reward
            + self.cfg.position_tan_reward_scale * position_error_tanh_reward
            + self.cfg.orientation_reward_scale * orientation_error_reward
            + self.cfg.action_rate_reward_scale * action_rate_l2_reward
            + self.cfg.joint_vel_reward_scale * joint_vel_l2_reward
        )
        # rewards = (
        #     self.cfg.position_reward_scale * position_error_reward
        #     + self.cfg.position_tan_reward_scale * position_error_tanh_reward
        #     + self.cfg.orientation_reward_scale * orientation_error_reward
        # )
        return rewards
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
                self.command_visualizer = VisualizationMarkers(self.cfg.command_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.command_visualizer.set_visibility(False)
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
        # self.command_visualizer_w, _ = combine_frame_transforms(self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7],self.command_visualizer_b)
        # self.command_visualizer.visualize(translations=self.command_visualizer_w)
        
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self._robot.data.body_state_w[:, self.ee_id]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
        


