from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass

# 自己编写的模块
from .utils.myfunc import *
from .utils.robot_ik_fun import DifferentialInverseKinematicsActionCfg


@configclass
class Ur3LiftPipeEnvCfg(DirectRLEnvCfg):
    
    # env
    episode_length_s = 4
    decimation = 2  # 5
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,))
    use_image_obs = False
    obs_groups = {"policy": ["policy"]}
    observation_space = 33
    # observation_space = 29
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 1/500
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)

        
    left_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Left_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/ur3TipCam_pro1_1_v0_debug.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,   # 真实重力下做 OSC
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -1.7139,
                "shoulder_lift_joint": 0.6371,
                "elbow_joint": 1.2418,
                "wrist_1_joint": 1.2720,
                "wrist_2_joint": -0.1490,
                "wrist_3_joint": -1.5632,
                "tip_joint": -0.1000,
            },
            pos=(0.15, -0.55, -0.16),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # arm 由 OSC 直接输出 effort，所以这里别再用高刚度位置驱动
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                ],
                velocity_limit=3.0,
                effort_limit=80.0,   # 建议起点，可再调
                stiffness=0.0,
                damping=0.0,
            ),
            # gripper / tip 先继续保留位置式控制
            "tip": ImplicitActuatorCfg(
                joint_names_expr=["tip_joint"],
                velocity_limit=3.0,
                effort_limit=10.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    
    # camera = CameraCfg( 
    #     prim_path="/World/envs/env_.*/Left_Robot/ur3_robot/camera_link/Camera",
    #     data_types=[ "rgb","distance_to_image_plane"],
    #     height=480,
    #     width=640,
    #     spawn=None,
    # )
    camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Left_Robot/ur3_robot/camera_link/Camera",
            update_period=0.01,
            height=128,
            width=128,
            data_types=["rgb"],
            # spawn=PinholeCameraCfg(
            #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            # ),
            spawn=None,
            offset=TiledCameraCfg.OffsetCfg(convention="opengl"),
        )

    # -------------------------
    # 计算 pipe 的姿态（只在这里用 tensor，结果转成 tuple[float]）
    # -------------------------
    # 角度用 Python float，便于 Hydra 存储
    roll = 0.0           # Roll in radians
    pitch = 0.0          # Pitch in radians
    yaw = 1.57           # Yaw in radians（或 math.pi / 2）

    # 只在这里临时用 tensor 做计算，返回后立刻转成 tuple of float
    pipe_quat = tuple(
        quat_from_euler_xyz(
            torch.tensor(roll),
            torch.tensor(pitch),
            torch.tensor(yaw),
        ).tolist()
    )

    pipe_pos = (0.0, -0.29, -0.2355)
    pipe = AssetBaseCfg(
        prim_path="/World/envs/env_.*/pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/pipe_stl.usd",
            scale=(0.5, 0.5, 0.05),

            # 关键三项
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=pipe_pos,
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
    )
    # -------------------------
    # Suture Needle object
    # -------------------------
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.29, -0.2351),
            rot=pipe_quat,   # tuple[float]，不会再是 tensor
        ),
        spawn=UsdFileCfg(
            usd_path="exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/object1.usd",
            # scale=(0.012, 0.02, 0.012),
            scale=(0.01, 0.018, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=32,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    table_robot = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table_R",
        spawn=sim_utils.UsdFileCfg(
            usd_path="exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/Props/Table/table.usd",
            scale=(1, 0.5, 0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.7, -0.53),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    table_operate = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table_O",
        spawn=sim_utils.UsdFileCfg(
            usd_path="exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/Props/Table/table.usd",
            scale=(1, 0.5, 0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.15, -0.603),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(usd_path=f"exts/ur3_lite/ur3_lite/tasks/manipulator/ur3_surgical/assets/default_environment.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.9),
        ),
    )

    # -------------------------
    # 可视化配置：都还是普通 Python 类型
    # -------------------------
    # goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
    #     prim_path="/World/Visuals/Command/goal_pose"
    # )
    # current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
    #     prim_path="/World/Visuals/Command/body_pose"
    # )

    # goal_pose_visualizer_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
    # current_pose_visualizer_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)

    # -------------------------
    # 指令输出范围：用 math.pi，避免 torch.pi 生成 tensor
    # -------------------------
    ranges = {
        "object_r": (0.002, 0.002),
        "object_theta": (1.73, 1.73),
        "z_height": (0.0, -0.2),
        "roll": (-0.75 * math.pi, -0.75 * math.pi),
        "pitch": (0.0, 0.0),
        "yaw": (-0.5 * math.pi, -0.5 * math.pi),
    }

    make_quat_unique = True
    debug_vis = False

        # 5mm 视觉球
    # range_vis = VisualizationMarkersCfg(
    #     prim_path="/Visuals/target_range",
    #     markers={
    #         "sphere": sim_utils.SphereCfg(
    #             radius=0.003,  # 0.005m = 5mm
    #             visual_material=sim_utils.GlassMdlCfg(
    #                 glass_color=(1.0, 0.2, 0.2),
    #                 frosting_roughness=0.1,
    #                 thin_walled=True,
    #             ),
    #         ),
    #     },
    # )
