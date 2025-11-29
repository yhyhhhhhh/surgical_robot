from __future__ import annotations
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg,RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.markers import  VisualizationMarkersCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# 自己编写的模块
from .utils.myfunc import *
from .utils.robot_ik_fun import DifferentialInverseKinematicsActionCfg, DifferentialInverseKinematicsAction
'''
修改了ecm的姿态和phantom的位置
'''
@configclass
class Ur3LiftNeedleEnvCfg(DirectRLEnvCfg):
    
    # env
    episode_length_s = 6
    decimation = 5
    action_space = 6

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # robot
    left_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Left_Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/ur3_scissor/ur3_withTip_edit.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -1.9898,
                "shoulder_lift_joint": -0.9985,
                "elbow_joint": 0.0242,
                "wrist_1_joint": -0.7092,
                "wrist_2_joint": -0.5679,
                "wrist_3_joint": -0.5806,
                "tip_joint":0,
            },
            pos=(-0.2, -0.6, -0.4), rot=(1, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "tip": ImplicitActuatorCfg(
                joint_names_expr=["tip_joint"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=2000.0,
                damping=40.0,
            ),
        },
    )
    
    right_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Right_Robot",
        spawn=sim_utils.UsdFileCfg(
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
                "shoulder_pan_joint": -2.6464,
                "shoulder_lift_joint": 0.1027,
                "elbow_joint": 1.4852,
                "wrist_1_joint": 0.9191,
                "wrist_2_joint": 1.1986,
                "wrist_3_joint": -1.0374,
            },
            pos=(0.2, -0.6, 1.4), rot=(0.0, 0.0, 0.0, 1.0),
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
    left_robot_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="left_robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        body_name = "scissors_tip",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    )
    # robot
    ecm = ArticulationCfg(
        prim_path="/World/envs/env_.*/Ecm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/ecm.usd",
            scale=(0.8, 0.8, 0.8),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "ecm_yaw_joint": -0.0,
            #     "ecm_pitch_front_joint": -0.49,
            #     "ecm_pitch_bottom_joint": -0.49,
            #     "ecm_pitch_end_joint": 0.15,
            #     "ecm_main_insertion_joint": 0.1,
            #     "ecm_tool_joint": 0.0,
            # },
            joint_pos={
                "ecm_yaw_joint": -0.0,
                "ecm_pitch_front_joint": -0.18,
                "ecm_pitch_bottom_joint": -0.37,
                "ecm_pitch_end_joint": 0.48,
                "ecm_main_insertion_joint": 0.0,
                "ecm_tool_joint": 0.0,
            },
            pos=(0.0, 0.25, -1.04), rot=(0.707, 0.0, 0.0, -0.707),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["ecm_yaw_joint","ecm_pitch_front_joint","ecm_pitch_bottom_joint","ecm_pitch_end_joint","ecm_tool_joint"],
                velocity_limit=100.0,
                effort_limit=80.0,
                stiffness=1e5,
                damping=623,
            ),
            "insertion": ImplicitActuatorCfg(
                joint_names_expr=["ecm_main_insertion_joint"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=1e5,
                damping=632,
            ),
        },
    )
    camera = CameraCfg( 
        prim_path="/World/envs/env_.*/Ecm/ecm_tool_link/cam_link/cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=25, focus_distance=300.0, horizontal_aperture=30, clipping_range=(0.1, 2000.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.707, 0, 0.0 ,0.707 ), convention="ros"),
    )
    table_robot = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table_R",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/Props/Table/table.usd",
            scale=(1, 0.5, 0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.7, -0.7), rot=(1, 0.0, 0.0, 0.0)),
    )
    
    table_operate = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table_O",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/Props/Table/Operating_table.usd",
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0.0, -0.92), rot=(0.35355,0.35355,0.6124,0.6124)),
    )
    
    room = AssetBaseCfg(
        prim_path="/World/envs/env_.*/room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/OR_scene_s0253_ct_relabel_resample1_syn_seed6_postprocess/operating_room.usd",
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.9), rot=(0.707, 0.707, 0.0, 0.0)),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.9)),
    )
    

    # Set Suture Needle as object
    needle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Needle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.2, 0.0), rot=(0.707, 0, 0, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/Props/Surgical_needle/needle.usd",
            scale=(0.4, 0.4, 0.4),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    
    plantom = AssetBaseCfg(
        prim_path="/World/envs/env_.*/plantom",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.2, -0.295)),
        spawn=UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/Props/Table/plantom_edit.usd",
            scale=(1.3, 1.3, 1.3),
        ),
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
        "needle_x": (-0.06, 0.06),
        "needle_y": (-0.25, -0.15),
        "z_height": (0.0, -0.2),
        "roll": (-torch.pi*0.75, -torch.pi*0.75),
        "pitch": (0, 0),
        "yaw": (-torch.pi*0.5, -torch.pi*0.5),
    }
    
    make_quat_unique = True
    debug_vis = True
