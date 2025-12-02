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
@configclass
class Ur3LiftPipeEnvCfg(DirectRLEnvCfg):
    
    # env
    episode_length_s = 5
    decimation = 5  # 5
    action_space = 3

    observation_space = 38
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 500,  # 1/500
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    
    # robot
    left_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Left_Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/yhy/DVRK/ur3_scissor/ur3TipCam_pro1_1.usd",
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/ur3TipCam_pro1_1_v0.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=256, solver_velocity_iteration_count=256
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -1.6665,
                "shoulder_lift_joint": 0.8053,
                "elbow_joint": 0.3794,
                "wrist_1_joint": 1.9702,
                "wrist_2_joint": -0.1016,
                "wrist_3_joint": -1.5677,
                "tip_joint":-0.2500,
            },
            # joint_pos={
            #             "shoulder_pan_joint": -1.5169,
            #             "shoulder_lift_joint": 0.8593,
            #             "elbow_joint": 0.8855,
            #             "wrist_1_joint": 1.3985,
            #             "wrist_2_joint": -0.5773,
            #             "wrist_3_joint": -1.5727,
            #             "tip_joint":-0.2500,
            #         },

            pos=(0.15, -0.55, -0.16), rot=(1, 0.0, 0.0, 0.0),
        ),
        # actuators={
        #     "arm": ImplicitActuatorCfg(
        #         joint_names_expr=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        #         velocity_limit=100.0,
        #         effort_limit=100.0,
        #         stiffness=800.0,
        #         damping=40.0,
        #     ),
        #     "tip": ImplicitActuatorCfg(
        #         joint_names_expr=["tip_joint"],
        #         velocity_limit=100.0,
        #         effort_limit=100.0,
        #         stiffness=800,
        #         damping=40.0,
        #     ),
        # },
        actuators={
            # "arm": ImplicitActuatorCfg(
            #     joint_names_expr=[
            #         "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
            #         "wrist_1_joint","wrist_2_joint","wrist_3_joint"
            #     ],
            #     velocity_limit=100.0,
            #     effort_limit=1000.0,          # ② 扭矩上限调高
            #     stiffness=10000.0,            # ③ KP/KD 更硬
            #     damping=100,
            # ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                    "wrist_1_joint","wrist_2_joint","wrist_3_joint"
                ],
                velocity_limit=3.0,
                effort_limit=100.0,          # ② 扭矩上限调高
                stiffness=800.0,            # ③ KP/KD 更硬
                damping=80,
            ),
            # "arm": ImplicitActuatorCfg(
            #     joint_names_expr=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
            #     velocity_limit=100.0,
            #     effort_limit=5000.0,
            #     stiffness=10000000.0,
            #     damping=100000.0,
            # ),
            "tip": ImplicitActuatorCfg(
                joint_names_expr=["tip_joint"],
                velocity_limit=3.0,
                effort_limit=5.0,
                stiffness=200.0,
                damping=20.0,
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
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    )
    
    camera = CameraCfg( 
        prim_path="/World/envs/env_.*/Left_Robot/ur3_robot/camera_link/Camera",
        data_types=[ "rgb","distance_to_image_plane"],
        height=480,
        width=640,
        spawn=None,
    )
    
        
    # 计算pipe的参数信息
    roll = torch.tensor(0, device=torch.device('cuda'))   # Roll in radians
    pitch = torch.tensor(0, device=torch.device('cuda'))   # Pitch in radians
    yaw = torch.tensor(1.57, device=torch.device('cuda'))     # Yaw in radians

    pipe_quat = quat_from_euler_xyz(roll,pitch,yaw)
    pipe_pos = torch.tensor([0.0, -0.29, -0.25], device=torch.device('cuda'))
    pipe  = AssetBaseCfg(
        prim_path="/World/envs/env_.*/pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/pipe_stl.usd",
            scale=(0.5, 0.5, 0.08),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.29, -0.25),rot =  (0.7071, 0.0000,0.0, 0.7071)),
    )

        # Set Suture Needle as object
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.29, -0.2354), rot=pipe_quat),
        spawn=UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/object1.usd",
            # scale=(0.008, 0.015, 0.01),
            scale=(0.012, 0.02, 0.012),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    
    table_robot = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table_R",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/IsaacLabExtensionTemplate/exts/my_ur3_project/my_ur3_project/tasks/manipulator/ur3_surgical/assets/Props/Table/table.usd",
            scale=(1, 0.5, 0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.7, -0.53), rot=(1, 0.0, 0.0, 0.0)),
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
    
    # 可视化部分
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/goal_pose")

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/World/Visuals/Command/body_pose"
    )

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
    
    # 指令输出范围
    # ranges = {
    #     "object_r": (0.0, 0.004),
    #     "object_theta": (0, 3.14),
    #     "z_height": (0.0, -0.2),
    #     "roll": (-torch.pi*0.75, -torch.pi*0.75),
    #     "pitch": (0, 0),
    #     "yaw": (-torch.pi*0.5, -torch.pi*0.5),
    # }
    ranges = {
        "object_r": (0.002, 0.002),
        "object_theta": (1.73, 1.73),
        "z_height": (0.0, -0.2),
        "roll": (-torch.pi*0.75, -torch.pi*0.75),
        "pitch": (0, 0),
        "yaw": (-torch.pi*0.5, -torch.pi*0.5),
    }
    make_quat_unique = True
    debug_vis = False
