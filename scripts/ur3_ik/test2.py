# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from pxr import PhysxSchema
from pxr import UsdPhysics, Sdf, Usd, UsdLux, UsdGeom, Gf, Tf
from omni.physx.scripts.physicsUtils import PhysicsSchemaTools
from omni.physx import  get_physx_simulation_interface
from scipy.spatial.transform import Rotation as R
import numpy as np

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, and yaw angles to a quaternion.

    Args:
        roll (float): Rotation around X-axis in radians.
        pitch (float): Rotation around Y-axis in radians.
        yaw (float): Rotation around Z-axis in radians.

    Returns:
        tuple: Quaternion (w, x, y, z).
    """
    # Create a Rotation object using RPY angles
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

    # Convert to quaternion
    # scipy returns quaternion in the order (x, y, z, w)
    quaternion = rotation.as_quat()
    return (quaternion[3], quaternion[0], quaternion[1], quaternion[2])  # Reorder to (w, x, y, z)


# Example usage
roll = np.radians(0)   # Roll in radians
pitch = np.radians(45)  # Pitch in radians
yaw = np.radians(0)    # Yaw in radians

quaternion = rpy_to_quaternion(roll, pitch, yaw)

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yhy/DVRK/ur3_scissor/ur3_withTip.usd",
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
                "shoulder_pan_joint": -3.2569,
                "shoulder_lift_joint": -2.1371,
                "elbow_joint": 2.3391,
                "wrist_1_joint": -2.5444,
                "wrist_2_joint": -1.4890,
                "wrist_3_joint": -1.6521,
                "tip_joint":0,
            },
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
                velocity_limit=1.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
        soft_joint_pos_limit_factor = 1.0,
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
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
       
    pipe  = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/pipe",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yhy/DVRK/ur3_scissor/pipe20_stl.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.5),rot = quaternion),
    )


def on_contact_report(contact_headers, contact_data):
    for contact_header in contact_headers:
        # 将 Actor0 和 Actor1 转换为字符串
        actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))  # type: ignore
        actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))  # type: ignore

        # # 输出未被过滤的接触信息
        print("Actor0: " + actor0)
        print("Actor1: " + actor1)
        
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.41, 0, 0.51, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))],
        [0.44, 0, 0.54, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))],
        [0.45, 0, 0.55, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))],        
        [0.5, 0, 0.6, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))], 
        [0.55, 0, 0.65, *rpy_to_quaternion(np.radians(-135), np.radians(0), np.radians(-90))], 
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'tip_joint'], body_names=["scissors_tip"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        # print(joint_pos_des,robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 为 Robot 应用碰撞报告
    robot_prim_path = "/World/envs/env_0/Robot/ur3_robot/scissor_link"
    robot_prim = scene.stage.GetPrimAtPath(robot_prim_path)
    contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(robot_prim)
    contact_report_api.CreateThresholdAttr().Set(0.0)  # type: ignore # 设置力阈值为 0，捕获所有碰撞
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
