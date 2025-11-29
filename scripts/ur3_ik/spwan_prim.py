# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.core.prims import XFormPrimView
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.physx.scripts.physicsUtils import PhysicsSchemaTools
from pxr import PhysxSchema
from pxr import UsdPhysics, Sdf, Usd, UsdLux, UsdGeom, Gf, Tf
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
class CartpoleSceneCfg(InteractiveSceneCfg):
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
                "shoulder_pan_joint": 0.4,
                "shoulder_lift_joint": -1.3,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0,
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
            usd_path="/home/yhy/DVRK/ur3_scissor/pipe_stl.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, 0.0)),
    )
    
import numpy as np
from pxr import Gf, UsdGeom

def sample_point_in_pipe(scene, pipe_prim_path, inner_radius=0.0125, height=0.2):
    """
    Sample a random point inside a pipe and transform it to world coordinates.

    Args:
        pipe_prim_path (str): The USD path to the pipe prim.
        inner_radius (float): Inner radius of the pipe in mm.
        outer_radius (float): Outer radius of the pipe in mm.
        height (float): Height of the pipe in mm.

    Returns:
        Gf.Vec3f: The sampled point in world coordinates.
    """

    pipe_prim = scene.stage.GetPrimAtPath(pipe_prim_path)

    if not pipe_prim.IsValid():
        raise ValueError(f"Invalid prim path: {pipe_prim_path}")

    # Sample a random point in cylindrical coordinates
    radius = np.random.uniform(0, inner_radius)  # Random radius within the pipe
    theta = np.random.uniform(0, 2 * np.pi)                # Random angle in radians
    z = np.random.uniform(0, height)         # Random height along the pipe

    # Convert to Cartesian coordinates in the local frame
    local_x = radius * np.cos(theta)
    local_y = radius * np.sin(theta)
    local_z = z
    local_point = Gf.Vec3f(local_x, local_y, local_z)

    # Transform to world coordinates
    xform_cache = UsdGeom.XformCache()
    world_transform = xform_cache.GetLocalToWorldTransform(pipe_prim)
    world_point = world_transform.Transform(local_point)
    
    # Convert world_point to a 1x3 PyTorch tensor
    world_point_tensor = torch.tensor([[world_point[0], world_point[1], world_point[2]]])
    
    return world_point_tensor

def is_point_in_pipe_isaaclab(scene, world_point, pipe_prim_path, pipe_radius_inner, pipe_radius_outer, pipe_height):
    """
    判断一个点是否在管道内，并返回到内壁的距离（IsaacLab实现版本）。
    
    Args:
        world_point (numpy.ndarray): 世界坐标系中的点，形状为 (3,)
        pipe_prim_path (str): 管道的路径，例如 "/World/pipe"
        pipe_radius_inner (float): 管道内半径
        pipe_radius_outer (float): 管道外半径
        pipe_height (float): 管道高度

    Returns:
        bool: 点是否在管道内
        float: 点到内壁的距离（如果不在管道内，返回 -1）
    """
    # 获取管道的世界变换
    pipe_prim = scene.stage.GetPrimAtPath(pipe_prim_path)
    
    # 创建 XformCache 实例
    xform_cache = UsdGeom.XformCache()

    # 获取 LocalToWorld 变换矩阵
    local_to_world = xform_cache.GetLocalToWorldTransform(pipe_prim)

    # 计算 WorldToLocal 变换矩阵
    world_to_local = local_to_world.GetInverse()

    local_point = world_to_local.Transform(world_point)  # 转换为局部坐标点
    
    # 在局部坐标系中计算
    x, y, z = local_point
    radial_distance = np.sqrt(x**2 + y**2)  # 点到管道轴线的径向距离

    # 判断是否在管道内部
    if radial_distance < pipe_radius_inner and 0 <= z <= pipe_height:
        distance_to_inner_wall = pipe_radius_inner - radial_distance
        return True, distance_to_inner_wall
    else:
        return False, -1


def on_physics_step(dt):
    print("this fun has been excuted")
    contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()

    for contact_header in contact_headers:
        print("Got contact header type: " + str(contact_header.type))
        print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))) # type: ignore
        print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))) # type: ignore
        print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0))) # type: ignore
        print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1))) # type: ignore
        print("StageId: " + str(contact_header.stage_id))
        print("Number of contacts: " + str(contact_header.num_contact_data))
        
        contact_data_offset = contact_header.contact_data_offset
        num_contact_data = contact_header.num_contact_data
        
        for index in range(contact_data_offset, contact_data_offset + num_contact_data, 1):
            print("Contact:")
            print("Contact position: " + str(contact_data[index].position))
            print("Contact normal: " + str(contact_data[index].normal))
            print("Contact impulse: " + str(contact_data[index].impulse))
            print("Contact separation: " + str(contact_data[index].separation))
            print("Contact faceIndex0: " + str(contact_data[index].face_index0))
            print("Contact faceIndex1: " + str(contact_data[index].face_index1))
            print("Contact material0: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material0))) # type: ignore
            print("Contact material1: " + str(PhysicsSchemaTools.intToSdfPath(contact_data[index].material1))) # type: ignore

def on_contact_report(contact_headers, contact_data):
    table_path = "/World/envs/env_0/Table"  # 假设table的路径为此路径，根据实际情况修改
    for contact_header in contact_headers:
        # 将 Actor0 和 Actor1 转换为字符串
        actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))  # type: ignore
        actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))  # type: ignore

        # 检查是否涉及 table
        if table_path in actor0 or table_path in actor1:
            continue  # 跳过该事件的处理

        # # 输出未被过滤的接触信息
        # print("Actor0: " + actor0)
        # print("Actor1: " + actor1)
        
        
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # Extract scene entities
    robot = scene["robot"]

    point_marker = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },)
    )
    sim_dt = sim.get_physics_dt()
    count = 0
    efforts = torch.zeros_like(robot.data.joint_pos)
    # Subscribe to collision events
    contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report)
    # 判断点是否在管道内
    # 世界坐标中的点
    world_point = Gf.Vec3d(0.35, 0.01, 0.023)  # 世界坐标点  

    # 管道路径
    pipe_prim_path = "/World/envs/env_0/pipe"

    # 管道参数
    pipe_radius_inner = 0.014  # 14 mm
    pipe_radius_outer = 0.015  # 15 mm
    pipe_height = 0.05         # 50 mm
    
    # stepping_sub = get_physx_interface().subscribe_physics_step_events(on_physics_step)
    while simulation_app.is_running():
        if count % 150 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos = torch.zeros_like(robot.data.joint_pos)
            joint_vel = torch.zeros_like(robot.data.joint_vel)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting robot state...")

        sample_point = sample_point_in_pipe(scene, "/World/envs/env_0/pipe")
        is_inside, distance = is_point_in_pipe_isaaclab(scene, world_point, pipe_prim_path, pipe_radius_inner, pipe_radius_outer, pipe_height)
    
        print(f"Is point inside pipe: {is_inside}")
        if is_inside:
            print(f"Distance to inner wall: {distance:.4f} meters")
            
        
        # Apply random action
        joint_pos = torch.zeros_like(robot.data.joint_pos)
        joint_pos[:,0]=0.3541
        joint_pos[:,1]=-1.7001
        joint_pos[:,2]=-0.8394
        joint_pos[:,3]=-0.7118
        joint_pos[:,4]=0.2397
        joint_pos[:,5]=2.5972
        
        # joint_pos[:, 0] = robot.data.joint_pos[:, 0]
        
        # efforts[:, 0] = torch.where(
        #     robot.data.joint_pos[:, 0] >= 4, torch.tensor(-10.0), 
        #     torch.where(robot.data.joint_pos[:, 0] <= 3, torch.tensor(10.0), 
        #                 torch.where(efforts[:, 0] >= 0, torch.tensor(10.0), torch.tensor(-10.0)))
        # )

        robot.set_joint_position_target(joint_pos)
        # robot.set_joint_velocity_target(efforts)
        
        point_marker.visualize(sample_point, torch.tensor([[1, 0, 0, 0]]))
        # print(robot.data.joint_pos)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
from omni.physx import get_physx_interface
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = CartpoleSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 为 Robot 应用碰撞报告
    robot_prim_path = "/World/envs/env_0/Robot/scissor_link"
    robot_prim = scene.stage.GetPrimAtPath(robot_prim_path)
    contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(robot_prim)
    contact_report_api.CreateThresholdAttr().Set(0.0)  # type: ignore # 设置力阈值为 0，捕获所有碰撞

    # pipe_prim_path = "/World/envs/env_0/pipe"
    # pipe_prim = scene.stage.GetPrimAtPath(pipe_prim_path)
    # contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(pipe_prim)
    # contact_report_api.CreateThresholdAttr().Set(0.0)  # type: ignore # 设置力阈值为 0，捕获所有碰撞
    
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
