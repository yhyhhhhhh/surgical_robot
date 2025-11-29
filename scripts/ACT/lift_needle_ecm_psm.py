import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reach state machine for psm platform environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save=True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from collections.abc import Sequence
import warp as wp
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.reach_dual.reach_env_cfg import ReachEnvCfg
from omni.isaac.lab.sensors.camera.utils import save_images_to_file
from omni.isaac.lab.utils import convert_dict_to_backend
import matplotlib.pyplot as plt
import cv2  # 添加OpenCV库的导入
import omni.replicator.core as rep
import os
# initialize warp
wp.init()

'''
lift needle 任务
'''
class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(0.7)
    GRASP_OBJECT = wp.constant(1.0)
    LIFT_OBJECT = wp.constant(2.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    print(state)
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.05
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    step=0
    # parse configuration
    env_cfg: ReachEnvCfg = parse_env_cfg(
        "Isaac-Needle-Dual-ECM-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Isaac-Needle-Dual-ECM-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    env.sim.step()
    camera = env.scene["camera"]
    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    # create state machine
    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    
    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    camera_index = args_cli.camera_id
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            step = step + 1
            # step environment
            dones = env.step(actions)[-2]
            
            # single_cam_data = convert_dict_to_backend(camera.data.output, backend="numpy")
            # save_images_to_file(camera.data.output, f"/home/yhy/orbit_surgical/source/standalone/environments/state_machine/camera/image_step_{step}.png")
            # Extract camera data
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
                )

                # Extract the other information
                single_cam_info = camera.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output = {"annotators": {}}
                rep_output["annotators"]["rgb"] = {"render_product": {"data": single_cam_data["rgb"]}}
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                rep_writer.write(rep_output)
            # observations
            robot_1: RigidObject = env.scene["robot_1"]
            robot_2: RigidObject = env.scene["robot_2"]
            # -- end-effector frame
            ee_1_frame_sensor = env.unwrapped.scene["ee_1_frame"]
            tcp_1_rest_position = ee_1_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_1_rest_position_b, _ = subtract_frame_transforms(
                robot_1.data.root_state_w[:, :3], robot_1.data.root_state_w[:, 3:7], tcp_1_rest_position
            )
            tcp_1_rest_orientation = ee_1_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # ee_2_frame_sensor = env.unwrapped.scene["ee_2_frame"]
            # tcp_2_rest_position = ee_2_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            # tcp_2_rest_position_b, _ = subtract_frame_transforms(
            #     robot_2.data.root_state_w[:, :3], robot_2.data.root_state_w[:, 3:7], tcp_2_rest_position
            # )
            # tcp_2_rest_orientation = ee_2_frame_sensor.data.target_quat_w[..., 0, :].clone()
            
            # -- target end-effector frame
            # -- object frame
            needle: RigidObject = env.scene["object"]
            object_data = needle.data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            object_position_b, _ = subtract_frame_transforms(
                robot_1.data.root_state_w[:, :3], robot_1.data.root_state_w[:, 3:7], object_position
            )
            print(object_position[0])
            object_orientation = object_data.root_quat_w
            # -- target object frame
            desired_pose = env.unwrapped.command_manager.get_command("object_pose")
            # desired_pose_1 = env.unwrapped.command_manager.get_command("ee_1_pose")
            # desired_pose_2 = env.unwrapped.command_manager.get_command("ee_2_pose")
            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_1_rest_position_b, tcp_1_rest_orientation], dim=-1),
                torch.cat([object_position_b, object_orientation], dim=-1),
                desired_pose,
            )
            combined_actions = torch.cat([
                actions, 
                torch.tensor([[0, 0.0, -0.05]],device=actions.device), 
                torch.tensor([[1.0, 0.0, 0.0, 0.0,0.0]],device=actions.device)
            ], dim=1)        
            # 在actions变量后拼接6个零扩展到1x14维度
            actions = combined_actions
            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                env.reset()

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
