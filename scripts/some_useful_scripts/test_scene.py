# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""



import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv

def main():
    """Main function."""
    # parse configuration
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )
    # create environment
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    env.sim.step()


    # Simulate physics
    while simulation_app.is_running():
        # perform step
        env.sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()