# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
UR3 抬管道任务（相对位置+相机）环境
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Ur3Lite-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl_cam_final:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cam_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Ur3Lite-PipeRelCamFinalGoal-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl_cam_final_goal:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cam_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="Ur3Lite-PipeRelCamFinalGoalForce-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl_cam_final_goal_force:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cam_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="Ur3Lite-PipeRelGoalForce-OSC-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl_osc_goal_force:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_osc_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)