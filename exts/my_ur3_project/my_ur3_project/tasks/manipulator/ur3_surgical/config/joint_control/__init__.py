# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="My-Isaac-Ur3-ACT-Joint-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_needle_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_env_cfg:Ur3LiftNeedleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="My-Isaac-Ur3-ACT-Joint-WithTip-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_needle_env_withtip:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_env_withtip_cfg:Ur3LiftNeedleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="My-Isaac-Ur3-Ik-Rel-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_needle_ik_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_ik_env_cfg:Ur3LiftNeedleEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="My-Isaac-Ur3-Ik-Abs-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_needle_ik_Abs_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_ik_Abs_env_cfg:Ur3LiftNeedleEnvCfg",
        # "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_ik_Abs_env_cfg_backup:Ur3LiftNeedleEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

gym.register(
    id="My-Isaac-Ur3-Ik-Abs-Position-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_needle_ik_Abs_position_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_ik_Abs_position_env_cfg:Ur3LiftNeedleEnvCfg",
        # "env_cfg_entry_point": f"{__name__}.ur3_lift_needle_ik_Abs_env_cfg_backup:Ur3LiftNeedleEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_pipe_ik_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_ik_env_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-Pipe-Ik-Act-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_pipe_Act_env:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_ik_env_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-Pipe-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_pipe_rl:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-PipeRel-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl:Ur3LiftNeedleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-Pick-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_lift_object_env:Ur3LiftObjectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3_lift_pipe_rl_cfg:Ur3LiftPipeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Ur3ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
gym.register(
    id="My-Isaac-Ur3-PipeRelCam-Ik-RL-Direct-v0",
    entry_point=f"{__name__}.ur3_Lift_pipe_rel_rl_cam:Ur3LiftNeedleEnv",
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