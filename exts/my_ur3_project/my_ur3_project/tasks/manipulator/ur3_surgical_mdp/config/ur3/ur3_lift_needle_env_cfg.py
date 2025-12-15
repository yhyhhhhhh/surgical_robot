from __future__ import annotations

import math
from typing import Sequence

import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.controllers import DifferentialIKControllerCfg

# manager-based env cfg + managers
try:
    from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    import omni.isaac.lab.envs.mdp as mdp
    from omni.isaac.lab.managers import (
        ActionTerm,
        ActionTermCfg,
        ObservationGroupCfg as ObsGroupCfg,
        ObservationTermCfg as ObsTermCfg,
        RewardTermCfg as RewTermCfg,
        TerminationTermCfg as DoneTermCfg,
        EventTermCfg as EventTermCfg,
        SceneEntityCfg,
    )
    from omni.isaac.lab.managers.manager_base import ManagerTermBase
except Exception as e:
    raise ImportError(
        "没有找到 Manager-based 的 IsaacLab API（ManagerBasedRLEnvCfg / managers）。"
        "请确认你安装的 IsaacLab 版本支持 manager-based workflow。原始报错：\n"
        f"{e}"
    )

# 你自己的模块
from .utils.myfunc import quat_from_euler_xyz
from .utils.robot_ik_fun import DifferentialInverseKinematicsActionCfg, DifferentialInverseKinematicsAction


# ======================================================================================
# MDP TERMS（为了让 cfg 文件“可直接用”，我把 terms 放在 cfg 文件里）
# ======================================================================================

@configclass
class PipeIkActionCfg(ActionTermCfg):
    """5维动作：pipe 坐标增量 + yaw + grip -> IK -> 关节 target"""

    class_type = None  # 下面赋值
    asset_name: str = "left_robot"
    ee_body_name: str = "scissors_tip"

    # 从 env.cfg 上取 IK cfg（避免 dataclass 互相引用导致的坑）
    ik_cfg_name: str = "left_robot_ik"

    # pipe 参数（你原 env 里写死的）
    pipe_radius: float = 0.0075
    pipe_length: float = 0.032
    pipe_safety_margin: float = 0.0
    s_min: float = -0.01
    s_max: float = 0.032

    # 动作步长（管外粗、管内细）: (ds, dr, dtheta, dyaw)
    step_outside: tuple[float, float, float, float] = (0.01, 0.003, 0.30, 0.6)
    step_inside: tuple[float, float, float, float] = (0.002, 0.0005, 0.04, 0.04)

    # gripper
    gripper_min: float = -0.28
    gripper_max: float = -0.10


class PipeIkAction(ActionTerm):
    cfg: PipeIkActionCfg

    def __init__(self, cfg: PipeIkActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # robot + ee id
        self.robot = env.scene.articulations[cfg.asset_name]
        self.ee_id = self.robot.data.body_names.index(cfg.ee_body_name)

        # 固定对齐：把末端局部 y 轴旋到世界 z 轴（复用你原逻辑）
        yaw0 = torch.tensor(-math.pi / 2.0, device=self.device)
        roll_align = torch.tensor(math.pi / 2.0, device=self.device)
        self.q_align_y_to_z = quat_from_euler_xyz(roll_align, 0.0 * yaw0, yaw0)  # (4,)

        # yaw buffer + gripper buffer
        self.ee_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.gripper_cmd = torch.full((self.num_envs, 1), cfg.gripper_max, device=self.device)

        # step scale buffers
        self.step_outside = torch.tensor(cfg.step_outside, device=self.device).unsqueeze(0)  # (1,4)
        self.step_inside = torch.tensor(cfg.step_inside, device=self.device).unsqueeze(0)    # (1,4)

        # IK helper（用你已有的类）
        ik_cfg = getattr(env.cfg, cfg.ik_cfg_name)
        self.ik = DifferentialInverseKinematicsAction(ik_cfg, env.scene)

        # 给 reward/obs/debu
