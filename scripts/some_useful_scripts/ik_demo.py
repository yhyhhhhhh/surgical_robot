# -*- coding: utf-8 -*-
"""
验证 DifferentialIK 在“目标末端位姿随时间变化”时的跟踪效果。
- 目标在机器人 base 坐标系下沿 y 方向做正弦来回摆动
- 每一步更新 IK 目标，并计算误差
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# -------------------------------------------------------------
# 启动 Isaac Sim
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Check Differential IK trajectory tracking.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Robot name: franka_panda or ur10")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------
# 仿真 / 控制逻辑
# -------------------------------------------------------------
import math
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

from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # 预定义机器人配置


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """桌面场景配置"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """主仿真循环：目标位置随时间变化"""

    robot = scene["robot"]

    # ---------- 1. 创建 IK 控制器（绝对位姿） ----------
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",      # 绝对末端 pose
        use_relative_mode=False,  # 不用相对模式
        ik_method="dls",
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # ---------- 2. 可视化 ----------
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # ---------- 3. 解析末端、关节索引 ----------
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # ---------- 4. 机器人重置 ----------
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    diff_ik.reset()

    # ---------- 5. 目标基准姿态（base 坐标系） ----------
    # 基本位置
    base_pos = torch.tensor([[0.4, 0.0, 0.4]], device=robot.device)  # (1,3)
    # 基本姿态：单位四元数 [w, x, y, z]
    base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device)  # (1,4)

    base_pos = base_pos.repeat(scene.num_envs, 1)
    base_quat = base_quat.repeat(scene.num_envs, 1)

    sim_dt = sim.get_physics_dt()
    step_count = 0

    print("[INFO] IK trajectory check started. Target moves in a sine along +Y/-Y.")

    # ---------- 6. 仿真循环 ----------
    while simulation_app.is_running():
        t = step_count * sim_dt  # 当前仿真时间

        # 6.1 根据时间生成目标位姿（在 base 坐标系）
        # y 方向正弦运动，振幅 0.15 m，角频率 0.5 rad/s
        y_offset = 0.15 * math.sin(0.5 * t)
        target_pos_b = base_pos.clone()
        target_pos_b[:, 1] += y_offset  # 修改 y 分量

        # 姿态先保持不变，你也可以改成随时间旋转
        target_quat_b = base_quat.clone()

        target_pose_b = torch.cat([target_pos_b, target_quat_b], dim=-1)  # (num_envs, 7)

        # 6.2 把目标传给 IK 控制器（这一行在 RL 环境里每 step 也要做）
        diff_ik.set_command(target_pose_b)

        # 6.3 从仿真读当前状态 & 计算关节目标
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        joint_pos_des = diff_ik.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )

        # 6.4 设置到底层 PD
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # 6.5 仿真前进一步
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # 6.6 可视化当前末端 & 目标末端
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(target_pos_b + scene.env_origins, target_quat_b)

        # 6.7 每隔一段打印误差
        if step_count % 50 == 0:
            pos_err = torch.norm(ee_pos_b - target_pos_b, dim=-1)  # (num_envs,)

            # 四元数误差转角度
            _, quat_err = subtract_frame_transforms(
                target_pos_b, target_quat_b,
                ee_pos_b, ee_quat_b,
            )
            w = torch.clamp(quat_err[:, 0].abs(), max=1.0)
            ang_err = 2.0 * torch.arccos(w) * 180.0 / math.pi

            print(
                f"[step {step_count:5d} | t={t:5.2f}s] "
                f"pos_err = {pos_err[0].item():.4f} m, "
                f"ori_err = {ang_err[0].item():.2f} deg"
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
