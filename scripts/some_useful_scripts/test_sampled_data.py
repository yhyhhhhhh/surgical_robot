import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
from einops import rearrange
import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
def load_and_inspect_traj(filename="trajectories_aligned.npz"):
    """
    加载 npz 文件，打印检查信息，并返回完整的数据字典。
    
    Returns:
        dataset (dict): 键为 UUID，值为包含 'action', 'pose_command', 'reward' 等的字典。
    """
    print(f"🔄 正在加载文件: {filename} ...")
    
    try:
        # 1. 加载文件 (必须开启 allow_pickle=True)
        raw_data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {filename}")
        return None

    # 初始化一个字典来存放提取后的数据
    dataset = {}
    
    all_uuids = raw_data.files
    print(f"✅ 文件加载成功！共包含 {len(all_uuids)} 条轨迹。\n")

    # 2. 遍历提取
    for i, uuid in enumerate(all_uuids):
        # --- [关键] 解包 ---
        # 从 npz 的 object array 中还原回 python 字典
        traj_data = raw_data[uuid].item()
        
        # 存入 dataset
        dataset[uuid] = traj_data

        # --- 打印前3条作为示例 ---
        if i < 3:
            pose = traj_data.get('pose_command', 'N/A')
            # 简单打印一下形状，确保数据对不对
            pose_shape = pose.shape if hasattr(pose, 'shape') else 'Unknown'
            print(f"UUID: {uuid} | Pose Shape: {pose_shape}")

    # 3. 关闭文件句柄
    raw_data.close()

    print(f"\n🎉 数据提取完毕，返回 {len(dataset)} 条数据。")
    return dataset


def main():
    num_envs = 1
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )

    env = gym.make("My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0", cfg=env_cfg)
    data = load_and_inspect_traj('trajectories_with_pose.npz')
    keys_list = list(data.keys())
    target_key = keys_list[3]
    target_value = data[target_key]['pose_command'].copy()
    pose_command = torch.tensor(target_value, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    env.env.set_external_command(pose_command)  # 设置第一条轨迹的 pose_command 作为外部命令输入
    env.reset()
    print(f"pose_command {pose_command} ")
    # ---------------------------------------------------------
    # 关键修改部分：准备动作张量
    # ---------------------------------------------------------
    # 定义单个机器人的动作 (假设维度是 5)
    # 比如: [x, y, z, rot_cmd, gripper]
    raw_actions = data[target_key]['action'] 

    # 2. 转为 GPU Tensor
    # 形状: (Total_Steps, 5)
    action_sequence = torch.tensor(raw_actions, dtype=torch.float32, device=env.device)

    # 获取总步数
    total_steps = action_sequence.shape[0]
    print(f"[Info] Loaded trajectory with {total_steps} steps.")

    step_idx = 0 # 初始化步数计数器

    while simulation_app.is_running():
        # --- 边界检查 ---
        if step_idx >= total_steps:
            print("轨迹回放结束")
            break # 或者在这里重置 step_idx = 0 让他循环播放

        # --- 核心修改开始 ---
        
        # 1. 取出【当前这一步】的动作
        # current_action 形状: (5,)
        current_action = action_sequence[step_idx]

        # 2. 扩展维度: (5,) -> (1, 5)
        current_action = current_action.unsqueeze(0)

        # 3. 广播给所有环境: (1, 5) -> (num_envs, 5)
        # 让所有环境在这一帧都执行相同的动作
        actions = current_action.repeat(num_envs, 1)
        
        # --- 核心修改结束 ---

        # 4. 执行环境步
        ret = env.step(actions)
        
        # 5. 步数 +1，准备读取下一帧动作
        step_idx += 1
        object_pos = env.env._object.data.body_pos_w[:, 0, :3]
        print(f"object_pos {object_pos} ")
    env.close()

if __name__ == "__main__":

    main()
    simulation_app.close()