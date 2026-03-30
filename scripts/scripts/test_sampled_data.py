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
import numpy as np
import torch

from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg


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
    env_cfg = parse_env_cfg(
        "My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )

    env = gym.make("My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0", cfg=env_cfg)
    data = load_and_inspect_traj('/home/yhy/IsaacLabExtensionTemplate_lite/data/random_trajectories_with_pose.npz')
    keys_list = list(data.keys())
    
    if not keys_list:
        print("未找到任何轨迹数据！")
        return

    # ---------------------------------------------------------
    # 辅助函数：加载指定索引的轨迹并重置环境
    # ---------------------------------------------------------
    def load_trajectory(idx):
        target_key = keys_list[idx]
        target_value = data[target_key]['pose_command'].copy()
        
        # 1. 设置命令并重置环境
        pose_command = torch.tensor(target_value, device=env.device).unsqueeze(0).repeat(num_envs, 1)
        env.env.set_external_command(pose_command)
        env.reset()
        
        # 2. 准备该轨迹的动作张量
        raw_actions = data[target_key]['action'] 
        action_sequence = torch.tensor(raw_actions, dtype=torch.float32, device=env.device)
        total_steps = action_sequence.shape[0]
        
        print(f"\n[{idx+1}/{len(keys_list)}] 开始播放轨迹: {target_key} | 总步数: {total_steps}")
        print(f"pose_command: {pose_command}")
        
        return action_sequence, total_steps

    # 初始化状态
    traj_idx = 0  # 当前播放的轨迹索引
    step_idx = 0  # 当前轨迹播放到的步数
    
    # 加载第一条轨迹
    action_sequence, total_steps = load_trajectory(traj_idx)

    # ---------------------------------------------------------
    # 仿真主循环
    # ---------------------------------------------------------
    while simulation_app.is_running():
        
        # --- 边界检查与循环切换逻辑 ---
        if step_idx >= total_steps:
            print(f"--- 轨迹 {keys_list[traj_idx]} 回放结束 ---")
            
            # 索引 +1，如果超出了 keys_list 长度，则取余回到 0 (实现无限循环)
            traj_idx = (traj_idx + 1) % len(keys_list)
            
            if traj_idx == 0:
                print(">>> 所有轨迹已播放完毕，开启新一轮循环！ <<<")
            
            # 加载新一条轨迹，重置步数计数器
            action_sequence, total_steps = load_trajectory(traj_idx)
            step_idx = 0
            
            # continue 跳过这一帧的其余逻辑，在下一帧直接应用新环境的第一步动作
            continue 

        # --- 核心动作提取与执行 ---
        # 1. 取出【当前这一步】的动作: (5,)
        current_action = action_sequence[step_idx]

        # 2. 扩展维度并广播给所有环境: (5,) -> (1, 5) -> (num_envs, 5)
        actions = current_action.unsqueeze(0).repeat(num_envs, 1)
        
        # 3. 执行环境步
        ret = env.step(actions)
        
        # 4. 步数 +1
        step_idx += 1
        
        # 5. 可选：打印信息
        # 避免输出刷屏太快，可以降低打印频率，例如：
        # if step_idx % 10 == 0:
        #     print(f"Step {step_idx}: object_pos {object_pos}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()