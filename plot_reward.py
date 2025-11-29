import re
import matplotlib.pyplot as plt
import numpy as np

def parse_reward_file(file_path):
    """解析奖励文件，返回奖励值列表"""
    rewards = []
    pattern = re.compile(r"tensor\(\[([-+]?\d+\.\d+)\],.*\)")
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                reward = float(match.group(1))
                rewards.append(reward)
            else:
                print(f"警告：无法解析行：{line.strip()}")
    return rewards

def plot_rewards(rewards, save_path=None):
    """绘制奖励曲线"""
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(rewards))
    
    plt.plot(steps, rewards, label='Reward per Step')
    plt.title('Reward Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    file_path = "all_rewards.txt"  # 修改为你的文件路径
    rewards = parse_reward_file(file_path)
    
    # 可选：计算移动平均（窗口大小=100）
    window = 100
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # 绘制原始数据和移动平均曲线
    plt.figure(figsize=(14, 7))
    plt.plot(rewards, alpha=0.5, label='Raw Reward')
    plt.plot(np.arange(window-1, len(rewards)), rewards_smooth, 'r', label=f'Smoothed (Window={window})')
    plt.title('Reward Progression with Smoothing')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_plot.png", dpi=300)
    plt.show()