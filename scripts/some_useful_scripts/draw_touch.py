import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载保存的动作
data = np.load("haptic_actions_20250507_160837.npz")  # 修改为你的文件名
actions = data["actions"]  # shape: (T, 8)

# 提取位置分量（前三维）
positions = actions[:, 0:3]

# 绘制三维轨迹
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Action Trajectory', color='blue')
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', label='Start')
ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', label='End')

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Action Position Trajectory in 3D")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
