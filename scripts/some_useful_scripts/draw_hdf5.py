import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 文件路径（你可以修改为其他 episode 文件）
file_path = "/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth_view/dataset/episode_1.hdf5"

# 打开 HDF5 文件并读取 touch_raw
with h5py.File(file_path, 'r') as f:
    touch_data = f['action/touch_raw'][:]  # shape: (T, 8)

# 提取前三个分量作为位置
pos = touch_data[:, 0:3]  # shape: (T, 3)
x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

# 创建 3D 轨迹图
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Touch Trajectory', color='blue')

# 起点终点标记
ax.scatter(x[0], y[0], z[0], c='green', label='Start')
ax.scatter(x[-1], y[-1], z[-1], c='red', label='End')

# 标签设置
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Touch Device Trajectory (Position)')
ax.legend()
plt.tight_layout()
plt.show()
