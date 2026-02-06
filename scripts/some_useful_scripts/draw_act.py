import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from matplotlib import cm  # colormap

# 读取 CSV 数据
csv_path = '/home/yhy/DVRK/IsaacLabExtensionTemplate/trajectory_logs/trajectory_18.csv'
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

x, y, z = data[:, 0], data[:, 1], data[:, 2]

# 构造线段（N-1段，每段由两点构成）
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 使用颜色渐变：从蓝（早）到红（晚）
norm = plt.Normalize(0, len(segments))
colors = cm.jet(norm(np.arange(len(segments))))

# 创建 Line3DCollection（可着色的线段）
lc = Line3DCollection(segments, colors=colors, linewidths=2)

# 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(lc)

# 设置坐标轴范围
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

# 坐标轴与标题
ax.set_title('3D Trajectory with Color Progression')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示颜色条（可选）
sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Step Index')

plt.tight_layout()
plt.show()
