# import numpy as np
# import matplotlib.pyplot as plt

# #力反馈
# #haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250506_235131.npz

# #精细操作
# #haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250507_004332.npz


# # 加载数据
# data = np.load("haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250506_235131.npz")

# pos_m = np.array(data["pos_m"])  # shape: (N, 3)
# pos_p = np.array(data["pos_p"])
# vel_m = np.array(data["vel_m"])
# vel_p = np.array(data["vel_p"])
# force = np.array(data["force"])
# time = np.array(data["time"])

# # 计算位置差
# dx = pos_m - pos_p[:,0,:]  # 相对位移
# # dx = pos_m - pos_p[:,:]  # 相对位移
# # ========== 1. 主/从端位置比较 ==========
# plt.figure()
# plt.title("Position Comparison: Master vs Slave")
# labels = ['X', 'Y', 'Z']
# for i in range(3):
#     plt.subplot(3, 1, i+1)
#     plt.plot(time, pos_m[:,i], label='Master')
#     plt.plot(time, pos_p[:,:, i], label='Slave')
#     # plt.plot(time, pos_p[:, i], label='Slave')
#     plt.ylabel(f'Pos {labels[i]} (m)')
#     plt.legend()
# plt.xlabel("Time (s)")
# plt.tight_layout()

# plt.figure()
# plt.title("Haptic Force Over Time")
# for i in range(3):
#     plt.plot(time, force[:, i], label=f'F_{labels[i]}')
# plt.xlabel("Time (s)")
# plt.ylabel("Force (N)")
# plt.legend()
# plt.grid(True)

# # ========== 3. 力 vs 相对位移（弹簧特性验证） ==========
# plt.figure()
# plt.title("Force vs Displacement (Virtual Spring Model)")
# for i in range(3):
#     plt.scatter(dx[:, i], force[:, i], label=f'Axis {labels[i]}', s=5)
#     plt.xlabel("Displacement (m)")
#     plt.ylabel("Force (N)")
# plt.legend()
# plt.grid(True)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# data = np.load("haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250507_004332.npz")

# pos_m = np.array(data["pos_m"])
# pos_p = np.array(data["pos_p"])
# vel_m = np.array(data["vel_m"])
# vel_p = np.array(data["vel_p"])
# force = np.array(data["force"])
# time = np.array(data["time"])

# # Compute relative displacement
# dx = pos_m - pos_p
# labels = ['X', 'Y', 'Z']
# time = np.array(data["time"])
# time = time - time[0]  # 设置横坐标从 0 开始

# fine_start = time[420]
# fine_end   = time[680]

# # === 1. Position Comparison ===
# plt.figure()
# plt.suptitle("Position Comparison: Master vs Slave")
# for i in range(3):
#     plt.subplot(3, 1, i+1)
#     plt.plot(time, pos_m[:, i], label='Master')
#     plt.plot(time, pos_p[:, i], label='Slave')
#     plt.axvspan(fine_start, fine_end, color='red', alpha=0.2, label='Fine Control Period')
#     plt.ylabel(f'Pos {labels[i]} (m)')
#     plt.legend()
# plt.xlabel("Time (s)")
# plt.tight_layout()

# # === 2. Haptic Force Over Time ===
# plt.figure()
# plt.title("Haptic Force Over Time")
# for i in range(3):
#     plt.plot(time, force[:, i], label=f'F_{labels[i]}')
# plt.axvspan(fine_start, fine_end, color='red', alpha=0.2, label='Fine Control Period')
# plt.xlabel("Time (s)")
# plt.ylabel("Force (N)")
# plt.legend()
# plt.grid(True)

# # === 3. Force vs Displacement ===
# plt.figure()
# plt.title("Force vs Displacement (Virtual Spring Model)")
# for i in range(3):
#     plt.scatter(dx[:, i], force[:, i], label=f'Axis {labels[i]}', s=5)
# plt.xlabel("Displacement (m)")
# plt.ylabel("Force (N)")
# plt.legend()
# plt.grid(True)

# import numpy as np

# # 假设你已加载以下变量：
# # pos_m, pos_p, time, fine_start_index, fine_end_index

# # 1. 计算误差向量（单位：毫米）
# error_all = (pos_m - pos_p)  # shape (N,3), 单位：mm

# # 2. 各阶段索引
# coarse_mask = np.ones(len(time), dtype=bool)
# coarse_mask[420:680] = False  # 非精细段 = 粗控制段

# error_fine = error_all[420:680]
# error_coarse = error_all[coarse_mask]

# # 3. 统计指标
# rmse_all    = np.sqrt(np.mean(error_all**2, axis=0))
# max_all     = np.max(np.abs(error_all), axis=0)
# rmse_fine   = np.sqrt(np.mean(error_fine**2, axis=0))
# rmse_coarse = np.sqrt(np.mean(error_coarse**2, axis=0))

# # 4. 打印表格
# labels = ['X轴', 'Y轴', 'Z轴']
# print(f"{'方向':<6} | {'RMSE (全)':>9} | {'最大误差':>9} | {'精细RMSE':>10} | {'粗控制RMSE':>12}")
# print("-" * 60)
# for i in range(3):
#     print(f"{labels[i]:<6} | {rmse_all[i]:9.2f} | {max_all[i]:9.2f} | {rmse_fine[i]:10.2f} | {rmse_coarse[i]:12.2f}")

# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt

# # 加载数据
# data = np.load("haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250506_235131.npz")

# pos_m = np.array(data["pos_m"])
# pos_p = np.array(data["pos_p"])
# force = np.array(data["force"])
# time = np.array(data["time"])

# labels = ['X', 'Y', 'Z']

# # 创建横向排列的 3 个子图
# fig, axes = plt.subplots(1, 3, figsize=(18, 4))
# fig.suptitle("Position and Force Over Time (Master vs Slave)")

# for i in range(3):
#     ax = axes[i]
#     ax.plot(time, pos_m[:, i], label='Master Pos', color='tab:blue')
#     ax.plot(time, pos_p[:,0, i], label='Slave Pos', color='tab:orange')
#     ax.set_ylabel("Position (m)")
#     ax.set_xlabel("Time (s)")
#     ax.set_title(f"{labels[i]} Axis")
#     ax.grid(True)

#     # 添加右轴显示力反馈
#     ax2 = ax.twinx()
#     ax2.plot(time, force[:, i], label='Force', color='tab:red', alpha=0.6)
#     ax2.set_ylabel("Force (N)", color='tab:red')

#     # 合并图例（主轴 + 副轴）
#     lines, labels_ = ax.get_legend_handles_labels()
#     lines2, labels2_ = ax2.get_legend_handles_labels()
#     ax.legend(lines + lines2, labels_ + labels2_, loc='upper right')

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = np.load("haptic_My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0_20250507_004332.npz")

pos_m = np.array(data["pos_m"])      # shape: (N, 3)
pos_p = np.array(data["pos_p"])      # shape: (N, 3) or (N, 1, 3)
if pos_p.ndim == 3:
    pos_p = pos_p[:, 0, :]           # squeeze dimension if needed

# 绘制三维轨迹
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_m[:, 0], pos_m[:, 1], pos_m[:, 2], label='Master Trajectory', color='tab:blue')
ax.plot(pos_p[:, 0], pos_p[:, 1], pos_p[:, 2], label='Slave Trajectory', color='tab:orange')

# 添加标签和图例
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory: Master vs Slave')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
