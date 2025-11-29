import matplotlib
matplotlib.use("TkAgg")  # 或 "Qt5Agg" 等，需在导入 pyplot 前调用

import numpy as np
import rospy
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# 全局变量
current_quat = [0, 0, 0, 1]   # 从 /phantom/phantom/pose 订阅到的四元数 (x,y,z,w)
last_joint_angle = 0.0        # 从 /phantom/phantom/joint_state 订阅到的最后一个关节角

def quaternion_to_angle_axis(quat):
    """
    将四元数转换为旋转角和旋转轴 (仅用于调试显示)
    quat: [x, y, z, w]
    返回: (angle, axis)
    """
    quat = np.array(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1 - w*w)
    if s < 1e-8:
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([x, y, z]) / s
    return angle, axis

def pose_callback(data):
    """
    订阅 /phantom/phantom/pose 回调
    ROS 中四元数顺序: x, y, z, w
    """
    global current_quat
    current_quat = [
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
        data.pose.orientation.w
    ]

def joint_callback(data):
    """
    订阅 /phantom/phantom/joint_state 回调
    取最后一个关节角度 (单位: 弧度)，作为绕 Z 轴的额外旋转
    """
    global last_joint_angle
    if data.position:
        # 取最后一个关节角
        last_joint_angle = data.position[-1]
        # print(last_joint_angle)
    else:
        last_joint_angle = 0.0


def ros_listener():
    """
    ROS 订阅线程函数
    使用 disable_signals=True 避免在子线程中注册信号出错
    """
    rospy.init_node('pose_listener', anonymous=True, disable_signals=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, pose_callback)
    rospy.Subscriber("/phantom/phantom/joint_states", JointState, joint_callback)
    rospy.spin()

# 启动 ROS 监听线程
ros_thread = threading.Thread(target=ros_listener)
ros_thread.daemon = True
ros_thread.start()

# 设置 Matplotlib 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

def update(frame):
    """
    Matplotlib 动画更新函数，每帧刷新姿态
    最终姿态 = 原 /phantom/phantom/pose 四元数 + 绕 Z 轴旋转 (last_joint_angle)
    """
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)
    
    global current_quat, last_joint_angle

    # 1) 将 /phantom/phantom/pose 的四元数转换为 Rotation
    R_pose = R.from_quat(current_quat)  # [x, y, z, w]

    # 2) 构造一个绕 Z 轴旋转的 Rotation
    #    last_joint_angle (单位：弧度)
    R_z = R.from_euler('z', -last_joint_angle, degrees=False)

    # 3) 组合旋转：先做 R_pose 再绕 Z 轴
    #    如果你想先绕 Z，再做 pose，可改为 R_final = R_z * R_pose
    R_final = R_pose * R_z

    # 4) 获取最终旋转矩阵
    R_mat = R_final.as_matrix()

    origin = np.array([0, 0, 0])
    scale = 0.8

    # 局部坐标系 (X=红, Y=绿, Z=蓝)
    x_local = np.array([1, 0, 0])
    y_local = np.array([0, 1, 0])
    z_local = np.array([0, 0, 1])

    x_global = origin + scale * R_mat.dot(x_local)
    y_global = origin + scale * R_mat.dot(y_local)
    z_global = origin + scale * R_mat.dot(z_local)

    # 绘制坐标轴
    ax.plot([origin[0], x_global[0]], [origin[1], x_global[1]], [origin[2], x_global[2]],
            color='r', linewidth=3)
    ax.plot([origin[0], y_global[0]], [origin[1], y_global[1]], [origin[2], y_global[2]],
            color='g', linewidth=3)
    ax.plot([origin[0], z_global[0]], [origin[1], z_global[1]], [origin[2], z_global[2]],
            color='b', linewidth=3)
    
    ax.text(x_global[0], x_global[1], x_global[2], 'X', color='r', fontsize=12)
    ax.text(y_global[0], y_global[1], y_global[2], 'Y', color='g', fontsize=12)
    ax.text(z_global[0], z_global[1], z_global[2], 'Z', color='b', fontsize=12)

    # 在标题中显示调试信息
    angle_pose, axis_pose = quaternion_to_angle_axis(current_quat)
    ax.set_title(
        f"Pose quat: {np.round(current_quat, 3)}\n"
        f"Last joint: {last_joint_angle:.3f} rad\n"
        f"Pose angle: {np.degrees(angle_pose):.1f}° about {np.round(axis_pose,2)}"
    )

ani = FuncAnimation(fig, update, interval=50)
plt.show()
