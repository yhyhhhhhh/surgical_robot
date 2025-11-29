import numpy as np
import rospy
import threading
import matplotlib
matplotlib.use("TkAgg")  # 或者 "Qt5Agg"，根据你的系统情况选择
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped

# 全局变量，存储最新接收到的四元数（格式： [x, y, z, w]）
current_quat = [0, 0, 0, 1]  # 初始为单位四元数

def quaternion_to_angle_axis(quat):
    """
    将四元数转换为旋转角和旋转轴
    参数:
        quat: 四元数 [x, y, z, w]，假设已归一化
    返回:
        angle: 旋转角（弧度）
        axis: 旋转轴（单位向量）
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

def ros_callback(data):
    """
    ROS 回调函数，从 PoseStamped 消息中提取四元数
    注意：ROS 中四元数格式为 (x, y, z, w)
    """
    global current_quat
    # 更新全局变量
    current_quat = [
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
        data.pose.orientation.w
    ]
    # 可选：打印转换信息（调试用）
    angle, axis = quaternion_to_angle_axis(current_quat)
    rospy.loginfo("旋转角（弧度）：%.3f, （度）：%.1f, 旋转轴：%s", angle, np.degrees(angle), np.round(axis,3))

def ros_listener():
    """
    ROS 订阅函数，运行在单独线程中
    """
    rospy.init_node('pose_listener', anonymous=True, disable_signals=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, ros_callback)
    rospy.spin()  # 保持节点运行

# 开启 ROS 监听线程
ros_thread = threading.Thread(target=ros_listener)
ros_thread.daemon = True
ros_thread.start()

# 设置 Matplotlib 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])  # 保证各轴比例相同

def update(frame):
    """
    Matplotlib 动画更新函数，每次刷新显示最新姿态
    """
    ax.cla()  # 清除当前轴
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X (右)')
    ax.set_ylabel('Y (前)')
    ax.set_zlabel('Z (上)')
    ax.view_init(elev=20, azim=30)
    
    # 使用全局变量 current_quat 获取最新姿态
    global current_quat
    # 利用四元数计算旋转矩阵
    r = R.from_quat(current_quat)
    R_mat = r.as_matrix()
    
    origin = np.array([0, 0, 0])
    scale = 0.8  # 坐标轴长度
    
    # 定义物体局部坐标系的单位轴（这里默认 X、Y、Z 分别为物体局部的右、前、上）
    x_axis_local = np.array([1, 0, 0])
    y_axis_local = np.array([0, 1, 0])
    z_axis_local = np.array([0, 0, 1])
    
    # 转换到全局坐标系
    x_axis_global = origin + scale * R_mat.dot(x_axis_local)
    y_axis_global = origin + scale * R_mat.dot(y_axis_local)
    z_axis_global = origin + scale * R_mat.dot(z_axis_local)
    
    # 绘制坐标轴
    ax.plot([origin[0], x_axis_global[0]], [origin[1], x_axis_global[1]], [origin[2], x_axis_global[2]], 
            color='r', linewidth=3, label='X')
    ax.plot([origin[0], y_axis_global[0]], [origin[1], y_axis_global[1]], [origin[2], y_axis_global[2]], 
            color='g', linewidth=3, label='Y')
    ax.plot([origin[0], z_axis_global[0]], [origin[1], z_axis_global[1]], [origin[2], z_axis_global[2]], 
            color='b', linewidth=3, label='Z')
    
    # 添加文字标注
    ax.text(x_axis_global[0], x_axis_global[1], x_axis_global[2], 'X', color='r', fontsize=12)
    ax.text(y_axis_global[0], y_axis_global[1], y_axis_global[2], 'Y', color='g', fontsize=12)
    ax.text(z_axis_global[0], z_axis_global[1], z_axis_global[2], 'Z', color='b', fontsize=12)
    
    # 在标题中显示当前四元数和旋转角信息
    angle, axis_vec = quaternion_to_angle_axis(current_quat)
    title_str = f"Quaternion: {np.round(current_quat,3)}\nAngle: {np.degrees(angle):.1f}° about {np.round(axis_vec,2)}"
    ax.set_title(title_str)

# 利用 FuncAnimation 实时更新图形（间隔 50 毫秒）
ani = FuncAnimation(fig, update, interval=50)
plt.show()
