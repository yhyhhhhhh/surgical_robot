import matplotlib
matplotlib.use("TkAgg")  # 或 "Qt5Agg" 等，需在导入 pyplot 之前
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
current_quat = [0, 0, 0, 1]   # /phantom/phantom/pose 获得的四元数 (x,y,z,w)
second_last_joint_angle = 0.0 # 倒数第二个关节角 (用于 Y 轴旋转)
last_joint_angle = 0.0        # 最后一个关节角 (用于 Z 轴旋转)

def quaternion_to_angle_axis(quat):
    """
    将四元数转换为 (旋转角, 旋转轴) 用于调试输出
    quat: [x, y, z, w]
    """
    quat = np.array(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
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
    ROS 四元数顺序: (x, y, z, w)
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
    - 倒数第二个关节角：绕 Y 轴
    - 最后一个关节角：绕 Z 轴
    """
    global second_last_joint_angle, last_joint_angle
    if len(data.position) >= 2:
        second_last_joint_angle = data.position[-2]
        last_joint_angle = data.position[-1]
    elif len(data.position) == 1:
        second_last_joint_angle = 0.0
        last_joint_angle = data.position[-1]
    else:
        second_last_joint_angle = 0.0
        last_joint_angle = 0.0

def ros_listener():
    """
    在子线程中初始化 ROS 节点并订阅话题
    disable_signals=True 避免子线程 signal 注册冲突
    """
    rospy.init_node('pose_listener', anonymous=True, disable_signals=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, pose_callback)
    rospy.Subscriber("/phantom/phantom/joint_states", JointState, joint_callback)
    rospy.spin()

# 启动 ROS 监听线程
ros_thread = threading.Thread(target=ros_listener)
ros_thread.daemon = True
ros_thread.start()

# 创建 Matplotlib 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

def update(frame):
    """
    Matplotlib 动画更新函数，每帧都根据最新数据绘制姿态：
      1) 原四元数 R_pose
      2) 倒数第二关节 -> R_y
      3) 最后一个关节   -> R_z
    最终合成: R_final = R_pose * R_y * R_z
    """
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)
    
    global current_quat, second_last_joint_angle, last_joint_angle

    # 1) 将 /phantom/phantom/pose 的四元数转换为 Rotation
    R_pose = R.from_quat(current_quat)  # (x, y, z, w)

    # 2) 构造绕 Y 轴旋转 (倒数第二个关节角)
    R_y = R.from_euler('y', second_last_joint_angle, degrees=False)

    # 3) 构造绕 Z 轴旋转 (最后一个关节角)
    R_z = R.from_euler('z', -last_joint_angle, degrees=False)

    # 4) 合成旋转: 先做 R_pose，再做 R_y，然后 R_z
    #    也可根据需求更改乘法顺序
    R_final = R_pose * R_z * R_x

    # 获取最终旋转矩阵
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
    
    # 文字标注
    ax.text(x_global[0], x_global[1], x_global[2], 'X', color='r', fontsize=12)
    ax.text(y_global[0], y_global[1], y_global[2], 'Y', color='g', fontsize=12)
    ax.text(z_global[0], z_global[1], z_global[2], 'Z', color='b', fontsize=12)

    # 在标题中显示调试信息
    angle_pose, axis_pose = quaternion_to_angle_axis(current_quat)
    title_str = (
        f"Pose quat: {np.round(current_quat,3)}\n"
        f"Joint[-2] (Y-axis) = {second_last_joint_angle:.3f} rad\n"
        f"Joint[-1] (Z-axis) = {last_joint_angle:.3f} rad\n"
        f"Pose angle: {np.degrees(angle_pose):.1f}° about {np.round(axis_pose,2)}"
    )
    ax.set_title(title_str)

ani = FuncAnimation(fig, update, interval=50)
plt.show()
