import numpy as np
import rospy
### ========== 导入torch msg ===========
import sys
import os

os.environ['ROS_PACKAGE_PATH'] = '/home/yhy/code/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:' + os.environ.get('ROS_PACKAGE_PATH', '')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/yhy/code/touch_ws/devel/lib/python3/dist-packages')
# ======================================
from omni_msgs.msg import OmniState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
def quaternion_to_angle_axis(quat):
    """
    将四元数转换为旋转角和旋转轴
    参数:
        quat: 四元数 [x, y, z, w]，假设已归一化
    返回:
        angle: 旋转角（弧度）
        axis: 旋转轴（单位向量）
    """
    # 确保四元数归一化
    quat = np.array(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    
    x, y, z, w = quat
    # 计算旋转角（注意数值误差可能导致 w 超出[-1, 1]范围）
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # 计算旋转轴
    s = np.sqrt(1 - w*w)
    if s < 1e-8:
        # 如果 s 非常小，则轴不可确定，此时返回默认轴（例如 [1, 0, 0]）
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([x, y, z]) / s
    
    return angle, axis

def callback(data):
    
    quat = [
        data.pose.orientation.w,
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
    ]
    angle, axis = quaternion_to_angle_axis(quat)
    
    print("旋转角（弧度）：", angle)
    print("旋转角（度）：", np.degrees(angle))
    print("旋转轴：", axis)
    
if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, callback)
    rate = rospy.Rate(100)  # 10Hz
    while not rospy.is_shutdown():


        rate.sleep()
        