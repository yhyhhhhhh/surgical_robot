import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
import torch
from omni.isaac.lab.utils import convert_dict_to_backend
import rospy
# from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
actions = torch.zeros(7).unsqueeze(0)  # 添加batch维度 → shape (1,7)
import math
import math
import torch
###################################### 导入torch msg ################################################### 
import sys
import os

os.environ['ROS_PACKAGE_PATH'] = '/home/yhy/code/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:' + os.environ.get('ROS_PACKAGE_PATH', '')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/yhy/code/touch_ws/devel/lib/python3/dist-packages')
########################################################################################################
from omni_msgs.msg import OmniState


# 预计算关节参数（元组存储加速访问）
JOINT_PARAMS = {
    # [Torch范围], [UR范围], 方向系数, [特殊分段]
    'shoulder_pan_joint': (
        (-1.02, 1.03), (-1.02, 1.03), 1.0
    ),
    'shoulder_lift_joint': (
        (0.0, 1.86), (1.8, 0.0), 1.0
    ),
    'elbow_joint': (
        (-0.85, 1.34), (-math.pi, math.pi), 1.0,
        # [  # 分段映射参数
        #     (-0.85, 0.32, math.pi, 0.0),    # 第一段
        #     (0.32, 1.34, 0.0, -math.pi)     # 第二段
        # ]
    ),
    'wrist_1_joint': (
        (0.59, 5.69), (-math.pi, math.pi), 1.0
    ),
    'wrist_2_joint': (
        (-3.46, -0.92), (-math.pi,0), 1.0
    ),
    'wrist_3_joint': (
        (-5.77, -0.46), (math.pi/2, -math.pi/2), 1.0
    )
}

# 重新定义关节顺序映射（目标索引，数据源索引，参数）
JOINT_ORDER = [
    # 前三个关节保持原样
    (0, 0, JOINT_PARAMS['shoulder_pan_joint']),
    (1, 1, JOINT_PARAMS['shoulder_lift_joint']),
    (2, 2, JOINT_PARAMS['elbow_joint']),
    
    # 特殊处理：将msg.position[4]映射到ur_angles[3]
    (3, 4, JOINT_PARAMS['wrist_2_joint']),  
    
    # 其余关节正常映射
    (4, 3, JOINT_PARAMS['wrist_1_joint']),
    (5, 5, JOINT_PARAMS['wrist_3_joint'])
]

def fast_mapper(value: float, params: tuple) -> float:
    """超高速映射函数（<1μs/次）"""
    # 解包参数
    (t_min, t_max), (u_min, u_max), direction, *segments = params
    
    # 处理特殊分段（如elbow）
    if segments:
        for seg in segments[0]:  # seg格式: (t_start, t_end, u_start, u_end)
            if seg[0] <= value <= seg[1]:
                ratio = (value - seg[0]) / (seg[1] - seg[0])
                return seg[2] + ratio * (seg[3] - seg[2])
    
    # 常规线性映射
    clamped = max(min(value, t_max), t_min)
    ratio = (clamped - t_min) / (t_max - t_min)
    return u_min + ratio * (u_max - u_min) * direction
# 保持fast_mapper和JOINT_PARAMS不变


def callback(msg):
    """精确控制每个索引的映射"""
    global actions
    ur_angles = [0.0] * 6
    
    # 修改后的循环逻辑
    for target_idx, source_idx, params in JOINT_ORDER:
        ur_angles[target_idx] = fast_mapper(msg.position[source_idx], params)
    
    actions[:,0:6] = torch.tensor(ur_angles, device='cuda').unsqueeze(0)

def button_callback(msg):
    global actions      # 添加全局变量      
    actions[:,6] = torch.tensor(msg.close_gripper, device='cuda').unsqueeze(0)
    
def main():
    rospy.init_node('joint_listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/joint_states", JointState, callback)
    rospy.Subscriber("/phantom/phantom/state", OmniState, button_callback)    
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-ACT-Joint-WithTip-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-ACT-Joint-WithTip-Direct-v0", cfg=env_cfg)

    camera = env.scene["Camera"]
    camera_index = 0

    env.reset()

    while not rospy.is_shutdown():
        

        # print(actions)
        env.step(actions)
        ## 处理图像
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )
        image = single_cam_data['rgb']
        # 假设 image 是 RGB 格式的图像
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Feed', image_bgr)
        cv2.waitKey(1)
        # print(env.scene.articulations["left_robot"].data.body_state_w[0,-1, 0:3])
        
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()
    simulation_app.close()