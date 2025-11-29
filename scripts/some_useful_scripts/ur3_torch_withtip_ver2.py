import cv2
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import torch
import rospy
from omni.isaac.lab.utils.math import *
from geometry_msgs.msg import PoseStamped
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image
import h5py
actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
prev_actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
sign = False
### ========== 导入torch msg ===========
import sys
import os

os.environ['ROS_PACKAGE_PATH'] = '/home/yhy/code/touch_ws/src/Geomagic_Touch_ROS_Drivers-hydro-devel:' + os.environ.get('ROS_PACKAGE_PATH', '')
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/yhy/code/touch_ws/devel/lib/python3/dist-packages')
# ======================================
from omni_msgs.msg import OmniState
# ========== 1. 定义线性映射函数 ==========
def map_range(value, src_min, src_max, dst_min, dst_max):
    """
    将 value 从 [src_min, src_max] 映射到 [dst_min, dst_max] 的线性函数。
    如果需要超出范围后 clamp，可以在此额外处理，这里只做最简单的映射。
    """
    # 避免除 0
    if src_max == src_min:
        return dst_min
    
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)

def callback(data):
    global actions
    # 保存上一个动作
    prev_actions = actions
    
    # ========== 2. 读取原始位置 + 乘以 10 ==========
    raw_x = data.pose.position.x * 10
    raw_y = data.pose.position.y * 10
    raw_z = data.pose.position.z * 10
    
    # 这里的范围要与你在硬件/输入端约定的相匹配：
    # 假设原先 Torch 端: x ∈ [-0.15, 0.20], y ∈ [-0.073, 0.126], z ∈ [-0.11, 0.165]
    # 乘以10后 => x ∈ [-1.5, 2.0], y ∈ [-0.73, 1.26], z ∈ [-1.1, 1.65]
    
    # ========== 3. 对 x, y, z 分别做线性映射 + z 的下限限制 ==========

    # 3.1 x 映射：[-1.5, 2.0] -> [-0.1, 0.1]
    mapped_x = map_range(raw_x, -1.5, 2.0, -0.1, 0.1)
    
    # 3.2 y 映射：[-0.73, 1.26] -> [-0.4, -0.0]
    mapped_y = map_range(raw_y, -0.73, 1.26, -0.4, -0.05)
    
    # 3.3 z 先映射到某个区间[-0.4, 0.2]，再做 z >= -0.3 的 clamp
    z_temp = map_range(raw_z, -1.1, 1.65, -0.3, 0.2)
    mapped_z = max(z_temp, -0.3)  # 低于 -0.3 则置为 -0.3

    # ========== 4. 获取 Orientation（保留原逻辑不变） ==========
    q_T = torch.tensor([
        data.pose.orientation.w,
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
    ], dtype=torch.float32, device=args_cli.device)
    q_offset = torch.tensor([0.4995,-0.5003,-0.4997,-0.5005],device=args_cli.device)
    q_R = quat_mul(q_offset, q_T)
    q_R = torch.tensor([-0.4995, -0.5003, -0.4997, -0.5005],device=args_cli.device)
    
    # ========== 5. 组装为张量 (x, y, z, qx, qy, qz, qw) 并在最后再加一个 0 ==========
    # 创建位置的 tensor，形状为 (3,)
    pos_tensor = torch.tensor([mapped_x, mapped_y, mapped_z], dtype=torch.float32, device=args_cli.device)

    # q_R 已经是 shape (4,)，直接拼接得到 shape (7,)
    base_tensor = torch.cat([pos_tensor, q_R], dim=0).unsqueeze(0)  # shape (1,7)
    actions[:, :7] = base_tensor  # 只覆盖前 7 列
    
def button_callback(msg):
    global actions,sign      # 添加全局变量      
    actions[:,7] = torch.tensor(msg.close_gripper, device='cuda').unsqueeze(0)
    sign = msg.locked
def main():
    global actions
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/phantom/phantom/pose", PoseStamped, callback)
    rospy.Subscriber("/phantom/phantom/state", OmniState, button_callback)  
    rate = rospy.Rate(10)  # 10Hz
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )

    env = gym.make("My-Isaac-Ur3-Ik-Abs-Direct-v0", cfg=env_cfg)

    env.reset()

    camera = env.scene["Camera"]
    camera_index = 0
    cam_pos_w = env.scene["Camera"].data.pos_w
    cam_quat_w = env.scene["Camera"].data.quat_w_world
    
    robot = env.scene["left_robot"].data
    needle = env.scene.rigid_objects["needle"]
    episode_len = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    episode_idx = 0
    sign_flag = sign
    while not rospy.is_shutdown():
        # print(sign,sign_flag)
        ### 控制每个episode的开始
        rate.sleep()
        if sign != sign_flag:
            sign_flag = sign
            success = False
            # ================================ 每个episode ======================================
            episode_idx = episode_idx+1
            print(f"{episode_idx}开始")
            env.reset()       
            ### 保存数据
            data_dict = {
                    '/observations/qpos': [],
                    '/observations/qvel': [],
                    '/action/touch_cam': [],
                    '/action/touch_raw': [],
                    '/action/TipPose_cam': [],
                    '/action/TipPose_raw': [],
                    '/action/gripper': [],
                    '/observations/images/top':[],
                    '/observations/images/depth':[],
                    '/observations/init_pos':[],
                }
            for step in range(episode_len):
                # print(env.scene.articulations["left_robot"].data.joint_pos[0,-1])
                # print(actions[:,0:3])
                env.step(actions)
                ### actions由于是末端位置，所以添加当前时刻的末端quat组成完整的actions (touch)
                actions[:,3:7] = robot.body_state_w[:,-1, 3:7]
                
                touch_raw = actions[:,0:7]
                Tip_raw = robot.body_state_w[:,-1, 0:7]
                gripper = actions[:,7]
                
                # print(env.scene.articulations["left_robot"].data.body_state_w[0,-1, 3:7])
                # print(env.scene.articulations["left_robot"].data.joint_pos)
                
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
                )
                image = single_cam_data['rgb']
                depth_image = enhance_depth_image(single_cam_data['distance_to_image_plane'])
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Camera Feed', image_bgr)
                cv2.waitKey(1)
                
                # ========== 1.相对相机坐标系 ==========
                pos_cam, quat_cam = subtract_frame_transforms(
                    cam_pos_w, 
                    cam_quat_w,
                    actions[:,0:3],
                    actions[:,3:7],
                )
                # 相机坐标系 → 世界坐标系
                pos_world_recovered, quat_world_recovered = combine_frame_transforms(
                    cam_pos_w, cam_quat_w,     # T_cam_w: 相机在世界中的位置
                    pos_cam, quat_cam          # T_obj_cam: 物体在相机中的位置
                )
                touch_cam =  torch.cat([pos_cam, quat_cam], dim=1)
                # ========== 2.相对相机坐标系（机械臂末端） ==========
                TipPos_cam, TipQuat_cam = subtract_frame_transforms(
                    cam_pos_w, 
                    cam_quat_w,
                    robot.body_state_w[:,-1, 0:3],
                    robot.body_state_w[:,-1, 3:7],
                )               
                Tip_cam =  torch.cat([TipPos_cam, TipQuat_cam], dim=1)
                # tensor([[[-0.0151, -0.1176, -0.2161]]], device='cuda:0')
                target_pos = torch.tensor([-0.0151, -0.1176, -0.2161], device='cuda:0')
                distance = torch.norm(needle.data.body_pos_w[0][:, :3] - target_pos, dim=1)
                # print(distance)
                is_within_threshold = torch.lt(distance, 0.018)
                # print(cam_pos_w)
                if is_within_threshold.item():
                    # print(f"抓取成功！")
                    success = True

                    
                if args_cli.save:
                
                    data_dict["/observations/qpos"].append(robot.joint_pos.cpu().numpy().reshape(7))
                    data_dict["/observations/qvel"].append(robot.joint_vel.cpu().numpy().reshape(7))
                    data_dict["/action/touch_cam"].append(touch_cam.cpu().numpy().reshape(7))
                    data_dict["/action/TipPose_cam"].append(Tip_cam.cpu().numpy().reshape(7))
                    data_dict["/action/touch_raw"].append(touch_raw.cpu().numpy().reshape(7))
                    data_dict["/action/TipPose_raw"].append(Tip_raw.cpu().numpy().reshape(7))         
                    data_dict["/action/gripper"].append(gripper.cpu().numpy().reshape(1))           
                    data_dict["/observations/images/top"].append(image)
                    data_dict["/observations/images/depth"].append(depth_image)
                    data_dict["/observations/init_pos"].append(needle.data.root_pos_w.cpu().numpy().reshape(3))
                    
            if success:
                print(f"成功")
                if args_cli.save:
                    # 创建保存目录
                    output_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth_view/dataset'
                    os.makedirs(output_dir, exist_ok=True)

                    with h5py.File(os.path.join(output_dir, "episode_"+ str(episode_idx)+".hdf5"), 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                        root.attrs['sim'] = True
                        obs = root.create_group('observations')
                        image = obs.create_group('images')
                        _ = image.create_dataset('top', (episode_len, 480, 640, 3), dtype='uint8',
                                                    chunks=(1, 480, 640, 3), )
                        _ = image.create_dataset('depth', (episode_len, 480, 640, 3), dtype='uint8',
                                                        chunks=(1, 480, 640, 3), )
                        qpos = obs.create_dataset('qpos', (episode_len, 7))
                        qvel = obs.create_dataset('qvel', (episode_len, 7))
                        
                        action = root.create_group('action')
                        touch_cam = action.create_dataset('touch_cam', (episode_len, 7))
                        TipPose_cam = action.create_dataset('TipPose_cam', (episode_len, 7))
                        touch_raw = action.create_dataset('touch_raw', (episode_len, 7))
                        TipPose_raw = action.create_dataset('TipPose_raw', (episode_len, 7))
                        gripper = action.create_dataset('gripper', (episode_len, 1))
                        
                        init_pos = obs.create_dataset('init_pos', (episode_len,3))
                        for name, array in data_dict.items():
                            root[name][...] = array
            print(f"{episode_idx}结束")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()
    simulation_app.close()