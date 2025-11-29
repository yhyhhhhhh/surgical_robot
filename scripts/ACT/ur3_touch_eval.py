import time
import cv2
import argparse
import numpy as np
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
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import torch
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image
import h5py
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
prev_actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
sign = False

### ====================================== ACT算法 ========================================
from policy import ACTPolicy 
import os
import pickle
from einops import rearrange
# =========================================================================================
# 使用 pynput 监听键盘按键
from pynput import keyboard
reset_requested = False

def setup_viewport():
    """配置 viewport 函数"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)


# 键盘监听回调函数
def on_press(key):
    global reset_requested
    try:
        if key.char == 'r':
            reset_requested = True
    except AttributeError:
        pass

def on_release(key):
    global reset_requested
    try:
        if key.char == 'r':
            reset_requested = False
    except AttributeError:
        pass


# 启动 pynput 键盘监听器（后台线程）
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def check_object_condition(obj, device='cuda:0'):
    """
    检测 object 的位置是否满足：
      1. 纵坐标（假设为 y 分量） > -0.15；
      2. (x, y) 与目标点 (0, -0.29) 之间的距离 > 0.005。
      
    参数：
      obj: 包含属性 data.body_pos_w 的对象，
           其中 data.body_pos_w 是一个张量，其 shape 至少为 (N, 3)；
      device: 张量所在设备（默认为 'cuda:0'）。
      
    返回：
      success: 布尔值，True 表示两个条件均满足；
      pos: 检测到的 object 位置（以第一个采样为准）；
      dist: 计算得到的 (x,y) 平面距离。
    """
    # 取 object 的位置张量（假设 shape 为 (N, 3)，这里选择第一个）
    pos_tensor = obj.data.body_pos_w[0][:, :3]  # 例如 shape 为 (num_bodies, 3)
    pos = pos_tensor[0]  # 取第一个 body's 坐标，形状 (3,)
   
    # 条件2：计算 (x, y) 平面上与目标点 (0, -0.29) 的欧氏距离是否大于 0.005
    xy = pos[:2]
    target_xy = torch.tensor([0.0, -0.29], device=device)
    dist = torch.norm(xy - target_xy)
    cond2 = dist > 0.01

    success = cond2
    return success

def main():
    global reset_requested
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    env.env.init_robot_ik()
    # 设置 viewport（注意要在环境初始化后调用）
    setup_viewport()
    env.reset()

    
    # data = parse_hdf5_file('/home/yhy/orbit_surgical/source/standalone/environments/state_machine/block_dataset1/episode_8.hdf5')
    
    camera = env.scene["Camera"]
    camera_index = 0
    cam_pos_w = env.scene["Camera"].data.pos_w
    cam_quat_w = env.scene["Camera"].data.quat_w_world
    
    # load policy and stats
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/'
    ckpt_path = ckpt_dir + 'policy_epoch_2300_seed_0.ckpt'
    policy_class = 'ACT'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    args = {
        'batch_size': 8, 
        'chunk_size': 200, 
        'ckpt_dir': '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/', 
        'dim_feedforward': 3200, 
        'eval': False, 
        'hidden_dim': 512, 
        'kl_weight': 10, 
        'lr': 1e-05, 
        'num_epochs': 2000, 
        'onscreen_render': True, 
        'policy_class': 'ACT', 
        'seed': 0, 
        'task_name': 'sim_needle', 
        'temporal_agg': False
        }
    
    policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names':['top']
                        }
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    set_seed(42)
    ### 100 - 50%
    query_frequency = 100
    # temporal_agg =  True
    temporal_agg =  False
    max_timesteps = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0
    t = 0

    while simulation_app.is_running():

        with torch.inference_mode():
            
            saved_actions = []  # 用于存储每一步的 action

            robot = env.unwrapped.scene["left_robot"]
            needle = env.unwrapped.scene.rigid_objects["object"]
            
            # env.reset()
            t = 0
            print('reset')
            total_cnt = total_cnt+1
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps + query_frequency, state_dim]).cuda()
            while t <= max_timesteps-1: # note: this will increase episode length by 1

                ### process previous timestep to get qpos and image_list
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(7))
                ee_id = robot.body_names.index("scissors_tip")
                # qpos_numpy = np.array(robot.data.body_state_w[:,ee_id,0:7].cpu().numpy().reshape(7))
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
                )
                curr_image = rearrange(single_cam_data["rgb"], 'h w c -> c h w')
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)
                depth_image = enhance_depth_image(single_cam_data['distance_to_image_plane'])
                image_bgr = cv2.cvtColor(single_cam_data["rgb"], cv2.COLOR_RGB2BGR)
                cv2.imshow('Camera Feed', image_bgr)
                depth_image = rearrange(depth_image, 'h w c -> c h w')
                depth_image_process = torch.from_numpy(depth_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)
                # 假设 curr_image 和 depth_image_process 的形状都是 (1, 1, 3, 480, 640)
                # curr_image = torch.cat((curr_image, depth_image_process), dim=1)
                curr_image= depth_image_process
                # cv2.imshow('Camera Feed', single_cam_data['rgb'])
                cv2.waitKey(1)  # 非阻塞显示
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                    # print("load_action")
                if temporal_agg:
                    # all_actions = policy(qpos, curr_image)
                    all_time_actions[[t], t:t + query_frequency] = all_actions[:, :query_frequency]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1) # type: ignore
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.03
                    # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))[::-1])
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = torch.tensor(action,device=args_cli.device).unsqueeze(0)

                # # 相机坐标系 → 世界坐标系
                # pos_world_recovered, quat_world_recovered = combine_frame_transforms(
                #     cam_pos_w, cam_quat_w,     # T_cam_w: 相机在世界中的位置
                #     action[:,0:3], action[:,3:7]          # T_obj_cam: 物体在相机中的位置
                # )
                # touch_cam =  torch.cat([pos_world_recovered, quat_world_recovered,action[:,7].unsqueeze(0)], dim=1)
                
                ### step the environment
                ts = env.step(action.to(torch.float32))
                saved_actions.append(action.cpu().numpy())  # 每步保存

                done = ts[2]
                time_out = ts[3]
                if done:
                    all_time_actions = torch.zeros([max_timesteps, max_timesteps + query_frequency, state_dim]).cuda()

                # print(action)
                # ts = env.step(touch_cam)
                # target_pos = torch.tensor([-0.0151, -0.1176, -0.2161], device=needle.data.body_pos_w.device)
                # distance = torch.norm(needle.data.body_pos_w[0][:, :3] - target_pos, dim=1)
                # is_within_threshold = torch.lt(distance, 0.02)
                
                # if is_within_threshold.item():
                    
                #     success_cnt = success_cnt+1
                #     break
                
                success = check_object_condition(needle, device='cuda:0')
                if success:
                    success_cnt = success_cnt+1
                    break
                
                t = t+1
                # 检测 reset 请求，若触发则重置环境并清空所有全局变量
                if reset_requested:
                    t = max_timesteps
                    print("检测到 'r' 键，重新开始")
                    
            saved_actions = np.array(saved_actions).squeeze()  # shape (T, action_dim)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"haptic_actions_{timestamp}.npz"
            np.savez(save_path, actions=saved_actions)
            print(f"动作序列已保存至 {save_path}")

            if not success:
                print(needle.data.body_state_w[0][:, 0:7])            
            print(f"当前成功率：{success_cnt / total_cnt:.2%}（{success_cnt}/{total_cnt}）")

                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()
