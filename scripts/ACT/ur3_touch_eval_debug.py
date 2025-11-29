import csv
from pathlib import Path
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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
args_cli.headless = False
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
from omni.isaac.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import random
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D图


def parse_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # 提取元信息
        sim = f.attrs.get('sim', None)

        # 提取动作数据（每个字段单独读取）
        action_group = f['action']
        touch_raw = action_group['touch_raw'][:]
        TipPose_raw = action_group['TipPose_raw'][:]

        # 返回结构化数据
        return {
            'touch_raw': touch_raw,
            'TipPose_raw': TipPose_raw,
        }


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

def setup_viewport():
    """配置 viewport 函数"""
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)

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
    num_envs = args_cli.num_envs
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=num_envs,
    )
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    setup_viewport()
    env.reset()

    
    data = parse_hdf5_file('/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth_view/dataset/episode_46.hdf5')
    tensor_data = data["touch_raw"][:, :3]  # 取前200个点和前三列（x,y,z）
    point_list = [tuple(point) for point in tensor_data.tolist()]  # 转换为元组列表
    
    camera = env.scene["Camera"]
    camera_index = 0
    cam_pos_w = env.scene["Camera"].data.pos_w
    cam_quat_w = env.scene["Camera"].data.quat_w_world
    
    # load policy and stats
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/'
    # ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250508_011539/'
    # ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250508_170000/'
    ckpt_path = ckpt_dir + 'policy_epoch_2300_seed_0.ckpt'
    # ckpt_path = ckpt_dir + 'policy_best.ckpt'
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
        # 'ckpt_dir': '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/', 
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
    temporal_agg =  True
    # temporal_agg =  False
    max_timesteps = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0
    t = 0
    num_sample = 1
    
    action_std = torch.as_tensor(stats['action_std'], device=args_cli.device)
    action_mean = torch.as_tensor(stats['action_mean'], device=args_cli.device)
    # 定义颜色列表，与 single_action 中的各个动作一一对应
    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(num_sample)]
    
    save_dir = "trajectory_logs_rl_tem1111"
    os.makedirs(save_dir, exist_ok=True)
    # =============================================================
    # 0️⃣  目录准备（放在脚本最前面即可）
    # =============================================================
    save_root = Path(save_dir)          # 你原来用的 save_dir
    img_root  = save_root / "images"    # 统一的图片根目录
    success_root = img_root / "success" # 只做根目录
    fail_root    = img_root / "fail"
    success_root.mkdir(parents=True, exist_ok=True)
    fail_root.mkdir(parents=True,    exist_ok=True)

    while simulation_app.is_running():

        with torch.inference_mode():
            
     
            
            robot = env.unwrapped.scene["left_robot"]
            needle = env.unwrapped.scene.rigid_objects["object"]
            
            env.reset()
            t = 0
            print('reset')
            total_cnt = total_cnt+1
            frames = []                 # ← 每个 episode 单独收集帧
            draw = omni_debug_draw.acquire_debug_draw_interface()

            colors = [(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1) for _ in range(query_frequency)]
            sizes = [8 for _ in range(query_frequency)]
            tem_point_list = []
            tem_point_list1 = []
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps + query_frequency, state_dim]).cuda()
            while t <= max_timesteps-1: # note: this will increase episode length by 1

                ### process previous timestep to get qpos and image_list
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(num_envs,7))
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
                cv2.waitKey(1)  # 非阻塞显示
                # ② 把当前帧缓存到列表
                frames.append(image_bgr.copy())
                depth_image = rearrange(depth_image, 'h w c -> c h w')
                depth_image_process = torch.from_numpy(depth_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)

                # curr_image = torch.cat((curr_image, depth_image_process), dim=1)
                curr_image= depth_image_process
                # 构造观测数据字典
                obs_dict = {'qpos': qpos, 'curr_image': curr_image}

                if t % query_frequency == 0:
                    B = obs_dict['qpos'].shape[0]
                    obs_dict_batch = {}
                    for key, value in obs_dict.items():
                        # 对除 batch 外的所有维度进行复制
                        obs_dict_batch[key] = value.unsqueeze(1).repeat(1, num_sample, *([1] * (value.dim()-1))).view(B * num_sample, *value.shape[1:])
                    qpos = obs_dict_batch['qpos'][0]
                    curr_image = obs_dict_batch['curr_image']
                    single_action = policy(qpos, curr_image)
                    all_actions = single_action[0].unsqueeze(0)

                    # print("load_action")
                if temporal_agg:
                    # 1. 扩展观测数据：对每个键在 batch 维度上复制 num_sample 次
                    B = obs_dict['qpos'].shape[0]
                    obs_dict_batch = {}
                    for key, value in obs_dict.items():
                        # 对除 batch 外的所有维度进行复制
                        obs_dict_batch[key] = value.unsqueeze(1).repeat(1, num_sample, *([1] * (value.dim()-1))).view(B * num_sample, *value.shape[1:])
                    
                    qpos = obs_dict_batch['qpos'][0]
                    curr_image = obs_dict_batch['curr_image']
                    single_action = policy(qpos, curr_image)
                    all_actions = single_action[0].unsqueeze(0)
                    all_time_actions[[t], t:t + query_frequency] = all_actions[:, :query_frequency]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]


                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                
                action = post_process(raw_action)
                tem_point_list.append(tuple(action[0:3]))
                tem_point_list1.append(tuple(robot.data.body_pos_w[0,ee_id,0:3].cpu().numpy()))
                action = torch.tensor(action,device=args_cli.device).unsqueeze(0)
                
                # 根据时间步 t 生成渐变颜色
                # color = [t / max_timesteps, 1, 1 - t / max_timesteps, 1]  # 从蓝色渐变到红色
                # # 遍历每个动作张量和对应颜色
                # for action_tensor in single_action:
                #     # 在 GPU 上进行后处理（乘法与加法操作）
                #     # 使用切片拿前 100 个点、前三个坐标
                #     vis_action = (action_tensor * action_std + action_mean)[:200, :3]
                    
                #     # detach 后一次性传输到 CPU
                #     vis_action_np = vis_action.detach().cpu().numpy()
                    
                #     # 转换为列表，每个坐标为 tuple（这里使用 list comprehension）
                #     point_list = [tuple(pt) for pt in vis_action_np.tolist()]
                    
                #     # 绘制平滑曲线
                #     draw.draw_lines_spline(point_list, color, 2, False)
                # draw.draw_lines_spline(tem_point_list, colors[0], 2, False)
                # draw.draw_lines_spline(tem_point_list1, colors[1], 2, False)
                ts = env.step(action.to(torch.float32))
                # draw.clear_lines()
                success = check_object_condition(needle, device='cuda:0')
                if success:
                    success_cnt = success_cnt+1
                    break
                
                t = t+1
                # 检测 reset 请求，若触发则重置环境并清空所有全局变量
                if reset_requested:
                    t = max_timesteps
                    print("检测到 'r' 键，重新开始")
             # =========================================================
            # 3️⃣  episode 结束后：按 success / fail 批量保存
            # =========================================================

            # root_dir = success_root if success else fail_root
            # run_tag  = f"{total_cnt:05d}"          # episode 编号
            # episode_dir = root_dir / run_tag       # ←★ 新建子目录
            # episode_dir.mkdir(parents=True, exist_ok=True)

            # # --- 保存所有帧 ---
            # for idx, frame in enumerate(frames):
            #     cv2.imwrite(str(episode_dir / f"frame_{idx:03d}.png"), frame)

            # ======= 保存轨迹图与数据 =======
            # if tem_point_list:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     x_vals = [p[0] for p in tem_point_list]
            #     y_vals = [p[1] for p in tem_point_list]
            #     z_vals = [p[2] for p in tem_point_list]
            #     ax.plot(x_vals, y_vals, z_vals, label='EE Trajectory')
            #     ax.set_title(f'Trajectory {total_cnt}')
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #     ax.legend()
            #     plt.savefig(os.path.join(save_dir, f"trajectory_{total_cnt}.png"))
            #     plt.close(fig)

            #     with open(os.path.join(save_dir, f"trajectory_{total_cnt}.csv"), mode='w', newline='') as file:
            #         writer = csv.writer(file)
            #         writer.writerow(["X", "Y", "Z"])
            #         for point in tem_point_list:
            #             writer.writerow(point)
            # draw.clear_lines()
            if not success:
                print(needle.data.body_state_w[0][:, 0:7])            
            print(f"当前成功率：{success_cnt / total_cnt:.2%}（{success_cnt}/{total_cnt}）")

                
    env.close()
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()
