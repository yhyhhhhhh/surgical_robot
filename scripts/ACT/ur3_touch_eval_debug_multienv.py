import time
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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
args_cli.headless = True
device = args_cli.device

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import torch
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image,enhance_depth_images_batch
import h5py
from omni.isaac.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
from viztracer import VizTracer
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
    env.env.init_robot_ik()
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
    query_frequency = 50
    temporal_agg =  True
    # temporal_agg =  False
    max_timesteps = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0
    t = 0
    num_sample = 7
    
    action_std = torch.as_tensor(stats['action_std'], device=args_cli.device)
    action_mean = torch.as_tensor(stats['action_mean'], device=args_cli.device)
    # 定义颜色列表，与 single_action 中的各个动作一一对应
    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(num_sample)]
    
    while simulation_app.is_running():

        with torch.inference_mode():
            
     
            
            robot = env.unwrapped.scene["left_robot"]
            needle = env.unwrapped.scene.rigid_objects["object"]
            
            # env.reset()
            t = 0
            print('reset')
            total_cnt = total_cnt+1
        
            ### evaluation loop
            if temporal_agg:
                # ---------------------------------------------------------------------------
                # 初始化全局缓冲区：env × src_t × tgt_t × action_dim
                # 注意：如果 max_timesteps 很大，可改环形缓冲区以节省显存
                # ---------------------------------------------------------------------------
                all_time_actions = torch.zeros(
                    num_envs,
                    max_timesteps,
                    max_timesteps + query_frequency,
                    state_dim,
                    device=args_cli.device,
                )
            while t <= max_timesteps-1: # note: this will increase episode length by 1

                ### process previous timestep to get qpos and image_list
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(num_envs,7))
                ee_id = robot.body_names.index("scissors_tip")
                # qpos_numpy = np.array(robot.data.body_state_w[:,ee_id,0:7].cpu().numpy().reshape(7))
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda()

                multi_cam_data = convert_dict_to_backend(
                    {k: v[:] for k, v in camera.data.output.items()},  # 移除索引选择
                    backend="numpy"
                )
                # ----------------------------------------------------------------------------
                # 1. 可视化图像
                # ----------------------------------------------------------------------------                              
                for i, img_rgb in enumerate(multi_cam_data['rgb']):
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"env_{i}", img_bgr)

                cv2.waitKey(1)            # 等待任意键
                # ----------------------------------------------------------------------------                
                
                # 处理多摄像头图像（假设RGB数据）
                # 使用einops处理多个摄像头维度
                curr_image = rearrange(multi_cam_data["rgb"], 'n h w c -> n c h w')
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda()
                curr_image = curr_image.unsqueeze(1)  # 添加batch维度
                
                depth_image = enhance_depth_images_batch(multi_cam_data['distance_to_image_plane'])
                depth_image = rearrange(depth_image, 'n h w c -> n c h w')
                depth_image_process = torch.from_numpy(depth_image / 255.0).float().cuda().unsqueeze(1)

                # curr_image = torch.cat((curr_image, depth_image_process), dim=1)
                curr_image= depth_image_process

                if temporal_agg:
                    # ----------------------------------------------------------------------------
                    # 2. 每隔 query_frequency 调一次策略网络，写入未来动作
                    # ----------------------------------------------------------------------------
                
                    future_actions = policy(qpos, curr_image)  # (N, Q, A)

                    # 写入 4D 缓冲区
                    env_ids = torch.arange(num_envs, device=device).unsqueeze(1).expand(-1, query_frequency)
                    src_t   = torch.full_like(env_ids, t)                       # (N, Q)
                    tgt_t   = t + torch.arange(query_frequency, device=device) # (Q,) broadcast
                    tgt_t   = tgt_t.unsqueeze(0).expand(num_envs, -1)          # (N, Q)

                    all_time_actions[env_ids, src_t, tgt_t] = future_actions[:, :query_frequency]

                    # ----------------------------------------------------------------------------
                    # 3. 取出针对当前时刻 t 的所有历史预测，并做加权平均
                    # ----------------------------------------------------------------------------
                    idx_target = (
                        torch.full(
                            (num_envs, 1, 1, 1), t, device=device, dtype=torch.long
                        ).expand(num_envs, max_timesteps, 1, state_dim)
                    )
                    actions_curr = torch.gather(all_time_actions, 2, idx_target).squeeze(2)  # (N, T, A)

                    # 加权平均得到 raw_action (N, A)
                    raw_action = torch.zeros(num_envs, state_dim, device=device)

                    for env_idx in range(num_envs):
                        # 有效行：该行任何元素非 0  即代表此 src_t 对当前 t 有预测
                        valid_mask = torch.any(actions_curr[env_idx] != 0, dim=1)
                        hist = actions_curr[env_idx][valid_mask]  # (L_i, A)

                        if hist.numel() == 0:
                            # 还没有任何预测时，就用最近一次网络输出的第一步动作
                            if 'future_actions' in locals():
                                raw_action[env_idx] = future_actions[env_idx, 0]
                            continue
                        
                        k_decay = 0.03
                        L = hist.size(0)
                        w = torch.exp(-k_decay * torch.arange(L - 1, -1, -1, device=device, dtype=hist.dtype))
                        w /= w.sum()
                        raw_action[env_idx] = (hist * w.unsqueeze(1)).sum(dim=0)
                else:
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    raw_action = all_actions[:, t % query_frequency]
                
                # ----------------------------------------------------------------------------
                # 4. 后处理 + 与环境交互
                # ----------------------------------------------------------------------------
                action_np = post_process(raw_action.cpu().numpy()).reshape(num_envs, state_dim)
                action = torch.tensor(action_np, device=device, dtype=torch.float32)

                ts = env.step(action)  # type: ignore[misc]  # 根据你的 env API 调整
                
                # ----------------------------------------------------------------------------
                # 5. reset 处理
                # ----------------------------------------------------------------------------                
                done = ts[2]
                time_out = ts[3]
                if temporal_agg:
                    all_time_actions[done] = 0
                    all_time_actions[time_out] = 0
                # ----------------------------------------------------------------------------
                
                t += 1

            # ---------------------------------------------------------------------------------
            # 统计信息
            # ---------------------------------------------------------------------------------
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
