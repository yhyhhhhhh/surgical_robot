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
from omni.isaac.lab.utils.math import *
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image
import h5py
actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
prev_actions = torch.zeros(8,device=args_cli.device).unsqueeze(0)  # 添加batch维度 → shape (1,7)
sign = False

### ====================================== ACT算法 ========================================
from ..ACT.policy import ACTPolicy 
import os
import pickle
from einops import rearrange
from omni_msgs.msg import OmniState
# =========================================================================================

def parse_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # 提取元信息
        sim = f.attrs.get('sim', None)

        # 提取观测数据
        obs = f['observations']
        images = obs['images']
        top_images = images['top'][:]
        depth_images = images['depth'][:]

        qpos = obs['qpos'][:]
        qvel = obs['qvel'][:]
        init_pos = obs['init_pos'][:]

        # 提取动作数据（每个字段单独读取）
        action_group = f['action']
        touch_cam = action_group['touch_cam'][:]
        TipPose_cam = action_group['TipPose_cam'][:]
        touch_raw = action_group['touch_raw'][:]
        TipPose_raw = action_group['TipPose_raw'][:]
        gripper = action_group['gripper'][:]

        # 返回结构化数据
        return {
            'sim': sim,
            'top_images': top_images,
            'depth_images': depth_images,
            'qpos': qpos,
            'qvel': qvel,
            'init_pos': init_pos,
            'action': {
                'touch_cam': touch_cam,
                'TipPose_cam': TipPose_cam,
                'touch_raw': touch_raw,
                'TipPose_raw': TipPose_raw,
                'gripper': gripper
            }
        }


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
    
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Ik-Abs-Direct-v0",
        device=args_cli.device,
        num_envs=1,
    )
    env = gym.make("My-Isaac-Ur3-Ik-Abs-Direct-v0", cfg=env_cfg)
    env.reset()

    
    # data = parse_hdf5_file('/home/yhy/orbit_surgical/source/standalone/environments/state_machine/block_dataset1/episode_8.hdf5')
    
    camera = env.scene["Camera"]
    camera_index = 0
    cam_pos_w = env.scene["Camera"].data.pos_w
    cam_quat_w = env.scene["Camera"].data.quat_w_world
    
    # load policy and stats
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250323_191249/'
    ckpt_path = ckpt_dir + 'policy_best.ckpt'
    policy_class = 'ACT'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    args = {
        'batch_size': 8, 
        'chunk_size': 100, 
        'ckpt_dir': '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250323_191249/', 
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
    # 10-40%
    set_seed(0)
    query_frequency = 100
    # temporal_agg =  True
    # temporal_agg =  False
    max_timesteps = int(env_cfg.episode_length_s/(env_cfg.sim.dt*env_cfg.decimation))
    success_cnt = 0
    total_cnt = 0


    while simulation_app.is_running():

        with torch.inference_mode():
            robot = env.scene["robot"]
            needle = env.scene["object"]
            
            env.reset()
            print('reset')
            total_cnt = total_cnt+1
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps + query_frequency, state_dim]).cuda()
            for t in range(max_timesteps): # note: this will increase episode length by 1
                ### process previous timestep to get qpos and image_list
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(8))
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                ### 处理图像
                single_cam_data = convert_dict_to_backend(
                    {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
                )
                depth_image = enhance_depth_image(single_cam_data['distance_to_image_plane'])
                curr_image = rearrange(depth_image, 'h w c -> c h w')
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)
                
                cv2.waitKey(1)  # 非阻塞显示
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_actions = policy(qpos, curr_image)
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
                 
                target_qpos = action
                if target_qpos[-1]<0.7: 
                    target_qpos[-1]=-1
                ### step the environment
                ts = env.step(torch.tensor(target_qpos).unsqueeze(0))
                
                target_pos = torch.tensor([-0.0151, -0.1176, -0.2161], device=needle.data.body_pos_w.device)
                distance = torch.norm(needle.data.body_pos_w[0][:, :3] - target_pos, dim=1)
                is_within_threshold = torch.lt(distance, 0.02)

                success = check_object_condition(needle, device='cuda:0')
                if success:
                    success_cnt = success_cnt+1
                    break
            print(f"当前成功率：{success_cnt / total_cnt:.2%}（{success_cnt}/{total_cnt}）")

                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()
