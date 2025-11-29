import time
import cv2
from omni.isaac.lab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--save", action="store_true", default=False,
                    help="Save the data from camera at index specified by ``--camera_id``.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import pickle
import os
from einops import rearrange
import h5py
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import *  # 包含euclidean_distance等工具
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnv
from omni.isaac.lab.utils import convert_dict_to_backend
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import enhance_depth_image, euclidean_distance
import gymnasium as gym
from policy import ACTPolicy
from pynput import keyboard
# from omni.isaac.debug_draw import _debug_draw
import random
# ---------------------------- bidirectional_sampler 函数 ----------------------------
def bidirectional_sampler(strong, weak, obs_dict, prior, num_sample=15, beta=0.5, num_mode=2):
    """
    使用对比的方法采样动作，保证与先验一致性并对比强弱策略的输出。
    参数:
      strong: 强策略（具有 predict_action 方法）
      weak: 弱策略（可选，如果不存在则传入 None）
      obs_dict: 当前时刻观测数据字典，例如包含 'qpos' 和 'curr_image'
      prior: 上一步预测得到的动作序列（用于后向一致性），如果无则为 None
      num_sample: 生成的采样数量
      beta: 后向一致性权重衰减因子
      num_mode: 控制挑选 top 样本的因子
    返回:
      dict: 包含采样动作的字典，键例如 'action'、'action_pred'
    """
    # 1. 扩展观测数据：对每个键在 batch 维度上复制 num_sample 次
    B = obs_dict['qpos'].shape[0]
    obs_dict_batch = {}
    for key, value in obs_dict.items():
        # 对除 batch 外的所有维度进行复制
        obs_dict_batch[key] = value.unsqueeze(1).repeat(1, num_sample, *([1] * (value.dim()-1))).view(B * num_sample, *value.shape[1:])
    
    # 2. 使用强策略生成动作（要求 strong 提供 predict_action 方法）
    action_strong_batch = strong.predict_action(obs_dict_batch)
    
    # 3. 将预测结果调整为 (B=1, num_sample, ···) 的形状
    AH, AD = action_strong_batch['action'].shape[1], action_strong_batch['action'].shape[2]
    action_strong_batch['action'] = action_strong_batch['action'].view(B, num_sample, AH, AD)
    
    if weak is not None:
        action_weak_batch = weak.predict_action(obs_dict_batch)
        action_weak_batch['action'] = action_weak_batch['action'].view(B, num_sample, AH, AD)
    else:
        action_weak_batch = None
    
    # 4. 后向一致性计算（若 prior 存在）
    if prior is not None:
        start_overlap = strong.n_obs_steps - 1
        end_overlap = prior.shape[1]
        num_sample = num_sample // num_mode
        dist_raw = euclidean_distance(
            action_strong_batch['action'][:, :, start_overlap:end_overlap],
            prior.unsqueeze(1)[:, :, start_overlap:], reduction='none'
        )
        weights = torch.tensor([beta**i for i in range(end_overlap - start_overlap)]).to(dist_raw.device)
        weights = weights / weights.sum()
        dist_weighted = dist_raw * weights.view(1, 1, end_overlap - start_overlap)
        dist_strong_sum = dist_weighted.sum(dim=2)
        _, cross_index = dist_strong_sum.sort(descending=False)
        index = cross_index[:, :num_sample]
        action_strong_batch = {k: v[torch.arange(B, device=index.device).unsqueeze(1), index] for k, v in action_strong_batch.items()}
        dist_avg_prior = dist_strong_sum[torch.arange(B, device=index.device).unsqueeze(1), index]
    
        if action_weak_batch is not None:
            dist_weak = euclidean_distance(
                action_weak_batch['action'][:, :, start_overlap:end_overlap],
                prior.unsqueeze(1)[:, :, start_overlap:], reduction='none'
            )
            dist_weighted_weak = dist_weak * weights.view(1, 1, end_overlap - start_overlap)
            dist_weak_sum = dist_weighted_weak.sum(dim=2)
            _, cross_index_weak = dist_weak_sum.sort(descending=False)
            index_weak = cross_index_weak[:, :num_sample]
            action_weak_batch = {k: v[torch.arange(B, device=index_weak.device).unsqueeze(1), index_weak] for k, v in action_weak_batch.items()}
        ratio = 0.5
    else:
        dist_avg_prior = 0.0
        ratio = 0.0
    
    # 5. 正样本对比：计算强策略内部样本间的欧氏距离，并取每个样本最近 topk 个距离的平均值
    dist_pos = euclidean_distance(
        action_strong_batch['action'].unsqueeze(1),
        action_strong_batch['action'].unsqueeze(2)
    ).view(B, num_sample, num_sample)
    topk = num_sample // 2 + 1
    values, _ = torch.topk(dist_pos, k=topk, largest=False, dim=-1)
    dist_avg_pos = values[:, :, 1:].mean(dim=-1)  # 跳过自身距离
    
    # 6. 负样本对比（若存在弱策略）
    if action_weak_batch is not None:
        dist_neg = euclidean_distance(
            action_strong_batch['action'].unsqueeze(1),
            action_weak_batch['action'].unsqueeze(2)
        ).view(B, num_sample, num_sample)
        topk_neg = num_sample // 2
        values_neg, _ = torch.topk(dist_neg, k=topk_neg, largest=False, dim=-1)
        dist_avg_neg = values_neg.mean(dim=-1)
    else:
        dist_avg_neg = 0
    
    # 7. 综合得分与样本选择
    dist_avg = dist_avg_prior * ratio + (dist_avg_pos - dist_avg_neg) * (1 - ratio)
    _, index_final = dist_avg.min(dim=-1)
    action_dict = {k: v[torch.arange(B, device=index_final.device), index_final] for k, v in action_strong_batch.items()}
    return action_dict

# ---------------------------- 键盘监听相关 ----------------------------
reset_requested = False
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
    pos_tensor = obj.data.body_pos_w[0][:, :3]
    pos = pos_tensor[0]
    xy = pos[:2]
    target_xy = torch.tensor([0.0, -0.29], device=device)
    dist = torch.norm(xy - target_xy)
    cond2 = dist > 0.01
    success = cond2
    return success

def main():
    global reset_requested
    env_cfg = parse_env_cfg("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", device=args_cli.device, num_envs=1)
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    env.reset()

    camera = env.scene["Camera"]
    camera_index = 0
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
    args_policy = {
        'batch_size': 8, 
        'chunk_size': 200, 
        'ckpt_dir': ckpt_dir, 
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
    
    policy_config = {
        'lr': args_policy['lr'],
        'num_queries': args_policy['chunk_size'],
        'kl_weight': args_policy['kl_weight'],
        'hidden_dim': args_policy['hidden_dim'],
        'dim_feedforward': args_policy['dim_feedforward'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': ['top']
    }
    weak_policy_config = policy_config.copy()
    weak_ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_1500_seed_0.ckpt')
    
    policy = make_policy(policy_class, policy_config)
    weak_policy = make_policy(policy_class, weak_policy_config)

    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    loading_status = weak_policy.load_state_dict(torch.load(weak_ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    weak_policy.cuda()
    weak_policy.eval()
    
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    set_seed(42)

    # ----------------- 包装策略接口：如果 policy 没有 predict_action 则进行包装 -----------------
    if not hasattr(policy, "predict_action"):
        def predict_action_wrapper(obs_dict):
            qpos = obs_dict['qpos']
            curr_image = obs_dict['curr_image']
            # 调用原有的 policy，得到单步动作（形状: (B, action_dim)）
            single_action = policy(qpos, curr_image)
            # 返回字典中 'action' 为当前动作，'action_pred' 为整个预测序列
            return {'action': single_action}
        policy.predict_action = predict_action_wrapper
        
    if not hasattr(weak_policy, "predict_action"):
        def predict_action_wrapper(obs_dict):
            qpos = obs_dict['qpos']
            curr_image = obs_dict['curr_image']
            single_action = policy(qpos, curr_image)
            return {'action': single_action}
        weak_policy.predict_action = predict_action_wrapper
    # 策略观察步数设置（用于后向一致性计算）
    policy.n_obs_steps = 1
    weak_policy.n_obs_steps = 1
    

    # 初始化 prior 为 None（首步无 prior）
    prior = None

    success_cnt = 0
    total_cnt = 0
    t = 0

    max_timesteps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    query_frequency = 100  # 可根据需要调整

    while simulation_app.is_running():
        with torch.inference_mode():
            robot = env.scene["left_robot"]
            needle = env.scene.rigid_objects["object"]
            env.reset()
            t = 0
            print('reset')
            total_cnt += 1
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + query_frequency, state_dim]).cuda()
            # draw = _debug_draw.acquire_debug_draw_interface()
            while t < max_timesteps:
                # 获取当前关节状态
                qpos_numpy = np.array(robot.data.joint_pos.cpu().numpy().reshape(7))
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # shape (1, 7)

                # 获取并处理摄像头数据
                single_cam_data = convert_dict_to_backend({k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy")
                depth_image = enhance_depth_image(single_cam_data['distance_to_image_plane'])
                depth_image = rearrange(depth_image, 'h w c -> c h w')
                depth_image_process = torch.from_numpy(depth_image / 255.0).float().cuda().unsqueeze(0).unsqueeze(1)  # shape (1, C, H, W)
                curr_image = depth_image_process

                cv2.imshow('Camera Feed', cv2.cvtColor(single_cam_data["rgb"], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)  # 非阻塞显示

                # 构造观测数据字典
                obs_dict = {'qpos': qpos, 'curr_image': curr_image}
                # 使用 bidirectional_sampler 生成采样动作
                sampled_action_dict = bidirectional_sampler(strong=policy, weak=weak_policy, obs_dict=obs_dict, prior=prior, num_sample=9, beta=0.99, num_mode=3)
                # 选择最终动作（例如：取 'action' 并去除多余维度）
                final_action = sampled_action_dict['action'].squeeze(1)  # shape (B, action_dim)

                # 更新 prior
                prior = sampled_action_dict['action']

                all_time_actions[[t], t:t + query_frequency] = final_action[:, :query_frequency]
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.03
                # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))[::-1])
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = torch.tensor(action,device=args_cli.device).unsqueeze(0)
                
                point_list_1 = [
                    (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, -0.1)) for _ in range(10)
                ]
                # draw.draw_lines_spline(point_list_1, (1, 1, 1, 1), 1, False)

                # 执行动作
                ts = env.step(action.to(torch.float32))
 
                success = check_object_condition(needle, device='cuda:0')
                if success:
                    success_cnt += 1
                    break
                
                t += 1
                if reset_requested:
                    t = max_timesteps
                    print("检测到 'r' 键，重新开始")
            if not success:
                print(needle.data.body_state_w[0][:, 0:7])            
            print(f"当前成功率：{success_cnt / total_cnt:.2%}（{success_cnt}/{total_cnt}）")
                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
    simulation_app.close()
