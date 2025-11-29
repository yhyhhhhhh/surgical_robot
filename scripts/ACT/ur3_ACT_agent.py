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
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.save = True
# args_cli.headless = True
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
import os
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.ur3_lift_needle_env import Ur3LiftNeedleEnvCfg
from my_ur3_project.tasks.manipulator.ur3_surgical.config.joint_control.utils.myfunc import (
    enhance_depth_image,
    enhance_depth_images_batch,
)
from policy import ACTPolicy
### ====================================== ACT算法 ========================================
from policy import ACTPolicy 
import os
import pickle
from einops import rearrange
enable_extension("omni.isaac.debug_draw")
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

# PolicyWrapper: 将所有 policy 相关逻辑封装
class ActPolicy:
    def __init__(
        self,
        policy: torch.nn.Module,
        stats: dict,
        device: str,
        num_envs: int,
        max_timesteps: int,
        state_dim: int,
        query_frequency: int = 50,
        temporal_agg: bool = True,
    ):
        self.policy = policy.to(device).eval()
        self.stats = stats
        self.device = device
        self.num_envs = num_envs
        self.max_timesteps = max_timesteps
        self.state_dim = state_dim
        self.query_frequency = query_frequency
        self.temporal_agg = temporal_agg

        if temporal_agg:
            self.all_time_actions = torch.zeros(
                num_envs,
                max_timesteps,
                max_timesteps + query_frequency,
                state_dim,
                device=device,
            )
        # ### 残差策略
        # self.residual_policy: ResidualPolicy = hydra.utils.instantiate(
        #     cfg.actor.residual_policy,
        #     obs_shape=(self.timestep_obs_dim,),
        #     action_shape=(self.action_dim,),
        # )


    def _pre_process(self, qpos_numpy: np.ndarray) -> torch.Tensor:
        x = (qpos_numpy - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return torch.from_numpy(x).float().to(self.device)

    def _post_process(self, raw_action: torch.Tensor) -> torch.Tensor:
        a_np = raw_action.cpu().numpy()
        a_np = a_np * self.stats["action_std"] + self.stats["action_mean"]
        return torch.tensor(a_np, device=self.device, dtype=torch.float32)

    def get_action(self, qpos_numpy: np.ndarray, curr_image: torch.Tensor, t) -> torch.Tensor:
        qpos = self._pre_process(qpos_numpy)
        if self.temporal_agg:

            self.future_actions = self.policy(qpos, curr_image)
            env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.query_frequency)
            src_t   = t.unsqueeze(1).expand(self.num_envs, self.query_frequency)
            # 构造 0,1,...,Q-1
            # 构造 0,1,...,Q-1
            offsets = torch.arange(self.query_frequency, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            tgt_t = t.unsqueeze(1).expand(self.num_envs, 1) + offsets   # (num_envs, Q)
            self.all_time_actions[env_ids, src_t, tgt_t] = self.future_actions[:, :self.query_frequency]

            idx = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                 # [N,1,1,1]
            idx = idx.expand(self.num_envs, self.max_timesteps, 1, self.state_dim)
            actions_curr = torch.gather(self.all_time_actions, 2, idx).squeeze(2)

            raw_action = torch.zeros(self.num_envs, self.state_dim, device=self.device)
            for i in range(self.num_envs):
                hist = actions_curr[i][torch.any(actions_curr[i] != 0, dim=1)]
                if hist.numel() == 0:
                    raw_action[i] = self.future_actions[i, 0]
                else:
                    L = hist.size(0)
                    w = torch.exp(-0.03 * torch.arange(L - 1, -1, -1, device=self.device, dtype=hist.dtype))
                    w /= w.sum()
                    raw_action[i] = (hist * w.unsqueeze(1)).sum(dim=0)
        else:
            if t % self.query_frequency == 0:
                self.future_actions = self.policy(qpos, curr_image)
            raw_action = self.future_actions[:, t % self.query_frequency]

        return self._post_process(raw_action)
    
    def reset(self):
        self.all_time_actions = torch.zeros(
            self.num_envs,
            self.max_timesteps,
            self.max_timesteps + self.query_frequency,
            self.state_dim,
            device=self.device,
        )

# 其它工具函数
def setup_viewport():
    viewport_api = get_active_viewport()
    viewport = ViewportCameraState("/OmniverseKit_Persp", viewport_api)
    viewport.set_target_world(Gf.Vec3d(0.1398, -0.2116, -0.1437), rotate=True)
    viewport.set_position_world(Gf.Vec3d(-1.2322, -0.9908, 0.4137), rotate=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_policy(policy_class: str, policy_config: dict) -> torch.nn.Module:
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    else:
        raise NotImplementedError


def make_optimizer(policy_class: str, policy: torch.nn.Module):
    if policy_class == 'ACT':
        return policy.configure_optimizers()
    else:
        raise NotImplementedError


def main():
    num_envs = args_cli.num_envs
    env_cfg: Ur3LiftNeedleEnvCfg = parse_env_cfg(
        "My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0",
        device=device,
        num_envs=num_envs,
    )
    env = gym.make("My-Isaac-Ur3-Pipe-Ik-Abs-Direct-v0", cfg=env_cfg)
    env.env.init_robot_ik()
    setup_viewport()
    env.reset()

    camera = env.scene["Camera"]

    # 加载 policy 和统计数据
    ckpt_dir = '/media/yhy/PSSD/DVRK_DATA/orbit_surgical/touch/needle_dataset_depth/ckpt/20250404_112024/'
    ckpt_path = os.path.join(ckpt_dir, 'policy_epoch_2300_seed_0.ckpt')
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    state_dim = 8
    lr_backbone = 1e-5
    backbone = 'resnet18'
    args = {
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
        'temporal_agg': False,
    }
    policy_config = {
        'lr': args['lr'],
        'num_queries': args['chunk_size'],
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'nheads': nheads,
        'camera_names': ['top'],
    }
    policy = make_policy(args['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()
    print(f'Loaded policy from {ckpt_path}')

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    set_seed(42)
    query_frequency = 100
    temporal_agg = True
    max_timesteps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    # 封装 policy 逻辑
    policy_wrapper = ActPolicy(
        policy=policy,
        stats=stats,
        device=device,
        num_envs=num_envs,
        max_timesteps=max_timesteps,
        state_dim=state_dim,
        query_frequency=query_frequency,
        temporal_agg=temporal_agg,
    )

    robot = env.unwrapped.scene["left_robot"]

    t = 0
    while simulation_app.is_running():
        
        with torch.inference_mode():

            # 读取关节位置
            qpos_numpy = robot.data.joint_pos.cpu().numpy().reshape(num_envs, 7)

            # 获取相机图像
            multi_cam_data = convert_dict_to_backend(
                {k: v[:] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # for i, img_rgb in enumerate(multi_cam_data['rgb']):
            #     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            #     cv2.imshow(f"env_{i}", img_bgr)
            # cv2.waitKey(1)

            # curr_image = rearrange(multi_cam_data["rgb"], 'n h w c -> n c h w')
            # curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(1)

            depth_image = enhance_depth_images_batch(multi_cam_data['distance_to_image_plane'])
            depth_image = rearrange(depth_image, 'n h w c -> n c h w')
            depth_image_process = torch.from_numpy(depth_image / 255.0).float().to(device).unsqueeze(1)

            curr_image = depth_image_process

            # 获取动作并与环境交互
            raw_action = policy_wrapper.get_action(qpos_numpy, curr_image, env.env.episode_length_buf)
            ts = env.step(raw_action)

            done, time_out = ts[2], ts[3]
            if temporal_agg:
                policy_wrapper.all_time_actions[done] = 0
                policy_wrapper.all_time_actions[time_out] = 0


            # 统计或日志输出可放这里

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
