import pathlib
import argparse
import functools
import numpy as np
import torch
import gymnasium as gym
import sys

sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools

import ur3_lite  # noqa: F401

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools

# -----------------------
# 三维动作扫描 + uncertainty 辅助函数
# -----------------------
def _slice_latent_one_env(latent, env_idx):
    return {k: v[env_idx:env_idx + 1] if torch.is_tensor(v) else v for k, v in latent.items()}

def _repeat_latent(latent_one, repeat_n):
    return {k: v.repeat(repeat_n, *[1]*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in latent_one.items()}

def _build_scanned_actions(action_base, env_idx, action_grid, scan_dims):
    base_action_one = action_base["action"][env_idx:env_idx + 1]
    n = action_grid.shape[0]
    scanned = {k: v for k, v in action_base.items() if k != "action"}
    scanned_action = base_action_one.repeat(n,1)
    scanned_action[:, scan_dims] = action_grid
    scanned["action"] = scanned_action
    return scanned

# -----------------------
# 三维扫描 + reward + uncertainty
# -----------------------
def compute_score_grid_3d(
    wm, ensemble, latent, base_action, reward_fn,
    env_idx=0,
    scan_dims=[0,1,2],
    action_min=-1.0,
    action_max=1.0,
    num_points_per_dim=11,
    alpha=0.7  # alpha: uncertainty权重, 1-alpha: reward权重
):
    device = base_action["action"].device
    grids = [torch.linspace(action_min, action_max, num_points_per_dim, device=device) for _ in range(3)]
    A0, A1, A2 = torch.meshgrid(*grids, indexing='ij') # A0,A1,A2分别是网格上的每个点对应的动作分量的值
    N = num_points_per_dim
    total_points = N**3

    # 构造动作 batch
    scanned_actions = {}
    base_a = base_action["action"][env_idx:env_idx+1].repeat(total_points,1)
    scanned_actions["action"] = base_a.clone()
    scanned_actions["action"][:, scan_dims] = torch.stack([A0.flatten(), A1.flatten(), A2.flatten()], dim=-1)

    # latent batch
    latent_one = {k:v[env_idx:env_idx+1] for k,v in latent.items()}
    latent_rep = {k:v.repeat(total_points, *[1]*(v.ndim-1)) for k,v in latent_one.items()}

    # uncertainty
    with torch.no_grad():
        unc = tools.get_uncertainty(wm, ensemble, latent_rep, scanned_actions)

    # reward proxy
    with torch.no_grad():
        base_action_one = base_action["action"][env_idx:env_idx + 1]
        reward_vals = reward_fn(scanned_actions, base_action_one)

    # 归一化
    unc_norm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
    reward_norm = ((reward_vals - reward_vals.min()) / (reward_vals.max() - reward_vals.min() + 1e-8)).reshape(-1, 1)

    # score
    score = alpha*unc_norm + (1-alpha)*reward_norm
    score_grid = score.reshape(N,N,N).cpu().numpy()
    unc_grid = unc.reshape(N,N,N)

    # 找最大 score
    max_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    max_action = torch.stack([A0[max_idx], A1[max_idx], A2[max_idx]]).detach().cpu().numpy()
    return A0.cpu().numpy(), A1.cpu().numpy(), A2.cpu().numpy(), unc_grid.cpu().numpy(), max_action

# -----------------------
# 3D 可视化类
# -----------------------
class LiveUncertaintyViewer3D:
    def __init__(self):
        self.initialized = False
    def update(self, A0, A1, A2, unc_grid, max_action):
        X = A0.flatten()
        Y = A1.flatten()
        Z = A2.flatten()
        U = unc_grid.flatten()

        if not self.initialized:
            plt.ion()
            self.fig = plt.figure(figsize=(8,6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.scatter = self.ax.scatter(X,Y,Z,c=U,cmap='viridis',marker='o')
            self.max_pt = self.ax.scatter(max_action[0], max_action[1], max_action[2], c='red', s=100, label='Selected action')
            self.fig.colorbar(self.scatter,label='Uncertainty')
            self.ax.set_xlabel("Action 0")
            self.ax.set_ylabel("Action 1")
            self.ax.set_zlabel("Action 2")
            self.ax.set_title("3D Action Space Uncertainty")
            self.ax.legend()
            self.initialized = True
        else:
            for coll in self.ax.collections:
                coll.remove()
            self.scatter = self.ax.scatter(X,Y,Z,c=U,cmap='viridis',marker='o')
            self.max_pt = self.ax.scatter(max_action[0], max_action[1], max_action[2], c='red', s=100, label='Selected action')
        plt.pause(0.001)

# -----------------------
# 修改后的评估函数
# -----------------------
@torch.no_grad()
def evaluate_world_model_vecenv(
    agent,
    vecenv,
    episodes,
    print_every=50,
    noise_std=0.0,
    use_noisy_action=False,
    enable_uncertainty_view=True,
    scan_dims=[0,1,2],
    num_points_per_dim=11,
    action_min=-1.0,
    action_max=1.0,
    alpha=0.7,
    reward_fn=None  # reward proxy 函数，输入 actions dict，输出 tensor shape=[N^3]
):
    num_env = vecenv.num_envs
    obs = vecenv.reset()
    done = np.ones(num_env, dtype=bool)
    agent_state = None

    wm = agent._wm
    data = wm.preprocess(obs)
    embed = wm.encoder(data)
    latent, _ = wm.dynamics.obs_step(None, None, embed, obs["is_first"], sample=False)

    eval_policy = functools.partial(agent, training=False)
    finished = 0
    steps = 0
    loop_idx = 0

    viewer = None
    if enable_uncertainty_view and (agent._disag_ensemble is not None):
        viewer = LiveUncertaintyViewer3D()

    while finished < episodes:
        action_base, agent_state = eval_policy(obs, done, agent_state)

        # -----------------------
        # 加噪动作
        # -----------------------
        action_noisy = {k: v.clone() if torch.is_tensor(v) else v for k,v in action_base.items()}
        if "action" in action_noisy and torch.is_tensor(action_noisy["action"]) and noise_std>0:
            noise = torch.randn_like(action_noisy["action"])*noise_std
            action_noisy["action"] = torch.clamp(action_noisy["action"]+noise,-1.0,1.0)

        # -----------------------
        # 扫描三维动作空间 + reward + uncertainty
        # -----------------------
        if agent._disag_ensemble is not None:
            A0,A1,A2, unc_grid, max_action = compute_score_grid_3d(
                wm, agent._disag_ensemble, latent, action_base,
                scan_dims=scan_dims,
                action_min=action_min,
                action_max=action_max,
                num_points_per_dim=num_points_per_dim,
                alpha=alpha,
                reward_fn=reward_fn
            )
            # print(f"Step {steps}: selected max score action={max_action}", flush=True)

            # 执行最大分数动作
            action_exec = {k:v.clone() for k,v in action_base.items()}
            action_exec["action"][0,0:3] = torch.tensor(max_action, device=action_exec["action"].device)

            # 可视化
            if enable_uncertainty_view:
                viewer.update(A0,A1,A2,unc_grid,max_action)
        else:
            action_exec = action_noisy if use_noisy_action else action_base

        next_obs, next_reward, next_done_t, _ = vecenv.step(action_exec)
        next_done = next_done_t.detach().cpu().numpy()
        steps += num_env
        loop_idx += 1

        data = wm.preprocess(next_obs)
        embed = wm.encoder(data)
        latent, _ = wm.dynamics.obs_step(latent, action_exec["action"], embed, next_obs["is_first"], sample=False)
        obs = next_obs

        if next_done.any():
            reset_obs = vecenv.reset(seed=next_done)
            obs = {k: (v.clone() if torch.is_tensor(v) else v) for k,v in next_obs.items()}
            for k,v in obs.items():
                if torch.is_tensor(v) and (k in reset_obs) and torch.is_tensor(reset_obs[k]):
                    obs[k][next_done] = reset_obs[k][next_done]
            if "is_first" in obs: obs["is_first"][next_done] = 1
            if "is_last" in obs: obs["is_last"][next_done] = 0
            if "is_terminal" in obs: obs["is_terminal"][next_done] = 0

            reset_data = wm.preprocess(obs)
            reset_embed = wm.encoder(reset_data)
            reset_latent, _ = wm.dynamics.obs_step(None,None,reset_embed,obs["is_first"],sample=False)
            done_mask = torch.as_tensor(next_done, device=reset_embed.device, dtype=torch.bool)
            for k in latent.keys():
                latent[k][done_mask] = reset_latent[k][done_mask]

        for i in np.where(next_done)[0]:
            finished += 1
            if finished >= episodes: break

        if print_every>0 and (steps%(print_every*num_env)==0):
            print(f"[wm-eval] steps={steps} finished={finished}/{episodes}", flush=True)

        done = next_done
    return
def main(config):
    # -----------------------
    # 基础设置
    # -----------------------
    tools.set_seed_everywhere(config.seed)
    config.evaldir = pathlib.Path(config.evaldir).expanduser()
    config.evaldir.mkdir(parents=True, exist_ok=True)

    print("Eval dir:", config.evaldir)

    # -----------------------
    # 创建环境
    # -----------------------
    envs = dreamer.make_env(config, num_envs=config.envs)
    acts = envs.single_action_space

    # Action normalization（必须和训练一致）
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high)

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # -----------------------
    # 创建空 replay（只为接口完整）
    # -----------------------
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    eval_dataset = dreamer.make_dataset(eval_eps, config)

    logger = tools.Logger(config.evaldir, step=0)

    # -----------------------
    # 创建 Agent
    # -----------------------
    agent = dreamer.Dreamer(
        envs.single_observation_space,
        acts,
        config,
        logger,
        eval_dataset,
    ).to(config.device)

    agent.requires_grad_(False)

    # -----------------------
    # 加载模型
    # -----------------------
    checkpoint = torch.load(config.model_path, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    # agent.eval()

    print(f"Loaded model from {config.model_path}")
   
    def reward_fn(actions, base_action_one):
        # actions["action"]: [N, act_dim]
        # base_action_one:   [1, act_dim]
        diff = actions["action"] - base_action_one
        return -torch.sum(diff * diff, dim=-1)
    # -----------------------
    # 世界模型测试（不走训练缓存逻辑）
    # -----------------------
    with torch.no_grad():

        evaluate_world_model_vecenv(
            agent=agent,
            vecenv=envs,
            episodes=5,
            noise_std=0.05,
            scan_dims=[0,1,2],
            num_points_per_dim=9,
            alpha=0.6,
            reward_fn=reward_fn,
            enable_uncertainty_view=True,
            action_min=-1.0,
            action_max=1.0
        )

    try:
        envs.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--task", type=str, default="Ur3Lite-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    # parser.add_argument("--model_path", type=str, default="/home/yhy/IsaacLabExtensionTemplate_lite/model/world_model/latest.pt")
    parser.add_argument("--model_path", type=str, default="/home/yhy/IsaacLabExtensionTemplate_lite/latent_safety/log/dreamerv3/world_model_only/0325/222850_test/latest.pt")
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--eval_print_every", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--use_noisy_action", action="store_true", default=False)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--evaldir", type=str, default="latent_safety/log/dreamerv3/1225/004059_test/eval_eps")
    parser.add_argument(
		"--enable_cameras", action="store_true", default=False
	)
    parser.add_argument(
		"--headless", action="store_true", default=False
	)

    args, remaining = parser.parse_known_args()
    args.enable_cameras = True
    args.rendering_mode = "quality"   # performance / balanced / quality
    args.headless = False   # 强制 headless
    # -----------------------
    # 读取 configs.yaml（与你训练一致）
    # -----------------------
    import pathlib
    import ruamel.yaml as yaml

    # 1. 初始化一个 YAML 解析器实例 (指定 typ='safe' 来替代原来的 safe_load)
    yaml_parser = yaml.YAML(typ='safe', pure=True)

    # 2. 使用解析器的 .load() 方法
    configs = yaml_parser.load(
        (pathlib.Path(__file__).parent / "../dreamerv3_torch/configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    defaults = {}
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    for name in name_list:
        recursive_update(defaults, configs[name])

    for k, v in vars(args).items():
        defaults[k] = v

    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)

    config = parser.parse_args(remaining)
    main(config)
