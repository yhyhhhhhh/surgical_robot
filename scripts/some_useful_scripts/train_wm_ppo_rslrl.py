# train_residual_wm_posterior_pool.py
from __future__ import annotations

import argparse
import pathlib
import time
import numpy as np
import ruamel.yaml as yaml
import torch
import sys
from typing import Dict, Any, List

# 让脚本能找到 dreamerv3_torch 相关模块
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer  # noqa: E402
import my_ur3_project.tasks  # noqa: F401, E402

from rsl_rl.env import VecEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


# -------------------------
# 配置加载工具
# -------------------------
def recursive_update(base: dict, update: dict):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def load_config_from_yaml(configs_yaml_path: str, selected_configs: list[str] | None, cli_overrides: dict):
    yaml_parser = yaml.YAML(typ="safe", pure=True)
    cfg_path = pathlib.Path(configs_yaml_path)
    if not cfg_path.is_absolute():
        cfg_path = (pathlib.Path(__file__).resolve().parent / cfg_path).resolve()

    configs = yaml_parser.load(cfg_path.read_text())
    name_list = ["defaults", *(selected_configs or [])]
    merged = {}
    for name in name_list:
        recursive_update(merged, configs[name])
    for k, v in cli_overrides.items():
        merged[k] = v
    return argparse.Namespace(**merged)


class NullLogger:
    def __init__(self):
        self.step = 0

    def scalar(self, *args, **kwargs): pass
    def video(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass


# -------------------------
# 真实 reset 观测池采集
# -------------------------
def _extract_obs(reset_out):
    # gymnasium: (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def _to_cpu_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(np.asarray(x)).cpu()


def _standardize_obs_dict_cpu(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    把 env.reset() 返回的 dict 统一成 CPU Tensor：
      - policy -> float32
      - is_* / failure -> bool
      - 其他 -> 原则上 float32
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        t = _to_cpu_tensor(v)
        if k == "policy":
            out[k] = t.to(dtype=torch.float32)
        elif k.startswith("is_") or k in ("failure",):
            # 兼容 int32 / uint8
            out[k] = (t != 0).to(dtype=torch.bool)
        else:
            # 例如 image / depth 等（你现在注释掉了，但保留兼容）
            if torch.is_floating_point(t):
                out[k] = t.to(dtype=torch.float32)
            else:
                out[k] = t.to(dtype=torch.float32)
    return out

@torch.no_grad()
def collect_init_obs_pool_from_env(env, pool_size: int = 4096) -> Dict[str, torch.Tensor]:
    """
    从已创建的真实 env 采集 reset 初始观测池（CPU tensors）。
    关键：避免重复 dreamer.make_env / 重复启动 Isaac Sim。
    """
    def _extract_obs(reset_out):
        if isinstance(reset_out, tuple) and len(reset_out) >= 1:
            return reset_out[0]
        return reset_out

    def _to_cpu_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu()
        return torch.as_tensor(np.asarray(x)).cpu()

    def _standardize_obs_dict_cpu(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            t = _to_cpu_tensor(v)
            if k == "policy":
                out[k] = t.to(dtype=torch.float32)
            elif k.startswith("is_") or k in ("failure",):
                out[k] = (t != 0).to(dtype=torch.bool)
            else:
                out[k] = t.to(dtype=torch.float32)
        return out

    # 预热一次拿 keys
    obs0 = _standardize_obs_dict_cpu(_extract_obs(env.reset()))
    keys = list(obs0.keys())
    bucket: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

    # env.num_envs 通常可用；否则用 policy.shape[0]
    seed_envs = int(obs0["policy"].shape[0])
    num_batches = int(np.ceil(pool_size / seed_envs))

    for _ in range(num_batches):
        obs = _standardize_obs_dict_cpu(_extract_obs(env.reset()))
        for k in keys:
            bucket[k].append(obs[k])

    pool: Dict[str, torch.Tensor] = {}
    for k in keys:
        cat = torch.cat(bucket[k], dim=0)
        pool[k] = cat[:pool_size].contiguous()

    # flags 统一化（posterior init 更稳）
    if "is_first" in pool:
        pool["is_first"] = torch.ones((pool["policy"].shape[0],), dtype=torch.bool)
    if "is_last" in pool:
        pool["is_last"] = torch.zeros((pool["policy"].shape[0],), dtype=torch.bool)
    if "is_terminal" in pool:
        pool["is_terminal"] = torch.zeros((pool["policy"].shape[0],), dtype=torch.bool)
    if "failure" in pool:
        pool["failure"] = torch.zeros((pool["policy"].shape[0],), dtype=torch.bool)

    return pool



# -------------------------
# World Model Imagination VecEnv (posterior_pool init + residual)
# -------------------------
class WorldModelImaginationVecEnv:
    def __init__(
        self,
        agent,
        act_low: np.ndarray,
        act_high: np.ndarray,
        num_envs: int,
        horizon: int = 15,
        device: str = "cuda",
        init_mode: str = "prior",               # "prior" | "posterior_pool"
        init_obs_pool: Dict[str, torch.Tensor] | None = None,
        disag_action_cond: bool = True,
        disag_log: bool = False,
        reward_reduce: str = "mean",
        clip_reward: float | None = None,
        residual_scale: float = 1.0,
    ):
        self.agent = agent
        self.wm = agent._wm
        self.ensemble = getattr(agent, "_disag_ensemble", None)
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.init_mode = init_mode
        self.init_obs_pool = init_obs_pool

        self.disag_action_cond = bool(disag_action_cond)
        self.disag_log = bool(disag_log)
        self.reward_reduce = reward_reduce
        self.clip_reward = clip_reward
        self.residual_scale = float(residual_scale)

        self.act_low = torch.as_tensor(act_low, device=self.device, dtype=torch.float32).view(1, -1)
        self.act_high = torch.as_tensor(act_high, device=self.device, dtype=torch.float32).view(1, -1)
        self.act_dim = self.act_low.shape[-1]

        self._latent = None
        self._t = None

        if self.init_mode == "posterior_pool" and self.init_obs_pool is None:
            raise ValueError("init_mode='posterior_pool' 时必须传入 init_obs_pool（真实 reset 观测池）")

        # 取 Dreamer 的 task actor 作为 base policy（冻结）
        self.base_policy = None
        if hasattr(self.agent, "_task_behavior"):
            tb = self.agent._task_behavior
        elif hasattr(self.agent, "task_behavior"):
            tb = self.agent.task_behavior
        else:
            tb = None

        if tb is not None:
            if hasattr(tb, "actor"):
                self.base_policy = tb.actor
                tb.requires_grad_(False)
                tb.eval()
            else:
                raise AttributeError("task_behavior 中找不到 actor；无法构造 base policy 用于残差")

    @torch.no_grad()
    def _intrinsic_reward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.ensemble is None:
            return torch.zeros((feat.shape[0],), device=self.device, dtype=torch.float32)

        inputs = torch.cat([feat, action], dim=-1) if self.disag_action_cond else feat
        div = self.ensemble.intrinsic_reward_penn(inputs)

        if div.ndim == 2:
            r = div.sum(dim=-1) if self.reward_reduce == "sum" else div.mean(dim=-1)
        else:
            r = div

        if self.disag_log:
            r = torch.log1p(r.clamp_min(0.0))

        if self.clip_reward is not None:
            r = torch.clamp(r, -float(self.clip_reward), float(self.clip_reward))
        return r.float()

    @torch.no_grad()
    def _init_latent_prior(self, B: int):
        if hasattr(self.wm.dynamics, "initial"):
            return self.wm.dynamics.initial(B)
        raise AttributeError("wm.dynamics 缺少 initial(B) 接口")

    @torch.no_grad()
    def _sample_init_obs_from_pool(self, B: int) -> Dict[str, torch.Tensor]:
        # pool 在 CPU，按 index 取出后再搬到 device
        pool = self.init_obs_pool
        assert pool is not None
        P = int(pool["policy"].shape[0])
        idx = torch.randint(0, P, (B,), device="cpu", dtype=torch.long)

        obs: Dict[str, torch.Tensor] = {}
        for k, v in pool.items():
            # v: (P, ...)
            picked = v.index_select(0, idx)
            if k == "policy":
                obs[k] = picked.to(self.device, dtype=torch.float32)
            elif k.startswith("is_") or k in ("failure",):
                obs[k] = picked.to(self.device, dtype=torch.bool)
            else:
                obs[k] = picked.to(self.device, dtype=torch.float32)
        return obs

    @torch.no_grad()
    def _init_latent_from_obs(self, obs: Dict[str, torch.Tensor]):
        """
        用真实观测 posterior 初始化 latent。
        兼容不同 dreamerv3_torch 版本的 dynamics 接口（obs_step / observe）。
        """
        # preprocess（如果存在）
        if hasattr(self.wm, "preprocess"):
            obs_pp = self.wm.preprocess(obs)
        else:
            obs_pp = obs

        # encoder
        if not hasattr(self.wm, "encoder"):
            raise AttributeError("wm 缺少 encoder，无法 posterior 初始化")
        embed = self.wm.encoder(obs_pp)
        if isinstance(embed, dict) and "embed" in embed:
            embed = embed["embed"]

        B = int(embed.shape[0])
        latent0 = self._init_latent_prior(B)
        action0 = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)
        is_first = torch.ones((B,), device=self.device, dtype=torch.bool)

        dyn = self.wm.dynamics

        # 常见：obs_step(state, action, embed, is_first) -> (post, prior)
        if hasattr(dyn, "obs_step"):
            try:
                post, _ = dyn.obs_step(latent0, action0, embed, is_first)
            except TypeError:
                post, _ = dyn.obs_step(latent0, action0, embed)
            return post

        # 另一种：observe(embed, action, is_first, state) -> (post, prior)
        if hasattr(dyn, "observe"):
            try:
                post, _ = dyn.observe(embed, action0, is_first, latent0)
            except TypeError:
                post, _ = dyn.observe(embed, action0, is_first)
            return post

        raise AttributeError("wm.dynamics 缺少 obs_step/observe；无法 posterior 初始化")

    @torch.no_grad()
    def reset(self):
        self.wm.eval()
        if self.ensemble: self.ensemble.eval()
        if hasattr(self.agent, "_task_behavior"): self.agent._task_behavior.eval()
        if hasattr(self.agent, "task_behavior"): self.agent.task_behavior.eval()

        if self.init_mode == "prior":
            self._latent = self._init_latent_prior(self.num_envs)
        elif self.init_mode == "posterior_pool":
            obs0 = self._sample_init_obs_from_pool(self.num_envs)
            self._latent = self._init_latent_from_obs(obs0)
        else:
            raise NotImplementedError(f"Unknown init_mode: {self.init_mode}")

        self._t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        return self.wm.dynamics.get_feat(self._latent).detach()

    @torch.no_grad()
    def step(self, action_residual: torch.Tensor):
        action_residual = action_residual.to(self.device).float()

        # 1) 当前状态特征（给 base actor 与 intrinsic reward）
        feat = self.wm.dynamics.get_feat(self._latent).detach()

        # 2) base action
        if self.base_policy is None:
            raise RuntimeError("base_policy 未初始化；无法做残差动作合成")
        actor_out = self.base_policy(feat)
        if hasattr(actor_out, "mode"):
            action_base = actor_out.mode()
        else:
            action_base = actor_out
        action_base = action_base.detach()

        # 3) 合成最终动作
        action_final = action_base + self.residual_scale * action_residual

        # 4) clip
        action_final = torch.max(torch.min(action_final, self.act_high), self.act_low)

        # 5) intrinsic reward（不确定性）
        reward = self._intrinsic_reward(feat, action_final)

        # 6) latent 步进（imagination）
        if hasattr(self.wm.dynamics, "img_step"):
            self._latent = self.wm.dynamics.img_step(self._latent, action_final)
        else:
            self._latent = self.wm.dynamics.imagine_step(self._latent, action_final)

        self._t += 1
        done = (self._t >= self.horizon)
        obs_next = self.wm.dynamics.get_feat(self._latent).detach()
        return obs_next, reward, done

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor):
        env_ids = env_ids.to(self.device).long()
        if env_ids.numel() == 0:
            feat_dim = int(self.wm.dynamics.get_feat(self._latent).shape[-1])
            return torch.empty((0, feat_dim), device=self.device)

        B = int(env_ids.numel())

        if self.init_mode == "prior":
            latent_new = self._init_latent_prior(B)
        elif self.init_mode == "posterior_pool":
            obs0 = self._sample_init_obs_from_pool(B)
            latent_new = self._init_latent_from_obs(obs0)
        else:
            raise NotImplementedError(f"Unknown init_mode: {self.init_mode}")

        # 递归赋值：把 latent_new（batch=B）写入 self._latent 的 env_ids 位置
        def _assign(dst, src):
            if isinstance(dst, dict):
                for k in dst.keys():
                    _assign(dst[k], src[k])
            elif torch.is_tensor(dst):
                dst[env_ids] = src
            elif isinstance(dst, (list, tuple)):
                for i in range(len(dst)):
                    _assign(dst[i], src[i])
            else:
                raise TypeError(f"Unsupported latent leaf type: {type(dst)}")

        _assign(self._latent, latent_new)
        self._t[env_ids] = 0

        obs = self.wm.dynamics.get_feat(self._latent).detach()
        return obs[env_ids]


# -------------------------
# Wrapper: 适配 rsl_rl VecEnv 接口
# -------------------------
class RslRlVecEnvFromWM(VecEnv):
    def __init__(self, env_model, clip_actions: float = 1.0):
        self.env_model = env_model
        self.device = env_model.device
        self.num_envs = env_model.num_envs
        self.num_actions = env_model.act_dim
        self.max_episode_length = env_model.horizon
        self.clip_actions = float(clip_actions)

        obs = self.env_model.reset()
        self.num_obs = int(obs.shape[-1])
        self.num_privileged_obs = self.num_obs

        self.obs_buf = obs
        self.privileged_obs_buf = obs

        self.extras = {"observations": {}}

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        obs = self.env_model.reset()
        self.obs_buf = obs
        self.privileged_obs_buf = obs

        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()

        return self.obs_buf, self.extras

    def step(self, actions):
        # actions 是 PPO 输出的残差
        actions = actions.to(self.device).float()
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_next, rew, done = self.env_model.step(actions)

        rew = rew.view(self.num_envs).to(torch.float32)
        done = done.view(self.num_envs).to(torch.bool)

        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            obs_reset = self.env_model.reset_idx(done_ids)
            obs_next[done_ids] = obs_reset
            self.episode_length_buf[done_ids] = 0

        self.obs_buf = obs_next
        self.privileged_obs_buf = obs_next
        self.rew_buf = rew
        self.reset_buf = done
        self.episode_length_buf += 1

        infos = {"observations": {}, "time_outs": done.clone()}
        return self.obs_buf, self.rew_buf, self.reset_buf, infos


# -------------------------
# Config Builder
# -------------------------
def build_train_cfg(args):
    return {
        "seed": 42,
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": args.num_steps_per_env,
            "max_iterations": args.max_iterations,
            "save_interval": args.save_interval,
            "experiment_name": args.experiment_name,
            "run_name": args.run_name,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "max_grad_norm": 1.0,
            "desired_kl": 0.01,
            "use_clipped_value_loss": True,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        "class_name": "OnPolicyRunner",
        "device": args.device,
        "num_steps_per_env": args.num_steps_per_env,
        "save_interval": 30,
        "empirical_normalization": True,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num_steps_per_env", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--logdir", type=str, default="logs_rslrl_wm")
    parser.add_argument("--run_name", type=str, default="test1_residual")
    parser.add_argument("--experiment_name", type=str, default="wm_disag_ppo")

    parser.add_argument("--residual_scale", type=float, default=1.0, help="Scale for the residual action")

    # 新增：posterior pool 参数
    parser.add_argument("--init_mode", type=str, default="posterior_pool", choices=["prior", "posterior_pool"])
    parser.add_argument("--init_pool_size", type=int, default=4096, help="Number of real reset observations in the pool")
    parser.add_argument("--init_seed_envs", type=int, default=32, help="Parallel envs used to collect the pool")

    args = parser.parse_args()

    cli_overrides = {"task": args.task, "device": args.device}
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # 1) 创建一次真实环境：既用来拿 specs，也用来采集 posterior init pool
    print("Creating ONE real env for specs + init_obs_pool ...")
    env_real = dreamer.make_env(config, num_envs=args.init_seed_envs)

    acts = env_real.single_action_space
    obs_space = env_real.single_observation_space

    # 动作空间标准化到 [-1, 1]
    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # 2) 采集真实 reset 观测池（不再内部 make_env）
    init_obs_pool = None
    if args.init_mode == "posterior_pool":
        print(f"Collecting init_obs_pool from the SAME real env: pool_size={args.init_pool_size}")
        init_obs_pool = collect_init_obs_pool_from_env(env_real, pool_size=args.init_pool_size)
        print("Init obs pool collected. policy shape:", tuple(init_obs_pool["policy"].shape))

    # 3) 关闭真实 env（只关闭一次）
    try:
        env_real.close()
        print("Real env closed.")
    except Exception as e:
        print(f"Warning closing env: {e}")


    # 3) 加载 Dreamer
    logger = NullLogger()
    agent = dreamer.Dreamer(
        obs_space,
        acts,
        config,
        logger,
        dataset=None,
    ).to(args.device)
    agent.requires_grad_(False)

    print(f"Loading checkpoint from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(ckpt, strict=False)

    # 4) 构建 WM 环境（posterior_pool init + residual）
    env_model = WorldModelImaginationVecEnv(
        agent=agent,
        act_low=acts.low,
        act_high=acts.high,
        num_envs=args.num_envs,
        horizon=args.horizon,
        device=args.device,
        init_mode=args.init_mode,
        init_obs_pool=init_obs_pool,
        disag_action_cond=getattr(config, "disag_action_cond", True),
        disag_log=getattr(config, "disag_log", False),
        reward_reduce="mean",
        clip_reward=None,
        residual_scale=args.residual_scale,
    )

    # 5) 构建 RSL-RL Wrapper
    rsl_env = RslRlVecEnvFromWM(env_model, clip_actions=1.0)

    # 6) 配置 PPO 并训练
    train_cfg = build_train_cfg(args)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else ts
    log_dir = str(pathlib.Path(args.logdir) / args.experiment_name / run_name)

    print(f"Starting PPO (Residual) training in {log_dir}...")
    runner = OnPolicyRunner(rsl_env, train_cfg, log_dir=log_dir, device=args.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
