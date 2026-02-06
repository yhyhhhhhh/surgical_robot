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

import my_ur3_project.tasks  # noqa: F401

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner


# -----------------------
# rsl_rl: 用于加载 PPO residual 的 Dummy VecEnv
# -----------------------
class DummyVecEnv(VecEnv):
    def __init__(self, num_envs: int, num_obs: int, num_actions: int, device: str):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.num_obs = int(num_obs)
        self.num_privileged_obs = int(num_obs)
        self.num_actions = int(num_actions)
        self.max_episode_length = 1

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = self.obs_buf
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.extras = {"observations": {}}

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        self.obs_buf.zero_()
        self.privileged_obs_buf.zero_()
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.episode_length_buf.zero_()
        return self.obs_buf, self.extras

    def step(self, actions):
        # Dummy：只为满足 runner 初始化/接口，不用于真实 step
        self.obs_buf.zero_()
        self.privileged_obs_buf.zero_()
        self.rew_buf.zero_()
        self.reset_buf.fill_(True)
        infos = {"observations": {}, "time_outs": self.reset_buf.clone()}
        return self.obs_buf, self.rew_buf, self.reset_buf, infos


def build_train_cfg_for_loading(device: str, num_steps_per_env: int = 32):
    # 与你训练时一致的 policy/algorithm 配置（只用于构建网络结构与加载 checkpoint）
    return {
        "seed": 42,
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": num_steps_per_env,
            "max_iterations": 1,
            "save_interval": 999999,
            "experiment_name": "eval",
            "run_name": "eval",
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
        "device": device,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 999999,
        "empirical_normalization": True,
    }


# -----------------------
# Dreamer + PPO Residual 组合策略（用于 tools.simulate_vecenv）
# -----------------------
class ResidualEvalPolicy:
    def __init__(
        self,
        agent,
        ppo_infer_policy,
        residual_scale: float,
        clip_actions: float,
        act_low: np.ndarray,
        act_high: np.ndarray,
        device: str,
        baseline_only: bool = False,
    ):
        self.agent = agent
        self.ppo = ppo_infer_policy
        self.residual_scale = float(residual_scale)
        self.clip_actions = float(clip_actions)
        self.device = torch.device(device)
        self.baseline_only = bool(baseline_only)

        self.act_low_t = torch.as_tensor(act_low, device=self.device, dtype=torch.float32).view(1, -1)
        self.act_high_t = torch.as_tensor(act_high, device=self.device, dtype=torch.float32).view(1, -1)
        self.ensemble = getattr(self.agent, "_disag_ensemble", None)

        # 这两个最好从 config 传进来；没有就先用默认
        self.disag_action_cond = True     # 训练时你一般是 True
        self.disag_log = False            # 训练时多半 False
        self.disag_reduce = "mean"        # "mean" 或 "sum"

    def _as_torch(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        if isinstance(x, (list, tuple)):
            return torch.as_tensor(x, device=self.device)
        if isinstance(x, dict):
            if "action" in x and torch.is_tensor(x["action"]):
                return x["action"].to(self.device)
            raise TypeError(f"Action dict missing tensor under key 'action': keys={list(x.keys())}")
        raise TypeError(f"Unsupported action type: {type(x)}")

    def _to_ref_type(self, x_t: torch.Tensor, ref):
        if torch.is_tensor(ref):
            return x_t
        if isinstance(ref, np.ndarray):
            return x_t.detach().cpu().numpy()
        if isinstance(ref, (list, tuple)):
            return x_t.detach().cpu().numpy()
        return x_t

    def _extract_feat(self, state):
        """
        兼容 state 是 dict 或 tuple(list) 的情况。
        你的 state 目前是 (rssm_dict, prev_action_tensor)。
        """
        dyn = self.agent._wm.dynamics

        # 情况 A：state 本身就是 RSSM dict
        if isinstance(state, dict):
            return dyn.get_feat(state)

        # 情况 B：state 是 tuple/list，优先找其中的 RSSM dict（含 stoch/deter）
        if isinstance(state, (tuple, list)):
            for elem in state:
                if isinstance(elem, dict) and ("stoch" in elem) and ("deter" in elem):
                    return dyn.get_feat(elem)
            # 次优：如果第 0 个就是 dict，也尝试
            if len(state) > 0 and isinstance(state[0], dict):
                return dyn.get_feat(state[0])

        raise RuntimeError(
            f"Cannot extract feat: unsupported state type/structure: {type(state)}. "
            f"Example state={state}"
        )

    def _inject_action_into_state(self, state, action_final_t: torch.Tensor):
        """
        把“实际执行动作”写回 carry/state。
        - state 是 dict：尝试覆盖常见 key
        - state 是 (rssm_dict, prev_action)：返回 (rssm_dict, action_final)
        """
        a = action_final_t.detach()

        # dict: 覆盖常见字段
        if isinstance(state, dict):
            for k in ["action", "prev_action", "last_action"]:
                if k in state:
                    state[k] = a
            return state

        # tuple/list: 你的情况通常是 (rssm_dict, prev_action_tensor)
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict) and torch.is_tensor(state[1]):
            return (state[0], a)

        if isinstance(state, (tuple, list)):
            # 更泛化：找到第一个“像 action”的 tensor 替换（按最后一维匹配动作维度）
            out = list(state)
            for i, elem in enumerate(out):
                if torch.is_tensor(elem) and elem.shape[-1] == a.shape[-1]:
                    out[i] = a
                    return tuple(out) if isinstance(state, tuple) else out
            return tuple(out) if isinstance(state, tuple) else out

        return state


    def __call__(self, *args, **kwargs):
        """
        保持与原 agent callable 接口兼容：tools.simulate_vecenv 会用同样方式调用 policy。
        """
        kwargs["training"] = False

        out = self.agent(*args, **kwargs)

        # 解析 Dreamer 输出：尽量兼容 (action, state) 或 dict 等
        if isinstance(out, tuple) and len(out) == 2:
            action_base, state = out
            pack = "tuple2"
        elif isinstance(out, dict) and "action" in out:
            action_base = out["action"]
            state = out.get("state", out.get("carry", None))
            pack = "dict_action"
        else:
            # 若你的 Dreamer 直接返回 action（不带 state），组合策略无法保持 filtering 一致性
            raise RuntimeError(
                "Dreamer agent output format not supported for residual eval. "
                "需要 agent 返回 (action, state) 或 dict{'action':..., 'state'/ 'carry':...}。"
            )

        # base action -> torch
        action_base_t = self._as_torch(action_base).float()

        # 如果没有 state，无法抽取 feat，也无法正确把 last_action 写回
        if state is None:
            raise RuntimeError("Dreamer did not return state/carry; cannot run residual policy correctly.")

        # feature for PPO residual
        feat = self._extract_feat(state).detach()
        feat = feat.to(self.device).float()

        # residual action
        if self.baseline_only:
            action_res_t = torch.zeros_like(action_base_t)
        else:
            res = self.ppo(feat)
            # ppo inference policy 通常直接返回 tensor；做一个兜底解析
            if torch.is_tensor(res):
                action_res_t = res
            elif isinstance(res, (tuple, list)) and len(res) > 0 and torch.is_tensor(res[0]):
                action_res_t = res[0]
            elif isinstance(res, dict):
                # 尝试取第一个 tensor
                v = None
                for vv in res.values():
                    if torch.is_tensor(vv):
                        v = vv
                        break
                if v is None:
                    raise RuntimeError("Cannot parse PPO policy output dict into tensor.")
                action_res_t = v
            else:
                raise RuntimeError(f"Cannot parse PPO policy output type: {type(res)}")

            action_res_t = action_res_t.to(self.device).float()
            action_res_t = torch.clamp(action_res_t, -self.clip_actions, self.clip_actions)

        # combine + clip to action bounds
        action_final_t = action_base_t + self.residual_scale * action_res_t
        action_final_t = torch.max(torch.min(action_final_t, self.act_high_t), self.act_low_t)

        # inject executed action into state (very important)
        state = self._inject_action_into_state(state, action_final_t)

        # 1) Dreamer forward
        base_out, state = self.agent(*args, **kwargs)

        # base_out 必须是 dict（你的就是 dict）
        if not isinstance(base_out, dict) or "action" not in base_out:
            raise RuntimeError(f"Dreamer base_out must be dict with key 'action'. Got: {type(base_out)}")

        action_base_t = base_out["action"].to(self.device).float()

        # 2) feature（你前面已修复 _extract_feat 支持 tuple）
        feat = self._extract_feat(state).detach().to(self.device).float()

        # 3) residual
        if self.baseline_only:
            action_res_t = torch.zeros_like(action_base_t)
        else:
            res = self.ppo(feat)
            if not torch.is_tensor(res):
                res = res[0]
            action_res_t = res.to(self.device).float()
            action_res_t = torch.clamp(action_res_t, -self.clip_actions, self.clip_actions)
        print("Residual action sample:", action_res_t.cpu().numpy())
        # 4) combine + clip
        action_final_t = action_base_t + self.residual_scale * action_res_t
        action_final_t = torch.max(torch.min(action_final_t, self.act_high_t), self.act_low_t)

        # 5) 用 final action 重算 disagreement（这一步就是你要的）
        new_disag = self._compute_disagreement(feat, action_final_t)

        # 6) 写回 carry/state 的 prev_action（你前面已修复支持 tuple）
        state = self._inject_action_into_state(state, action_final_t)

        # 7) 返回 dict：覆盖 action 和 disagreement
        out = dict(base_out)               # 保留其他字段（比如 logprob）
        out["action"] = action_final_t.detach()
        out["disagreement"] = new_disag.detach()

        # 注意：out["logprob"] 此时仍对应 base action，不再严格正确。
        # 评估一般不需要 logprob；若你担心误导，可选择 del out["logprob"]（看 tools.simulate_vecenv 是否依赖它）
        # if "logprob" in out: del out["logprob"]

        return out, state



    
    @torch.no_grad()
    def _compute_disagreement(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        返回 shape [B, 1]，与你看到的 tensor([[...]] ) 对齐
        """
        if self.ensemble is None:
            return torch.zeros((feat.shape[0], 1), device=feat.device, dtype=torch.float32)

        x = torch.cat([feat, action], dim=-1) if self.disag_action_cond else feat
        div = self.ensemble.intrinsic_reward_penn(x)

        # div 可能是 [B] 或 [B, D]
        if div.ndim == 2:
            r = div.mean(dim=-1) if self.disag_reduce == "mean" else div.sum(dim=-1)
        else:
            r = div

        if self.disag_log:
            r = torch.log1p(r.clamp_min(0.0))

        return r.view(-1, 1).float()

def main(config, ppo_ckpt: str, residual_scale: float, clip_actions: float, baseline_only: bool):
    tools.set_seed_everywhere(config.seed)
    config.evaldir = pathlib.Path(config.evaldir).expanduser()
    config.evaldir.mkdir(parents=True, exist_ok=True)
    print("Eval dir:", config.evaldir)

    # -----------------------
    # 创建仿真环境（真实 step）
    # -----------------------
    envs = dreamer.make_env(config, num_envs=config.envs)
    acts = envs.single_action_space

    # Action normalization（必须与训练一致）
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high) * 1

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # -----------------------
    # 创建空 replay（接口完整）
    # -----------------------
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    eval_dataset = dreamer.make_dataset(eval_eps, config)
    logger = tools.Logger(config.evaldir, step=0)

    # -----------------------
    # 创建 Dreamer agent
    # -----------------------
    agent = dreamer.Dreamer(
        envs.single_observation_space,
        acts,
        config,
        logger,
        eval_dataset,
    ).to(config.device)
    agent.requires_grad_(False)

    # 加载 Dreamer checkpoint
    checkpoint = torch.load(config.model_path, map_location=config.device)
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(checkpoint, strict=False)

    print(f"Loaded Dreamer model from {config.model_path}")

    # -----------------------
    # 先用 Dreamer 跑一步，探测 PPO 需要的 feat_dim
    # -----------------------
    # 这里通过调用一次 policy（不真正 step）来拿到 state->feat 维度。
    # 注意：simulate_vecenv 内部会 reset，这里仅做维度探测。
    dummy_obs = envs.reset()
    # tools / env wrapper 可能返回 dict/tuple，这里直接交给 agent 处理
    with torch.no_grad():
        latent0 = agent._wm.dynamics.initial(config.envs)
        feat0 = agent._wm.dynamics.get_feat(latent0)
        feat_dim = int(feat0.shape[-1])

    act_dim = int(config.num_actions)
    print(f"[Probe] PPO residual obs_dim(feat_dim)={feat_dim}, act_dim={act_dim}")

    # -----------------------
    # 用 DummyVecEnv 构造 rsl_rl runner 并加载 PPO checkpoint
    # -----------------------
    dummy_env = DummyVecEnv(num_envs=config.envs, num_obs=feat_dim, num_actions=act_dim, device=config.device)
    train_cfg = build_train_cfg_for_loading(device=config.device, num_steps_per_env=32)

    runner = OnPolicyRunner(dummy_env, train_cfg, log_dir=None, device=config.device)

    print(f"Loading PPO residual checkpoint from: {ppo_ckpt}")
    runner.load(ppo_ckpt)

    # 推理策略（包含 normalization 等 runner 内部逻辑）
    ppo_policy = runner.get_inference_policy(device=config.device)

    # -----------------------
    # 组合策略：Dreamer base + PPO residual
    # -----------------------
    eval_policy = ResidualEvalPolicy(
        agent=agent,
        ppo_infer_policy=ppo_policy,
        residual_scale=residual_scale,
        clip_actions=clip_actions,
        act_low=acts.low,
        act_high=acts.high,
        device=config.device,
        baseline_only=baseline_only,
    )

    # -----------------------
    # 跑评估（真实仿真环境）
    # -----------------------
    with torch.no_grad():
        tools.simulate_vecenv(
            eval_policy,
            envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
            save_success=True,
        )

    logger.write()

    try:
        envs.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")

    parser.add_argument("--ppo_ckpt", type=str, default="/home/yhy/IsaacLabExtensionTemplate/logs_rslrl_wm/wm_disag_ppo/test1_residual/model_2999.pt", help="rsl_rl PPO residual checkpoint (.pt)")
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--clip_actions", type=float, default=1.0)
    parser.add_argument("--baseline_only", action="store_true", default=False)

    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--evaldir", type=str, default="latent_safety/log/dreamerv3/1225/eval_eps")
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)

    args, remaining = parser.parse_known_args()
    args.enable_cameras = True
    args.rendering_mode = "quality"
    args.headless = False

    # -----------------------
    # 读取 configs.yaml（与你原脚本一致）
    # -----------------------
    import ruamel.yaml as yaml

    yaml_parser = yaml.YAML(typ='safe', pure=True)
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

    main(
        config=config,
        ppo_ckpt=args.ppo_ckpt,
        residual_scale=args.residual_scale,
        clip_actions=args.clip_actions,
        baseline_only=args.baseline_only,
    )
