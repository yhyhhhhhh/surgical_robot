from __future__ import annotations

import argparse
import pathlib
import numpy as np
import ruamel.yaml as yaml
import torch
import sys

# 让脚本能找到 dreamerv3_torch 相关模块
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools

import my_ur3_project.tasks  # noqa: F401


# -------------------------
# 1) 配置加载：沿用你训练脚本的 configs.yaml 合并逻辑
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

    # 关键：相对路径以“脚本所在目录”解析，避免 cwd 不一致导致找不到文件
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


# -------------------------
# 2) 一个最小 Logger（Dreamer 初始化需要 logger，但这里不做训练）
# -------------------------
class NullLogger:
    def __init__(self):
        self.step = 0

    def scalar(self, *args, **kwargs):
        pass

    def video(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass


# -------------------------
# 3) WorldModelImaginationVecEnv（无 episodes 版本）
#    - prior 初始化：只用 wm.dynamics.initial
#    - bootstrap 初始化：从真实 env 临时采一段序列（不保存到磁盘）
# -------------------------
class WorldModelImaginationVecEnv:
    """
    测试用 Env-Model：
    - obs = wm.dynamics.get_feat(latent)
    - step: latent <- img_step(latent, action)
    - reward: ensemble disagreement（intrinsic）
    - done: 固定 horizon 截断（只为测试链路）
    """

    def __init__(
        self,
        agent,                     # dreamer.Dreamer（已加载权重）
        env_real,                  # 真实仿真 env（bootstrap 初始化用；prior 模式可传 None）
        act_low: np.ndarray,
        act_high: np.ndarray,
        num_envs: int,
        horizon: int = 15,
        device: str = "cuda",
        init_mode: str = "prior",          # "prior" or "bootstrap"
        bootstrap_steps: int = 8,          # bootstrap 序列长度（>=2 推荐）
        bootstrap_policy: str = "task",    # "task" or "random"
        disag_action_cond: bool = True,
        disag_log: bool = False,
        reward_reduce: str = "mean",       # div为向量时 mean/sum
        clip_reward: float | None = None,
        eps: float = 1e-8,
    ):
        self.agent = agent
        self.wm = agent._wm
        self.ensemble = getattr(agent, "_disag_ensemble", None)

        self.env_real = env_real
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = torch.device(device)

        self.init_mode = init_mode
        self.bootstrap_steps = int(bootstrap_steps)
        self.bootstrap_policy = bootstrap_policy

        self.disag_action_cond = bool(disag_action_cond)
        self.disag_log = bool(disag_log)
        self.reward_reduce = reward_reduce
        self.clip_reward = clip_reward
        self.eps = eps

        self.act_low = torch.as_tensor(act_low, device=self.device, dtype=torch.float32).view(1, -1)
        self.act_high = torch.as_tensor(act_high, device=self.device, dtype=torch.float32).view(1, -1)
        self.act_dim = self.act_low.shape[-1]

        self._latent = None
        self._t = None

    @torch.no_grad()
    def _intrinsic_reward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """ensemble disagreement -> 标量 reward (N,)"""
        if self.ensemble is None:
            return torch.zeros((feat.shape[0],), device=self.device, dtype=torch.float32)

        if self.disag_action_cond:
            inputs = torch.cat([feat, action], dim=-1)
        else:
            inputs = feat

        div = self.ensemble.intrinsic_reward_penn(inputs)  # (N,) 或 (N,D)

        if div.ndim == 2:
            if self.reward_reduce == "sum":
                r = div.sum(dim=-1)
            else:
                r = div.mean(dim=-1)
        else:
            r = div

        if self.clip_reward is not None:
            r = torch.clamp(r, -float(self.clip_reward), float(self.clip_reward))

        return r.float()

    @torch.no_grad()
    def _init_latent_prior(self):
        """prior 初始化：直接从 dynamics.initial 开始"""
        if hasattr(self.wm.dynamics, "initial"):
            latent = self.wm.dynamics.initial(self.num_envs)
        else:
            raise AttributeError("wm.dynamics 没有 initial(B) 接口，无法做 prior 初始化。")
        return latent

    @torch.no_grad()
    def _bootstrap_sequence_from_real_env(self):
        """
        从真实 env 临时采集一段序列（不落盘）：
        - obs_0..obs_{K-1}
        - action_0..action_{K-2}
        并用 obs_step 得到 posterior latent（最后一步）
        """
        if self.env_real is None:
            raise ValueError("bootstrap 初始化需要传入 env_real，但当前为 None。")

        K = max(self.bootstrap_steps, 2)

        # reset 得到 obs_0
        obs = self.env_real.reset()
        # 兼容 IsaacLab/Gymnasium reset 返回 (obs, info)
        if isinstance(obs, (tuple, list)) and len(obs) == 2:
            obs = obs[0]

        # 存 obs 序列：这里只用 policy（如你有 image 也可扩展）
        if "policy" not in obs:
            raise KeyError("env_real.reset() 的 obs dict 中找不到 'policy'，无法 bootstrap。")

        obs_policy_seq = [obs["policy"].to(self.device)]  # list of (N, D)
        is_first_seq = [obs.get("is_first", torch.ones(self.num_envs, device=self.device, dtype=torch.int32)).to(self.device).bool()]

        action_seq = []  # (K-1) 个动作

        # 采集 K-1 次 step，得到 obs_1..obs_{K-1}
        latent_state = None  # Dreamer __call__ 的 state（latent, action）
        for t in range(K - 1):
            if self.bootstrap_policy == "random":
                u = torch.rand((self.num_envs, self.act_dim), device=self.device)
                action = self.act_low + (self.act_high - self.act_low) * u
                action_dict = {"action": action}
            else:
                # 使用当前 Dreamer policy 产生动作（training=False 不更新参数）
                action_dict, latent_state = self.agent(obs, reset=None, state=latent_state, training=False)

            action_seq.append(action_dict["action"].to(self.device))

            step_out = self.env_real.step(action_dict)
            # 兼容 wrapper step 返回 (obs, reward, done, info) 或 gymnasium (obs,reward,terminated,truncated,info)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                obs, _, _, _ = step_out
            elif isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, _, _, _, _ = step_out
            else:
                raise RuntimeError("env_real.step() 返回格式不符合预期。")

            if "policy" not in obs:
                raise KeyError("env_real.step() 的 obs dict 中找不到 'policy'。")

            obs_policy_seq.append(obs["policy"].to(self.device))
            is_first_seq.append(obs.get("is_first", torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)).to(self.device).bool())

        # 组装 tensor
        obs_policy = torch.stack(obs_policy_seq, dim=1)   # (N, K, D)
        is_first = torch.stack(is_first_seq, dim=1)       # (N, K)
        actions = torch.stack(action_seq, dim=1)          # (N, K-1, A)

        # 用 obs_step 构建 posterior latent（对应 Dreamer 的时间对齐：obs_t 使用 action_{t-1}）
        if hasattr(self.wm.dynamics, "initial"):
            latent = self.wm.dynamics.initial(self.num_envs)
        else:
            latent = None

        prev_action = torch.zeros((self.num_envs, self.act_dim), device=self.device, dtype=torch.float32)

        for t in range(K):
            # 构造 obs_t dict（至少 policy）
            obs_t = {
                "policy": obs_policy[:, t],
                "is_first": is_first[:, t].to(torch.int32),
                "is_terminal": torch.zeros_like(is_first[:, t], dtype=torch.int32),
                "failure": torch.zeros_like(is_first[:, t], dtype=torch.int32),
            }
            obs_t = self.wm.preprocess(obs_t)
            embed = self.wm.encoder(obs_t)

            latent, _ = self.wm.dynamics.obs_step(latent, prev_action, embed, is_first[:, t])

            # 更新 prev_action：t=0 用零动作；t>=1 用 action_{t-1}
            if t >= 1:
                prev_action = actions[:, t - 1]

        return latent

    @torch.no_grad()
    def reset(self):
        """返回 obs: torch (N, feat_dim)"""
        # self.agent.eval()
        self.wm.eval()
        if self.ensemble is not None:
            self.ensemble.eval()

        if self.init_mode == "prior":
            latent = self._init_latent_prior()
        elif self.init_mode == "bootstrap":
            latent = self._bootstrap_sequence_from_real_env()
        else:
            raise ValueError(f"未知 init_mode={self.init_mode}，请用 prior/bootstrap")

        self._latent = latent
        self._t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        obs = self.wm.dynamics.get_feat(self._latent).detach()
        return obs

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """
        action: torch (N, act_dim)
        返回：
        - obs_next: (N, feat_dim)
        - reward: (N,)
        - done: (N,) 仅用 horizon 截断
        """
        action = action.to(self.device).float()
        action = torch.max(torch.min(action, self.act_high), self.act_low)

        feat = self.wm.dynamics.get_feat(self._latent).detach()
        reward = self._intrinsic_reward(feat, action)

        if hasattr(self.wm.dynamics, "img_step"):
            self._latent = self.wm.dynamics.img_step(self._latent, action)
        elif hasattr(self.wm.dynamics, "imagine_step"):
            self._latent = self.wm.dynamics.imagine_step(self._latent, action)
        else:
            raise AttributeError("找不到 dynamics.img_step / dynamics.imagine_step。")

        self._t += 1
        done = (self._t >= self.horizon)

        obs_next = self.wm.dynamics.get_feat(self._latent).detach()
        return obs_next, reward, done

    @torch.no_grad()
    def reset_idx(self, env_ids: torch.Tensor):
        """只重置部分环境，并返回这些环境的 obs (len(env_ids), feat_dim)"""
        env_ids = env_ids.to(self.device)
        if env_ids.numel() == 0:
            feat_dim = self.wm.dynamics.get_feat(self._latent).shape[-1]
            return torch.empty((0, feat_dim), device=self.device)

        # 1) 新 latent（用 prior，避免 bootstrap 的 env 交互开销）
        latent_new = self._init_latent_prior()

        # 2) 把 latent_new 的对应 env_ids 写回 self._latent
        def _assign(dst, src):
            if isinstance(dst, dict):
                for k in dst.keys():
                    _assign(dst[k], src[k])
            elif torch.is_tensor(dst):
                dst[env_ids] = src[env_ids]
            elif isinstance(dst, (list, tuple)):
                for i in range(len(dst)):
                    _assign(dst[i], src[i])
            else:
                raise TypeError(f"Unsupported latent type: {type(dst)}")

        _assign(self._latent, latent_new)

        # 3) 重置计时
        self._t[env_ids] = 0

        # 4) 返回这些 env 的 obs
        obs = self.wm.dynamics.get_feat(self._latent).detach()
        return obs[env_ids]

# -------------------------
# 4) 主程序：加载模型 -> 构建 Env-Model -> reset/step rollout
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # configs.yaml 加载
    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=None)

    # 环境与模型
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt", help="训练好的 latest.pt 或模型文件（含 agent_state_dict）")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")

    # Env-Model 初始化模式
    parser.add_argument("--init_mode", type=str, default="bootstrap", choices=["prior", "bootstrap"])
    parser.add_argument("--bootstrap_steps", type=int, default=8)
    parser.add_argument("--bootstrap_policy", type=str, default="task", choices=["task", "random"])

    args = parser.parse_args()

    # 1) 载入 config（需要用来创建 Dreamer / env）
    cli_overrides = {
        "task": args.task,
        "device": args.device,
    }
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # 2) 创建真实 env（bootstrap 模式需要；prior 模式也可用来拿 action space）
    env_real = dreamer.make_env(config, num_envs=args.num_envs)

    # 3) 动作空间归一化（与你训练脚本一致）
    acts = env_real.single_action_space
    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # 4) 创建 Dreamer agent（不训练，只为加载权重并拿 wm/ensemble）
    logger = NullLogger()
    dummy_dataset = None  # 不使用 dataset
    agent = dreamer.Dreamer(
        env_real.single_observation_space,
        acts,
        config,
        logger,
        dummy_dataset,
    ).to(args.device)
    agent.requires_grad_(requires_grad=False)
    # agent.eval()

    # 5) 加载 checkpoint
    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        # 兼容你直接保存了 state_dict
        agent.load_state_dict(ckpt, strict=False)

    # 6) 创建 Env-Model（prior 模式 env_real 可以不用，但保留不影响）
    env_model = WorldModelImaginationVecEnv(
        agent=agent,
        env_real=env_real if args.init_mode == "bootstrap" else None,
        act_low=acts.low,
        act_high=acts.high,
        num_envs=args.num_envs,
        horizon=args.horizon,
        device=args.device,
        init_mode=args.init_mode,
        bootstrap_steps=args.bootstrap_steps,
        bootstrap_policy=args.bootstrap_policy,
        disag_action_cond=getattr(config, "disag_action_cond", True),
        disag_log=getattr(config, "disag_log", False),
        reward_reduce="mean",
        clip_reward=None,
    )

    # 7) rollout 测试
    print(f"\n[Env-Model] init_mode={args.init_mode} reset() ...")
    obs = env_model.reset()
    print(f"[Env-Model] obs shape: {tuple(obs.shape)}")

    total_r = torch.zeros((args.num_envs,), device=args.device, dtype=torch.float32)

    for t in range(args.horizon):
        # 随机动作（只做链路测试）
        u = torch.rand((args.num_envs, acts.shape[0]), device=args.device)
        action = env_model.act_low + (env_model.act_high - env_model.act_low) * u

        obs, r, done = env_model.step(action)
        print(f"[Env-Model] step {t} reward: {r.float().mean().item():.3f}")
        total_r += r

        if t < 3:
            print(
                f"[Env-Model] step={t} "
                f"reward_mean={float(r.mean().cpu()):.6f} "
                f"reward_max={float(r.max().cpu()):.6f} "
                f"done_frac={float(done.float().mean().cpu()):.3f}"
            )

    print("\n[Env-Model] rollout 完成")
    print(
        f"  return(mean)={float(total_r.mean().cpu()):.6f} "
        f"return(max)={float(total_r.max().cpu()):.6f} "
        f"return(min)={float(total_r.min().cpu()):.6f}"
    )

    # 8) 清理
    try:
        env_real.close()
    except Exception:
        pass

    if hasattr(dreamer, "simulation_app") and dreamer.simulation_app is not None:
        dreamer.simulation_app.close()


if __name__ == "__main__":
    main()

