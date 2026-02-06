# mpc_collect_and_update_wm.py
from __future__ import annotations

import argparse
import pathlib
import time
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import ruamel.yaml as yaml
import torch

# 让脚本能找到 dreamerv3_torch 相关模块
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer  # noqa: E402
import my_ur3_project.tasks  # noqa: F401, E402


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
# Gymnasium/Env 输出兼容
# -------------------------
def extract_obs_reset(reset_out):
    # gymnasium: (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def unpack_step(step_out):
    """
    兼容：
      - gymnasium: obs, reward, terminated, truncated, info
      - gym: obs, reward, done, info
      - 一些封装：obs, reward, done
    """
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, rew, terminated, truncated, info = step_out
        done = np.asarray(terminated) | np.asarray(truncated)
        return obs, rew, done, info
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs, rew, done, info = step_out
        return obs, rew, done, info
    if isinstance(step_out, tuple) and len(step_out) == 3:
        obs, rew, done = step_out
        return obs, rew, done, {}
    raise TypeError(f"Unsupported step() output format: {type(step_out)} len={len(step_out) if isinstance(step_out, tuple) else 'NA'}")


def to_torch(x, device, dtype=None):
    if torch.is_tensor(x):
        t = x.to(device)
        return t if dtype is None else t.to(dtype)
    t = torch.as_tensor(np.asarray(x), device=device)
    return t if dtype is None else t.to(dtype)


# -------------------------
# 简单 Replay（time-major ring buffer，支持序列采样）
# -------------------------
class DictRingReplay:
    """
    存储每个时间步的 obs(dict)、action、reward、done。
    shape 约定（按时间维存储）：
      - obs[k]: (T, B, ...)
      - action: (T, B, A)
      - reward: (T, B)
      - done:   (T, B) bool
    """
    def __init__(self, capacity_steps: int, num_envs: int, act_dim: int):
        self.T = int(capacity_steps)
        self.B = int(num_envs)
        self.A = int(act_dim)

        self.ptr = 0
        self.size = 0

        self.obs_store: Dict[str, np.ndarray] = {}
        self.action = np.zeros((self.T, self.B, self.A), dtype=np.float32)
        self.reward = np.zeros((self.T, self.B), dtype=np.float32)
        self.done = np.zeros((self.T, self.B), dtype=np.bool_)

    def _ensure_obs_keys(self, obs: Dict[str, Any]):
        if self.obs_store:
            return
        for k, v in obs.items():
            arr = np.asarray(v)
            # 强制保存为 numpy
            self.obs_store[k] = np.zeros((self.T, self.B, *arr.shape[1:]), dtype=arr.dtype)

    def add_step(self, obs: Dict[str, Any], action: np.ndarray, reward: np.ndarray, done: np.ndarray):
        """
        obs: dict，key-> (B, ...)
        action: (B, A)
        reward: (B,)
        done: (B,) bool
        """
        self._ensure_obs_keys(obs)

        t = self.ptr
        for k, v in obs.items():
            self.obs_store[k][t] = np.asarray(v)
        self.action[t] = action.astype(np.float32)
        self.reward[t] = reward.astype(np.float32).reshape(self.B)
        self.done[t] = done.astype(np.bool_).reshape(self.B)

        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        return self.size >= seq_len + 1 and batch_size > 0

    def sample_sequences(self, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        返回 time-major batch：
          - obs[k]: (L, N, ...)
          - action: (L, N, A)
          - reward: (L, N)
          - done:   (L, N)
        采样时避免跨 done 边界：窗口内 done 必须全为 False（可按需放宽）。
        """
        if not self.can_sample(batch_size, seq_len):
            raise RuntimeError("Not enough data to sample")

        L = int(seq_len)
        N = int(batch_size)

        # 将 ring buffer 映射成线性时间轴 [0, size)
        # 线性索引 i -> ring index (ptr - size + i) mod T
        def ring_index(i: int) -> int:
            return (self.ptr - self.size + i) % self.T

        # 预取 done 的线性视图，便于检查
        done_lin = np.stack([self.done[ring_index(i)] for i in range(self.size)], axis=0)  # (size, B)

        starts: List[Tuple[int, int]] = []
        max_tries = 20000
        tries = 0
        while len(starts) < N and tries < max_tries:
            tries += 1
            t0 = np.random.randint(0, self.size - L)
            e = np.random.randint(0, self.B)
            window_done = done_lin[t0 : t0 + L, e]
            if window_done.any():
                continue
            starts.append((t0, e))

        if len(starts) < N:
            raise RuntimeError("Failed to sample enough non-terminal sequences. Consider increasing buffer or relaxing constraints.")

        # 组装 batch
        batch: Dict[str, torch.Tensor] = {}

        # obs keys
        for k, store in self.obs_store.items():
            # store: (T, B, ...)
            seqs = []
            for (t0, e) in starts:
                frames = []
                for dt in range(L):
                    rr = ring_index(t0 + dt)
                    frames.append(store[rr, e])
                seqs.append(np.stack(frames, axis=0))  # (L, ...)
            arr = np.stack(seqs, axis=1)  # (L, N, ...)
            batch[k] = torch.as_tensor(arr, device=device)

        # action/reward/done
        act_seqs = []
        rew_seqs = []
        don_seqs = []
        for (t0, e) in starts:
            a_frames = []
            r_frames = []
            d_frames = []
            for dt in range(L):
                rr = ring_index(t0 + dt)
                a_frames.append(self.action[rr, e])
                r_frames.append(self.reward[rr, e])
                d_frames.append(self.done[rr, e])
            act_seqs.append(np.stack(a_frames, axis=0))
            rew_seqs.append(np.stack(r_frames, axis=0))
            don_seqs.append(np.stack(d_frames, axis=0))

        batch["action"] = torch.as_tensor(np.stack(act_seqs, axis=1), device=device, dtype=torch.float32)  # (L,N,A)
        batch["reward"] = torch.as_tensor(np.stack(rew_seqs, axis=1), device=device, dtype=torch.float32)  # (L,N)
        batch["done"] = torch.as_tensor(np.stack(don_seqs, axis=1), device=device, dtype=torch.bool)       # (L,N)

        # 常用派生字段：is_first（Dreamer 常用）
        # is_first[t] = done[t-1]，t=0 置 True（表示序列起点）
        done_tm1 = torch.zeros_like(batch["done"])
        done_tm1[1:] = batch["done"][:-1]
        batch["is_first"] = torch.zeros_like(batch["done"])
        batch["is_first"][0] = True
        batch["is_first"][1:] = done_tm1[1:]

        return batch


# -------------------------
# RSSM posterior 跟踪器（真实环境用）
# -------------------------
class PosteriorTracker:
    def __init__(self, agent, act_dim: int, device: torch.device):
        self.agent = agent
        self.wm = agent._wm
        self.dyn = self.wm.dynamics
        self.act_dim = int(act_dim)
        self.device = device

        self.state = None  # latent
        self.prev_action = None

    @torch.no_grad()
    def reset_from_obs(self, obs: Dict[str, Any]) -> Any:
        obs_t = self._preprocess_obs(obs)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed

        B = int(embed.shape[0])
        self.state = self._initial(B)
        self.prev_action = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)

        is_first = torch.ones((B,), device=self.device, dtype=torch.bool)
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)
        return self.state

    @torch.no_grad()
    def update(self, action: torch.Tensor, obs_next: Dict[str, Any], done: torch.Tensor) -> Any:
        obs_t = self._preprocess_obs(obs_next)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed

        done = done.to(self.device).bool()
        is_first = done  # done 后下一观测当作新 episode 的 first

        # 为 done 的 env 强制重置 prior state（更稳健）
        if done.any():
            ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            Bn = int(ids.numel())
            new0 = self._initial(Bn)
            self._assign_state(self.state, new0, ids)

        self.prev_action = action.detach()
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)
        return self.state

    def get_feat(self) -> torch.Tensor:
        return self.dyn.get_feat(self.state)

    def _preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # obs dict -> torch on device
        obs_t: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if k == "policy":
                obs_t[k] = to_torch(v, self.device, dtype=torch.float32)
            elif k.startswith("is_") or k in ("failure",):
                obs_t[k] = to_torch(v, self.device).bool()
            else:
                # 兼容 image 等
                obs_t[k] = to_torch(v, self.device)
        if hasattr(self.wm, "preprocess"):
            obs_t = self.wm.preprocess(obs_t)
        return obs_t

    def _initial(self, B: int):
        if hasattr(self.dyn, "initial"):
            return self.dyn.initial(B)
        raise AttributeError("wm.dynamics 缺少 initial(B)")

    def _obs_step(self, state, action, embed, is_first):
        # 兼容 obs_step / observe
        if hasattr(self.dyn, "obs_step"):
            try:
                post, _ = self.dyn.obs_step(state, action, embed, is_first)
            except TypeError:
                post, _ = self.dyn.obs_step(state, action, embed)
            return post
        if hasattr(self.dyn, "observe"):
            try:
                post, _ = self.dyn.observe(embed, action, is_first, state)
            except TypeError:
                post, _ = self.dyn.observe(embed, action, is_first)
            return post
        raise AttributeError("wm.dynamics 缺少 obs_step/observe")

    def _assign_state(self, dst, src, ids):
        # 将 src(batch=|ids|) 写到 dst[ids]
        if isinstance(dst, dict):
            for k in dst.keys():
                self._assign_state(dst[k], src[k], ids)
        elif torch.is_tensor(dst):
            dst[ids] = src
        elif isinstance(dst, (list, tuple)):
            for i in range(len(dst)):
                self._assign_state(dst[i], src[i], ids)
        else:
            raise TypeError(f"Unsupported latent leaf type: {type(dst)}")


# -------------------------
# MPC：Random Shooting（receding horizon）
# -------------------------
@torch.no_grad()
def intrinsic_disagreement(ensemble, feat: torch.Tensor, action: torch.Tensor, action_cond: bool = True,
                           reduce: str = "mean", log1p: bool = False) -> torch.Tensor:
    """
    返回 (B,) intrinsic reward
    """
    if ensemble is None:
        return torch.zeros((feat.shape[0],), device=feat.device, dtype=torch.float32)

    inp = torch.cat([feat, action], dim=-1) if action_cond else feat
    div = ensemble.intrinsic_reward_penn(inp)  # (B,) or (B,K)

    if div.ndim == 2:
        r = div.sum(dim=-1) if reduce == "sum" else div.mean(dim=-1)
    else:
        r = div

    if log1p:
        r = torch.log1p(torch.clamp_min(r, 0.0))
    return r.float()


def repeat_latent(latent, N: int):
    # 将 latent 的 batch 维 repeat_interleave N 次：B -> B*N
    if isinstance(latent, dict):
        return {k: repeat_latent(v, N) for k, v in latent.items()}
    if torch.is_tensor(latent):
        return latent.repeat_interleave(N, dim=0)
    if isinstance(latent, (list, tuple)):
        return type(latent)(repeat_latent(v, N) for v in latent)
    raise TypeError(f"Unsupported latent type: {type(latent)}")


@torch.no_grad()
def plan_action_mpc_residual(
    wm,
    ensemble,
    base_actor,                 # Dreamer task actor（可选，作为安全基线）
    latent_post,                # 当前 posterior latent（batch=B）
    act_low: torch.Tensor,      # (1,A) on device
    act_high: torch.Tensor,     # (1,A) on device
    plan_horizon: int = 5,
    num_candidates: int = 128,
    residual_scale: float = 0.2,
    residual_sigma: float = 0.5,
    action_cond: bool = True,
    reduce: str = "mean",
    log1p: bool = False,
    action_l2_pen: float = 0.0,
    residual_l2_pen: float = 0.01,
) -> torch.Tensor:
    """
    Random Shooting MPC:
      - 采样 N 条残差序列 Δa_{t:t+H-1}
      - 在 WM imagination 上 rollout H 步，累加 disagreement
      - 选最优序列，只返回第 1 步动作 a_t
    """
    device = act_low.device
    dyn = wm.dynamics

    # 当前真实 posterior 的 feat（用于合成第 1 步动作）
    feat0 = dyn.get_feat(latent_post)
    B = int(feat0.shape[0])
    A = int(act_low.shape[-1])
    H = int(plan_horizon)
    N = int(num_candidates)

    # 采样残差序列： (B,N,H,A) in [-1,1]
    res = torch.randn((B, N, H, A), device=device, dtype=torch.float32) * float(residual_sigma)
    res = torch.clamp(res, -1.0, 1.0)

    # 扩展 latent：B -> B*N
    z = repeat_latent(latent_post, N)  # (B*N,...)
    total = torch.zeros((B * N,), device=device, dtype=torch.float32)

    # rollout
    for t in range(H):
        feat = dyn.get_feat(z)  # (B*N, F)

        # base action（随 imagined state 变化）
        if base_actor is None:
            a_base = torch.zeros((B * N, A), device=device, dtype=torch.float32)
        else:
            out = base_actor(feat)
            a_base = actor_out_to_action(out).detach()
            a_base = a_base.detach()

        # residual at step t
        res_t = res[:, :, t, :].reshape(B * N, A)
        a = a_base + float(residual_scale) * res_t
        a = torch.max(torch.min(a, act_high), act_low)

        # intrinsic reward
        r = intrinsic_disagreement(ensemble, feat, a, action_cond=action_cond, reduce=reduce, log1p=log1p)

        # 轻量正则：避免“投机到奇怪动作”
        if action_l2_pen > 0:
            r = r - float(action_l2_pen) * (a ** 2).sum(dim=-1)
        if residual_l2_pen > 0:
            r = r - float(residual_l2_pen) * (res_t ** 2).sum(dim=-1)

        total += r

        # imagine step
        if hasattr(dyn, "img_step"):
            z = dyn.img_step(z, a)
        else:
            z = dyn.imagine_step(z, a)

    # 选最优 candidate
    total = total.view(B, N)
    best = torch.argmax(total, dim=1)  # (B,)

    # 合成第 1 步动作（在真实 posterior latent 上）
    if base_actor is None:
        base0 = torch.zeros((B, A), device=device, dtype=torch.float32)
    else:
        out0 = base_actor(feat0)
        base0 = actor_out_to_action(out0)
        base0 = base0.detach()

    res0 = res[torch.arange(B, device=device), best, 0, :]  # (B,A)
    a0 = base0 + float(residual_scale) * res0
    a0 = torch.max(torch.min(a0, act_high), act_low)
    return a0


# -------------------------
# 世界模型更新：接口自动探测（必要时你改 1 行即可）
# -------------------------
def update_world_model(agent, batch: Dict[str, torch.Tensor], grad_steps: int = 1):
    """
    这部分由于 dreamerv3_torch 版本差异较大，采用“探测可用入口”的方式。
    你如果运行时报错，只需要把这里替换成你项目中实际的 WM 更新调用即可。
    """
    # 允许训练
    agent.requires_grad_(True)
    agent.train()

    # 常见候选入口（按优先级尝试）
    candidates = [
        "train_step",
        "_train_step",
        "update",
        "_update",
        "_train",
        "learn",
        "world_model_update",
        "train_world_model",
        "_train_world_model",
    ]

    fn = None
    for name in candidates:
        if hasattr(agent, name) and callable(getattr(agent, name)):
            fn = getattr(agent, name)
            break

    if fn is None:
        agent.eval()
        agent.requires_grad_(False)
        raise RuntimeError(
            "找不到 Dreamer agent 的训练入口函数。请在 update_world_model() 里把 fn(...) 替换为你工程里的 WM 更新调用。\n"
            "你可以打印 dir(agent) 或查看 dreamerv3_torch 的训练脚本，通常会有 train_step/update/_train 之类的方法。"
        )

    # 进行若干梯度步
    for _ in range(int(grad_steps)):
        out = fn(batch)  # 你的版本可能返回 loss dict；不强依赖
        _ = out

    agent.eval()
    agent.requires_grad_(False)

def actor_out_to_action(actor_out: Any) -> torch.Tensor:
    """兼容 Dreamer actor 输出为 dict / dist / tensor 的情况，返回 (B, A)"""
    if isinstance(actor_out, dict):
        if "action" in actor_out:
            return actor_out["action"]
        # 有些实现叫 mean
        if "mean" in actor_out:
            return actor_out["mean"]
        raise KeyError(f"Actor output dict has no 'action' key. Keys={list(actor_out.keys())}")

    # 分布对象（有 .mode()）
    if hasattr(actor_out, "mode") and callable(actor_out.mode):
        return actor_out.mode()

    # 退化：直接就是 tensor
    if torch.is_tensor(actor_out):
        return actor_out

    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")

# -------------------------
# Main：MPC 采集 + 周期性更新 WM
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")

    # 环境与采集
    parser.add_argument("--num_envs", type=int, default=16, help="真实采集并行环境数；MPC 很吃算力，建议 8~32 起步")
    parser.add_argument("--total_real_steps", type=int, default=20000, help="总真实交互步数（每个 env 每步算 1）")
    parser.add_argument("--collect_per_update", type=int, default=2000, help="累计多少真实步数触发一次 WM 更新")
    parser.add_argument("--replay_capacity", type=int, default=50000, help="replay 存储的时间步容量（time dimension）")

    # WM 训练
    parser.add_argument("--wm_grad_steps", type=int, default=200, help="每次触发更新时做多少个梯度步")
    parser.add_argument("--batch_size", type=int, default=32, help="WM 更新的 batch（序列条数）")
    parser.add_argument("--seq_len", type=int, default=50, help="WM 更新的序列长度")

    # MPC
    parser.add_argument("--plan_horizon", type=int, default=5, help="MPC 规划视野（未来 H 步）")
    parser.add_argument("--num_candidates", type=int, default=128, help="每步候选序列数量 N")
    parser.add_argument("--residual_scale", type=float, default=0.2, help="残差叠加系数 alpha")
    parser.add_argument("--residual_sigma", type=float, default=0.5, help="残差采样方差（random shooting）")

    parser.add_argument("--use_base_actor", action="store_true", help="使用 Dreamer task actor 作为 base policy（更安全）")
    # 改成显式 bool：
    parser.add_argument("--action_cond", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--disag_reduce", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--disag_log1p", action="store_true")

    parser.add_argument("--action_l2_pen", type=float, default=0.0)
    parser.add_argument("--residual_l2_pen", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)

    # 0) load config
    cli_overrides = {"task": args.task, "device": args.device}
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # 1) 创建真实环境（用于采集）
    print("Creating real env ...")
    env = dreamer.make_env(config, num_envs=args.num_envs)

    # 2) 获取 spec & 归一化动作边界到 [-1,1]（与你前面对齐）
    acts = env.single_action_space
    obs_space = env.single_observation_space

    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0
    act_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
    config.num_actions = act_dim

    act_low = torch.as_tensor(acts.low, device=device, dtype=torch.float32).view(1, -1)
    act_high = torch.as_tensor(acts.high, device=device, dtype=torch.float32).view(1, -1)

    # 3) 加载 Dreamer agent（包含 WM + ensemble）
    logger = NullLogger()
    agent = dreamer.Dreamer(obs_space, acts, config, logger, dataset=None).to(args.device)

    print(f"Loading checkpoint from {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(ckpt, strict=False)

    agent.requires_grad_(False)


    wm = agent._wm
    ensemble = getattr(agent, "_disag_ensemble", None)

    # base actor（可选）
    base_actor = None
    if args.use_base_actor:
        if hasattr(agent, "_task_behavior") and hasattr(agent._task_behavior, "actor"):
            base_actor = agent._task_behavior.actor
        elif hasattr(agent, "task_behavior") and hasattr(agent.task_behavior, "actor"):
            base_actor = agent.task_behavior.actor
        else:
            raise AttributeError("use_base_actor=True 但找不到 task_behavior.actor")

        base_actor.eval()

    # 4) posterior tracker
    tracker = PosteriorTracker(agent, act_dim=act_dim, device=device)

    # 5) replay
    replay = DictRingReplay(
        capacity_steps=args.replay_capacity,
        num_envs=args.num_envs,
        act_dim=act_dim,
    )

    # 6) reset env & init posterior
    reset_out = env.reset()
    obs = extract_obs_reset(reset_out)
    if not isinstance(obs, dict):
        raise TypeError(f"env.reset() 应返回 dict obs，但得到 {type(obs)}")
    tracker.reset_from_obs(obs)

    collected = 0
    collected_since_update = 0

    print("Start MPC data collection ...")
    t0 = time.time()

    while collected < args.total_real_steps:
        # (a) 当前 posterior feat
        feat = tracker.get_feat()

        # (b) MPC 规划动作（未来 H=5 不确定性最大）
        a = plan_action_mpc_residual(
            wm=wm,
            ensemble=ensemble,
            base_actor=base_actor if args.use_base_actor else None,
            latent_post=tracker.state,
            act_low=act_low,
            act_high=act_high,
            plan_horizon=args.plan_horizon,
            num_candidates=args.num_candidates,
            residual_scale=args.residual_scale,
            residual_sigma=args.residual_sigma,
            action_cond=args.action_cond,
            reduce=args.disag_reduce,
            log1p=args.disag_log1p,
            action_l2_pen=args.action_l2_pen,
            residual_l2_pen=args.residual_l2_pen,
        )

        # (c) 与真实 env 交互（优先用 numpy）
        a_np = a.detach().cpu().numpy().astype(np.float32)
        step_out = env.step({"action": a_np})
        obs_next, rew, done, info = unpack_step(step_out)

        # (d) 写入 replay（存当前 obs 与动作；也可存 rew/done）
        # rew/done 统一成 (B,)
        rew_np = np.asarray(rew).reshape(args.num_envs).astype(np.float32)
        done_np = np.asarray(done).reshape(args.num_envs).astype(np.bool_)

        replay.add_step(
            obs=obs,
            action=a_np,
            reward=rew_np,
            done=done_np,
        )

        collected += args.num_envs
        collected_since_update += args.num_envs

        # (e) posterior 跟踪：用真实 obs_next 更新 latent
        done_t = to_torch(done_np, device).bool()
        tracker.update(action=a, obs_next=obs_next, done=done_t)

        # (f) 前进一步
        obs = obs_next

        # (g) 到达阈值则更新世界模型
        if collected_since_update >= args.collect_per_update:
            collected_since_update = 0

            # 足够数据才更新
            if replay.can_sample(args.batch_size, args.seq_len):
                print(f"[UPDATE] collected={collected}, sampling batch and updating world model ...")

                # 采样一个 batch（time-major）
                batch = replay.sample_sequences(
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                )

                # 更新 WM（grad_steps 次）
                update_world_model(agent, batch, grad_steps=args.wm_grad_steps)

                # 更新后切回 eval（MPC 用）
                agent.requires_grad_(False)
                if base_actor is not None:
                    base_actor.eval()

                print("[UPDATE] done.")

        if collected % (args.num_envs * 50) == 0:
            elapsed = time.time() - t0
            print(f"Progress: collected_steps={collected} (env_steps*B), elapsed={elapsed:.1f}s")

    # close
    try:
        env.close()
    except Exception:
        pass

    print("Finished.")


if __name__ == "__main__":
    main()
