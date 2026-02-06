# mpc_active_collect_train_wm_gpu.py
from __future__ import annotations

import argparse
import pathlib
import time
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ruamel.yaml as yaml
import torch

# 让脚本能找到 dreamerv3_torch
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer  # noqa: E402
import my_ur3_project.tasks  # noqa: F401, E402


# -------------------------
# 配置加载
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
    def __init__(self): self.step = 0
    def scalar(self, *args, **kwargs): pass
    def video(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass


# -------------------------
# reset/step 输出兼容（保持 torch）
# -------------------------
def extract_obs_reset(reset_out):
    # gymnasium: (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def unpack_step(step_out):
    """
    返回 obs, rew, done, info；done 为 torch.bool 或 np.bool，后续统一转 torch
    """
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, rew, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, rew, done, info
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs, rew, done, info = step_out
        return obs, rew, done, info
    raise TypeError(f"Unsupported step() output: {type(step_out)} len={len(step_out) if isinstance(step_out, tuple) else 'NA'}")


def wrap_action_for_env(env, action_tensor: torch.Tensor):
    # Dreamer wrapper 通常需要 dict action
    key = getattr(env, "_key", None)
    if key is not None:
        return {key: action_tensor}
    return {"action": action_tensor}


# -------------------------
# actor 输出适配（你已确认输出是 dict）
# -------------------------
def actor_out_to_action(actor_out: Any) -> torch.Tensor:
    if isinstance(actor_out, dict):
        if "action" in actor_out:
            return actor_out["action"]
        if "mean" in actor_out:
            return actor_out["mean"]
        raise KeyError(f"Actor output dict has no 'action'. Keys={list(actor_out.keys())}")
    if hasattr(actor_out, "mode") and callable(actor_out.mode):
        return actor_out.mode()
    if torch.is_tensor(actor_out):
        return actor_out
    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


# -------------------------
# ensemble 输入维度推断（避免 action_cond 维度不匹配）
# -------------------------
def infer_ensemble_in_dim(ensemble) -> Optional[int]:
    if ensemble is None:
        return None
    net = getattr(ensemble, "_networks", None)
    if net is None:
        return None
    lin1 = getattr(net, "lin1", None)
    if lin1 is not None:
        w = getattr(lin1, "weights", None)
        if w is not None and torch.is_tensor(w):
            return int(w.shape[1])
        w2 = getattr(lin1, "weight", None)
        if w2 is not None and torch.is_tensor(w2):
            return int(w2.shape[-1])
    return None


@torch.no_grad()
def intrinsic_disagreement(ensemble, feat: torch.Tensor, action: torch.Tensor,
                           action_cond: bool, reduce: str = "mean", log1p: bool = False) -> torch.Tensor:
    if ensemble is None:
        return torch.zeros((feat.shape[0],), device=feat.device, dtype=torch.float32)

    expected = infer_ensemble_in_dim(ensemble)
    feat_dim = int(feat.shape[-1])
    act_dim = int(action.shape[-1])
    cand_dim = feat_dim + act_dim if action_cond else feat_dim

    if expected is not None and cand_dim != expected:
        action_cond = False

    inp = torch.cat([feat, action], dim=-1) if action_cond else feat
    div = ensemble.intrinsic_reward_penn(inp)
    if div.ndim == 2:
        r = div.sum(dim=-1) if reduce == "sum" else div.mean(dim=-1)
    else:
        r = div
    if log1p:
        r = torch.log1p(torch.clamp_min(r, 0.0))
    return r.float()


# -------------------------
# GPU Episode Replay：存 torch.cuda.Tensor，不做 numpy
# -------------------------
class EpisodeReplayGPU:
    def __init__(self, max_episodes: int = 5000):
        self.max_episodes = int(max_episodes)
        self.episodes: List[Dict[str, torch.Tensor]] = []

    def add_episode(self, ep: Dict[str, torch.Tensor]):
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def __len__(self):
        return len(self.episodes)

class VecEpisodeWriterGPU:
    """
    每个 env 写一个 episode buffer（list of tensors），done 时 stack 成 (T, ...) 存入 replay
    关键点：reward/discount/is_* 等标量全部存成 0-d 标量，保证 batch 采样后是 (L,B)，
    让 Dreamer preprocess 自己扩到 (L,B,1)。
    """
    def __init__(self, num_envs: int, replay, device: torch.device, obs_keys=("policy",)):
        self.B = int(num_envs)
        self.replay = replay
        self.device = device
        self.obs_keys = tuple(obs_keys)

        self.cur = [dict() for _ in range(self.B)]
        self.started = [False] * self.B

    def _scalar0d(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        """把任意形状但 numel==1 的输入压成 0-d 标量 tensor。"""
        if torch.is_tensor(x):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(x, device=self.device)
        t = t.to(dtype)
        if t.numel() != 1:
            raise ValueError(f"Expected scalar (numel==1), got shape={tuple(t.shape)}")
        return t.reshape(())  # 0-d

    def _ensure_keys(self, i: int):
        if self.started[i]:
            return
        for k in self.obs_keys:
            self.cur[i][k] = []
        # Dreamer 训练常用字段
        self.cur[i]["action"] = []
        self.cur[i]["reward"] = []
        self.cur[i]["discount"] = []
        self.cur[i]["is_first"] = []
        self.cur[i]["is_last"] = []
        self.cur[i]["is_terminal"] = []
        self.cur[i]["failure"] = []
        self.started[i] = True

    def add_step(
        self,
        obs: dict,
        action: torch.Tensor,       # (B,A)
        reward: torch.Tensor,       # (B,) or (B,1)
        done: torch.Tensor,         # (B,) or (B,1) bool
        first_flags: torch.Tensor,  # (B,) bool
        failure: torch.Tensor | None = None,
    ):
        B = self.B
        if failure is None:
            failure = torch.zeros_like(done, dtype=torch.bool, device=self.device)

        # 保证一维索引一致
        reward = reward.view(-1)
        done = done.view(-1).to(torch.bool)
        first_flags = first_flags.view(-1).to(torch.bool)
        failure = failure.view(-1).to(torch.bool)

        for i in range(B):
            self._ensure_keys(i)

            # 只存真正的观测模态（例如 policy / image），不要直接存 env 给的 is_first/is_last 等
            for k in self.obs_keys:
                v = obs[k]
                tv = v if torch.is_tensor(v) else torch.as_tensor(v, device=self.device)
                self.cur[i][k].append(tv[i])  # policy: (obs_dim,)

            # action: (A,)
            self.cur[i]["action"].append(action[i].to(torch.float32))

            # 标量：全部压成 0-d
            r0 = self._scalar0d(reward[i], torch.float32)
            d0 = self._scalar0d(done[i], torch.bool)
            f0 = self._scalar0d(first_flags[i], torch.bool)
            fail0 = self._scalar0d(failure[i], torch.bool)

            # reward/discount 必须是 float32 标量
            self.cur[i]["reward"].append(r0)
            self.cur[i]["discount"].append(self._scalar0d((~d0).to(torch.float32), torch.float32))

            self.cur[i]["is_first"].append(f0)
            self.cur[i]["is_last"].append(d0)
            self.cur[i]["is_terminal"].append(d0)
            self.cur[i]["failure"].append(fail0)

            if bool(done[i].item()):
                self._finalize(i)

    def _finalize(self, i: int):
        ep_lists = self.cur[i]
        if not ep_lists:
            self.started[i] = False
            return
        ep = {k: torch.stack(v, dim=0) for k, v in ep_lists.items()}  # time-major
        self.replay.add_episode(ep)
        self.cur[i] = dict()
        self.started[i] = False



class InfiniteSequenceDatasetGPU:
    """
    next(self._dataset) 直接返回 (L, N, ...) 的 torch.cuda batch
    """
    def __init__(self, replay: EpisodeReplayGPU, batch_size: int, batch_length: int,
                 device: torch.device, min_episodes: int = 10):
        self.replay = replay
        self.batch_size = int(batch_size)
        self.batch_length = int(batch_length)
        self.device = device
        self.min_episodes = int(min_episodes)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if len(self.replay) < self.min_episodes:
            raise StopIteration("Not enough episodes yet")

        L = self.batch_length
        N = self.batch_size

        samples: List[Tuple[Dict[str, torch.Tensor], int]] = []
        tries = 0
        while len(samples) < N and tries < 20000:
            tries += 1
            ep = self.replay.episodes[np.random.randint(0, len(self.replay.episodes))]
            T = int(ep["action"].shape[0])
            if T < L:
                continue
            t0 = np.random.randint(0, T - L + 1)
            # 避免跨终止边界
            if ep["is_last"][t0:t0 + L].any():
                continue
            samples.append((ep, t0))

        if len(samples) < N:
            raise StopIteration("Failed to sample sequences; collect more data or relax terminal constraint.")

        keys = samples[0][0].keys()
        batch: Dict[str, torch.Tensor] = {}
        for k in keys:
            seqs = []
            for ep, t0 in samples:
                seqs.append(ep[k][t0:t0 + L])  # (L,...)
            batch[k] = torch.stack(seqs, dim=1).to(self.device)  # (L,N,...)

        return batch


# -------------------------
# PosteriorTracker：全 torch，不用 np.asarray
# -------------------------
class PosteriorTracker:
    def __init__(self, agent, act_dim: int, device: torch.device):
        self.agent = agent
        self.wm = agent._wm
        self.dyn = self.wm.dynamics
        self.act_dim = int(act_dim)
        self.device = device
        self.state = None
        self.prev_action = None

    def _initial(self, B: int):
        if hasattr(self.dyn, "initial"):
            return self.dyn.initial(B)
        raise AttributeError("wm.dynamics 缺少 initial(B)")

    def _obs_step(self, state, action, embed, is_first):
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

    def _to_device_tensor(self, v: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if torch.is_tensor(v):
            t = v.to(self.device)
            return t if dtype is None else t.to(dtype)
        t = torch.as_tensor(v, device=self.device)
        return t if dtype is None else t.to(dtype)

    def _preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs_t: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if k == "policy":
                obs_t[k] = self._to_device_tensor(v, dtype=torch.float32)
            elif k.startswith("is_") or k in ("failure",):
                obs_t[k] = self._to_device_tensor(v).to(torch.bool)
            else:
                obs_t[k] = self._to_device_tensor(v)
        if hasattr(self.wm, "preprocess"):
            obs_t = self.wm.preprocess(obs_t)
        return obs_t

    @torch.no_grad()
    def reset_from_obs(self, obs: Dict[str, Any]):
        obs_t = self._preprocess_obs(obs)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed
        B = int(embed.shape[0])
        self.state = self._initial(B)
        self.prev_action = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)
        is_first = torch.ones((B,), device=self.device, dtype=torch.bool)
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    @torch.no_grad()
    def update(self, action: torch.Tensor, obs_next: Dict[str, Any], done: torch.Tensor):
        obs_t = self._preprocess_obs(obs_next)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed
        done_t = done.to(self.device).to(torch.bool)
        is_first = done_t
        self.prev_action = action.detach()
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    def feat(self) -> torch.Tensor:
        return self.dyn.get_feat(self.state)


# -------------------------
# MPC：Random Shooting（GPU）
# -------------------------
def repeat_latent(latent, N: int):
    if isinstance(latent, dict):
        return {k: repeat_latent(v, N) for k, v in latent.items()}
    if torch.is_tensor(latent):
        return latent.repeat_interleave(N, dim=0)
    if isinstance(latent, (list, tuple)):
        return type(latent)(repeat_latent(v, N) for v in latent)
    raise TypeError(type(latent))


@torch.no_grad()
def plan_action_mpc(
    wm,
    ensemble,
    base_actor,          # 可为 None
    latent_post,         # posterior latent
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    H: int,
    N: int,
    residual_scale: float,
    residual_sigma: float,
    action_cond: bool,
    reduce: str,
    log1p: bool,
    residual_l2_pen: float = 0.01,
) -> torch.Tensor:
    device = act_low.device
    dyn = wm.dynamics

    feat0 = dyn.get_feat(latent_post)
    B = int(feat0.shape[0])
    A = int(act_low.shape[-1])

    res = torch.randn((B, N, H, A), device=device, dtype=torch.float32) * float(residual_sigma)
    res = torch.clamp(res, -1.0, 1.0)

    z = repeat_latent(latent_post, N)
    total = torch.zeros((B * N,), device=device, dtype=torch.float32)

    for t in range(H):
        feat = dyn.get_feat(z)
        if base_actor is None:
            a_base = torch.zeros((B * N, A), device=device, dtype=torch.float32)
        else:
            a_base = actor_out_to_action(base_actor(feat)).detach()

        res_t = res[:, :, t, :].reshape(B * N, A)
        a = a_base + float(residual_scale) * res_t
        a = torch.max(torch.min(a, act_high), act_low)

        r = intrinsic_disagreement(ensemble, feat, a, action_cond=action_cond, reduce=reduce, log1p=log1p)
        if residual_l2_pen > 0:
            r = r - float(residual_l2_pen) * (res_t ** 2).sum(dim=-1)

        total += r

        if hasattr(dyn, "img_step"):
            z = dyn.img_step(z, a)
        else:
            z = dyn.imagine_step(z, a)

    total = total.view(B, N)
    best = torch.argmax(total, dim=1)

    if base_actor is None:
        base0 = torch.zeros((B, A), device=device, dtype=torch.float32)
    else:
        base0 = actor_out_to_action(base_actor(feat0)).detach()

    res0 = res[torch.arange(B, device=device), best, 0, :]
    a0 = base0 + float(residual_scale) * res0
    a0 = torch.max(torch.min(a0, act_high), act_low)
    return a0


# -------------------------
# Main：采集（GPU）+ 周期更新（train_model_only/train_uncertainty_only）
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_env_steps", type=int, default=200000)  # 这里是“总 env-steps（B*steps）”
    parser.add_argument("--collect_per_update", type=int, default=5000)
    parser.add_argument("--max_episodes", type=int, default=5000)

    parser.add_argument("--plan_horizon", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--residual_sigma", type=float, default=0.5)
    parser.add_argument("--use_base_actor", action="store_true")
    parser.add_argument("--action_cond", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--disag_reduce", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--disag_log1p", action="store_true")

    parser.add_argument("--wm_updates", type=int, default=200)
    parser.add_argument("--unc_updates", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)

    cli_overrides = {"task": args.task, "device": args.device}
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # 创建真实 env（期望其 obs/reward/done 都是 torch on GPU）
    env = dreamer.make_env(config, num_envs=args.num_envs)

    # action spec（low/high 来源是 numpy，但只用来初始化 torch bounds，不参与循环）
    acts = env.single_action_space
    obs_space = env.single_observation_space

    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0
    act_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
    config.num_actions = act_dim

    act_low = torch.as_tensor(acts.low, device=device, dtype=torch.float32).view(1, -1)
    act_high = torch.as_tensor(acts.high, device=device, dtype=torch.float32).view(1, -1)

    # replay/dataset（GPU）
    replay = EpisodeReplayGPU(max_episodes=args.max_episodes)
    dataset = InfiniteSequenceDatasetGPU(
        replay=replay,
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        device=device,
        min_episodes=max(10, int(config.batch_size)),
    )

    logger = NullLogger()

    # agent：dataset 直接传 GPU dataset iterator
    agent = dreamer.Dreamer(obs_space, acts, config, logger, dataset).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(ckpt, strict=False)

    agent.requires_grad_(False)


    wm = agent._wm
    ensemble = getattr(agent, "_disag_ensemble", None)

    base_actor = None
    if args.use_base_actor:
        if hasattr(agent, "_task_behavior") and hasattr(agent._task_behavior, "actor"):
            base_actor = agent._task_behavior.actor
        else:
            raise AttributeError("use_base_actor=True 但找不到 agent._task_behavior.actor")
        base_actor.eval()

    tracker = PosteriorTracker(agent, act_dim=act_dim, device=device)
    writer = VecEpisodeWriterGPU(num_envs=args.num_envs, replay=replay, device=device)

    obs = extract_obs_reset(env.reset())
    if not isinstance(obs, dict):
        raise TypeError(f"env.reset() must return dict, got {type(obs)}")

    tracker.reset_from_obs(obs)

    # is_first 由我们自己维护（torch on GPU）
    first_flags = torch.ones((args.num_envs,), device=device, dtype=torch.bool)

    collected = 0
    since_update = 0
    t0 = time.time()

    while collected < args.total_env_steps:
        # MPC 规划（GPU）
        a = plan_action_mpc(
            wm=wm,
            ensemble=ensemble,
            base_actor=base_actor if args.use_base_actor else None,
            latent_post=tracker.state,
            act_low=act_low,
            act_high=act_high,
            H=args.plan_horizon,
            N=args.num_candidates,
            residual_scale=args.residual_scale,
            residual_sigma=args.residual_sigma,
            action_cond=args.action_cond,
            reduce=args.disag_reduce,
            log1p=args.disag_log1p,
            residual_l2_pen=0.01,
        )

        # 与真实 env 交互：不转 numpy，直接传 torch dict
        step_out = env.step(wrap_action_for_env(env, a))
        obs_next, rew, done, info = unpack_step(step_out)

        rew_t = rew if torch.is_tensor(rew) else torch.as_tensor(rew, device=device)
        done_t = done if torch.is_tensor(done) else torch.as_tensor(done, device=device)

        rew_t = rew_t.view(-1).to(torch.float32)      # (B,)
        done_t = done_t.view(-1).to(torch.bool)       # (B,)
        first_flags = first_flags.view(-1).to(torch.bool)

        writer.add_step(obs=obs, action=a, reward=rew_t, done=done_t, first_flags=first_flags)


        # posterior 更新（GPU）
        tracker.update(action=a, obs_next=obs_next, done=done_t)

        # 下一步
        first_flags = done_t.clone()
        obs = obs_next

        collected += args.num_envs
        since_update += args.num_envs

        # 周期性更新
        if since_update >= args.collect_per_update and len(replay) >= max(10, int(config.batch_size)):
            since_update = 0

            # agent.train()
            agent.requires_grad_(True)

            # dataset 不足时，train_* 会 StopIteration；这里容错一次即可
            for _ in range(int(args.wm_updates)):
                try:
                    agent.train_model_only(training=True)
                except StopIteration:
                    break

            for _ in range(int(args.unc_updates)):
                try:
                    agent.train_uncertainty_only(training=True)
                except StopIteration:
                    break

        
            agent.requires_grad_(False)
            if base_actor is not None:
                base_actor.eval()

        if collected % (args.num_envs * 200) == 0:
            print(f"collected_env_steps={collected}, episodes={len(replay)}, elapsed={time.time()-t0:.1f}s")

    try:
        env.close()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
