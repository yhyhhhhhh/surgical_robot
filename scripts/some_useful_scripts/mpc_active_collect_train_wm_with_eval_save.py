# mpc_active_collect_train_wm_with_eval_save_single_env.py
# Single SimulationContext version:
# - Uses ONE Isaac Lab env instance (no env_eval) to avoid:
#     "Simulation context already exists. Cannot create a new one."
# - Collects a fixed eval replay on the same env BEFORE training.
# - Trains online with MPC-driven active collection.
# - Periodically evaluates baseline vs current on the fixed eval replay.
# - Saves latest/best/final checkpoints.

from __future__ import annotations

import argparse
import pathlib
import time
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ruamel.yaml as yaml
import torch

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer  # noqa: E402
import my_ur3_project.tasks  # noqa: F401, E402


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Env IO utilities (robust to dict/tuple)
# -----------------------------------------------------------------------------
def _as_torch(x: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device)
    else:
        t = torch.as_tensor(x, device=device)
    return t if dtype is None else t.to(dtype)


def extract_done(done: Any, device: torch.device) -> torch.Tensor:
    """
    Returns bool tensor shape (B,).
    Handles:
      - torch.Tensor
      - tuple(terminated, truncated)
      - dict-like {key: tensor}
    """
    if isinstance(done, (tuple, list)):
        if len(done) == 2:
            term = _as_torch(done[0], device, torch.bool).view(-1)
            trunc = _as_torch(done[1], device, torch.bool).view(-1)
            return (term | trunc)
        done = done[0]

    if isinstance(done, dict):
        for k in ("done", "is_last", "terminated", "truncated"):
            if k in done:
                return _as_torch(done[k], device, torch.bool).view(-1)
        v = next(iter(done.values()))
        return _as_torch(v, device, torch.bool).view(-1)

    return _as_torch(done, device, torch.bool).view(-1)


def extract_reward(rew: Any, device: torch.device) -> torch.Tensor:
    """
    Returns float tensor shape (B,).
    """
    if isinstance(rew, dict):
        for k in ("reward", "rew"):
            if k in rew:
                return _as_torch(rew[k], device, torch.float32).view(-1)
        v = next(iter(rew.values()))
        return _as_torch(v, device, torch.float32).view(-1)
    return _as_torch(rew, device, torch.float32).view(-1)


def wrap_action_for_env(env, action_tensor: torch.Tensor):
    key = getattr(env, "_key", None)
    if key is not None:
        return {key: action_tensor}
    return {"action": action_tensor}


# -----------------------------------------------------------------------------
# Actor output parsing
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Disagreement (intrinsic reward) helpers
# -----------------------------------------------------------------------------
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
def intrinsic_disagreement(
    ensemble,
    feat: torch.Tensor,
    action: torch.Tensor,
    action_cond: bool,
    reduce: str = "mean",
    log1p: bool = False,
) -> torch.Tensor:
    if ensemble is None:
        return torch.zeros((feat.shape[0],), device=feat.device, dtype=torch.float32)

    expected = infer_ensemble_in_dim(ensemble)
    feat_dim = int(feat.shape[-1])
    act_dim = int(action.shape[-1])
    cand_dim = feat_dim + act_dim if action_cond else feat_dim

    # safeguard: disable action-conditioning if dims mismatch
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


# -----------------------------------------------------------------------------
# GPU episode replay & dataset (time-major)
# -----------------------------------------------------------------------------
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
    Stores episodes fully on GPU.
    IMPORTANT: store is_first/is_last/is_terminal/failure as float32 masks (0/1),
    so Dreamer dynamics can safely do arithmetic (1.0 - is_first).
    """
    def __init__(self, num_envs: int, replay: EpisodeReplayGPU, device: torch.device, obs_keys=("policy",)):
        self.B = int(num_envs)
        self.replay = replay
        self.device = device
        self.obs_keys = tuple(obs_keys)
        self.cur = [dict() for _ in range(self.B)]
        self.started = [False] * self.B

    def _scalar0d(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(x, device=self.device)
        t = t.to(dtype)
        if t.numel() != 1:
            raise ValueError(f"Expected scalar (numel==1), got shape={tuple(t.shape)}")
        return t.reshape(())

    def _ensure_keys(self, i: int):
        if self.started[i]:
            return
        for k in self.obs_keys:
            self.cur[i][k] = []
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
        reward: torch.Tensor,       # (B,) float
        done: torch.Tensor,         # (B,) bool
        first_flags: torch.Tensor,  # (B,) bool
        failure: torch.Tensor | None = None,
    ):
        B = self.B
        if failure is None:
            failure = torch.zeros_like(done, dtype=torch.bool, device=self.device)

        reward = reward.view(-1)
        done = done.view(-1).to(torch.bool)
        first_flags = first_flags.view(-1).to(torch.bool)
        failure = failure.view(-1).to(torch.bool)

        for i in range(B):
            self._ensure_keys(i)

            for k in self.obs_keys:
                v = obs[k]
                tv = v if torch.is_tensor(v) else torch.as_tensor(v, device=self.device)
                self.cur[i][k].append(tv[i].to(torch.float32))

            self.cur[i]["action"].append(action[i].to(torch.float32))

            r0 = self._scalar0d(reward[i], torch.float32)
            d0 = self._scalar0d(done[i], torch.bool)
            f0 = self._scalar0d(first_flags[i], torch.bool)
            fail0 = self._scalar0d(failure[i], torch.bool)

            # float masks
            f0f = self._scalar0d(f0.to(torch.float32), torch.float32)      # 0/1
            d0f = self._scalar0d(d0.to(torch.float32), torch.float32)      # 0/1
            fail0f = self._scalar0d(fail0.to(torch.float32), torch.float32)

            self.cur[i]["reward"].append(r0)
            self.cur[i]["discount"].append(self._scalar0d((~d0).to(torch.float32), torch.float32))
            self.cur[i]["is_first"].append(f0f)
            self.cur[i]["is_last"].append(d0f)
            self.cur[i]["is_terminal"].append(d0f)
            self.cur[i]["failure"].append(fail0f)

            if bool(done[i].item()):
                self._finalize(i)

    def _finalize(self, i: int):
        ep_lists = self.cur[i]
        if not ep_lists:
            self.started[i] = False
            return
        ep = {k: torch.stack(v, dim=0) for k, v in ep_lists.items()}  # (T, ...)
        self.replay.add_episode(ep)
        self.cur[i] = dict()
        self.started[i] = False


class InfiniteSequenceDatasetGPU:
    def __init__(
        self,
        replay: EpisodeReplayGPU,
        batch_size: int,
        batch_length: int,
        device: torch.device,
        min_episodes: int = 10,
        avoid_terminal: bool = True,
    ):
        self.replay = replay
        self.batch_size = int(batch_size)
        self.batch_length = int(batch_length)
        self.device = device
        self.min_episodes = int(min_episodes)
        self.avoid_terminal = bool(avoid_terminal)

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
            if self.avoid_terminal and ep["is_last"][t0:t0 + L].any():
                continue
            samples.append((ep, t0))

        if len(samples) < N:
            raise StopIteration("Failed to sample sequences; collect more data or relax terminal constraint.")

        keys = samples[0][0].keys()
        batch: Dict[str, torch.Tensor] = {}
        for k in keys:
            seqs = [ep[k][t0:t0 + L] for ep, t0 in samples]   # list of (L, ...)
            batch[k] = torch.stack(seqs, dim=1).to(self.device)  # (L, N, ...)
        # 强制 chunk 起点 reset
        if "is_first" in batch:
            batch["is_first"] = batch["is_first"].clone()
            batch["is_first"][0, :] = 1.0

        # 可选：如果你担心起点 cont / terminal 语义干扰，可以把起点 terminal 清 0
        # if "is_terminal" in batch:
        #     batch["is_terminal"] = batch["is_terminal"].clone()
        #     batch["is_terminal"][0, :] = 0.0

        return batch
# -----------------------------------------------------------------------------
# Posterior tracker (keeps RSSM state synced with real env)
# -----------------------------------------------------------------------------
class PosteriorTracker:
    """
    IMPORTANT: pass is_first as float mask (B,1) to avoid:
      - bool subtraction (1.0 - is_first)
      - in-place broadcast explosions
    """
    def __init__(self, agent, act_dim: int, device: torch.device, obs_keys=("policy",)):
        self.agent = agent
        self.wm = agent._wm
        self.dyn = self.wm.dynamics
        self.act_dim = int(act_dim)
        self.device = device
        self.obs_keys = tuple(obs_keys)

        self.state = None
        self.prev_action = None

    def _initial(self, B: int):
        if hasattr(self.dyn, "initial"):
            return self.dyn.initial(B)
        raise AttributeError("wm.dynamics missing initial(B)")

    def _to_device_tensor(self, v: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return _as_torch(v, self.device, dtype)
    
    def _preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """对原始 obs 进行预处理（转 Tensor，类型转换）"""
        obs_t: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if k == "policy":
                obs_t[k] = self._to_device_tensor(v, dtype=torch.float32)
            elif k.startswith("is_") or k in ("failure",):
                obs_t[k] = self._to_device_tensor(v).to(torch.bool)
            else:
                obs_t[k] = self._to_device_tensor(v)
        # 调用 Dreamer 模型自带的 preprocess（例如图像归一化）
        if hasattr(self.wm, "preprocess"):
            obs_t = self.wm.preprocess(obs_t)
        return obs_t
    
    def _obs_step(self, state, action, embed, is_first: torch.Tensor):
        if is_first.dtype != torch.float32:
            is_first = is_first.to(torch.float32)
        if is_first.ndim == 1:
            is_first = is_first.view(-1, 1)
        if is_first.ndim > 2:
            reduce_dims = tuple(range(1, is_first.ndim))
            is_first = (is_first > 0.5).any(dim=reduce_dims).to(torch.float32).view(-1, 1)

        if hasattr(self.dyn, "obs_step"):
            post, _ = self.dyn.obs_step(state, action, embed, is_first)
            return post
        raise AttributeError("wm.dynamics missing obs_step")

    @torch.no_grad()
    def reset_from_obs(self, obs: Dict[str, Any]):
        obs_t = self._preprocess_obs(obs)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed
        B = int(embed.shape[0])

        self.state = self._initial(B)
        self.prev_action = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)

        is_first = torch.ones((B, 1), device=self.device, dtype=torch.float32)
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    @torch.no_grad()
    def update(self, action: torch.Tensor, obs_next: Dict[str, Any], done: torch.Tensor):
        obs_t = self._preprocess_obs(obs_next)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed

        done_t = done.to(self.device)
        if done_t.ndim > 1:
            done_t = done_t.any(dim=tuple(range(1, done_t.ndim)))
        done_t = done_t.to(torch.bool).view(-1)

        is_first = done_t.to(torch.float32).view(-1, 1)

        self.prev_action = action.detach()
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    def feat(self) -> torch.Tensor:
        return self.dyn.get_feat(self.state)


# -----------------------------------------------------------------------------
# MPC planner: random shooting (maximize disagreement)
# -----------------------------------------------------------------------------
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
    base_actor,
    latent_post,
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

    def img_step(state, action):
        if hasattr(dyn, "img_step"):
            return dyn.img_step(state, action)
        return dyn.imagine_step(state, action)

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
        z = img_step(z, a)

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


# -----------------------------------------------------------------------------
# Evaluation: baseline vs current on fixed eval replay (no extra sim context)
# -----------------------------------------------------------------------------
def dist_mean(dist_obj):
    if hasattr(dist_obj, "mean"):
        m = dist_obj.mean
        if callable(m):
            return m()
        return m
    if hasattr(dist_obj, "mode") and callable(dist_obj.mode):
        return dist_obj.mode()
    raise AttributeError("Distribution object has neither mean nor mode")


@torch.no_grad()
def eval_models_on_dataset(
    baseline_agent,
    current_agent,
    dataset: InfiniteSequenceDatasetGPU,
    num_batches: int,
    action_cond: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    def eval_one(agent, batch):
        wm = agent._wm
        dyn = wm.dynamics
        ensemble = getattr(agent, "_disag_ensemble", None)

        wm.eval()
        if ensemble is not None:
            ensemble.eval()

        policy = batch["policy"]          # (L,B,obs_dim)
        action = batch["action"]          # (L,B,act_dim)
        is_first = batch["is_first"]      # (L,B) float mask (0/1)
        reward = batch["reward"]          # (L,B)

        L, B, _ = policy.shape
        act_dim = action.shape[-1]
        device = policy.device

        state = dyn.initial(B)
        prev_action = torch.zeros((B, act_dim), device=device)

        mse_obs_sum = 0.0
        mse_rew_sum = 0.0
        n_steps = 0

        dis_list = []
        err_list = []

        def img_step(s, a):
            if hasattr(dyn, "img_step"):
                return dyn.img_step(s, a)
            return dyn.imagine_step(s, a)

        for t in range(L - 1):
            f = is_first[t]
            if f.ndim > 1:
                f = (f > 0.5).any(dim=tuple(range(1, f.ndim)))
            f = f.to(torch.float32).view(B, 1)

            obs_t = {
                "policy": policy[t].to(torch.float32),
                # Dreamer preprocess 要求至少有 is_first
                "is_first": batch["is_first"][t].to(torch.float32),
                # 下面这些一般不是强制，但建议带上保持格式一致
                "is_last": batch["is_last"][t].to(torch.float32) if "is_last" in batch else torch.zeros_like(batch["is_first"][t]).to(torch.float32),
                "is_terminal": batch["is_terminal"][t].to(torch.float32) if "is_terminal" in batch else torch.zeros_like(batch["is_first"][t]).to(torch.float32),
                "failure": batch["failure"][t].to(torch.float32) if "failure" in batch else torch.zeros_like(batch["is_first"][t]).to(torch.float32),
            }
            if hasattr(wm, "preprocess"):
                obs_t = wm.preprocess(obs_t)

            embed = wm.encoder(obs_t)
            embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed

            post, _ = dyn.obs_step(state, prev_action, embed, f)
            feat_post = dyn.get_feat(post)

            a_t = action[t].to(torch.float32)
            nxt = img_step(post, a_t)
            feat_nxt = dyn.get_feat(nxt)

            dec = wm.heads["decoder"](feat_nxt)
            if isinstance(dec, dict):
                obs_dist = dec["policy"] if "policy" in dec else next(iter(dec.values()))
            else:
                obs_dist = dec

            pred_obs = dist_mean(obs_dist)
            targ_obs = policy[t + 1].to(torch.float32)

            obs_err = (pred_obs - targ_obs).pow(2).mean(dim=-1)
            mse_obs_sum += float(obs_err.mean().item())

            if "reward" in wm.heads:
                rdist = wm.heads["reward"](feat_nxt)
                pred_r = dist_mean(rdist).view(-1)
                targ_r = reward[t].to(torch.float32).view(-1)
                mse_rew_sum += float((pred_r - targ_r).pow(2).mean().item())

            if ensemble is not None:
                dis = intrinsic_disagreement(ensemble, feat_post, a_t, action_cond=action_cond)
                dis_list.append(dis)
                err_list.append(obs_err)

            state = post
            prev_action = a_t
            n_steps += 1

        out = {
            "mse_obs": mse_obs_sum / max(1, n_steps),
            "mse_rew": mse_rew_sum / max(1, n_steps),
        }

        if dis_list:
            dis_all = torch.cat(dis_list, dim=0)
            err_all = torch.cat(err_list, dim=0)
            dis_c = dis_all - dis_all.mean()
            err_c = err_all - err_all.mean()
            corr = (dis_c * err_c).mean() / (dis_c.std() * err_c.std() + 1e-8)
            out["unc_corr"] = float(corr.item())
        else:
            out["unc_corr"] = 0.0

        return out

    def run(agent):
        acc = {"mse_obs": 0.0, "mse_rew": 0.0, "unc_corr": 0.0}
        cnt = 0
        for _ in range(num_batches):
            try:
                batch = next(dataset)
            except StopIteration:
                break
            met = eval_one(agent, batch)
            for k in acc:
                acc[k] += float(met.get(k, 0.0))
            cnt += 1
        if cnt == 0:
            return {k: float("nan") for k in acc}
        return {k: v / cnt for k, v in acc.items()}

    base = run(baseline_agent)
    curr = run(current_agent)
    return base, curr


@torch.no_grad()
def collect_eval_data_random_on_same_env(
    env,
    eval_writer: VecEpisodeWriterGPU,
    steps: int,
    act_dim: int,
    device: torch.device,
):
    obs = env.reset()
    first_flags = torch.ones((env.num_envs,), device=device, dtype=torch.bool)

    collected = 0
    while collected < steps:
        a = torch.empty((env.num_envs, act_dim), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)

        obs_next, rew, done, info = env.step(wrap_action_for_env(env, a))
        rew_t = extract_reward(rew, device)
        done_t = extract_done(done, device)

        eval_writer.add_step(obs=obs, action=a, reward=rew_t, done=done_t, first_flags=first_flags)

        first_flags = done_t.clone()
        obs = obs_next
        collected += env.num_envs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    parser.add_argument("--device", type=str, default="cuda")

    # Collect/train
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_env_steps", type=int, default=200000)
    parser.add_argument("--collect_per_update", type=int, default=5000)
    parser.add_argument("--max_episodes", type=int, default=5000)

    # MPC
    parser.add_argument("--plan_horizon", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.0)
    parser.add_argument("--residual_sigma", type=float, default=1.0)
    parser.add_argument("--use_base_actor", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--action_cond", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--disag_reduce", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--disag_log1p", action="store_true")
    parser.add_argument("--residual_l2_pen", type=float, default=0.01)

    # Updates per cycle
    parser.add_argument("--wm_updates", type=int, default=10)
    parser.add_argument("--unc_updates", type=int, default=10)

    # Eval (fixed replay collected BEFORE training, on SAME env)
    parser.add_argument("--eval_collect_steps", type=int, default=20000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--eval_every_updates", type=int, default=1)

    # Save
    parser.add_argument("--save_dir", type=str, default="logs_wm_mpc_ckpt")

    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)
    np.random.seed(0)

    cli_overrides = {"task": args.task, "device": args.device}
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # Create ONE env only
    env = dreamer.make_env(config, num_envs=args.num_envs)

    acts = env.single_action_space
    obs_space = env.single_observation_space

    # Normalize action space to [-1,1]
    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0
    act_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
    config.num_actions = act_dim

    act_low = torch.as_tensor(acts.low, device=device, dtype=torch.float32).view(1, -1)
    act_high = torch.as_tensor(acts.high, device=device, dtype=torch.float32).view(1, -1)

    # Training replay/dataset
    replay = EpisodeReplayGPU(max_episodes=args.max_episodes)
    dataset = InfiniteSequenceDatasetGPU(
        replay=replay,
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        device=device,
        min_episodes=max(10, int(config.batch_size)),
        avoid_terminal=True,
    )

    # Eval replay/dataset (fixed, collected before training)
    eval_replay = EpisodeReplayGPU(max_episodes=max(2000, args.max_episodes // 5))
    eval_dataset = InfiniteSequenceDatasetGPU(
        replay=eval_replay,
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        device=device,
        min_episodes=max(10, int(config.batch_size)),
        avoid_terminal=True,
    )

    logger = NullLogger()

    # Agent (trainable)
    agent = dreamer.Dreamer(obs_space, acts, config, logger, dataset).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(ckpt, strict=False)

    # Baseline agent (frozen copy of initial)
    baseline_agent = dreamer.Dreamer(obs_space, acts, config, logger, dataset).to(args.device)
    baseline_agent.load_state_dict(agent.state_dict(), strict=False)
    baseline_agent.requires_grad_(False)

    wm = agent._wm
    ensemble = getattr(agent, "_disag_ensemble", None)

    base_actor = None
    if args.use_base_actor:
        if hasattr(agent, "_task_behavior") and hasattr(agent._task_behavior, "actor"):
            base_actor = agent._task_behavior.actor
        else:
            raise AttributeError("use_base_actor=True but agent._task_behavior.actor not found")
        base_actor.eval()

    tracker = PosteriorTracker(agent, act_dim=act_dim, device=device, obs_keys=("policy",))
    writer = VecEpisodeWriterGPU(num_envs=args.num_envs, replay=replay, device=device, obs_keys=("policy",))
    eval_writer = VecEpisodeWriterGPU(num_envs=args.num_envs, replay=eval_replay, device=device, obs_keys=("policy",))

    # -------------------------------------------------------------------------
    # 1) Collect fixed eval data on SAME env (random policy), then reset for train
    # -------------------------------------------------------------------------
    print(f"[EVAL] collecting fixed eval replay on SAME env: {args.eval_collect_steps} env steps ...")
    collect_eval_data_random_on_same_env(env, eval_writer, steps=args.eval_collect_steps, act_dim=act_dim, device=device)
    print(f"[EVAL] eval episodes collected: {len(eval_replay)}")

    # Reset env for training start
    obs = env.reset()
    tracker.reset_from_obs(obs)
    first_flags = torch.ones((args.num_envs,), device=device, dtype=torch.bool)

    # Save directory
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_mse_obs = float("inf")

    def save_ckpt(tag: str, collected_steps: int, num_updates: int, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "agent_state_dict": agent.state_dict(),
            "collected_env_steps": int(collected_steps),
            "num_updates": int(num_updates),
            "time": time.time(),
        }
        if extra:
            payload.update(extra)
        path = save_dir / f"{tag}.pt"
        torch.save(payload, path)
        print(f"[CKPT] saved: {path}")

    agent.requires_grad_(False)


    collected = 0
    since_update = 0
    num_updates = 0
    t0 = time.time()

    print("[RUN] start MPC active collection + online training (single env) ...")
    while collected < args.total_env_steps:
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
            residual_l2_pen=args.residual_l2_pen,
        )

        obs_next, rew, done, info = env.step(wrap_action_for_env(env, a))
        rew_t = extract_reward(rew, device)
        done_t = extract_done(done, device)

        # 1) 准备写入 replay 的 obs_store，默认用 obs_next
        obs_store = obs_next

        # 2) 如果 done 时 obs_next 是 reset obs，需要替换成 terminal obs
        if done_t.any():
            done_ids = torch.nonzero(done_t, as_tuple=False).squeeze(-1)

            # 优先从 info 里拿 terminal obs（你需要按实际 wrapper 的字段名改这里）
            term = None
            if isinstance(info, dict):
                term = info.get("terminal_observation", None) or info.get("final_observation", None)

            if term is not None:
                # term 可能也是 dict(obs_key -> tensor[B,...])
                for k in obs_store.keys():
                    obs_store[k][done_ids] = term[k][done_ids]
            else:
                # 退化策略：至少覆盖 policy，模仿你 baseline 的逻辑
                if "policy" in obs_store and "policy" in obs:
                    obs_store["policy"][done_ids] = obs["policy"][done_ids]
                # 如果你还有 image 等，也可以一并覆盖

        # 3) 写入：对齐到 obs_store（=真正的 next/terminal obs）
        # 注意：这里的 first_flags 应该描述“obs_store 是否是 episode 的 first”
        # - 如果你把 done 的 obs_store 替换成 terminal obs，那么它不是 first，应置 0
        first_store = done_t.clone()
        if done_t.any():
            first_store[done_ids] = False  # 因为 done_ids 的 obs_store 已被改成 terminal obs

        writer.add_step(obs=obs_store, action=a, reward=rew_t, done=done_t, first_flags=first_store)

        # tracker 仍然用 reset 后的 obs_next 做后验（这是对的）
        tracker.update(action=a, obs_next=obs_next, done=done_t)

        first_flags = done_t.clone()
        obs = obs_next

        collected += args.num_envs
        since_update += args.num_envs

        # Periodic train
        if since_update >= args.collect_per_update and len(replay) >= max(10, int(config.batch_size)):
            since_update = 0
            num_updates += 1

            agent.requires_grad_(True)


            for _ in range(int(args.wm_updates)):
                try:
                    agent.train_world_model_only(training=True)
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

            save_ckpt("latest", collected_steps=collected, num_updates=num_updates)

            if num_updates % int(args.eval_every_updates) == 0:
                np.random.seed(123)
                torch.manual_seed(123)

                base_met, cur_met = eval_models_on_dataset(
                    baseline_agent=baseline_agent,
                    current_agent=agent,
                    dataset=eval_dataset,
                    num_batches=int(args.eval_batches),
                    action_cond=args.action_cond,
                )
                print(f"[EVAL@update={num_updates}] baseline={base_met}  current={cur_met}")

                cur_mse = float(cur_met.get("mse_obs", float("inf")))
                if cur_mse < best_mse_obs:
                    best_mse_obs = cur_mse
                    save_ckpt(
                        "best_mse_obs",
                        collected_steps=collected,
                        num_updates=num_updates,
                        extra={"best_mse_obs": best_mse_obs},
                    )

        if collected % (args.num_envs * 200) == 0:
            print(f"[PROG] collected_env_steps={collected} episodes={len(replay)} updates={num_updates} elapsed={time.time()-t0:.1f}s")

    save_ckpt("final", collected_steps=collected, num_updates=num_updates)

    np.random.seed(999)
    torch.manual_seed(999)
    base_met, cur_met = eval_models_on_dataset(
        baseline_agent=baseline_agent,
        current_agent=agent,
        dataset=eval_dataset,
        num_batches=int(args.eval_batches),
        action_cond=args.action_cond,
    )
    print(f"[EVAL@final] baseline={base_met}  current={cur_met}")

    try:
        env.close()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
