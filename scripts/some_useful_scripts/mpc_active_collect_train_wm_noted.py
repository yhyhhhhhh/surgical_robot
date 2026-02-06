# mpc_active_collect_train_wm_gpu.py
# 启用 Python 3.10+ 的类型注解特性（如 list[str] 而非 List[str]）
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
# 路径设置与导入
# -----------------------------------------------------------------------------
# 将 scripts 目录加入系统路径，以便能导入 dreamerv3_torch 库
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer  # noqa: E402
# 导入自定义的任务环境（Isaac Lab / Omniverse 相关任务）
import my_ur3_project.tasks  # noqa: F401, E402


# -----------------------------------------------------------------------------
# 配置加载模块
# -----------------------------------------------------------------------------
def recursive_update(base: dict, update: dict):
    """递归更新字典：将 update 中的配置覆盖到 base 中"""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_update(base[k], v)
        else:
            base[k] = v


def load_config_from_yaml(configs_yaml_path: str, selected_configs: list[str] | None, cli_overrides: dict):
    """
    从 YAML 文件加载配置，并合并命令行参数覆盖
    1. 读取 YAML
    2. 合并 'defaults' 和用户选定的配置组
    3. 应用命令行参数覆盖 (cli_overrides)
    """
    yaml_parser = yaml.YAML(typ="safe", pure=True)
    cfg_path = pathlib.Path(configs_yaml_path)
    # 处理相对路径
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
    """空日志记录器：用于禁用 Dreamer 内部繁杂的日志输出，提升采集速度"""
    def __init__(self): self.step = 0
    def scalar(self, *args, **kwargs): pass
    def video(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass


# -----------------------------------------------------------------------------
# 环境接口适配器 (Wrappers)
# -----------------------------------------------------------------------------




def wrap_action_for_env(env, action_tensor: torch.Tensor):
    """
    将动作 Tensor 包装为环境需要的字典格式。
    Dreamer 通常输出 action tensor，但某些环境 wrapper 需要 {'action': tensor} 或 {'robot_0': tensor}。
    """
    key = getattr(env, "_key", None)
    if key is not None:
        return {key: action_tensor}
    return {"action": action_tensor}


# -----------------------------------------------------------------------------
# Actor 输出处理
# -----------------------------------------------------------------------------
def actor_out_to_action(actor_out: Any) -> torch.Tensor:
    """
    从 Actor 网络的输出中提取动作 Tensor。
    Actor 可能输出一个分布对象、字典或直接的 Tensor。
    """
    if isinstance(actor_out, dict):
        if "action" in actor_out:
            return actor_out["action"]
        if "mean" in actor_out:
            return actor_out["mean"]
        raise KeyError(f"Actor output dict has no 'action'. Keys={list(actor_out.keys())}")
    # 如果是分布对象（如 Normal），通常有 mode() 方法获取确定性动作
    if hasattr(actor_out, "mode") and callable(actor_out.mode):
        return actor_out.mode()
    if torch.is_tensor(actor_out):
        return actor_out
    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


# -----------------------------------------------------------------------------
# 主动学习核心：内在奖励 (Intrinsic Disagreement) 计算
# -----------------------------------------------------------------------------
def infer_ensemble_in_dim(ensemble) -> Optional[int]:
    """推断 Ensemble 网络的输入维度，用于判断是否需要拼接动作 (Action Conditioned)"""
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
    """
    计算 Ensemble 模型预测的不确定性（作为好奇心奖励）。
    
    参数:
        ensemble: 预测未来的多个神经网络集合
        feat: 当前特征状态
        action: 拟采取的动作
    返回:
        r: 不确定性得分 (Variance/Disagreement)
    """
    if ensemble is None:
        return torch.zeros((feat.shape[0],), device=feat.device, dtype=torch.float32)

    # 自动检查输入维度，防止配置错误
    expected = infer_ensemble_in_dim(ensemble)
    feat_dim = int(feat.shape[-1])
    act_dim = int(action.shape[-1])
    cand_dim = feat_dim + act_dim if action_cond else feat_dim

    if expected is not None and cand_dim != expected:
        action_cond = False

    # 构造输入：是否将动作与特征拼接
    inp = torch.cat([feat, action], dim=-1) if action_cond else feat
    
    # 计算 Ensemble 成员输出的方差 (intrinsic_reward_penn 内部通常计算方差)
    div = ensemble.intrinsic_reward_penn(inp)
    
    # 聚合结果
    if div.ndim == 2:
        r = div.sum(dim=-1) if reduce == "sum" else div.mean(dim=-1)
    else:
        r = div
    
    # 可选的 log1p 平滑处理
    if log1p:
        r = torch.log1p(torch.clamp_min(r, 0.0))
    return r.float()


# -----------------------------------------------------------------------------
# GPU 数据存储与重放 (Episode Replay) - 核心优化部分
# -----------------------------------------------------------------------------
class EpisodeReplayGPU:
    """
    简单的显存 Replay Buffer。
    完全存储在 GPU 上，是一个 List[Dict[str, Tensor]]。
    """
    def __init__(self, max_episodes: int = 5000):
        self.max_episodes = int(max_episodes)
        self.episodes: List[Dict[str, torch.Tensor]] = []

    def add_episode(self, ep: Dict[str, torch.Tensor]):
        self.episodes.append(ep)
        # 如果超过容量，移除最旧的 episode
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def __len__(self):
        return len(self.episodes)

class VecEpisodeWriterGPU:
    """
    向量化环境的数据写入器。
    功能：
    1. 接收并行环境 (Batch=B) 的每一步输出。
    2. 在显存中缓存这些步 (Steps)。
    3. 当某个环境 Done 时，将缓存的 Steps 堆叠成 Episode，存入 Replay。
    
    **关键修正**: 强制将 reward/done 等标量转为 0-d Tensor，修复 PyTorch 广播错误。
    """
    def __init__(self, num_envs: int, replay, device: torch.device, obs_keys=("policy",)):
        self.B = int(num_envs)
        self.replay = replay
        self.device = device
        self.obs_keys = tuple(obs_keys)

        # 为每个环境维护一个暂存列表
        self.cur = [dict() for _ in range(self.B)]
        self.started = [False] * self.B

    def _scalar0d(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        """
        工具函数：把任意形状但只有一个元素的输入（如 [1], [1,1]）强制压成 0-d 标量 tensor (shape=[])。
        这是为了防止数据出现 (L, B, 1, 1) 这种奇怪形状，导致后续训练出错。
        """
        if torch.is_tensor(x):
            t = x.to(self.device)
        else:
            t = torch.as_tensor(x, device=self.device)
        t = t.to(dtype)
        if t.numel() != 1:
            raise ValueError(f"Expected scalar (numel==1), got shape={tuple(t.shape)}")
        return t.reshape(())  # 变为 0-d

    def _ensure_keys(self, i: int):
        """初始化某个环境的暂存字典"""
        if self.started[i]:
            return
        for k in self.obs_keys:
            self.cur[i][k] = []
        # Dreamer 训练必须字段
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

        # 展平 tensor，方便通过索引 i 访问
        reward = reward.view(-1)
        done = done.view(-1).to(torch.bool)
        first_flags = first_flags.view(-1).to(torch.bool)
        failure = failure.view(-1).to(torch.bool)

        for i in range(B):
            self._ensure_keys(i)

            # 1. 存储观测值 (Observation)
            for k in self.obs_keys:
                v = obs[k]
                tv = v if torch.is_tensor(v) else torch.as_tensor(v, device=self.device)
                self.cur[i][k].append(tv[i]) 

            # 2. 存储动作
            self.cur[i]["action"].append(action[i].to(torch.float32))

            # 3. 存储标量 (Reward, Done 等)
            # 使用 _scalar0d 强制修复形状问题，这是修复 "TypeError/RuntimeError" 的关键
            r0 = self._scalar0d(reward[i], torch.float32)
            d0 = self._scalar0d(done[i], torch.bool)
            f0 = self._scalar0d(first_flags[i], torch.bool)
            fail0 = self._scalar0d(failure[i], torch.bool)

            self.cur[i]["reward"].append(r0)
            # Discount: 如果 done，discount 为 0，否则通常为 1.0 (或 gamma)
            self.cur[i]["discount"].append(self._scalar0d((~d0).to(torch.float32), torch.float32))

            self.cur[i]["is_first"].append(f0)
            self.cur[i]["is_last"].append(d0)
            self.cur[i]["is_terminal"].append(d0)
            self.cur[i]["failure"].append(fail0)

            # 如果当前步是 Done，则结束这个 episode 并归档
            if bool(done[i].item()):
                self._finalize(i)

    def _finalize(self, i: int):
        """将暂存列表堆叠为 Tensor 并存入 Replay"""
        ep_lists = self.cur[i]
        if not ep_lists:
            self.started[i] = False
            return
        # stack 之后 shape: (Time, ...)
        ep = {k: torch.stack(v, dim=0) for k, v in ep_lists.items()}  
        self.replay.add_episode(ep)
        self.cur[i] = dict() # 重置暂存区
        self.started[i] = False


class InfiniteSequenceDatasetGPU:
    """
    GPU 专用数据集迭代器。
    替代 PyTorch 原生的 DataLoader，避免多进程和 CPU copy 开销。
    直接在 GPU 上采样 Batch。
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
        # 数据量不足时不开始训练
        if len(self.replay) < self.min_episodes:
            raise StopIteration("Not enough episodes yet")

        L = self.batch_length
        N = self.batch_size

        samples: List[Tuple[Dict[str, torch.Tensor], int]] = []
        tries = 0
        # 随机采样 N 个序列片段
        while len(samples) < N and tries < 20000:
            tries += 1
            # 随机选一个 episode
            ep = self.replay.episodes[np.random.randint(0, len(self.replay.episodes))]
            T = int(ep["action"].shape[0])
            if T < L:
                continue
            # 随机选起始时间点
            t0 = np.random.randint(0, T - L + 1)
            # 避免选中的片段跨越了 is_last (即不跨越 episode 边界)
            if ep["is_last"][t0:t0 + L].any():
                continue
            samples.append((ep, t0))

        if len(samples) < N:
            raise StopIteration("Failed to sample sequences; collect more data or relax terminal constraint.")

        # 堆叠 Batch: (L, N, ...)
        keys = samples[0][0].keys()
        batch: Dict[str, torch.Tensor] = {}
        for k in keys:
            seqs = []
            for ep, t0 in samples:
                seqs.append(ep[k][t0:t0 + L]) 
            batch[k] = torch.stack(seqs, dim=1).to(self.device)

        return batch


# -----------------------------------------------------------------------------
# 状态跟踪器 (PosteriorTracker)
# -----------------------------------------------------------------------------
class PosteriorTracker:
    """
    用于在推理/交互阶段维护世界模型的隐藏状态 (Recurrent State)。
    它不进行训练，只是不断根据 obs 更新状态，以便 MPC 规划器知道“现在在哪”。
    """
    def __init__(self, agent, act_dim: int, device: torch.device):
        self.agent = agent
        self.wm = agent._wm
        self.dyn = self.wm.dynamics
        self.act_dim = int(act_dim)
        self.device = device
        self.state = None
        self.prev_action = None

    def _initial(self, B: int):
        """获取初始隐藏状态"""
        if hasattr(self.dyn, "initial"):
            return self.dyn.initial(B)
        raise AttributeError("wm.dynamics 缺少 initial(B)")

    def _obs_step(self, state, action, embed, is_first):
        """调用 RSSM/Dynamics 模型的一步推理"""
        if hasattr(self.dyn, "obs_step"):
            try:
                post, _ = self.dyn.obs_step(state, action, embed, is_first)
            except TypeError:
                post, _ = self.dyn.obs_step(state, action, embed)
            return post
        raise AttributeError("wm.dynamics 缺少 obs_step/observe")

    def _to_device_tensor(self, v: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """辅助函数：转 Tensor 并移至 GPU"""
        if torch.is_tensor(v):
            t = v.to(self.device)
            return t if dtype is None else t.to(dtype)
        t = torch.as_tensor(v, device=self.device)
        return t if dtype is None else t.to(dtype)

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

    @torch.no_grad()
    def reset_from_obs(self, obs: Dict[str, Any]):
        """环境 Reset 后，重置 Tracker 状态"""
        obs_t = self._preprocess_obs(obs)
        embed = self.wm.encoder(obs_t)
        # 处理 Encoder 输出可能是 dict 的情况
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed
        B = int(embed.shape[0])
        self.state = self._initial(B)
        self.prev_action = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)
        is_first = torch.ones((B,), device=self.device, dtype=torch.bool)
        # 执行第一步 observe
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    @torch.no_grad()
    def update(self, action: torch.Tensor, obs_next: Dict[str, Any], done: torch.Tensor):
        """环境 Step 后，更新 Tracker 状态"""
        obs_t = self._preprocess_obs(obs_next)
        embed = self.wm.encoder(obs_t)
        embed = embed["embed"] if isinstance(embed, dict) and "embed" in embed else embed
        done_t = done.to(self.device).to(torch.bool)
        is_first = done_t # 如果上一帧 Done，则这一帧视为 First
        self.prev_action = action.detach()
        self.state = self._obs_step(self.state, self.prev_action, embed, is_first)

    def feat(self) -> torch.Tensor:
        """获取当前用于决策的特征向量 (h, z)"""
        return self.dyn.get_feat(self.state)


# -----------------------------------------------------------------------------
# MPC 规划器：随机打靶法 (Random Shooting)
# -----------------------------------------------------------------------------
def repeat_latent(latent, N: int):
    """辅助函数：将 latent state 复制 N 份，以便进行 N 条轨迹并行预测"""
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
    base_actor,          # 可选：基础策略网络，用于提供初始动作建议
    latent_post,         # 当前的后验状态 (起点)
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    H: int,              # 规划视界 (Horizon)
    N: int,              # 候选轨迹数量 (Candidates)
    residual_scale: float,
    residual_sigma: float,
    action_cond: bool,
    reduce: str,
    log1p: bool,
    residual_l2_pen: float = 0.01,
) -> torch.Tensor:
    """
    MPC 规划主函数。
    逻辑：
    1. 生成 N 条候选动作序列（随机噪声 或 BaseActor + 噪声）。
    2. 使用 World Model (dyn) 想象未来 H 步。
    3. 计算每条轨迹的内在奖励 (ensemble disagreement)。
    4. 选择奖励最高的轨迹的第一步动作。
    """
    device = act_low.device
    dyn = wm.dynamics

    # 获取当前特征
    feat0 = dyn.get_feat(latent_post)
    B = int(feat0.shape[0])
    A = int(act_low.shape[-1])

    # 1. 生成随机噪声
    res = torch.randn((B, N, H, A), device=device, dtype=torch.float32) * float(residual_sigma)
    res = torch.clamp(res, -1.0, 1.0)

    # 复制状态以并行模拟
    z = repeat_latent(latent_post, N)
    total = torch.zeros((B * N,), device=device, dtype=torch.float32)

    # 2. 逐步想象 (Dreaming Loop)
    for t in range(H):
        feat = dyn.get_feat(z)
        
        # 确定基准动作 (base_actor 或 零)
        if base_actor is None:
            a_base = torch.zeros((B * N, A), device=device, dtype=torch.float32)
        else:
            a_base = actor_out_to_action(base_actor(feat)).detach()

        # 叠加噪声得到候选动作
        res_t = res[:, :, t, :].reshape(B * N, A)
        a = a_base + float(residual_scale) * res_t
        a = torch.max(torch.min(a, act_high), act_low) # 裁剪到动作空间

        # 3. 计算内在奖励 (好奇心/分歧)
        r = intrinsic_disagreement(ensemble, feat, a, action_cond=action_cond, reduce=reduce, log1p=log1p)
        
        # 加上动作惩罚 (类似正则化，防止动作过大)
        if residual_l2_pen > 0:
            r = r - float(residual_l2_pen) * (res_t ** 2).sum(dim=-1)

        total += r

        # World Model 向前推演一步状态
        z = dyn.img_step(z, a)


    # 4. 选择最佳轨迹
    total = total.view(B, N)
    best = torch.argmax(total, dim=1)

    # 计算最终要执行的动作 (t=0)
    if base_actor is None:
        base0 = torch.zeros((B, A), device=device, dtype=torch.float32)
    else:
        base0 = actor_out_to_action(base_actor(feat0)).detach()

    res0 = res[torch.arange(B, device=device), best, 0, :]
    a0 = base0 + float(residual_scale) * res0
    a0 = torch.max(torch.min(a0, act_high), act_low)
    return a0


# -----------------------------------------------------------------------------
# Main 函数：程序入口
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument("--configs_yaml", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    # 预训练模型路径 (Checkpoint)
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    parser.add_argument("--device", type=str, default="cuda")

    # 训练/采集参数
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_env_steps", type=int, default=200000)  # 总采集步数
    parser.add_argument("--collect_per_update", type=int, default=5000) # 每采集多少步更新一次模型
    parser.add_argument("--max_episodes", type=int, default=5000)       # Replay Buffer 大小

    # MPC 参数
    parser.add_argument("--plan_horizon", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--residual_sigma", type=float, default=1.0)
    parser.add_argument("--use_base_actor", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--action_cond", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--disag_reduce", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--disag_log1p", action="store_true")

    # 更新次数
    parser.add_argument("--wm_updates", type=int, default=200)   # 每次循环更新 World Model 的次数
    parser.add_argument("--unc_updates", type=int, default=200)  # 每次循环更新 Disagreement Ensemble 的次数
    parser.add_argument("--headless", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载并合并配置
    cli_overrides = {"task": args.task, "device": args.device}
    config = load_config_from_yaml(args.configs_yaml, args.configs, cli_overrides)

    # 1. 创建环境 (Dreamer 封装版)
    env = dreamer.make_env(config, num_envs=args.num_envs)

    # 2. 获取动作空间信息，初始化上下界 Tensor (用于 MPC 裁剪)
    acts = env.single_action_space
    obs_space = env.single_observation_space

    acts.low = np.ones_like(acts.low) * -1.0
    acts.high = np.ones_like(acts.high) * 1.0
    act_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
    config.num_actions = act_dim

    act_low = torch.as_tensor(acts.low, device=device, dtype=torch.float32).view(1, -1)
    act_high = torch.as_tensor(acts.high, device=device, dtype=torch.float32).view(1, -1)

    # 3. 初始化数据存储 (Replay Buffer & Dataset)
    replay = EpisodeReplayGPU(max_episodes=args.max_episodes)
    dataset = InfiniteSequenceDatasetGPU(
        replay=replay,
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        device=device,
        min_episodes=max(10, int(config.batch_size)),
    )

    logger = NullLogger()

    # 4. 初始化 Dreamer Agent 并加载权重
    agent = dreamer.Dreamer(obs_space, acts, config, logger, dataset).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    else:
        agent.load_state_dict(ckpt, strict=False)

    # 默认冻结 Agent 梯度，仅在训练阶段开启
    agent.requires_grad_(False)

    # 提取子模块方便后续调用
    wm = agent._wm
    ensemble = getattr(agent, "_disag_ensemble", None)

    # 设置 Base Actor (如果有)
    base_actor = None
    if args.use_base_actor:
        if hasattr(agent, "_task_behavior") and hasattr(agent._task_behavior, "actor"):
            base_actor = agent._task_behavior.actor
        else:
            raise AttributeError("use_base_actor=True 但找不到 agent._task_behavior.actor")
        base_actor.eval()

    # 5. 初始化状态追踪器和写入器
    tracker = PosteriorTracker(agent, act_dim=act_dim, device=device)
    writer = VecEpisodeWriterGPU(num_envs=args.num_envs, replay=replay, device=device)

    # 6. 环境重置 (Reset)
    obs = env.reset()

    # Tracker 依据初始 obs 初始化状态
    tracker.reset_from_obs(obs)

    # 记录是否为首帧 (is_first)
    first_flags = torch.ones((args.num_envs,), device=device, dtype=torch.bool)

    collected = 0
    since_update = 0
    t0 = time.time()

    # ==========================
    # 主循环 (采集 + 训练)
    # ==========================
    while collected < args.total_env_steps:
        # --- A. MPC 规划 (Plan) ---
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

        # --- B. 环境交互 (Step) ---
        # 包装动作并执行
        step_out = env.step(wrap_action_for_env(env, a))
        obs_next, rew, done, info = step_out

        # 确保数据为 Torch Tensor 并展平
        rew_t = rew if torch.is_tensor(rew) else torch.as_tensor(rew, device=device)
        done_t = done if torch.is_tensor(done) else torch.as_tensor(done, device=device)

        rew_t = rew_t.view(-1).to(torch.float32)      # (B,)
        done_t = done_t.view(-1).to(torch.bool)       # (B,)
        first_flags = first_flags.view(-1).to(torch.bool)

        # --- C. 数据存储 (Store) ---
        writer.add_step(obs=obs, action=a, reward=rew_t, done=done_t, first_flags=first_flags)

        # --- D. 状态更新 (Update Tracker) ---
        tracker.update(action=a, obs_next=obs_next, done=done_t)

        # 准备下一帧
        first_flags = done_t.clone() # 如果 Done，下一帧是 First
        obs = obs_next

        collected += args.num_envs
        since_update += args.num_envs

        # --- E. 周期性训练 (Train) ---
        if since_update >= args.collect_per_update and len(replay) >= max(10, int(config.batch_size)):
            since_update = 0
            
            # 开启梯度
            agent.requires_grad_(True)

            # 1. 训练世界模型 (WM) - 学习环境动力学
            for _ in range(int(args.wm_updates)):
                try:
                    agent.train_model_only(training=True)
                except StopIteration: # 防止数据不足
                    break
            
            # 2. 训练不确定性集合 (Ensemble) - 学习"我不懂哪里"
            for _ in range(int(args.unc_updates)):
                try:
                    agent.train_uncertainty_only(training=True)
                except StopIteration:
                    break

            # 关闭梯度，恢复 Eval 模式
            agent.requires_grad_(False)
            if base_actor is not None:
                base_actor.eval()

        # 打印进度
        if collected % (args.num_envs * 200) == 0:
            print(f"collected_env_steps={collected}, episodes={len(replay)}, elapsed={time.time()-t0:.1f}s")

    try:
        env.close()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()