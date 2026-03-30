from __future__ import annotations

import gymnasium as gym  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261

# from rl_games.common.vecenv import IVecEnv
import numpy as np
import torch

"""
Vectorized environment wrapper.
"""


class DreamerVecEnvWrapper(gym.Wrapper):
    """
    适配 Isaac Lab DirectRLEnv / Gymnasium env 到 Dreamer 采集/训练习惯：

    输入 env._get_observations() 期望提供：
        {
          "policy": Tensor[N, D] (float32),
          "image":  Tensor[N, H, W, 3] (uint8),
          "is_first": Tensor[N] (int/bool)   # 你在 env 里已算好
          "is_last": Tensor[N] (占位也行)
          "is_terminal": Tensor[N] (占位也行)
        }

    本 wrapper 在 step() 里用 terminated/truncated 生成：
        is_last = done
        is_terminal = terminated

    并把它们写回 obs dict，保持与 env 输出对齐。
    """

    def __init__(self, env, device: str | torch.device = "cuda"):
        super().__init__(env)
        self.device = torch.device(device)
        # IsaacLab 环境一般有 unwrapped.num_envs
        self._num_envs = getattr(env.unwrapped, "num_envs", None)
        
        if self._num_envs is None:
            raise ValueError("Underlying env must have attribute `unwrapped.num_envs`.")

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        # 直接透传也可以；如果你需要严格 space，可自己补 Dict space
        return self.env.observation_space
    # ---------- 你要补的：single_observation_space ----------
    @property
    def single_observation_space(self):
        """
        返回“单个子环境”的观测空间（去掉 batch 维 num_envs）。
        与你当前 obs key 对齐：policy/image/is_first/is_last/is_terminal
        """
        base = self.env.env._observation_space

        # 1) 先把底层观测空间复制出来，并去掉 batch 维
        if hasattr(base, "spaces"):  # Dict
            new_spaces = {}
            for k, sp in base.spaces.items():
                if isinstance(sp, gym.spaces.Box):
                    low = sp.low
                    high = sp.high
                    shape = sp.shape

                    # 如果第一维是 num_envs，就裁掉
                    if len(shape) >= 1 and shape[0] == self.num_envs:
                        low0 = np.array(low[0])
                        high0 = np.array(high[0])
                        new_shape = shape[1:]
                    else:
                        low0 = np.array(low)
                        high0 = np.array(high)
                        new_shape = shape

                    new_spaces[k] = gym.spaces.Box(
                        low=low0, high=high0, shape=new_shape, dtype=sp.dtype
                    )
                else:
                    # 其他类型空间（Discrete 等）就原样放进去
                    new_spaces[k] = sp
        else:
            # 如果底层不是 Dict，就把它当作 policy
            new_spaces = {"policy": base}

        # 2) 确保 Dreamer 所需的 flags 一定存在（即使底层没声明）
        for flag in ["is_first", "is_last", "is_terminal","failure"]:
            if flag not in new_spaces:
                new_spaces[flag] = gym.spaces.Box(0, 1, (), dtype=bool)

        # 3) （可选）你也可以在这里强制要求有 image/policy
        # if "policy" not in new_spaces or "image" not in new_spaces:
        #     raise ValueError("Expected observation keys: policy and image")

        return gym.spaces.Dict(new_spaces)

    # ---------- 你要补的：action_space ----------
    @property
    def action_space(self):
        """
        返回向量化 action space。
        如果设定了 ac_lim，则把动作限制到 [-ac_lim, ac_lim]（不修改底层 env.action_space）。
        """
        sp = self._env.action_space
        if self.ac_lim is None:
            return sp

        low = -self.ac_lim * np.ones_like(sp.low)
        high = self.ac_lim * np.ones_like(sp.high)
        return gym.spaces.Box(low=low, high=high, dtype=sp.dtype)

    # ---------- 你要补的：single_action_space ----------
    @property
    def single_action_space(self):
        """
        返回单个子环境的 action space（去掉 batch 维 num_envs）。
        """
        sp = self.action_space
        low = sp.low
        high = sp.high

        # 如果是 vectorized (num_envs, act_dim)，取第 0 个
        if low.ndim >= 2 and low.shape[0] == self.num_envs:
            low0 = low[0]
            high0 = high[0]
        else:
            low0 = low
            high0 = high

        return gym.spaces.Box(low=low0, high=high0, dtype=sp.dtype)
    def _to_tensor(self, x, dtype=None):
        if torch.is_tensor(x):
            t = x
        else:
            t = torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(self.device)

    def reset(self,seed=None,options=None):
        seed = self._to_tensor(seed) if seed is not None else None
        # 第一次必须真的 reset 一次底层 env
        if not getattr(self, "_has_reset_once", False):
            obs, extras = self.env.reset(seed=seed,options=options)
            for k, v in obs.items():
                if torch.is_tensor(v):
                    obs[k] = v.to(self.device)

            any_tensor = next(v for v in obs.values() if torch.is_tensor(v))
            B = any_tensor.shape[0]
            self._reset_obs_cache = obs
            self._has_reset_once = True

            # 初次 reset：全 env is_first=1
            is_first = torch.ones(B, device=self.device, dtype=torch.bool)
        else:
            # 非首次：不调用底层 reset，只返回 step() 缓存的 auto-reset 后 obs
            obs = self._reset_obs_cache
            any_tensor = next(v for v in obs.values() if torch.is_tensor(v))
            B = any_tensor.shape[0]

            if seed is None:
                # 默认：不建议全量，用 mask 更稳；这里兜底为全 0
                is_first = torch.zeros(B, device=self.device, dtype=torch.bool)
            else:
                is_first = seed.to(self.device).bool()

        obs_out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}
        obs_out["is_first"] = is_first.to(torch.int32)
        obs_out["is_last"] = torch.zeros(B, device=self.device, dtype=torch.int32)
        obs_out["is_terminal"] = torch.zeros(B, device=self.device, dtype=torch.int32)
        return obs_out


    def step(self, action):
        if isinstance(action, dict):
            action = action.get("action", action)

        action = self._to_tensor(action, dtype=torch.float32)

        out = self.env.step(action)

        obs_reset, reward, terminated, truncated, info = out
        terminated = self._to_tensor(terminated).bool()
        truncated = self._to_tensor(truncated).bool()

        done = terminated | truncated

        # move obs_reset to device
        for k, v in obs_reset.items():
            if torch.is_tensor(v):
                obs_reset[k] = v.to(self.device)

        # ✅ 缓存：这就是 IsaacLab auto-reset 后的新 episode 初始帧（对 done env）
        self._reset_obs_cache = obs_reset
        self._next_is_first = done.to(self.device)
        self._has_reset_once = True

        # ---- 构造 terminal 对齐的 obs_out：done env 用 terminal_observation 替换 ----
        obs_out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs_reset.items()}

        # 你需要 env/extras 里提供 terminal_observation（我们前面讨论的那套）
        term_ids = info.get("terminal_env_ids", None)
        term_obs = info.get("terminal_observation", None)

        if torch.is_tensor(term_ids) and isinstance(term_obs, dict) and term_ids.numel() > 0:
            term_ids = term_ids.to(self.device)
            for k, tv in term_obs.items():
                if k in obs_out and torch.is_tensor(obs_out[k]) and torch.is_tensor(tv):
                    obs_out[k][term_ids] = tv.to(self.device)

        # ---- Dreamer flags（step 帧不标 is_first）----
        obs_out["is_first"] = torch.zeros_like(done, device=self.device, dtype=torch.int32)
        obs_out["is_last"] = done.to(torch.int32)
        obs_out["is_terminal"] = terminated.to(torch.int32)

        reward = self._to_tensor(reward, dtype=torch.float32)
        return obs_out, reward, done, info




# class WorldModelImaginationVecEnv(gym.Env):
#     """
#     基于世界模型（World Model / RSSM）的“想象环境（Env-Model）”，用于 PPO 在模型里 rollout 训练探索策略。

#     设计目标：
#     - 观测（obs）：RSSM 的特征向量 feat = wm.dynamics.get_feat(latent)
#       （通常就是 Dreamer actor/value 的输入，维度一般为 dyn_deter + dyn_stoch*(dyn_discrete?)）
#     - 转移（transition）：latent' = wm.dynamics.img_step(latent, action)
#       （纯想象步进，不与真实仿真交互）
#     - 奖励（reward）：来自 ensemble 的 disagreement（你的 OneStepPredictor），作为 intrinsic reward
#     - 终止（done）：推荐用固定 horizon（最稳定），也可选用 cont/failure head 作为额外终止条件

#     注意：
#     - 这个 Env-Model 不是替代真实仿真 env，它只是 PPO 的“训练环境”。
#     - 新数据采集仍然必须回到真实仿真 Env-Real（IsaacLab）执行探索策略得到，然后更新 WM/ensemble。
#     """

#     metadata = {"render_modes": []}

#     def __init__(
#         self,
#         world_model,
#         ensemble,  # 你的 OneStepPredictor；若为 None 则 intrinsic reward 恒为 0
#         act_space: gym.spaces.Box,
#         sample_batch_fn,  # 可调用对象：返回一个 batch dict（来自 replay/dataset）
#         num_envs: int,
#         horizon: int = 15,
#         device: str | torch.device = "cuda",
#         disag_action_cond: bool = True,   # 必须与 config.disag_action_cond 对齐
#         reward_reduce: str = "mean",      # 若 div 是向量，如何聚合成标量："mean" 或 "sum"
#         log_reward: bool = False,         # 是否对 disagreement 取 log（与你 config.disag_log 类似）
#         clip_reward: float | None = None, # 奖励裁剪，防止 PPO 被极端值支配
#         eps: float = 1e-8,
#         use_cont_head: bool = False,      # 可选：用 cont head 作为终止信号之一
#         cont_threshold: float = 0.5,
#         use_failure_head: bool = False,   # 可选：用 failure head 作为终止信号之一
#         failure_threshold: float = 0.5,
#         start_seq_len: int | None = None, # reset 时从真实序列使用多少步来构建 posterior latent
#     ):
#         super().__init__()
#         self.wm = world_model
#         self.ensemble = ensemble
#         self.sample_batch_fn = sample_batch_fn

#         self.device = torch.device(device)
#         self.num_envs = int(num_envs)
#         self.horizon = int(horizon)
#         self.disag_action_cond = bool(disag_action_cond)
#         self.reward_reduce = reward_reduce
#         self.log_reward = bool(log_reward)
#         self.clip_reward = clip_reward
#         self.eps = eps

#         self.use_cont_head = bool(use_cont_head)
#         self.cont_threshold = float(cont_threshold)
#         self.use_failure_head = bool(use_failure_head)
#         self.failure_threshold = float(failure_threshold)

#         self.start_seq_len = start_seq_len

#         # 动作空间：建议直接用真实仿真 env 的 normalized action space（通常是 [-1,1]）
#         assert isinstance(act_space, gym.spaces.Box)
#         self.action_space = act_space
#         self.act_dim = int(np.prod(self.action_space.shape))

#         # 内部状态：RSSM latent + 时间步计数 + done 标记
#         self._latent = None      # dict[str, Tensor]
#         self._t = None           # Tensor[num_envs]
#         self._done = None        # Tensor[num_envs]

#         # 先 dry-run reset 一次，推断 feat_dim，从而构造 observation_space
#         with torch.no_grad():
#             obs, _ = self.reset()
#             feat_dim = obs.shape[-1]

#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32
#         )

#     # =========================
#     # 1) 从真实 batch 构建 posterior latent（reset 用）
#     # =========================

#     def _get_actions_from_batch(self, batch: dict) -> torch.Tensor:
#         """
#         从 batch dict 里找到动作序列，常见 key：'action'/'actions'/'act'
#         期望 shape: (B, T, act_dim)
#         """
#         for k in ["action", "actions", "act"]:
#             if k in batch:
#                 return batch[k]
#         raise KeyError("batch 中找不到动作序列，请确认 key 是否为 action/actions/act 之一。")

#     def _get_obs_for_encoder_at_t(self, batch: dict, t: int) -> dict:
#         """
#         从 batch 里取出第 t 步的观测，构造给 wm.preprocess/wm.encoder 用的 obs_dict。

#         你需要根据你项目里 batch 的 key 做对齐。
#         常见：
#           - batch['policy']: (B,T,D)
#           - batch['image']:  (B,T,H,W,C)
#           - batch['is_first']:(B,T)
#         """
#         obs = {}

#         # 你目前主要是 policy 向量观测
#         if "policy" in batch:
#             obs["policy"] = batch["policy"][:, t]

#         # 如果你还有 image 观测，可以一并带上
#         if "image" in batch:
#             obs["image"] = batch["image"][:, t]

#         # 标志位（有则带上，preprocess/obs_step 可能会用）
#         for flag in ["is_first", "is_last", "is_terminal", "failure"]:
#             if flag in batch:
#                 obs[flag] = batch[flag][:, t]

#         if len(obs) == 0:
#             raise KeyError("batch 中没有可用观测（至少需要 'policy' 或 'image'）。")
#         return obs

#     def _get_is_first_seq(self, batch: dict, T: int) -> torch.Tensor:
#         """
#         取出 is_first 序列用于 obs_step，shape: (B,T)
#         若 batch 中没有 is_first，则默认仅 t=0 为 True。
#         """
#         if "is_first" in batch:
#             return batch["is_first"][:, :T]
#         B = next(iter(batch.values())).shape[0]
#         is_first = torch.zeros((B, T), device=self.device, dtype=torch.bool)
#         is_first[:, 0] = True
#         return is_first

#     def _initial_latent(self, B: int):
#         """
#         获取 RSSM 初始 latent。
#         大多数 Dreamer/RSSM 实现提供 dynamics.initial(B)。
#         若你的实现允许 latent=None，则可返回 None 让 obs_step 自己处理。
#         """
#         if hasattr(self.wm.dynamics, "initial"):
#             return self.wm.dynamics.initial(B)
#         return None

#     @torch.no_grad()
#     def _posterior_latent_from_batch(self, batch: dict) -> dict:
#         """
#         reset 时，从 replay/dataset 采一段真实序列，通过 obs_step 构建 posterior latent，
#         然后用最后一步 latent 作为 imagination 的起点。

#         这样能保证 imagination 起点“可达”，避免 PPO 在不可达状态刷分歧。
#         """
#         actions = self._get_actions_from_batch(batch)  # (B,T,act_dim)
#         B, T, _ = actions.shape

#         # 选择用多少步来构建 posterior
#         T_use = self.start_seq_len or T
#         T_use = int(min(T_use, T))
#         if T_use < 2:
#             T_use = min(2, T)

#         is_first_seq = self._get_is_first_seq(batch, T_use)

#         latent = self._initial_latent(B)
#         prev_action = torch.zeros((B, self.act_dim), device=self.device, dtype=torch.float32)

#         # 逐步执行 obs_step
#         for t in range(T_use):
#             obs_t = self._get_obs_for_encoder_at_t(batch, t)
#             obs_t = {k: v.to(self.device) for k, v in obs_t.items()}

#             # 预处理（归一化/类型转换等），与你 Dreamer _policy 里一致
#             obs_t = self.wm.preprocess(obs_t)

#             # 编码得到 embed
#             embed = self.wm.encoder(obs_t)

#             # 用上一时刻动作 prev_action 做观测更新（这符合常见 RSSM 设计）
#             latent, _ = self.wm.dynamics.obs_step(
#                 latent, prev_action, embed, is_first_seq[:, t]
#             )

#             prev_action = actions[:, t].to(self.device).float()

#         return latent

#     # =========================
#     # 2) 计算 intrinsic reward（ensemble disagreement）
#     # =========================

#     @torch.no_grad()
#     def _intrinsic_reward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#         """
#         返回 shape: (N,) 的标量 intrinsic reward。
#         - 如果 ensemble 返回向量 div（N,D），则做 mean/sum 聚合。
#         - 可选 log/clip，避免 PPO 不稳定。
#         """
#         if self.ensemble is None:
#             return torch.zeros((feat.shape[0],), device=self.device, dtype=torch.float32)

#         if self.disag_action_cond:
#             inputs = torch.cat([feat, action], dim=-1)
#         else:
#             inputs = feat

#         div = self.ensemble.intrinsic_reward_penn(inputs)  # (N,) 或 (N,D)

#         # 聚合为标量
#         if div.ndim == 2:
#             if self.reward_reduce == "sum":
#                 r = div.sum(dim=-1)
#             else:
#                 r = div.mean(dim=-1)
#         else:
#             r = div

#         if self.log_reward:
#             r = torch.log(r + self.eps)

#         if self.clip_reward is not None:
#             r = torch.clamp(r, -float(self.clip_reward), float(self.clip_reward))

#         return r.float()

#     @torch.no_grad()
#     def _termination_signals(self, feat: torch.Tensor) -> torch.Tensor:
#         """
#         可选终止信号：
#         - cont head：若 continuation 概率过低则终止
#         - failure head：若失败概率过高则终止

#         返回 shape: (N,) 的 bool terminated。
#         """
#         N = feat.shape[0]
#         terminated = torch.zeros((N,), device=self.device, dtype=torch.bool)

#         if self.use_cont_head and "cont" in self.wm.heads:
#             cont = self.wm.heads["cont"](feat).mean()  # (N,1) 或 (N,)
#             if cont.ndim == 2:
#                 cont = cont.squeeze(-1)
#             terminated |= (cont < self.cont_threshold)

#         if self.use_failure_head and "failure" in self.wm.heads:
#             fail = self.wm.heads["failure"](feat).mean()  # (N,1) 或 (N,)
#             if fail.ndim == 2:
#                 fail = fail.squeeze(-1)
#             terminated |= (fail > self.failure_threshold)

#         return terminated

#     # =========================
#     # 3) Gym API
#     # =========================

#     def reset(self, *, seed=None, options=None):
#         """
#         reset：
#         1) 从 sample_batch_fn() 取 batch（batch_size 必须等于 num_envs）
#         2) 用 obs_step 构建 posterior latent
#         3) 返回 obs = feat(latent)
#         """
#         super().reset(seed=seed)

#         batch = self.sample_batch_fn()
#         batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

#         # 构建 posterior latent 作为 imagination 起点
#         self._latent = self._posterior_latent_from_batch(batch)

#         feat = self.wm.dynamics.get_feat(self._latent).detach()

#         # 计数与 done 标记清零
#         self._t = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
#         self._done = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

#         # 要求 batch_size == num_envs，便于向量化 PPO
#         if feat.shape[0] != self.num_envs:
#             raise ValueError(
#                 f"Env-Model 期望 batch_size == num_envs，但得到 feat batch={feat.shape[0]}，num_envs={self.num_envs}。"
#                 "请让 sample_batch_fn() 返回 batch_size=num_envs。"
#             )

#         obs = feat.cpu().numpy().astype(np.float32)
#         info = {}
#         return obs, info

#     def step(self, action):
#         """
#         step：
#         - action: (num_envs, act_dim) 的 numpy 或 torch
#         - 返回 gymnasium 格式：(obs, reward, terminated, truncated, info)

#         说明：
#         - reward 使用当前状态 feat 和当前 action 计算 disagreement（对应“采取动作的探索价值”）
#         - transition 用 img_step 推进 latent
#         - truncated 用固定 horizon（推荐）
#         - terminated 可选用 cont/failure head（可关掉）
#         """
#         # 转成 torch
#         if torch.is_tensor(action):
#             act = action.to(self.device).float()
#         else:
#             act = torch.as_tensor(action, device=self.device, dtype=torch.float32)

#         # 动作裁剪到 action_space 范围
#         low = torch.as_tensor(self.action_space.low, device=self.device, dtype=torch.float32)
#         high = torch.as_tensor(self.action_space.high, device=self.device, dtype=torch.float32)
#         act = torch.max(torch.min(act, high), low)

#         # 当前 feat（用于计算 intrinsic reward）
#         feat = self.wm.dynamics.get_feat(self._latent).detach()

#         # intrinsic reward：disagreement(feat, action)
#         reward = self._intrinsic_reward(feat, act)  # (N,)

#         # imagination dynamics：latent <- img_step(latent, action)
#         with torch.no_grad():
#             if hasattr(self.wm.dynamics, "img_step"):
#                 next_latent = self.wm.dynamics.img_step(self._latent, act)
#             elif hasattr(self.wm.dynamics, "imagine_step"):
#                 next_latent = self.wm.dynamics.imagine_step(self._latent, act)
#             else:
#                 raise AttributeError("找不到 dynamics.img_step 或 dynamics.imagine_step，请按你的实现对齐函数名。")

#         self._latent = next_latent

#         # 时间步推进
#         self._t += 1

#         # terminated：可选信号（cont/failure）
#         feat_next = self.wm.dynamics.get_feat(self._latent).detach()
#         terminated = self._termination_signals(feat_next)

#         # truncated：固定 horizon
#         truncated = (self._t >= self.horizon)

#         # 输出
#         obs = feat_next.cpu().numpy().astype(np.float32)
#         rew = reward.detach().cpu().numpy().astype(np.float32)
#         terminated_np = terminated.detach().cpu().numpy()
#         truncated_np = truncated.detach().cpu().numpy()

#         info = {}
#         return obs, rew, terminated_np, truncated_np, info
