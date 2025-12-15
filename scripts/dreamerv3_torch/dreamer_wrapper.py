from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import torch
import numpy as np
from rl_games.common.vecenv import IVecEnv
import matplotlib.pyplot as plt
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

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

    def reset(self, seed=None, **kwargs):
        out = self.env.reset(seed=seed, **kwargs)
        # IsaacLab reset 通常返回 (obs, info)
        obs = out[0] if isinstance(out, (tuple, list)) else out

        # 保证都在 device 上
        for k, v in obs.items():
            if torch.is_tensor(v):
                obs[k] = v.to(self.device)

        # reset 后通常应当是 first
        # 你 env 里已经提供 is_first，这里兜底一下
        if "is_first" not in obs:
            obs["is_first"] = torch.ones(self.num_envs, device=self.device, dtype=torch.int32)
        else:
            obs["is_first"] = obs["is_first"].to(self.device)

        obs["is_last"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        obs["is_terminal"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        return obs

    def step(self, action):
        # 兼容 Dreamer 可能传 dict action：{"action": Tensor}
        if isinstance(action, dict):
            action = action.get("action", action)

        # action -> torch on device
        action = self._to_tensor(action, dtype=torch.float32)

        out = self.env.step(action)

        # 兼容两种常见 step 返回：
        # 1) Gymnasium: (obs, reward, terminated, truncated, info)
        # 2) IsaacLab DirectRLEnv: (obs, reward, reset_terminated, reset_time_outs, extras)
        if isinstance(out, (tuple, list)) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = self._to_tensor(terminated).bool()
            truncated = self._to_tensor(truncated).bool()
        elif isinstance(out, (tuple, list)) and len(out) == 4:
            obs, reward, done, info = out
            done = self._to_tensor(done).bool()
            terminated = done
            truncated = torch.zeros_like(done)
        else:
            raise RuntimeError(f"Unexpected env.step() return format: type={type(out)}, len={len(out) if isinstance(out,(tuple,list)) else 'NA'}")

        done = terminated | truncated

        # obs 放到 device
        for k, v in obs.items():
            if torch.is_tensor(v):
                obs[k] = v.to(self.device)

        # 你 env 里 is_first 已经给了（reset 后 episode_length_buf==0），这里不强行覆盖
        # 但如果缺失，就做一个合理兜底：auto-reset 环境里 returned obs 通常是 first
        if "is_first" not in obs:
            obs["is_first"] = done.to(torch.int32)
        else:
            obs["is_first"] = obs["is_first"].to(self.device)

        # 关键：由 terminated/truncated 填充 is_last / is_terminal
        obs["is_last"] = done.to(torch.int32)
        obs["is_terminal"] = terminated.to(torch.int32)

        # reward / done 也保持 torch（你的 simulate_vecenv 里会 .cpu().numpy())
        reward = self._to_tensor(reward, dtype=torch.float32)
        done_out = done  # bool tensor

        return obs, reward, done_out, info



import gym
import copy
class df_takeoff_wrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.device = env.device

		self.original_obs = None

	@property
	def observation_space(self):
		spaces = self.get_takeoff_obs_space
		pol_pos_space = gym.spaces.Box(spaces['policy']['eef_pos'].low[0], spaces['policy']['eef_pos'].high[0], spaces['policy']['eef_pos'].shape[1:], dtype=spaces['policy']['eef_pos'].dtype)
		pol_quat_space = gym.spaces.Box(spaces['policy']['eef_quat'].low[0], spaces['policy']['eef_quat'].high[0], spaces['policy']['eef_quat'].shape[1:], dtype=spaces['policy']['eef_quat'].dtype)
		new_spaces = {"agent_pos": pol_pos_space, 'agent_quat': pol_quat_space}
		for k in spaces["rgb_camera"].keys():
			img_shape = [*spaces['rgb_camera'][k].shape[1:]]
			img_shape[-1] = 3
			img_space = gym.spaces.Box(
							  np.moveaxis(np.zeros_like(spaces['rgb_camera'][k].low)[0, :, :, :3], -1, 0), \
							  np.moveaxis(1*np.ones_like(spaces['rgb_camera'][k].high)[0, :, :, :3], -1, 0), 
							  [3, 128, 128], dtype='uint8'
							  )
			new_spaces[k] = img_space

		obs_space_dict = gym.spaces.Dict(
							{
								**new_spaces,
							}
						)

		return obs_space_dict

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)

		self.original_obs = copy.deepcopy(obs)

		obs['front_cam'] = obs['front_cam'].permute(0, 3, 1, 2) / 255.0
		obs['wrist_cam'] = obs['wrist_cam'].permute(0, 3, 1, 2) / 255.0
		obs['agent_pos'] = obs['eef_pos']
		obs['agent_quat'] = obs['eef_quat']

		obs.pop('eef_pos')
		obs.pop('eef_quat')

		for k, v in obs.items():
			obs[k] = v.cpu().numpy()

		return obs

	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		self.original_obs = copy.deepcopy(obs)

		obs_new = {}
		obs_new['front_cam'] = obs['front_cam'].permute(0, 3, 1, 2) / 255.0
		obs_new['wrist_cam'] = obs['wrist_cam'].permute(0, 3, 1, 2) / 255.0
		obs_new['agent_pos'] = obs['eef_pos']
		obs_new['agent_quat'] = obs['eef_quat']

		for k, v in obs_new.items():
			obs_new[k] = v.cpu().numpy() # Need to squeeze

		reward = reward.cpu().numpy()
		done = done.item() # .cpu().numpy()

		return obs_new, reward, done, info
