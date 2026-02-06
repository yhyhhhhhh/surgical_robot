import numpy as np
import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

class RslRlImaginationVecEnv(VecEnv):
    def __init__(self, env_model, obs_key="policy", clip_actions: float | None = 1.0):
        self.env = env_model
        self.obs_key = obs_key
        self.clip_actions = clip_actions

        self.num_envs = int(env_model.num_envs)
        self.device = env_model.device
        self.max_episode_length = int(env_model.horizon)
        self.num_actions = int(env_model.act_dim)

        # rsl_rl/isaaclab 通常使用 [-clip, clip] 的连续动作空间
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # runner 需要 episode_length_buf 可读写（用于随机初始化 episode length 等）
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # 注意：runner 不一定会主动 reset，所以这里先 reset 一次（与 IsaacLab wrapper 一致）:contentReference[oaicite:5]{index=5}
        obs = self.env.reset()
        self.num_obs = int(obs.shape[-1])
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

        self._obs = obs

    def reset(self):
        obs = self.env.reset()
        self.episode_length_buf.zero_()
        self._obs = obs
        return TensorDict({self.obs_key: obs}, batch_size=[self.num_envs]), {}

    def get_observations(self):
        return TensorDict({self.obs_key: self._obs}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -float(self.clip_actions), float(self.clip_actions))

        obs, rew, done = self.env.step(actions)   # obs:(N,D) rew:(N,) done:(N,)bool
        self.episode_length_buf += 1

        dones = done.to(torch.long)  # rsl_rl 期望 long 类型:contentReference[oaicite:6]{index=6}

        # 关键：auto-reset（让下一个 obs 对应 reset 后的初始态）
        if done.any():
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            obs_reset = self.env.reset_idx(env_ids)
            obs[env_ids] = obs_reset
            self.episode_length_buf[env_ids] = 0

        self._obs = obs
        extras = {}  # 如果你需要 time_outs，可在这里加（finite horizon 通常不需要）:contentReference[oaicite:7]{index=7}
        return TensorDict({self.obs_key: obs}, batch_size=[self.num_envs]), rew, dones, extras

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
