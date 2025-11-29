from collections import deque
import hydra
from omegaconf import DictConfig
from .residual import ResidualPolicy
import torch
import torch.nn as nn

import numpy as np
from ipdb import set_trace as bp  # noqa
from typing import Dict, Union

from collections import namedtuple
from omegaconf import DictConfig
from hydra import initialize, initialize_config_dir, compose
from typing import Optional
import os
from .residual import ResidualPolicy
ResidualTrainingValues = namedtuple(
    "ResidualTrainingValues",
    [
        "residual_naction_samp",
        "residual_naction_mean",
        "logprob",
        "entropy",
        "value",
        "env_action",
        "next_residual_nobs",
    ],
)


# 解析 Hydra 配置文件
# config_path: 配置文件目录（绝对或相对）
# config_name: 配置文件名（不含 .yaml）
# version_base: Hydra 版本基准
# 返回 OmegaConf DictConfig
def load_cfg(
    config_path: str,
    config_name: str,
    version_base: Optional[str] = "1.2",
) -> DictConfig:
    # 根据路径类型选择初始化方式
    if os.path.isabs(config_path):
        # 绝对文件夹路径，使用 initialize_config_dir
        with initialize_config_dir(config_dir=config_path, version_base=version_base):
            cfg = compose(config_name=config_name)
    else:
        # 相对路径，使用 initialize
        with initialize(config_path=config_path, version_base=version_base):
            cfg = compose(config_name=config_name)
    return cfg

class ResidualACTPolicy:
    def __init__(
        self,
        policy: torch.nn.Module,
        stats: dict,
        device: str,
        num_envs: int,
        max_timesteps: int,
        state_dim: int,
        query_frequency: int = 50,
        temporal_agg: bool = True,
    ):
        self.policy = policy.to(device).eval()
        self.stats = stats
        self.device = device
        self.num_envs = num_envs
        self.max_timesteps = max_timesteps
        self.state_dim = state_dim
        self.query_frequency = query_frequency
        self.temporal_agg = temporal_agg

        if temporal_agg:
            self.all_time_actions = torch.zeros(
                num_envs,
                max_timesteps,
                max_timesteps + query_frequency,
                state_dim,
                device=device,
            )
        self.obs_dim = 31
        self.action_dim = self.state_dim
  
    def process_obs(self):
        pass

    def _pre_process(self, qpos_numpy: np.ndarray) -> torch.Tensor:
        x = (qpos_numpy - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return torch.from_numpy(x).float().to(self.device)

    def _post_process(self, raw_action: torch.Tensor) -> torch.Tensor:
        a_np = raw_action.cpu().numpy()
        a_np = a_np * self.stats["action_std"] + self.stats["action_mean"]
        return torch.tensor(a_np, device=self.device, dtype=torch.float32)

    def get_action(self, qpos_numpy: np.ndarray, curr_image: torch.Tensor, t) -> torch.Tensor:
        qpos = self._pre_process(qpos_numpy)
        if self.temporal_agg:

            self.future_actions = self.policy(qpos, curr_image)
            env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.query_frequency)
            src_t   = t.unsqueeze(1).expand(self.num_envs, self.query_frequency)
            # 构造 0,1,...,Q-1
            # 构造 0,1,...,Q-1
            offsets = torch.arange(self.query_frequency, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            tgt_t = t.unsqueeze(1).expand(self.num_envs, 1) + offsets   # (num_envs, Q)
            self.all_time_actions[env_ids, src_t, tgt_t] = self.future_actions[:, :self.query_frequency]

            idx = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                 # [N,1,1,1]
            idx = idx.expand(self.num_envs, self.max_timesteps, 1, self.state_dim)
            actions_curr = torch.gather(self.all_time_actions, 2, idx).squeeze(2)

            raw_action = torch.zeros(self.num_envs, self.state_dim, device=self.device)
            for i in range(self.num_envs):
                hist = actions_curr[i][torch.any(actions_curr[i] != 0, dim=1)]
                if hist.numel() == 0:
                    raw_action[i] = self.future_actions[i, 0]
                else:
                    L = hist.size(0)
                    w = torch.exp(-0.03 * torch.arange(L - 1, -1, -1, device=self.device, dtype=hist.dtype))
                    w /= w.sum()
                    raw_action[i] = (hist * w.unsqueeze(1)).sum(dim=0)
        else:
            if t % self.query_frequency == 0:
                self.future_actions = self.policy(qpos, curr_image)
                self.future_actions = self._post_process(self.future_actions)
            raw_action = self.future_actions[:, t % self.query_frequency]

        return self._post_process(raw_action)
    
    def get_chunked_action(self, t):
        return self.future_actions[:, t % self.query_frequency]
    def reset(self,env_ids):
        if self.temporal_agg:
            self.all_time_actions[env_ids, ...] = 0.0
