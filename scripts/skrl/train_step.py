import torch
import torch.nn as nn
import numpy as np

# ---------------- skrl imports ----------------
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

# seed
set_seed(42)

# ---------------- utils (optional) ----------------
def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ---------------- models ----------------
# PPO policy: Gaussian actor (mean + log_std)
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 256), "ReLU"),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128), "ReLU"),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.num_actions), "Tanh", std=0.01),
            nn.Tanh()
        )
        # log_std as parameter
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # return mean, log_std, extra_info
        return self.net(inputs["states"]), self.log_std_parameter, {}

# PPO value: state -> V(s)
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            layer_init(nn.Linear(self.num_observations, 256), "ReLU"),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128), "ReLU"),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), "Tanh", std=1.0)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ---------------- load env ----------------
# 你换成自己的 task
env = load_isaaclab_env(
    task_name="My-Isaac-Ur3-PipeRel-Ik-RL-Direct-v0",
    num_envs=16,           # 建议训练时不要 1
    headless=True
)
env = wrap_env(env)

device = env.device

# ---------------- memory ----------------
# PPO 是 on-policy，memory_size 通常设置为 rollouts
# skrl 的配置示例也常用 memory_size 与 rollouts 对齐（或在 Runner yaml 中用 -1 自动匹配）:contentReference[oaicite:1]{index=1}
# 这里我们手写脚本，就显式用 rollouts
rollouts = 16

memory = RandomMemory(
    memory_size=rollouts,
    num_envs=env.num_envs,
    device=device
)

# ---------------- models dict ----------------
models = {
    "policy": StochasticActor(env.observation_space, env.action_space, device),
    "value":  Value(env.observation_space, env.action_space, device)
}

# ---------------- PPO config ----------------
cfg = PPO_DEFAULT_CONFIG.copy()

# 核心训练参数
cfg["rollouts"] = rollouts
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 1
cfg["discount_factor"] = 0.98
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4

# 稳定性
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True

# entropy/value 权重（可按你任务调）
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 2.0

# 预处理（对你这种精细任务通常有帮助）
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": (1,), "device": device}

# reward scale（可选）
SCALE = 1.0
cfg["rewards_shaper"] = lambda r, *_: r * SCALE

# logging / checkpoints
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/ur3_pipe_ppo_step"
cfg["experiment"]["experiment_name"] = "PPO"

# ---------------- agent ----------------
agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device
)
STEP_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# ---------------- StepTrainer config ----------------
cfg_trainer = STEP_TRAINER_DEFAULT_CONFIG.copy()
cfg_trainer.update({
    "timesteps": 500000,
    "headless": True,
    # 这个值决定 trainer 会不会自动处理 env 的 info/log
    # 你环境里有 self.extras["log"] 的话可以试 "log"
    "environment_info": "log",
    "close_environment_at_exit": False,
})
# StepTrainer 每次 train 会返回 obs/reward/... 方便你自定义统计 :contentReference[oaicite:2]{index=2}
trainer = StepTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ---------------- training loop ----------------
timesteps = cfg_trainer["timesteps"]
episode_rewards_env0 = []
episode_ee_dist_env0 = []
for t in range(timesteps):
    obs, rewards, terminated, truncated, infos = trainer.train(timestep=t, timesteps=timesteps)

    episode_ee_dist_env0.append(infos["log"]["metrics/ee_obj_dist"].item())
    # ===== 2) env0 单 episode 内 reward 分布 =====
    episode_rewards_env0.append(rewards[0].item())
    done0 = (terminated[0] | truncated[0]).item()
    
    if done0:
        if agent.writer is not None:
            # 1. 处理 Reward 数据
            # 将 list 转为 tensor，并增加一个维度变为 [1, Length]
            reward_signal = torch.tensor(episode_rewards_env0, dtype=torch.float32).unsqueeze(0)
            agent.writer.add_audio(
                "reward/episode_env0_curve",  # 建议改名为 curve 以示区分
                reward_signal,
                global_step=t,
                sample_rate=1  # 关键：1秒对应1步
            )

            # 2. 处理 Distance 数据
            dist_signal = torch.tensor(episode_ee_dist_env0, dtype=torch.float32).unsqueeze(0)
            agent.writer.add_audio(
                "infos/episode_env0_dist_curve", 
                dist_signal,
                global_step=t,
                sample_rate=1
            )

            agent.writer.add_scalar(
                "reward/episode_env0_return",
                sum(episode_rewards_env0),
                global_step=t
            )
        episode_rewards_env0 = []
        episode_ee_dist_env0 = []
# close
env.close()
