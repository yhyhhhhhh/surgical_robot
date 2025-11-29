import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

#2073331757
# 1344147256
# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

import numpy as np
def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer

def build_mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation,
    output_std=1.0,
    bias_on_last_layer=True,
    last_layer_bias_const=0.0,
):
    act_func = getattr(nn, activation)
    layers = []
    layers.append(
        layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation)
    )
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            layer_init(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation
            )
        )
        layers.append(act_func())
    layers.append(
        layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
            bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)

# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=-1):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions),
                                 nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# load and wrap the Isaac Lab environment
# env = load_isaaclab_env(task_name="Template-Isaac-Ur3-ReachInPipe-Direct-v0", num_envs=64,headless=False)
env = load_isaaclab_env(task_name="My-Isaac-Ur3-Pipe-Ik-Act-Direct-v0", num_envs=1,headless=True)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.98
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-3
cfg["critic_learning_rate"] = 5e-3
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-3
cfg["initial_entropy_value"] = 0.2
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/Isaac-Ant-v0"
# ── 2. 新增 rewards_shaper，充当 reward_scale ───────────
SCALE = 0.2                 # 把 +100 → +2， -1~+5 → -0.02~+0.10
cfg["rewards_shaper"] = lambda r, *_: r * SCALE
agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# agent.load("/home/yhy/DVRK/IsaacLabExtensionTemplate/runs/torch/Isaac-Ant-v0/24-12-24_11-14-10-399590_SAC/checkpoints/best_agent.pt")
# agent.load("/home/yhy/DVRK/IsaacLabExtensionTemplate/runs/torch/Isaac-Ant-v0/24-12-24_23-44-38-701720_SAC/checkpoints/best_agent.pt")
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 500000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()