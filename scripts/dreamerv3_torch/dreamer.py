import argparse
import os
from omni.isaac.lab.app import AppLauncher

# Check if simulation app is already running (e.g., from Jupyter notebook)
# 检查仿真应用是否已经在运行（例如由 Jupyter notebook 启动）
def _is_app_already_running():
    try:
        import omni.kit.app
        return omni.kit.app.get_app() is not None
    except:
        return False

# Flag indicating whether an app is already running or launched from Jupyter
# 标志位：指示应用是否已经在运行，或者是否是从 Jupyter 中启动
_APP_ALREADY_RUNNING = _is_app_already_running() or os.environ.get("ISAAC_JUPYTER_KERNEL", "0") == "1"

# Only launch app if no app is already running
# 仅在没有已有应用运行时才启动新的仿真应用
if not _APP_ALREADY_RUNNING:
    # add argparse arguments
    # 添加命令行参数解析器
    parser = argparse.ArgumentParser(description="Isaac Lab environments.")
    # append AppLauncher cli args
    # 追加 AppLauncher 所需的命令行参数
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    # 解析命令行参数
    args_cli, remaining = parser.parse_known_args()

    # launch omniverse app
    # 启动 Omniverse 仿真应用
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    # App is already running (e.g., from Jupyter notebook)
    # 仿真应用已经在运行（例如由 Jupyter notebook 启动）
    simulation_app = None
    print("[INFO] Simulation app already running - skipping duplicate launch")
    # 打印提示：仿真应用已在运行，跳过重复启动

import functools
import os
import pathlib
import sys

# 使用 OSMesa 作为 Mujoco 的渲染后端（无显示渲染）
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

# 将当前文件所在目录加入 Python 搜索路径，方便本地模块导入
sys.path.append(str(pathlib.Path(__file__).parent))

import gymnasium as gym
import torch

import carb

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

# 导入 Isaac Lab 任务集合（保持导入以触发注册）
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import sys
sys.path.append('latent_safety')  # 添加自定义 latent_safety 模块路径

from dreamer_wrapper import DreamerVecEnvWrapper
import dreamerv3_torch.exploration as expl
import dreamerv3_torch.models as models
import dreamerv3_torch.tools as tools
import dreamerv3_torch.uncertainty as uncertainty
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd

# 将 Tensor 转为 NumPy 数组的辅助函数
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    """
    Dreamer 算法主类，封装世界模型、策略行为、探索行为及训练逻辑。
    Main Dreamer class that encapsulates world model, task behavior,
    exploration behavior, and training logic.
    """
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger

        # Logging scheduler (log every config.log_every steps)
        # 日志记录调度器（每 config.log_every 步记录一次）
        self._should_log = tools.Every(config.log_every)

        # 用于计算训练步长的批大小 = batch_size * batch_length
        batch_steps = config.batch_size * config.batch_length

        # Training scheduler (train every batch_steps / train_ratio env steps)
        # 训练调度器（每 batch_steps / train_ratio 个环境步长训练一次）1024/512
        self._should_train = tools.Every(batch_steps / config.train_ratio)

        # Pretraining is executed only once at the beginning
        # 预训练调度器：只在开始时执行一次预训练
        self._should_pretrain = tools.Once()

        # Reset scheduler (for periodic env reset if需要)
        # 重置调度器（每 config.reset_every 步执行一次重置）
        self._should_reset = tools.Every(config.reset_every)

        # 控制探索策略使用的步数上限（除以 action_repeat 以转换成环境步）
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))

        # 字典形式保存各种训练指标
        self._metrics = {}

        # this is update step
        # 这里的 step 是按 action_repeat 缩放后的“更新步”
        self._step = logger.step // config.action_repeat
        self._update_count = 0

        # 在线采样的数据集生成器（来自经验回放或收集的 trajectories）
        self._dataset = dataset

        # World model 包含编码器、动力学模型以及各种预测 head（reward 等）
        # World model that learns latent dynamics and predicts reward, etc.
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        # Imaginary rollout based behavior for task optimization
        # 基于“想象轨迹”（latent rollout）的任务行为（策略优化）
        self._task_behavior = models.ImagBehavior(config, self._wm)

        # 如果配置允许且不是 Windows，使用 torch.compile 对网络进行编译优化
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        # reward 函数：在 latent 特征空间上预测 reward 的均值
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()

        # Exploration behavior selector
        # 构造探索行为，根据 config.expl_behavior 选择具体策略
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,                     # 贪心：直接使用任务策略
            random=lambda: expl.Random(config, act_space),          # 随机策略
            plan2explore=lambda: expl.Plan2Explore(                 # Plan2Explore 探索策略
                config, self._wm, reward
            ),
        )[config.expl_behavior]().to(self._config.device)

        # 如果使用 ensemble，不确定性模型使用 OneStepPredictor
        if config.use_ensemble:
            self._disag_ensemble = uncertainty.OneStepPredictor(config, self._wm)
        else:
            self._disag_ensemble = None

    def __call__(self, obs, reset, state=None, training=True):
        """
        主调用接口：在与环境交互时被调用。
        Main call: called during environment interaction.
        - 进行必要的训练步骤
        - 输出当前策略给出的动作
        """
        step = self._step
        if training:
            # 决定本次要执行多少次训练迭代（预训练或正常训练）
            # 先预训练一次，
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            # 执行训练步骤
            for _ in range(steps):
                self._train(next(self._dataset))   # dataset是train_eps的一个包装器，从train_eps的数据中采样，所以train_eps更新后dataset也会更新
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            # 达到日志记录周期则写入日志与视频
            if self._should_log(step):
                # 标量指标写入
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                # 可选的视频预测日志
                if self._config.video_pred_log:
                    if self._config.use_ensemble:
                        video_pred = self._wm.video_pred(
                            next(self._dataset),
                            ensemble=self._disag_ensemble,
                        )
                        self._logger.video("train_openl", video_pred)
                    else:
                        openl = self._wm.video_pred(next(self._dataset))
                        self._logger.video("train_openl", to_np(openl))
                # 写入日志（包含 FPS 等）
                self._logger.write(fps=True)

        # 根据当前观测和内部状态生成策略输出和新状态
        policy_output, state = self._policy(obs, state, training)

        if training:
            # 按环境并行实例数量推进 step 计数
            self._step += len(reset)
            # logger.step 按 action_repeat 还原为环境真实步数
            self._logger.step = self._config.action_repeat * self._step

        return policy_output, state

    def train_model_only(self, training=True):
        """
        仅训练世界模型（不更新策略），通常用于单独预训练世界模型。
        Train only the world model without updating the policy.
        """
        step = self._step
        if training:
            # 单次世界模型训练
            self._train(next(self._dataset))
            self._update_count += 1
            self._metrics["update_count"] = self._update_count

            # 每 1000 步可选地记录视频预测
            if (step + 1) % 1000 == 0:
                if self._config.video_pred_log:
                    if self._config.use_ensemble:
                        video_pred = self._wm.video_pred(
                            next(self._dataset),
                            ensemble=self._disag_ensemble,
                        )
                        self._logger.video("train_openl", video_pred)
                    else:
                        openl = self._wm.video_pred(next(self._dataset))
                        self._logger.video("train_openl", to_np(openl))

            # 写入标量日志
            for name, values in self._metrics.items():
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []

            # 写日志但不在命令行打印
            self._logger.write(fps=True, print_cli=False)

        if training:
            self._step += 1
            self._logger.step = self._step

    def train_uncertainty_only(self, training=True):
        """
        仅训练不确定性相关模块（如 ensemble），不更新主世界模型。
        Train only the uncertainty components (e.g. ensemble).
        """
        step = self._step
        if training:
            # 使用世界模型的专用接口训练不确定性模块
            met = self._wm.train_uncertainty_only(
                data=next(self._dataset),
                ensemble=self._disag_ensemble,
            )
            self._update_count += 1
            self._metrics["update_count"] = self._update_count

            # 每 1000 步可选地记录视频预测
            if (step + 1) % 1000 == 0:
                if self._config.video_pred_log:
                    if self._config.use_ensemble:
                        video_pred = self._wm.video_pred(
                            next(self._dataset),
                            ensemble=self._disag_ensemble,
                        )
                        self._logger.video("train_openl", video_pred)
                    else:
                        openl = self._wm.video_pred(next(self._dataset))
                        self._logger.video("train_openl", to_np(openl))

            # 记录不确定性训练的各项指标
            for name, value in met.items():
                if name not in self._metrics.keys():
                    self._metrics[name] = [value]
                else:
                    self._metrics[name].append(value)

            # 写入标量日志
            for name, values in self._metrics.items():
                self._logger.scalar(name, float(np.mean(values)))
                self._metrics[name] = []

            # 写日志但不在命令行打印
            self._logger.write(fps=True, print_cli=False)

        if training:
            self._step += 1
            self._logger.step = self._step

    def _policy(self, obs, state, training):
        """
        根据当前观测和内部 latent state 计算策略输出（动作和对数概率）。
        Compute the policy output (action and logprob) given observation and state.
        """
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # 预处理观测（归一化、类型转换等）
        obs = self._wm.preprocess(obs)
        # 编码器将观测映射到 embedding
        embed = self._wm.encoder(obs)
        # 使用动力学模型进行观测步更新（结合上一 latent、动作、当前 embed）
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        # 在评估模式下可以使用状态均值代替随机部分
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]

        # 提取用于策略的特征向量 feat
        feat = self._wm.dynamics.get_feat(latent)

        # 选择行为：评估 / 探索 / 正常训练
        if not training:
            # 评估时使用策略分布的 mode（确定性）
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            # 在探索阶段使用探索行为（如随机或 Plan2Explore）
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            # 正常训练阶段使用任务策略的采样
            actor = self._task_behavior.actor(feat)
            action = actor.sample()

        # 计算动作在策略分布下的对数概率
        logprob = actor.log_prob(action)

        # 分离图，使后续环境交互不反向传播梯度
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        # 如果使用 onehot_gumble 分布，则将动作索引转为 one-hot
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1),
                self._config.num_actions,
            )

        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        """
        单次训练步骤：更新世界模型和行为策略（以及探索策略）。
        Single training step: update world model, behavior, and optionally exploration.
        """
        metrics = {}

        # 训练世界模型，得到 posterior 状态、上下文以及训练指标
        post, context, mets = self._wm._train(data, ensemble=self._disag_ensemble)
        metrics.update(mets)
        start = post

        # 在想象轨迹中使用的 reward 函数（基于 latent 状态）
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        # 训练任务行为（策略和价值网络等），返回的最后一个元素是指标
        metrics.update(self._task_behavior._train(start, reward)[-1])

        # 如果探索策略不是 greedy，则训练探索行为
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        # 将所有指标累积到 self._metrics 中，用于后续平均与记录日志
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    """
    统计给定目录中所有 .npz 轨迹文件的总步数。
    Count total steps from all *.npz episodes in a folder.
    约定文件名形如 xxx-<step>.npz，取其中 step-1 相加。
    """
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    """
    基于 episodes 生成可迭代的数据集（用于训练）。
    Create a dataset generator from episodes with given batch_length and batch_size.
    """
    # 从 episode 中按 batch_length 采样序列
    generator = tools.sample_episodes(episodes, config.batch_length)
    # 再将生成器打包为批数据流
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, num_envs):
    """
    创建并包装 Isaac Lab 多环境实例，适配 Dreamer 接口。
    Create and wrap Isaac Lab vectorized environment compatible with Dreamer.
    """
    # 从任务名解析出环境配置
    env_cfg = parse_env_cfg(
        config.task,
        device='cuda',
        num_envs=num_envs,
        use_fabric=True,
    )
    env_cfg.seed = 0  # 固定随机种子以便复现

    # create environment
    # 创建原始 Isaac Lab Gym 环境
    env = gym.make(config.task, cfg=env_cfg)

    # 用 DreamerVecEnvWrapper 包装环境，适配 Dreamer 所需接口
    env = DreamerVecEnvWrapper(env, device=env_cfg.sim.device)

    # 对动作做归一化处理（将动作范围规范到 [-1, 1]）
    env = wrappers.NormalizeActions(env)

    # 从环境返回的字典中选取 "action" 键对应的动作
    env = wrappers.SelectAction(env, key="action")

    # 为每个环境实例分配唯一 ID，方便记录和区分
    env = wrappers.UUID(env)

    return env
