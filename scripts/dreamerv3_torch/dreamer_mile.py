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
    args_cli.headless = True  # 强制无头模式运行

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

import os
import pathlib
import sys

# 使用 OSMesa 作为 Mujoco 的渲染后端（无显示渲染）
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

# 将当前文件所在目录加入 Python 搜索路径，方便本地模块导入
sys.path.append(str(pathlib.Path(__file__).parent))

import gymnasium as gym
import sys
import torch

# 导入 Isaac Lab 任务集合（保持导入以触发注册）
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

sys.path.append('latent_safety')  # 添加自定义 latent_safety 模块路径

import torch
from torch import nn

import dreamerv3_torch.models_mile as models
import dreamerv3_torch.tools as tools
import dreamerv3_torch.uncertainty as uncertainty
import envs.wrappers as wrappers
from dreamer_wrapper import DreamerVecEnvWrapper

# 将 Tensor 转为 NumPy 数组的辅助函数
to_np = lambda x: x.detach().cpu().numpy()


class DreamerMile(nn.Module):
    """
    Dreamer 算法主类，封装世界模型、策略行为、探索行为及训练逻辑。
    Main Dreamer class that encapsulates world model, task behavior,
    exploration behavior, and training logic.
    """
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(DreamerMile, self).__init__()
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
        self._wm = models.WorldModelMile(obs_space, act_space, self._step, config)

        # Imaginary rollout based behavior for task optimization
        # 基于“想象轨迹”（latent rollout）的任务行为（策略优化）

        # 如果配置允许且不是 Windows，使用 torch.compile 对网络进行编译优化
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)

        # 如果使用 ensemble，不确定性模型使用 OneStepPredictor
        if config.use_ensemble:
            self._disag_ensemble = uncertainty.OneStepPredictor(config, self._wm)
        else:
            self._disag_ensemble = None

    # def __call__(self, obs, reset, state=None, training=True):
    #     """
    #     主调用接口：在与环境交互时被调用。
    #     Main call: called during environment interaction.
    #     - 进行必要的训练步骤
    #     - 输出当前策略给出的动作
    #     """
    #     step = self._step
    #     if training:
    #         # 决定本次要执行多少次训练迭代（预训练或正常训练）
    #         # 先预训练一次，
    #         steps = (
    #             self._config.pretrain
    #             if self._should_pretrain()
    #             else self._should_train(step)
    #         )
    #         # 执行训练步骤
    #         for _ in range(steps):
    #             self._train(next(self._dataset))   # dataset是train_eps的一个包装器，从train_eps的数据中采样，所以train_eps更新后dataset也会更新
    #             self._update_count += 1
    #             self._metrics["update_count"] = self._update_count

    #         # 达到日志记录周期则写入日志与视频
    #         if self._should_log(step):
    #             # 标量指标写入
    #             for name, values in self._metrics.items():
    #                 self._logger.scalar(name, float(np.mean(values)))
    #                 self._metrics[name] = []
    #             # 可选的视频预测日志
    #             if self._config.video_pred_log:
    #                 if self._config.use_ensemble:
    #                     video_pred = self._wm.video_pred(
    #                         next(self._dataset),
    #                         ensemble=self._disag_ensemble,
    #                     )
    #                     self._logger.video("train_openl", video_pred)
    #                 else:
    #                     openl = self._wm.video_pred(next(self._dataset))
    #                     self._logger.video("train_openl", to_np(openl))
    #             # 写入日志（包含 FPS 等）
    #             self._logger.write(fps=True)

    #     # 根据当前观测和内部状态生成策略输出和新状态
    #     policy_output, state = self._policy(obs, state, training)

    #     if training:
    #         # 按环境并行实例数量推进 step 计数
    #         self._step += len(reset)
    #         # logger.step 按 action_repeat 还原为环境真实步数
    #         self._logger.step = self._config.action_repeat * self._step

    #     return policy_output, state
    def __call__(self, obs, reset, state=None, training=True):
        """
        与环境交互时调用：
        - 训练模式下先按调度训练 world model
        - 然后根据当前观测更新 posterior latent state
        - 再由 policy 输出动作
        """
        step = self._step

        # -------------------------
        # 1) 训练部分（保持你原来的逻辑）
        # -------------------------
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []

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

                self._logger.write(fps=True)

        # -------------------------
        # 2) 在线动作选择
        # -------------------------
        with torch.no_grad():
            # reset -> is_first
            reset = torch.tensor(reset, device=self._config.device, dtype=torch.float32)
            if reset.ndim == 0:
                reset = reset.unsqueeze(0)

            batch_size = len(reset)

            # 初始化 recurrent state
            if state is None:
                state = self._init_policy_state(batch_size)

            # 预处理当前单步观测
            obs = {
                k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
                for k, v in obs.items()
            }
            for k in obs.keys():
                if "cam" in k:
                    obs[k] = obs[k] / 255.0

            # 给 encoder / obs_step 补 is_first
            obs["is_first"] = reset

            # 编码当前观测，得到当前时刻 embedding x_t
            embed = self._wm.encoder(obs)

            # 单步 posterior 更新：
            # prev_action 用上一步真实执行动作（交互时就是上一时刻 agent 输出的动作）
            post, prior = self._wm.dynamics.obs_step(
                prev_state=state["latent"],
                prev_action=state["action"],
                embed=embed,
                is_first=reset,
                sample=training,   # 训练时可采样；评估时可设 False
                policy=self._wm.heads["action"],   # policy / action head
            )

            # 从 posterior latent state 提取 feature
            feat = self._wm.dynamics.get_feat(post)

            # policy 输出动作分布
            action_dist = self._wm.heads["action"](feat)

            # 执行动作：
            # 如果 action head 是分布，这里取 mode() 更稳
            # 若你想训练时更随机，也可改成 sample()
            if training and hasattr(action_dist, "sample"):
                action = action_dist.sample()
            elif hasattr(action_dist, "mode"):
                action = action_dist.mode()
            elif hasattr(action_dist, "mean"):
                action = action_dist.mean
            else:
                # 万一 action head 直接返回 tensor
                action = action_dist
            
            logprob = action_dist.log_prob(action)
            # 更新 recurrent state，供下一步使用
            new_state = {
                "latent": post,
                "action": action,
            }
        policy_output = {"action": action, "logprob": logprob}
        # -------------------------
        # 3) step 计数
        # -------------------------
        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step

        return policy_output, new_state
        
    def _init_policy_state(self, batch_size):
        latent = self._wm.dynamics.initial(batch_size)
        action = torch.zeros(
            batch_size,
            self._config.num_actions,
            device=self._config.device,
            dtype=torch.float32,
        )
        return {"latent": latent, "action": action}
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
        # ===== 这里：直接算 disagreement（不确定性）=====
        if self._config.use_ensemble and (self._disag_ensemble is not None):
            with torch.no_grad():
                if self._config.disag_action_cond:
                    inputs = torch.cat([feat.detach(), action.detach()], dim=-1)
                else:
                    inputs = feat.detach()

                # 你已有的接口：直接用
                # disagreement shape 通常是 [B, 1] 或 [B, ...]（取决于 EnsembleStochasticLinear 的最后一项）

        policy_output = {"action": action, "logprob": logprob}
        # if disagreement is not None:
        #     policy_output["disagreement"] = disagreement

        state = ({k: v.detach() for k, v in latent.items()}, action)
        return policy_output, state
        # policy_output = {"action": action, "logprob": logprob}
        # state = (latent, action)
        # return policy_output, state

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
        # 训练BC策略
        # bc_mets = self._task_behavior.train_bc(context["feat"], data["action"])
        # metrics.update(bc_mets)

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
                
    def train_world_model_only(self, training=True):
        """
        只更新 world model（encoder/dynamics/heads），不更新 task_behavior/expl_behavior。
        """
        if not training:
            return

        data = next(self._dataset)

        # 只跑 world model 的 _train
        post, context, mets = self._wm._train(data, ensemble=None)
        met = self._wm.train_uncertainty_only(
            data,
            ensemble=self._disag_ensemble,
        )
        # 记录 metrics（按你现有写法）
        self._update_count += 1
        self._metrics["update_count"] = self._update_count
        for k, v in mets.items():
            self._metrics.setdefault(k, []).append(v)
            
        # 记录不确定性训练的各项指标
        for name, value in met.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

        for name, values in self._metrics.items():
            self._logger.scalar(name, float(np.mean(values)))
            self._metrics[name] = []
        self._logger.write(fps=True, print_cli=False)

        self._step += 1
        self._logger.step = self._step


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
