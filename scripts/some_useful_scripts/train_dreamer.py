import argparse
import pathlib
import functools
import numpy as np
import ruamel.yaml as yaml
import gymnasium as gym
import torch
import sys
from tqdm import trange
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools

# 将 latent_safety 目录加入 Python 搜索路径，便于后续模块导入


from torch import nn
from torch import distributions as torchd
import my_ur3_project.tasks  # noqa: F401
# 工具函数：将 torch 张量转为 numpy 数组（常用于日志或可视化）
to_np = lambda x: x.detach().cpu().numpy()


def count_steps(folder):
	"""
	统计给定目录下所有 .npz 文件的总步数。
	假设文件名形如 xxx-000010.npz，则从中解析出 10，并做累加。
	"""
	return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def main(config):
	"""
	主训练入口函数：
	1. 设置随机种子和确定性运行（可选）
	2. 构建日志 / 数据目录
	3. 加载离线数据（如有）
	4. 创建环境并对动作空间做归一化
	5. 视情况用随机策略预填充数据集（prefill）
	6. 创建 Dreamer 智能体并进行训练（纯模型训练或在线训练）
	"""
	# 设置所有随机种子，保证实验可复现
	tools.set_seed_everywhere(config.seed)
	# 可选：启用确定性运行（会牺牲部分性能）
	if config.deterministic_run:
		tools.enable_deterministic_run()
	
	# 日志目录（支持 ~ 展开）
	logdir = pathlib.Path(config.logdir).expanduser()
	# 若未指定 traindir / evaldir，则默认放在 logdir 下
	config.traindir = config.traindir or logdir / "train_eps"
	config.evaldir = config.evaldir or logdir / "eval_eps"

	# 由于环境内部每次 step 会重复同一个动作 action_repeat 次，
	# 因此要把配置里的步数等除以 action_repeat，统一成“环境 step”尺度
	config.steps //= config.action_repeat
	config.eval_every //= config.action_repeat
	config.log_every //= config.action_repeat
	config.time_limit //= config.action_repeat

	print("Logdir", logdir)
	# 创建目录（若已存在则忽略）
	logdir.mkdir(parents=True, exist_ok=True)
	config.traindir.mkdir(parents=True, exist_ok=True)
	config.evaldir.mkdir(parents=True, exist_ok=True)

	# 统计训练数据中已存在的步数，用于日志起始 step
	step = count_steps(config.traindir)
	# logger 中的 step 是“环境步数”，所以乘以 action_repeat
	logger = tools.Logger(logdir, config.action_repeat * step)

	# 保存当前配置（便于复现实验）
	logger.config(vars(config))
	logger.write()

	print("Create envs.")
	# ---------- 加载训练 episodes（离线数据） ----------
	if config.offline_traindir:
		# 从多个离线 traindir 中加载数据
		train_eps = None
		for offline_dir in config.offline_traindir:
			# 支持在路径中使用格式化占位符（如 {seed} 等）
			directory = offline_dir.format(**vars(config))
			# 将多目录的数据合并到同一个 train_eps 中
			train_eps = tools.load_episodes(
				directory,
				limit=config.dataset_size,
				episodes=train_eps
			)
	else:
		# 若没有指定离线数据目录，则从当前 traindir 中加载已有数据
		directory = config.traindir
		train_eps = tools.load_episodes(directory, limit=config.dataset_size)
		
	# ---------- 加载评估 episodes（离线数据） ----------
	if config.offline_evaldir:
		directory = config.offline_evaldir.format(**vars(config))
	else:
		directory = config.evaldir
	# eval_eps 一般只需要少量 episode（这里限制为 1）
	eval_eps = tools.load_episodes(directory, limit=1)

	# ---------- 创建向量化环境 ----------
	train_envs = dreamer.make_env(config, num_envs=config.envs)
	# 单环境的动作空间（所有 envs 共用）
	acts = train_envs.single_action_space

	# Normalized Action!
	# 将动作空间统一缩放到 [-0.5, 0.5]，便于网络训练与数值稳定
	acts.low = 0.5 * np.ones_like(acts.low) * -1
	acts.high = 0.5 * np.ones_like(acts.high) 
	print("Action Space", acts)
	# 记录动作维度：如果是离散动作，用 n；否则用连续动作维度
	config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

	# 用于保存 simulate_vecenv 返回的 RNN/环境内部状态
	state = None

	# 若不是纯离线数据训练，则需要用随机策略预填充数据集
	if not config.offline_traindir:
		# 需要填充的步数 = 目标 prefill 步数 - 已经存在的步数
		prefill = max(0, config.prefill - count_steps(config.traindir))
		print(f"Prefill dataset ({prefill} steps).")
		# 若为离散动作空间，使用 one-hot 分布随机采样
		if hasattr(acts, "discrete"):
			random_actor = tools.OneHotDist(
				torch.zeros(config.num_actions).repeat(config.envs, 1)
			)
		else:
			# 连续动作空间，使用 Uniform(low, high) 随机采样
			random_actor = torchd.independent.Independent(
				torchd.uniform.Uniform(
					torch.tensor(acts.low).repeat(config.envs, 1),
					torch.tensor(acts.high).repeat(config.envs, 1),
				),
				1,
			)

		# 随机策略：只负责生成随机动作和对应的 logprob
		def random_agent(o, d, s):
			"""
			o: 观测 observation
			d: done 标志
			s: 内部状态（这里不用）
			"""
			action = random_actor.sample()
			logprob = random_actor.log_prob(action)
			return {"action": action, "logprob": logprob}, None

		# 使用随机策略在环境中采集 prefill 步的数据，并写入 traindir
		state = tools.simulate_vecenv(
			random_agent,
			train_envs,
			train_eps,
			config.traindir,
			logger,
			limit=config.dataset_size,
			steps=prefill,
		)
		# 更新 logger 的 step（环境总步数）
		logger.step += prefill * config.action_repeat
		print(f"Logger: ({logger.step} steps).")

	print("Simulate agent.")
	# 将 episodes 转为 Dreamer 可用的数据集（迭代器 / buffer）
	train_dataset = dreamer.make_dataset(train_eps, config)
	eval_dataset = dreamer.make_dataset(eval_eps, config)

	# ---------- 创建 Dreamer 智能体 ----------
	agent = dreamer.Dreamer(
		train_envs.single_observation_space,  # 观测空间
		acts,                                 # 动作空间
		config,
		logger,
		train_dataset,                        # 训练数据集
	).to(config.device)
	# 先禁用所有参数梯度，训练时由内部方法控制需要梯度的部分
	agent.requires_grad_(requires_grad=False)
	
	# Load a pretrained model
	# 若指定了 model_path，则加载一个已训练好的模型权重
	if config.model_path:
		checkpoint = torch.load(config.model_path)
		agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
		del checkpoint
		# 清理 GPU 显存缓存
		torch.cuda.empty_cache()  # Clear GPU memory cache
		# 允许再次预训练（取消只预训练一次的限制）
		agent._should_pretrain._once = False

	# ---------- 仅训练模型（不与环境交互）模式 ----------
	if config.model_only:
		# 使用离线数据训练世界模型（model-only）
		for idx_step in trange(
			int(config.steps),
			desc="Training Dreamer with Offline Dataset",
			ncols=0,
			leave=False
		):
			# 只训练世界模型、编码器等部分
			agent.train_model_only(training=True)
			# 如需单独训练不确定性模型，可开启下面这行
			# agent.train_uncertainty_only(training=True)

			# 按 log_every 间隔保存最新模型 latest.pt
			if ((idx_step + 1) % config.log_every) == 0:
				items_to_save = {
					"agent_state_dict": agent.state_dict(),
					"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
				}
				torch.save(items_to_save, logdir / "latest.pt")
			
			# 按 save_every 间隔保存打了步数标记的模型快照
			if ((idx_step + 1) % config.save_every) == 0:
				items_to_save = {
					"agent_state_dict": agent.state_dict(),
					"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
				}
				torch.save(items_to_save, logdir / "model_{:04d}".format(idx_step + 1))
			
	else:
		# ---------- 在线训练模式：交替评估 + 训练 ----------
		while agent._step < config.steps + config.eval_every:
			# 写出积累的日志（标量、图片、视频等）
			logger.write()

			# 若配置要求进行评估，则先跑评估 episode
			if config.eval_episode_num > 0:
				print("Start evaluation.")

				with torch.no_grad():
					# 构造评估策略：training=False 表示不更新参数
					eval_policy = functools.partial(agent, training=False)
					# 使用当前策略与环境交互若干评估 episode
					tools.simulate_vecenv(
						eval_policy,
						train_envs,
						eval_eps,
						config.evaldir,
						logger,
						is_eval=True,
						episodes=config.eval_episode_num,
						save_success=True
					)
					# 可选：记录世界模型的 open-loop 视频预测
					if config.video_pred_log:
						if config.use_ensemble:
							# 使用不确定性集成模型生成多模型的预测视频
							video_pred = agent._wm.video_pred(
								next(eval_dataset),
								ensemble=agent._disag_ensemble
							)
							logger.video("eval_openl", video_pred)
						else:
							# 单世界模型的视频预测
							video_pred = agent._wm.video_pred(next(eval_dataset))
							logger.video("eval_openl", to_np(video_pred))
						
			print("Start training.")
			# 使用当前 agent 与环境交互，收集训练数据并执行训练
			state = tools.simulate_vecenv(
				agent,
				train_envs,
				train_eps,
				config.traindir,
				logger,
				limit=config.dataset_size,
				steps=config.eval_every,  # 每次训练 eval_every 个环境 step
				state=state,              # 传递 RNN / 环境状态
				save_success=True
			)
			# 每轮训练后将最新模型保存为 latest.pt
			items_to_save = {
				"agent_state_dict": agent.state_dict(),
				"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
			}
			torch.save(items_to_save, logdir / "latest.pt")
			# TODO: 可加入类似 model_only 中的 save_every 逻辑

	# 训练结束后，尝试关闭环境（防止资源泄露）
	try:
		train_envs.close()
	except Exception:
		pass


from datetime import datetime

if __name__ == "__main__":
	# ------------------ 第一阶段：基础命令行参数解析 ------------------
	parser = argparse.ArgumentParser()
	parser.add_argument("--configs", nargs="+")  # 指定要使用的配置块名（在 configs.yaml 中）
	parser.add_argument(
		"--enable_cameras", action="store_true", default=True
	)
	parser.add_argument(
		"--headless", action="store_true", default=True
	)
	parser.add_argument(
		"--disable_fabric", action="store_true", default=False,
		help="Disable fabric and use USD I/O operations."
	)
	# 任务名称（Isaac 仿真任务）
	parser.add_argument(
		"--task",
		type=str,
		default="My-Isaac-Ur3-PipeRelCam-Ik-RL-Direct-v0",
		help="Name of the task."
	)
	# 另一种任务难度的默认值示例（当前注释掉）
	# parser.add_argument("--task", type=str, default="Isaac-Takeoff-Hard-Franka-IK-Rel-v0", help="Name of the task.")

	# 先解析已知参数，其余参数留待后面解析
	args, remaining = parser.parse_known_args()

	# 从 configs.yaml 中加载所有配置（如 defaults / specific config 名）
	configs = yaml.safe_load(
		(pathlib.Path(sys.argv[0]).parent / "../dreamerv3_torch/configs.yaml").read_text()
	)
	
	# 递归更新字典，用于将多个配置块叠加
	def recursive_update(base, update):
		for key, value in update.items():
			if isinstance(value, dict) and key in base:
				recursive_update(base[key], value)
			else:
				base[key] = value

	# 组合需要加载的配置块列表：["defaults", ...]
	name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
	defaults = {}
	for name in name_list:
		recursive_update(defaults, configs[name])

	# Overwrite defaults with command-line arguments
	# 用第一阶段解析到的命令行参数覆盖 defaults 的同名字段（命令行优先）
	for key, value in vars(args).items():
		defaults[key] = value

	# ------------------ 第二阶段：根据 defaults 自动生成命令行参数 ------------------
	parser = argparse.ArgumentParser()
	for key, value in sorted(defaults.items(), key=lambda x: x[0]):
		# tools.args_type 用来推断参数类型（int / float / bool / str 等）
		arg_type = tools.args_type(value)
		parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
	
	# 使用这个完整的 parser 解析剩余参数，得到最终配置
	final_config = parser.parse_args(remaining)

	# 为本次实验创建带时间戳的子目录
	curr_time = datetime.now().strftime("%m%d/%H%M%S")
	expt_name = (curr_time + "_" + final_config.remark)
	final_config.logdir = f"{final_config.logdir}/{expt_name}"

	# 启动主训练流程
	main(final_config)

	# Close simulation app if it exists (won't exist in Jupyter environments)
	# 如果存在 Isaac 的 simulation_app（例如在独立进程中运行的仿真器），则关闭它
	if hasattr(dreamer, 'simulation_app') and dreamer.simulation_app is not None:
		dreamer.simulation_app.close()
