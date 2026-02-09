import pathlib
import argparse
import functools
import numpy as np
import torch
import gymnasium as gym
import sys

sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools

import my_ur3_project.tasks  # noqa: F401


def main(config):
    # -----------------------
    # 基础设置
    # -----------------------
    tools.set_seed_everywhere(config.seed)
    config.evaldir = pathlib.Path(config.evaldir).expanduser()
    config.evaldir.mkdir(parents=True, exist_ok=True)

    print("Eval dir:", config.evaldir)

    # -----------------------
    # 创建环境
    # -----------------------
    envs = dreamer.make_env(config, num_envs=config.envs)
    acts = envs.single_action_space

    # Action normalization（必须和训练一致）
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high)

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # -----------------------
    # 创建空 replay（只为接口完整）
    # -----------------------
    eval_eps = tools.load_episodes(config.evaldir, limit=1)
    eval_dataset = dreamer.make_dataset(eval_eps, config)

    logger = tools.Logger(config.evaldir, step=0)

    # -----------------------
    # 创建 Agent
    # -----------------------
    agent = dreamer.Dreamer(
        envs.single_observation_space,
        acts,
        config,
        logger,
        eval_dataset,
    ).to(config.device)

    agent.requires_grad_(False)

    # -----------------------
    # 加载模型
    # -----------------------
    checkpoint = torch.load(config.model_path, map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    # agent.eval()

    print(f"Loaded model from {config.model_path}")
    print(f"Loaded model from {config.model_path}")

    # ================= 复制以下代码块 =================
    def inspect_model_structure(model):
        print(f"\n{'='*20} 模型层级结构 (Layer Hierarchy) {'='*20}")
        # 1. 打印基础层级 (如果模型太深，这部分可能很长)
        print(model)
        
        print(f"\n{'='*20} 详细参数统计 (Detailed Params) {'='*20}")
        # 2. 打印整齐的参数表格
        header = f"{'Layer Name':<50} | {'Shape':<25} | {'Params':>10}"
        print(header)
        print("-" * len(header))
        
        total_params = 0
        trainable_params = 0
        
        # 遍历所有参数
        for name, param in model.named_parameters():
            # 获取形状字符串，例如 [64, 3, 7, 7]
            shape_str = str(list(param.shape))
            param_count = param.numel()
            
            # 统计
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            
            # 打印每一行
            print(f"{name:<50} | {shape_str:<25} | {param_count:>10,}")
            
        print("-" * len(header))
        print(f"Total Params: {total_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print("="*85 + "\n")

    # 执行查看
    inspect_model_structure(agent)
    # -----------------------
    # 构造评估策略
    # -----------------------
    eval_policy = functools.partial(agent, training=False)

    # -----------------------
    # 跑评估
    # -----------------------
    with torch.no_grad():
        tools.simulate_vecenv(
            eval_policy,
            envs,
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
            save_success=True,     # 保存 success 标记
        )

    logger.write()

    try:
        envs.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--task", type=str, default="My-Isaac-Ur3-PipeRelCamFinal-Ik-RL-Direct-v0")
    parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    # parser.add_argument("--model_path", type=str, default="latent_safety/log/dreamerv3/world_model_only/0208/202025_test/latest.pt")
    parser.add_argument("--eval_episode_num", type=int, default=20)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--evaldir", type=str, default="latent_safety/log/dreamerv3/1225/004059_test/eval_eps")
    parser.add_argument(
		"--enable_cameras", action="store_true", default=False
	)
    parser.add_argument(
		"--headless", action="store_true", default=False
	)

    args, remaining = parser.parse_known_args()
    args.enable_cameras = True
    args.rendering_mode = "quality"   # performance / balanced / quality
    args.headless = False   # 强制 headless
    # -----------------------
    # 读取 configs.yaml（与你训练一致）
    # -----------------------
    import pathlib
    import ruamel.yaml as yaml

    # 1. 初始化一个 YAML 解析器实例 (指定 typ='safe' 来替代原来的 safe_load)
    yaml_parser = yaml.YAML(typ='safe', pure=True)

    # 2. 使用解析器的 .load() 方法
    configs = yaml_parser.load(
        (pathlib.Path(__file__).parent / "../dreamerv3_torch/configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                recursive_update(base[k], v)
            else:
                base[k] = v

    defaults = {}
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    for name in name_list:
        recursive_update(defaults, configs[name])

    for k, v in vars(args).items():
        defaults[k] = v

    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", type=tools.args_type(v), default=v)

    config = parser.parse_args(remaining)
    main(config)
