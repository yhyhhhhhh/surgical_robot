import collections
import datetime
import io
import json
import math
import numpy as np
import os
import pathlib
import random
import time
import torch
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F

to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)

class DummyLogger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

import numpy as np
import os
import pathlib
import tempfile
import time
import uuid

import imageio

import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class Logger:
    def __init__(
        self,
        logdir,
        step,
        use_wandb=True,
        use_tensorboard=True,
        tensorboard_subdir="tb",
    ):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step
        self._use_wandb = bool(use_wandb)
        self._use_tensorboard = bool(use_tensorboard)
        self._tb_writer = None
        self._tb_logdir = pathlib.Path(logdir) / tensorboard_subdir

        if self._use_wandb:
            name = str(logdir).split("/")[-2] + "_" + str(logdir).split("/")[-1]
            # Initialize WandB
            wandb.init(project="isaac_takeoff", config={"logdir": str(logdir)}, name=name)

        if self._use_tensorboard:
            if SummaryWriter is None:
                print("[Logger] TensorBoard disabled: SummaryWriter not available.")
                self._use_tensorboard = False
            else:
                self._tb_logdir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(self._tb_logdir))
                print(f"[Logger] TensorBoard logging to: {self._tb_logdir}")

    def config(self, config_dict):
        # Convert PosixPath objects to strings
        config_dict = {
            k: str(v) if isinstance(v, pathlib.PosixPath) else v
            for k, v in config_dict.items()
        }
        # Log the config
        if self._use_wandb:
            wandb.config.update(config_dict)
        if self._tb_writer is not None:
            config_text = json.dumps(config_dict, indent=2, ensure_ascii=False, default=str)
            self._tb_writer.add_text("config", f"```json\n{config_text}\n```", self.step)

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        # value: 期望是 (B, T, H, W, C)
        self._videos[name] = np.array(value)

    # ------------ 关键新增：把 numpy 视频写成 mp4 文件 ------------
    def _array_to_mp4(self, value, fps=20):
        """
        value: numpy 数组，形状 (T, H, W, C)，dtype uint8
        返回：临时 mp4 文件路径
        """
        value = np.asarray(value)
        assert value.ndim == 4, f"video array must be 4D (T,H,W,C), got {value.shape}"

        # 创建临时目录和文件名
        tmpdir = tempfile.mkdtemp(prefix="wandb_video_")
        filename = os.path.join(tmpdir, f"{uuid.uuid4().hex}.mp4")

        writer = imageio.get_writer(filename, fps=fps)
        for frame in value:
            writer.append_data(frame)
        writer.close()

        return filename

    def _to_tb_video_tensor(self, value):
        """Convert video from (B,T,H,W,C) to torch tensor (B,T,C,H,W), float in [0,1]."""
        value = np.asarray(value)
        if value.ndim != 5:
            raise ValueError(f"Expected video value with 5 dims (B,T,H,W,C), got {value.shape}")

        if np.issubdtype(value.dtype, np.floating):
            video = value.astype(np.float32)
            # If model outputs roughly [-1, 1], map to [0, 1].
            if video.min() < 0.0 and video.min() >= -1.1 and video.max() <= 1.1:
                video = 0.5 * (video + 1.0)
            # If values are in [0,255] float, normalize to [0,1].
            elif video.max() > 1.5:
                video = video / 255.0
            video = np.clip(video, 0.0, 1.0)
        else:
            video = value.astype(np.float32) / 255.0

        return torch.from_numpy(video).permute(0, 1, 4, 2, 3).contiguous()

    def write(self, fps=False, step=False, fps_namespace="", print_cli=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            fps_str = fps_namespace + "fps"
            scalars.append((fps_str, self._compute_fps(step)))
        if print_cli:
            print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))

        # Log metrics
        metrics = {"step": step, **dict(scalars)}
        if self._use_wandb:
            wandb.log(metrics, step=step)
        if self._tb_writer is not None:
            for k, v in scalars:
                self._tb_writer.add_scalar(k, float(v), global_step=step)

        # ----- images -----
        for name, value in self._images.items():
            if np.shape(value)[0] == 3:  # (C,H,W) -> (H,W,C)
                value = np.transpose(value, (1, 2, 0))
            if self._use_wandb:
                wandb.log({name: [wandb.Image(value, caption=name)]}, step=step)
            if self._tb_writer is not None:
                img = np.asarray(value)
                try:
                    if img.ndim == 2:
                        self._tb_writer.add_image(name, img, global_step=step, dataformats="HW")
                    elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
                        self._tb_writer.add_image(name, img, global_step=step, dataformats="CHW")
                    elif img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                        self._tb_writer.add_image(name, img, global_step=step, dataformats="HWC")
                except Exception as e:
                    print(f"[Logger] Failed to log image {name} to TensorBoard: {e}")

        # ----- videos -----
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")

            # TensorBoard video: keep batch as batch.
            if self._tb_writer is not None:
                try:
                    tb_video = self._to_tb_video_tensor(value)
                    self._tb_writer.add_video(name, tb_video, global_step=step, fps=20)
                except Exception as e:
                    print(f"[Logger] Failed to log video {name} to TensorBoard: {e}")

            # WandB video: tile batch along width then encode mp4.
            if self._use_wandb:
                wandb_value = np.asarray(value)
                if np.issubdtype(wandb_value.dtype, np.floating):
                    wandb_value = np.clip(255 * wandb_value, 0, 255).astype(np.uint8)

                if wandb_value.ndim != 5:
                    raise ValueError(
                        f"Expected video value with 5 dims (B,T,H,W,C), got {wandb_value.shape}"
                    )

                B, T, H, W, C = wandb_value.shape
                # (B, T, H, W, C) -> (T, H, B*W, C)
                wandb_value = wandb_value.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)
                try:
                    video_path = self._array_to_mp4(wandb_value, fps=20)
                    wandb.log({name: wandb.Video(video_path)}, step=step)
                except Exception as e:
                    print(f"[Logger] Failed to log video {name}: {e}")

        # 清空缓存
        self._scalars = {}
        self._images = {}
        self._videos = {}
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        if self._use_wandb:
            wandb.log({f"scalars/{name}": value}, step=step)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(f"scalars/{name}", float(value), global_step=step)

    def offline_video(self, name, value, step):
        if self._tb_writer is not None:
            try:
                tb_video = self._to_tb_video_tensor(value)
                self._tb_writer.add_video(name, tb_video, global_step=step, fps=20)
            except Exception as e:
                print(f"[Logger] Failed to log offline video {name} to TensorBoard: {e}")

        # value: 预期 (B, T, H, W, C)
        wandb_value = np.asarray(value)
        if np.issubdtype(wandb_value.dtype, np.floating):
            wandb_value = np.clip(255 * wandb_value, 0, 255).astype(np.uint8)

        if wandb_value.ndim != 5:
            raise ValueError(f"Expected video value with 5 dims (B,T,H,W,C), got {wandb_value.shape}")

        B, T, H, W, C = wandb_value.shape
        wandb_value = wandb_value.transpose(1, 2, 0, 3, 4).reshape(T, H, B * W, C)

        if self._use_wandb:
            try:
                video_path = self._array_to_mp4(wandb_value, fps=20)
                wandb.log({name: wandb.Video(video_path)}, step=step)
            except Exception as e:
                print(f"[Logger] Failed to log offline video {name}: {e}")


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    num_env = envs.num_envs
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(envs.num_envs, bool)
        length = np.zeros(envs.num_envs, np.int32)
        obs = [None] * envs.num_envs
        agent_state = None
        reward = [0] * envs.num_envs
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)
                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1: linear RGB -> sRGB"""
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    # piecewise
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * np.power(x, 1.0 / 2.4) - a)

def save_as_npz(data, filename="trajectories_test.npz"):
    # 将 OrderedDict 扁平化处理，方便存储
    # key 为 UUID，value 为内部的字典
    save_dict = {key: np.array(value) for key, value in data.items()}
    np.savez_compressed(filename, **save_dict)
    print(f"数据已压缩保存至 {filename}")



def save_traj_drop_first_and_last(trajectories, pose_data, filename="trajectories_aligned.npz"):
    """
    逻辑修改：
    1. 必须满足 轨迹数 == Pose数，否则报错。
    2. 如果相等：同时舍弃第一条和最后一条轨迹 (以及对应的 Pose)，保留中间段。
    3. 保存结果。
    """
    
    # 1. 获取所有轨迹的 UUID
    all_uuids = list(trajectories.keys())
    num_traj = len(all_uuids)
    
    # 2. 获取 Pose 的数量 (兼容 Tensor, List, Array)
    num_pose = pose_data.shape[0] if hasattr(pose_data, 'shape') else len(pose_data)

    print(f"数据检查: 轨迹 {num_traj} 条, Pose {num_pose} 个")

    # 3. 严格校验：必须相等
    if num_traj != num_pose:
        raise ValueError(f"数量不符合预期！要求严格相等，实际：Traj={num_traj}, Pose={num_pose}")
    
    # 安全检查：如果数据少于2条，掐头去尾后就没了
    if num_traj < 2:
        print("警告：数据量不足 2 条，舍弃首尾后将为空！")
        uuids_to_save = []
    else:
        # --- 核心逻辑：舍弃第一条 [0] 和最后一条 [-1] ---
        print("-> 数量对齐。正在舍弃第一条和最后一条轨迹...")
        uuids_to_save = all_uuids[1:-1] 

    # 4. 开始对齐合并
    save_dict = {}
    
    for i, uuid in enumerate(uuids_to_save):
        # 4.1 获取轨迹内容
        content = trajectories[uuid]
        
        # 4.2 获取 Pose 
        # 【关键点】因为 uuids_to_save 跳过了第 0 个 uuid
        # 所以这里的 Pose 也必须跳过第 0 个，取 i + 1
        raw_pose = pose_data[i + 1]
        
        # --- 类型清洗 (Tensor/List -> Numpy) ---
        if isinstance(raw_pose, torch.Tensor):
            clean_pose = raw_pose.detach().cpu().numpy()
        elif isinstance(raw_pose, (list, tuple)):
            clean_pose = np.array(raw_pose)
        else:
            clean_pose = np.array(raw_pose) # numpy or other

        # 4.3 注入 Pose 到字典中
        # 确保 content 是字典或可以被修改的对象，如果它是 tensor/array 需要先根据你的数据结构处理
        # 这里假设 content 是一个字典
        if isinstance(content, dict):
            content['pose_command'] = clean_pose
            final_data = content
        else:
            # 如果 content 本身是数组，可能需要把它包装一下
            # 根据你之前的代码逻辑，这里直接转 array
            # 但通常建议保存为 dict 结构更清晰
            final_data = content 
            # 警告：如果 content 不是 dict，这里无法插入 pose_command
            # 假设你的数据结构允许这样做，或者你是在这里构造 save_dict 的 value
        
        # 4.4 准备保存
        save_dict[uuid] = final_data

    # 5. 保存
    # 这里的 save_dict 结构是 {uuid: {traj_data..., 'pose_command': ...}}
    np.savez_compressed(filename, **save_dict)
    
    print(f"保存成功！原始 {num_traj} 条，已保存 {len(save_dict)} 条 (舍弃了首尾)。")
    if num_traj >= 2:
        print(f"被舍弃的 UUID: {all_uuids[0]} (首), {all_uuids[-1]} (尾)")
        

def _seq_to_np(seq):
    arrs = [np.asarray(x) for x in seq]
    try:
        return np.stack(arrs, axis=0)
    except ValueError:
        return np.array(arrs, dtype=object)

def save_one_episode_npz(ep, save_path):
    ep_data = {k: _seq_to_np(v) for k, v in ep.items()}
    np.savez_compressed(save_path, **ep_data)

def save_new_finished_episodes(cache, outdir):
    os.makedirs(outdir, exist_ok=True)

    for ep_id, ep in cache.items():
        save_path = os.path.join(outdir, f"{ep_id}.npz")

        # 已经保存过就跳过，避免覆盖
        if os.path.exists(save_path):
            continue

        # 只保存完整结束的轨迹
        is_finished = False
        if "is_last" in ep and len(ep["is_last"]) > 0:
            is_finished = is_finished or bool(np.asarray(ep["is_last"][-1]))
        if "is_terminal" in ep and len(ep["is_terminal"]) > 0:
            is_finished = is_finished or bool(np.asarray(ep["is_terminal"][-1]))

        if not is_finished:
            continue

        save_one_episode_npz(ep, save_path)
        # print(f"[saved] {save_path}")

def simulate_vecenv(
    agent,  # 智能体
    vecenv,  # 一个包含N个子环境的单一向量化环境
    cache,  # 数据缓存
    directory,  # 保存数据的目录
    logger,  # 日志记录器
    is_eval=False,  # 是否为评估模式
    limit=None,  # 数据集大小限制
    steps=0,  # 总步数
    episodes=0,  # 总回合数
    state=None,  # 初始化状态
    save_success=False  # 是否保存成功的轨迹
):

    num_env = vecenv.num_envs
    id_bank = [str(uuid.uuid4()) for _ in range(num_env)]  # 为每个子环境生成唯一的ID

    # 解包或初始化仿真状态
    if state is None:
        step, episode = 0, 0
        done = np.ones(num_env, bool)  
        length = np.zeros(num_env, np.int32) 
        obs = None  
        agent_state = None  
        reward = np.zeros(num_env, dtype=np.float32) 
    else:
        step, episode, done, length, obs, agent_state, reward = state  

    # 如果环境在done时自动重置子环境，初始时只需要进行一次重置
    if obs is None:
        obs = vecenv.reset()
        for i in range(num_env):
            t = {key: value[i].detach().cpu() for key, value in obs.items()}
            t["reward"] = 0.0
            t["discount"] = 1.0
            t["failure"] = 0.0
            add_to_cache(cache, id_bank[i], t)  
    
    pose_buffer = []
    pose_buffer.append('Initial placeholder for pose buffer')
    while ((steps and step < steps) or (episodes and episode < episodes)):

        # step_info = f"{step + 1}/{steps}" if steps is not None else f"{step + 1} (没有上限)"
        # episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (没有上限)"
        # print(f"当前步骤: {step_info} (步长: {length}) | 当前回合: {episode_info}", end="\r", flush=True)

        obs_dict = obs
        action, agent_state = agent(obs_dict, done, agent_state)
        next_obs, next_reward, next_done, info = vecenv.step(action)
        next_done = next_done.cpu().numpy()
        episode += int(next_done.sum())
        length += 1
        step += num_env
        length *= (1 - next_done)

        for i in range(num_env):
            transition = {key: value[i].detach().cpu().numpy() for key, value in next_obs.items()}
            if next_done[i]:
                # overwrite_keys = ["image", "policy"]
                overwrite_keys = ["policy"]
                for key in overwrite_keys:
                    transition[key] = obs[key][i].detach().cpu().numpy()
            if isinstance(action, dict):
                for k, v in action.items():
                    transition[k] = v[i].cpu().numpy()
            else:
                transition["action"] = np.array(action)
            transition["reward"] = next_reward[i].item()
            transition["discount"] = np.array(1.0, dtype=np.float32)
            add_to_cache(cache, id_bank[i], transition)

        # === 2) 再生成下一步 policy 用的 obs_policy，并只对 done env 替换 reset 帧 ===
        obs_policy = next_obs
        if next_done.any():
            reset_obs = vecenv.reset(seed=next_done) 
            obs_policy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in next_obs.items()}
            for k, v in obs_policy.items():
                if torch.is_tensor(v) and k in reset_obs and torch.is_tensor(reset_obs[k]):
                    obs_policy[k][next_done] = reset_obs[k][next_done]
            obs_policy["is_first"][next_done] = 1
            obs_policy["is_last"][next_done] = 0
            obs_policy["is_terminal"][next_done] = 0

        done_indices = np.where(next_done)[0]
        if len(done_indices) > 0:
            for i in done_indices:
                index = id_bank[i]
                length = len(cache[index]["reward"]) - 1
                score = float(np.array(cache[index]["reward"]).sum())
                # video_front = cache[index]["image"]
                # video = video_front
                pose_tensor = info['pose']['metrics/pose_command']
                clean_pose = pose_tensor.detach().cpu().numpy()
                pose_buffer.append(clean_pose)

                log_data = info['log']
                for key, value in log_data.items():
                    logger.scalar(key, float(value.item()))

                if not is_eval:
                    save_new_finished_episodes(cache, "/home/yhy/IsaacLabExtensionTemplate_lite/data2/")
                    logger.scalar(f"训练返回", score)
                    logger.scalar(f"训练步长", length)
                    logger.scalar(f"训练回合", len(cache))
                    logger.scalar("训练/已完成环境步", step)
                    logger.scalar("训练/已完成回合数", episode)
                    if steps:
                        logger.scalar("训练/累计步数进度(%)", 100.0 * float(step) / float(steps))
                    if episodes:
                        logger.scalar("训练/累计回合进度(%)", 100.0 * float(episode) / float(episodes))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    # logger.video(f"评估策略", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"评估返回", score)
                        logger.scalar(f"评估步长", length)
                        logger.scalar(f"评估回合", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True

                id_bank[i] = str(uuid.uuid4())
                # ✅ 立刻往新 id 里插入 reset 起始帧（否则 episode 会从 o1 开始）
                if reset_obs is not None:
                    first = {k: reset_obs[k][i].detach().cpu().numpy() for k in reset_obs if torch.is_tensor(reset_obs[k])}
                    first["is_first"] = np.array(1, dtype=np.int32)
                    first["is_last"] = np.array(0, dtype=np.int32)
                    first["is_terminal"] = np.array(0, dtype=np.int32)
                    first["reward"] = np.array(0.0, dtype=np.float32)
                    first["discount"] = np.array(1.0, dtype=np.float32)
                    # reset 步：action 填 0
                    if isinstance(action, dict):
                        for k, v in action.items():
                            first[k] = np.zeros_like(v[i].detach().cpu().numpy())
                    else:
                        first["action"] = np.zeros_like(action[i].detach().cpu().numpy())
                    add_to_cache(cache, id_bank[i], first)
            if not is_eval:
                # 裁剪在整批 done 处理完成后再进行，避免删掉本轮仍会访问的 episode。
                step_in_dataset = erase_over_episodes(
                    cache, limit, protected_keys=set(id_bank)
                )
                logger.scalar(f"数据集大小", step_in_dataset)

        obs = obs_policy
        reward = next_reward
        done = next_done
    # save_traj_drop_first_and_last(cache, pose_buffer, filename=os.path.join(directory, "random_trajectories_with_pose.npz"))
    # 如果是评估模式，保留最小的缓存
    if is_eval:
        if len(cache) > 0:
            cache.popitem(last=True)   # 先删除最后一个
        while len(cache) > 1:
            cache.popitem(last=False)  # 再从前面删，直到只剩 1 个
    
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)



def collect_vecenv(
    agent, vecenv, cache, directory,
    logger=None, steps=0, episodes=0, state=None,
    save_success=False, success_key="success"
):
    os.makedirs(directory, exist_ok=True)

    num_env = vecenv.num_envs
    id_bank = [str(uuid.uuid4()) for _ in range(num_env)]

    if state is None:
        step, episode = 0, 0
        done = np.ones(num_env, bool)
        length = np.zeros(num_env, np.int32)
        obs = None
        agent_state = None  # 不用也保留，方便对齐你原返回
        reward = np.zeros(num_env, dtype=np.float32)
    else:
        step, episode, done, length, obs, agent_state, reward = state

    # reset + t=0
    if obs is None:
        obs = vecenv.reset()
        for i in range(num_env):
            t0 = {k: obs[k][i].detach().cpu().numpy() for k in obs.keys()}
            t0["reward"] = np.array(0.0, np.float32)
            t0["discount"] = np.array(1.0, np.float32)
            t0["failure"] = np.array(0.0, np.float32)
            add_to_cache(cache, id_bank[i], t0)

    # 1. 初始化一个空列表（对齐 simulate_vecenv）
    pose_buffer = []
    pose_buffer.append('Initial placeholder for pose buffer')

    while ((steps and step < steps) or (episodes and episode < episodes)):
        # 1) 推理动作：agent(obs) -> tensor or dict
        with torch.inference_mode():
            act = agent(obs['policy'])

        action = act if isinstance(act, dict) else {"action": act}

        # 2) env step
        next_obs, next_reward, next_done, info = vecenv.step(action)
        next_done = next_done.cpu().numpy() if hasattr(next_done, "cpu") else np.asarray(next_done)

        # counters
        episode += int(next_done.sum())
        length += 1
        step += num_env
        length *= (1 - next_done)

        # 3) 写 transition
        for i in range(num_env):
            transition = {k: next_obs[k][i].detach().cpu().numpy() for k in next_obs.keys()}

            # done 时覆写 policy（保持你 simulate 的习惯）
            if next_done[i] and ("policy" in obs):
                transition["policy"] = obs["policy"][i].detach().cpu().numpy()

            # 存 action（dict 的每个 key 都存）
            for k, v in action.items():
                transition[k] = v[i].detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)[i]

            transition["reward"] = next_reward[i].item() if hasattr(next_reward[i], "item") else float(next_reward[i])
            transition["discount"] = np.array(1.0, dtype=np.float32)  # 按你 simulate：永远 1

            add_to_cache(cache, id_bank[i], transition)

        # 4) done 的 env：partial reset -> obs_policy；并插入新 episode 的 reset 起始帧
        obs_policy = next_obs
        reset_obs = None
        if next_done.any():
            reset_obs = vecenv.reset(seed=next_done)  # 只 reset done 的 env（按你的约定）

            obs_policy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in next_obs.items()}
            for k, v in obs_policy.items():
                if torch.is_tensor(v) and (reset_obs is not None) and (k in reset_obs) and torch.is_tensor(reset_obs[k]):
                    obs_policy[k][next_done] = reset_obs[k][next_done]

            # 标记 reset 帧（只对 done env）
            if "is_first" in obs_policy:    obs_policy["is_first"][next_done] = 1
            if "is_last" in obs_policy:     obs_policy["is_last"][next_done] = 0
            if "is_terminal" in obs_policy: obs_policy["is_terminal"][next_done] = 0

        done_indices = np.where(next_done)[0]
        if len(done_indices) > 0:
            for i in done_indices:

                # pose 记录（完全对齐 simulate_vecenv：不吞异常，直接取并 append 整批）
                pose_tensor = info['pose']['metrics/pose_command']
                clean_pose = pose_tensor.detach().cpu().numpy()
                pose_buffer.append(clean_pose)

                # 更新 ID（开新 episode）
                id_bank[i] = str(uuid.uuid4())

                # ✅ 插入 reset 起始帧到新 id（按你的 simulate）
                if reset_obs is not None:
                    first = {k: reset_obs[k][i].detach().cpu().numpy()
                             for k in reset_obs if torch.is_tensor(reset_obs[k])}

                    first["is_first"] = np.array(1, np.int32)
                    first["is_last"] = np.array(0, np.int32)
                    first["is_terminal"] = np.array(0, np.int32)

                    first["reward"] = np.array(0.0, np.float32)
                    first["discount"] = np.array(1.0, np.float32)

                    # reset 步 action 置 0（对每个 action key）
                    for k, v in action.items():
                        vv = v[i].detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)[i]
                        first[k] = np.zeros_like(vv)

                    add_to_cache(cache, id_bank[i], first)

        obs = obs_policy
        reward = next_reward
        done = next_done

    # ✅ 最终统一保存（对齐 simulate_vecenv 的文件名）
    save_traj_drop_first_and_last(
        cache, pose_buffer,
        filename=os.path.join(directory, "random_trajectories_with_pose.npz")
    )

    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)
def _stack_sequence(seq):
    """
    seq: list of scalars/ndarrays
    Return a numeric ndarray (avoid dtype=object) so np.load works normally.
    """
    arr = np.array(seq)
    if arr.dtype == object:
        arr = np.stack(seq, axis=0)
    return arr


def save_episode_npz(cache, episode_id, directory, suffix_int=None):
    """
    Save cache[episode_id] (dict of lists) to a .npz episode file.

    IMPORTANT:
    - Adds numeric suffix to filename to be compatible with your count_steps()
      which parses the trailing "-000010.npz" integer.
    """
    ep = cache[episode_id]
    out = {k: _stack_sequence(v) for k, v in ep.items()}

    # Consistency check: all keys must have same time length as reward
    if "reward" not in out:
        raise KeyError("Episode has no 'reward' key; cannot save.")
    T = len(out["reward"])
    for k, v in out.items():
        if len(v) != T:
            raise ValueError(f"Episode length mismatch: key={k}, len={len(v)}, reward_len={T}")

    os.makedirs(directory, exist_ok=True)
    if suffix_int is None:
        fname = f"{episode_id}.npz"
    else:
        fname = f"{episode_id}-{int(suffix_int):06d}.npz"  # compatible with count_steps()
    path = os.path.join(str(directory), fname)
    np.savez_compressed(path, **out)
    return path


def _extract_success(info, env_i, success_key):
    """
    Robustly extract success flag for env_i from vectorized env info.
    Returns float in [0,1] if found, else None.
    """
    try:
        # Case 1: info is list/tuple of per-env dicts
        if isinstance(info, (list, tuple)):
            log = info[env_i].get("log", {})
            v = log.get(success_key, None)
        else:
            # Case 2: info is a dict; log fields may be vectors
            log = info.get("log", {})
            v = log.get(success_key, None)
            if v is not None and hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                v = v[env_i]

        if v is None:
            return None
        return float(v.item()) if hasattr(v, "item") else float(v)
    except Exception:
        return None


def get_uncertainty(wm, ensemble, latent, actions):

    with torch.no_grad():
        feats = wm.dynamics.get_feat(latent)
        
        inputs = torch.concat([feats, actions['action']], -1)
        uncertainty = ensemble.intrinsic_reward_penn(inputs)

        # FIXME: nf
        # uncertainty = 1 - ensemble.calculate_likelihood(feats)

        return uncertainty
    
def get_kld(wm, post, prior):

    with torch.no_grad():
        kl_free = 1.0
        dyn_scale = 0.5
        rep_scale = 0.1
        _, _, value, _ = wm.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale
                )

        return value
    
def get_recon_loss(wm, post, obs):

    with torch.no_grad():

        data = wm.preprocess(obs)
        feat = wm.dynamics.get_feat(post)
        pred = wm.heads["decoder"](feat.unsqueeze(0))
        loss_front = -pred['front_cam'].log_prob(data["front_cam"].unsqueeze(0)).item()
        loss_wrist = -pred['wrist_cam'].log_prob(data["wrist_cam"].unsqueeze(0)).item()
    
        return loss_front + loss_wrist

def forward_latent(wm, latent_prev, act_prev, obs):
    with torch.no_grad():
        data = wm.preprocess(obs)
        embed = wm.encoder(data)

        # Latent and action is None for the initial
        latent_post, latent_prior = wm.dynamics.obs_step(latent_prev, act_prev['action'], embed, obs["is_first"], sample=False)

        return latent_post, latent_prior

def post_process_actions(action_cmd: torch.Tensor) -> torch.Tensor:

    action_cmd = torch.clamp(action_cmd, -0.5, 0.5)
    neg_mask = action_cmd[:, -1] < 0
    action_cmd[neg_mask, -1] = -1.0
    action_cmd[~neg_mask, -1] = 1.0
    
    return action_cmd

def filter_action(wm, ensemble, latent, actions, latent_filter, uq_thr=4.2, reachability_thr=0.1, denom=1):
    with torch.no_grad():
        action = actions['action']
        uncertainty = get_uncertainty(wm, ensemble, latent, actions)
        feats = wm.dynamics.get_feat(latent)

        pred_next_latent = wm.dynamics.img_step(latent, action)
        pred_next_feat = wm.dynamics.get_feat(pred_next_latent)
        next_best_act, log_prob = latent_filter.select_action(pred_next_feat, eval_mode=True)

        reachability_value_Q1 = latent_filter.critic1(pred_next_feat, next_best_act).item()
        reachability_value_Q2 = latent_filter.critic2(pred_next_feat, next_best_act).item()
        reachability_value = min(reachability_value_Q1, reachability_value_Q2)

        # # 3. Filter action with optimal policy
        is_filtered = False
        if (uncertainty > uq_thr) or (reachability_value < reachability_thr) :
            new_actions, log_prob = latent_filter.select_action(feats, eval_mode=True)
            action[:, :6] = post_process_actions(new_actions)[:, :6] / denom
            is_filtered = True
        
    return action, is_filtered

import omni.isaac.core.utils.torch as torch_utils


def reset_episode(vecenv, latent_dynamics, seed=None):
    vecenv.seed(seed)
    obs = vecenv.reset()

    # torch_utils.set_seed(seed, torch_deterministic=True)
    if seed is None:
        torch_utils.set_seed(-1, torch_deterministic=True)
    else:
        torch_utils.set_seed(seed, torch_deterministic=True)

    data = latent_dynamics.preprocess(obs)
    embed = latent_dynamics.encoder(data)
    latent = None
    action = None
    latent, _ = latent_dynamics.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=False)

    return obs, latent, None

def reset_episode_df(vecenv, latent_dynamics, seed=None):
    vecenv.seed(seed)
    obs = vecenv.reset()

    if seed is None:
        torch_utils.set_seed(-1, torch_deterministic=True)
    else:
        torch_utils.set_seed(seed, torch_deterministic=True)

    obs_original = vecenv.original_obs
    data = latent_dynamics.preprocess(obs_original)
    embed = latent_dynamics.encoder(data)
    latent = None
    action = None
    latent, _ = latent_dynamics.dynamics.obs_step(latent, action, embed, obs_original["is_first"], sample=False)

    return obs, latent, obs_original

# Convert NumPy data types to Python native types
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):  # Convert NumPy integers to Python int
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):  # Convert NumPy floats to Python float
        return float(obj)
    else:
        return obj
    
def evaluate_venv_filtering(
    agent_base,
    latent_dynamics,
    latent_ensemble,
    latent_filter,
    config,
    vecenv,        # A single vectorized environment with N sub-envs
    episodes=10,  # total number of finished trajectories to test
    state=None,
    is_filter=False
):

    total_episode_cnt = 0
    global_results = []
    result_file_path = os.path.join(config.outFolder, "results.json")
    # === Unpack or initialize simulation state ===
    num_env = vecenv.num_envs
    if state is None:
        step = 0
        episode = 0
        done = np.ones(num_env, bool)
        length = np.zeros(num_env, np.int32)
        obs = None
        agent_state = None
    else:
        step, episode, done, length, obs, agent_state, reward = state

    # Reset environment if needed and initialize the latent state.
    obs, latent, agent_state = reset_episode(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)

    # --- Containers for per-episode uncertainty logging ---
    # For each sub-environment, we will accumulate the uncertainty values over the trajectory.
    traj_uncertainties = [[] for _ in range(num_env)]
    traj_filtered = [[] for _ in range(num_env)]
    traj_kld = [[] for _ in range(num_env)]
    traj_recon_loss = [[] for _ in range(num_env)]

    # To record finished episode details:
    finished_episodes_log = []
    success_count = 0
    failure_count = 0
    timeout_count = 0

    # === Main evaluation loop ===
    # (Stop when we have finished the desired number of episodes or reached the max step count.)

    while (episode < episodes):

        # === (1) Nominal Policy ===
        obs_dict = obs
        # Get action from the nominal policy.
        action, agent_state = agent_base._policy(obs_dict, agent_state, training=False)
        action['action'][:, -1] *= 2 # gripper action

        # === (2) Filter Action ===
        is_filtered = False
        if is_filter:
            new_action, is_filtered = \
                filter_action(latent_dynamics, latent_ensemble, latent, action, latent_filter, \
                  uq_thr=config.ood_threshold, reachability_thr=config.reachability_threshold, denom=config.action_denom)
            action['action'] = new_action

        filtered_action = action
        next_obs, next_reward, next_done, info = vecenv.step(filtered_action)
        next_done = next_done.cpu().numpy()

        # === (3) Calculate and record uncertainty values ===
        final_uncertainties = get_uncertainty(latent_dynamics, latent_ensemble, latent, filtered_action)
        for i in range(num_env):
            traj_uncertainties[i].append(final_uncertainties[i].item())
            traj_filtered[i].append(is_filtered)

        # === (4) Update Latent ===
        latent, latent_prior = forward_latent(latent_dynamics, latent, filtered_action, next_obs)

        for i in range(num_env):
            kld_uncertaity = get_kld(latent_dynamics, latent, latent_prior)
            traj_kld[i].append(kld_uncertaity[i].item())
            recon_loss = get_recon_loss(latent_dynamics, latent, next_obs)
            traj_recon_loss[i].append(recon_loss)

        # === Update counters ===
        length += 1
        step += num_env
        # Capture the current length values (these represent the step count for this episode)
        current_lengths = length.copy()

        # Identify which environments have just finished an episode.
        done_indices = np.where(next_done)[0]
        if len(done_indices) > 0:
            # For each finished sub-env, determine the termination outcome and log uncertainty stats.
            for i in done_indices:

                # Success
                if info['log']['Episode_Termination/success']:
                    success_count += 1
                    outcome = 'success'
                
                # Failure
                elif info['log']['Episode_Termination/failure']:
                    failure_count += 1
                    outcome = 'failure'

                # Timeout.
                elif info['log']['Episode_Termination/time_out']:
                    timeout_count += 1
                    outcome = 'incompletion'

                else:
                    failure_count += 1
                    outcome = 'failure' #'incompletion'
                    
                # --- Compute uncertainty statistics for this episode ---
                traj_unc = traj_uncertainties[i]
                num_filtered = np.sum(traj_filtered[i])
                max_unc = np.max(traj_unc) if len(traj_unc) > 0 else 0.0
                mean_unc = np.mean(traj_unc) if len(traj_unc) > 0 else 0.0

                traj_kld_value = traj_kld[i]
                max_kld = np.max(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
                mean_kld = np.mean(traj_kld_value) if len(traj_kld_value) > 0 else 0.0

                traj_recon_loss_value = traj_recon_loss[i]
                max_recon_loss = np.max(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0
                mean_recon_loss = np.mean(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0

                result = {
                    'episode_index': episode,
                    'length': current_lengths[i],
                    'max_uncertainty': max_unc,
                    'mean_uncertainty': mean_unc,
                    'max_kld': max_kld,
                    'mean_kld': mean_kld,
                    'max_recon': max_recon_loss,
                    'mean_recon': mean_recon_loss,
                    'num_filtered': num_filtered,
                    'outcome': outcome
                }
                finished_episodes_log.append(result)
                # Reset uncertainty storage for that environment (so the next episode starts fresh)
                traj_uncertainties[i] = []
                traj_filtered[i] = []
                traj_kld[i] = []
                traj_recon_loss[i] = [] 
                # print(result)

                # Save result
                total_episode_cnt += 1 
                converted_result = {k: convert_numpy(v) for k, v in result.items()}
                global_results.append(converted_result)

            # Increase our episode count by the number of environments that finished this step.
            episode += len(done_indices)

            next_obs, latent, agent_state = reset_episode(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)

            if episode % 10 == 0 :
                lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
                max_uncs = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
                mean_uncs = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
                max_klds = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
                mean_klds = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
                max_recon = np.array([r['max_recon'] for r in finished_episodes_log], dtype=np.float32)
                mean_recon = np.array([r['mean_recon'] for r in finished_episodes_log], dtype=np.float32)
                num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

                # Compute means
                length_mean = np.mean(lengths)
                max_unc_mean = np.mean(max_uncs)
                mean_unc_mean = np.mean(mean_uncs)
                max_klds_mean = np.mean(max_klds)
                mean_klds_mean = np.mean(mean_klds)
                max_recon_mean = np.mean(max_recon)
                mean_recon_mean = np.mean(mean_recon)
                num_filtered_mean = np.mean(num_filtered)

                # Compute standard deviations
                length_std = np.std(lengths)
                max_unc_std = np.std(max_uncs)
                mean_unc_std = np.std(mean_uncs)
                max_recon_std = np.std(max_recon)
                mean_recon_std = np.std(mean_recon)
                num_filtered_std = np.std(num_filtered)

                # Print or log the results
                success_rate = success_count / len(finished_episodes_log)
                failure_rate = failure_count / len(finished_episodes_log)
                timeout_rate = timeout_count / len(finished_episodes_log)
                print("=== Episode Results Summary ===")
                print(f"Count of Episodes: {len(finished_episodes_log)}")
                print(f"Success Rate: {success_rate:.2f}")
                print(f"Failure Rate: {failure_rate:.2f}")
                print(f"Timeout Rate: {timeout_rate:.2f}")
                print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")
                print(f"Length (mean ± std): {length_mean:.2f} ± {length_std:.2f}")
                print(f"Max Uncertainty (mean ± std): {max_unc_mean:.4f} ± {max_unc_std:.4f}")
                print(f"Mean Uncertainty (mean ± std): {mean_unc_mean:.4f} ± {mean_unc_std:.4f}")
                print(f"Max KLD (mean): {max_klds_mean:.4f}")
                print(f"Mean KLD (mean): {mean_klds_mean:.4f}")
                print(f"Max recon (mean ± std): {max_recon_mean:.4f} ± {max_recon_std:.4f}")
                print(f"Mean recon (mean ± std): {mean_recon_mean:.4f} ± {mean_recon_std:.4f}")
                print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")

        # Reset the length counters for those environments that just finished.
        length *= (1 - next_done)

        # Prepare for next step.
        obs = next_obs

        episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (no upper limit)"
        print(f"Filtered: {is_filtered} | (Lengths: {length}) | Current Episode: {episode_info}", end="\r", flush=True)


    # === Log overall results after evaluation ===
    total_episodes = episodes
    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    failure_rate = failure_count / total_episodes if total_episodes > 0 else 0.0
    timeout_rate = timeout_count / total_episodes if total_episodes > 0 else 0.0

    print("\n===== Evaluation Results =====")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Failure Rate: {failure_rate:.2f}")
    print(f"Timeout Rate: {timeout_rate:.2f}")

    wandb.log({
            "eval/total_episodes": total_episodes,
            "eval/success_rate": success_rate,
            "eval/failure_rate": failure_rate,
            "eval/incompletion_rate": timeout_rate,
        })
    
    with open(result_file_path, "w") as f:
        json.dump(global_results, f, indent=4)

    return total_episodes, success_rate, failure_rate, timeout_rate, finished_episodes_log

def evaluate_venv_filtering_df(
    agent_df,
    latent_dynamics,
    latent_ensemble,
    latent_filter,
    config,
    vecenv,        # A single vectorized environment with N sub-envs
    dict_apply,
    episodes=10,  # total number of finished trajectories to test
    state=None,
    is_filter=False,
):

    total_episode_cnt = 0
    global_results = []

    # === Unpack or initialize simulation state ===
    num_env = vecenv.num_envs
    if state is None:
        step = 0
        episode = 0
        length = 0
        obs = None

    # Reset environment if needed and initialize the latent state.
    obs, latent, obs_original = reset_episode_df(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)
    agent_df.reset()

    # --- Containers for per-episode uncertainty logging ---
    # For each sub-environment, we will accumulate the uncertainty values over the trajectory.
    traj_uncertainties = []
    traj_filtered = []
    traj_kld = []
    traj_recon_loss = []

    # To record finished episode details:
    finished_episodes_log = []
    success_count = 0
    failure_count = 0
    timeout_count = 0

    # === Main evaluation loop ===
    # (Stop when we have finished the desired number of episodes or reached the max step count.)
    while (episode < episodes):

        # === (1) Nominal Policy ===
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device="cuda"))
        
        for k, v in obs_dict.items():
            obs_dict[k] = v.transpose(1,0)

        # run policy
        with torch.no_grad():
            action_dict = agent_df.predict_action(obs_dict)

        np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

        action = torch.tensor(np_action_dict['action'][0,:1], device="cuda")
        action[:, -1] *= 2

        # === (2) Filter Action ===
        is_filtered = False
        action_dict = {'action': action}
        if is_filter:
            new_action, is_filtered = \
                filter_action(latent_dynamics, latent_ensemble, latent, action_dict, latent_filter, \
                  uq_thr=config.ood_threshold, reachability_thr=config.reachability_threshold, denom=config.action_denom)
            action_dict['action'] = new_action

        filtered_action = action_dict['action'].cpu().numpy()
        next_obs, reward, next_done, info = vecenv.step(filtered_action)

        # === (3) Calculate and record uncertainty values ===
        action_dict = {'action': torch.tensor(filtered_action, device="cuda")}
        final_uncertainties = get_uncertainty(latent_dynamics, latent_ensemble, latent, action_dict)

        traj_uncertainties.append(final_uncertainties.item())
        traj_filtered.append(is_filtered)

        # === (4) Update Latent ===
        next_obs_original = vecenv.original_obs
        latent, latent_prior = forward_latent(latent_dynamics, latent, action_dict, next_obs_original)

        kld_uncertaity = get_kld(latent_dynamics, latent, latent_prior)
        traj_kld.append(kld_uncertaity.item())
        recon_loss = get_recon_loss(latent_dynamics, latent, next_obs_original)
        traj_recon_loss.append(recon_loss)

        # === Update counters ===
        length += 1
        step += num_env
        # Capture the current length values (these represent the step count for this episode)
        current_lengths = length

        if next_done:
            # Success
            if info['log'][-1]['Episode_Termination/success']:
                success_count += 1
                outcome = 'success'
            
            # Failure
            elif info['log'][-1]['Episode_Termination/failure']:
                failure_count += 1
                outcome = 'failure'

            # Timeout.
            elif info['log'][-1]['Episode_Termination/time_out']:
                timeout_count += 1
                outcome = 'incompletion'

            else:
                failure_count += 1
                outcome = 'failure' #'incompletion'
                    
            num_filtered = np.sum(traj_filtered)
            max_unc = np.max(traj_uncertainties) if len(traj_uncertainties) > 0 else 0.0
            mean_unc = np.mean(traj_uncertainties) if len(traj_uncertainties) > 0 else 0.0
            traj_kld_value = traj_kld
            max_kld = np.max(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
            mean_kld = np.mean(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
            traj_recon_loss_value = traj_recon_loss
            max_recon_loss = np.max(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0
            mean_recon_loss = np.mean(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0

            # result = {
            # 		'episode_index': episode,
            # 		'length': current_lengths,
            # 		'max_uncertainty': max_unc,
            # 		'mean_uncertainty': mean_unc,
            # 		'num_filtered': num_filtered,
            # 		'outcome': outcome
            # 	}

            result = {
                    'episode_index': episode,
                    'length': current_lengths,
                    'max_uncertainty': max_unc,
                    'mean_uncertainty': mean_unc,
                    'max_kld': max_kld,
                    'mean_kld': mean_kld,
                    'max_recon': max_recon_loss,
                    'mean_recon': mean_recon_loss,
                    'num_filtered': num_filtered,
                    'outcome': outcome
                }
            finished_episodes_log.append(result)
            # Reset uncertainty storage for that environment (so the next episode starts fresh)
            traj_uncertainties = []
            traj_filtered = []
            traj_kld = []
            traj_recon_loss = [] 
            length = 0
            print(result)

            # Save result
            total_episode_cnt += 1 
            converted_result = {k: convert_numpy(v) for k, v in result.items()}
            global_results.append(converted_result)
                
            # Increase our episode count by the number of environments that finished this step.
            episode += 1

            next_obs, latent, _ = reset_episode_df(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)


            if episode % 10 == 0 :
                lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
                max_uncs = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
                mean_uncs = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
                max_klds = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
                mean_klds = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
                max_recon = np.array([r['max_recon'] for r in finished_episodes_log], dtype=np.float32)
                mean_recon = np.array([r['mean_recon'] for r in finished_episodes_log], dtype=np.float32)
                num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

                # Compute means
                length_mean = np.mean(lengths)
                max_unc_mean = np.mean(max_uncs)
                mean_unc_mean = np.mean(mean_uncs)
                max_klds_mean = np.mean(max_klds)
                mean_klds_mean = np.mean(mean_klds)
                max_recon_mean = np.mean(max_recon)
                mean_recon_mean = np.mean(mean_recon)
                num_filtered_mean = np.mean(num_filtered)

                # Compute standard deviations
                length_std = np.std(lengths)
                max_unc_std = np.std(max_uncs)
                mean_unc_std = np.std(mean_uncs)
                max_recon_std = np.std(max_recon)
                mean_recon_std = np.std(mean_recon)
                num_filtered_std = np.std(num_filtered)

                # Print or log the results
                success_rate = success_count / len(finished_episodes_log)
                failure_rate = failure_count / len(finished_episodes_log)
                timeout_rate = timeout_count / len(finished_episodes_log)
                print("=== Episode Results Summary ===")
                print(f"Count of Episodes: {len(finished_episodes_log)}")
                print(f"Success Rate: {success_rate:.2f}")
                print(f"Failure Rate: {failure_rate:.2f}")
                print(f"Timeout Rate: {timeout_rate:.2f}")
                print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")
                print(f"Length (mean ± std): {length_mean:.2f} ± {length_std:.2f}")
                print(f"Max Uncertainty (mean ± std): {max_unc_mean:.4f} ± {max_unc_std:.4f}")
                print(f"Mean Uncertainty (mean ± std): {mean_unc_mean:.4f} ± {mean_unc_std:.4f}")
                print(f"Max KLD (mean): {max_klds_mean:.4f}")
                print(f"Mean KLD (mean): {mean_klds_mean:.4f}")
                print(f"Max recon (mean ± std): {max_recon_mean:.4f} ± {max_recon_std:.4f}")
                print(f"Mean recon (mean ± std): {mean_recon_mean:.4f} ± {mean_recon_std:.4f}")
                print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")

        # Prepare for next step.
        obs = next_obs

        episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (no upper limit)"
        print(f"Filtered: {is_filtered} | Uncertainty {final_uncertainties.item():.4f} (Lengths: {length}) | Current Episode: {episode_info}", end="\r", flush=True)


    # === Log overall results after evaluation ===
    total_episodes = episodes
    success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
    failure_rate = failure_count / total_episodes if total_episodes > 0 else 0.0
    timeout_rate = timeout_count / total_episodes if total_episodes > 0 else 0.0

    print("\n===== Evaluation Results =====")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Failure Rate: {failure_rate:.2f}")
    print(f"Timeout Rate: {timeout_rate:.2f}")

    wandb.log({
            "eval/total_episodes": total_episodes,
            "eval/success_rate": success_rate,
            "eval/failure_rate": failure_rate,
            "eval/incompletion_rate": timeout_rate,
        })
    
    result_file_path = os.path.join(config.outFolder, "results.json")
    with open(result_file_path, "w") as f:
        json.dump(global_results, f, indent=4)

    return total_episodes, success_rate, failure_rate, timeout_rate, finished_episodes_log



def add_to_cache(cache, id, transition):

    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size, protected_keys=None):
    step_in_dataset = 0
    protected_keys = set() if protected_keys is None else set(protected_keys)

    # OrderedDict: 从最新到最旧遍历，优先保留最近轨迹。
    for key, ep in reversed(list(cache.items())):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
            # print(key, (len(ep["reward"]) - 1), step_in_dataset)
        else:
            #FIXME
            if key.startswith("expert_") or key in protected_keys:
                continue
            del cache[key]

    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


import cv2
import imageio


def save_video(directory, filename, video, failure):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    length = len(video)
    filename = directory / f"{filename}-{length}.mp4"
        
    # Get video dimensions
    height, width, channels = video[0].shape
    fps = 20  # You can adjust FPS if needed

    # Initialize video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    # video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

    writer = imageio.get_writer(str(filename), fps=fps, codec='libx264')

    # Write frames to the video file
    # for i, frame in video:
    # 	video_writer.write(frame)

    for i, frame in enumerate(video):
        failure_i = failure[i]
        frame = frame.astype(np.uint8)

        if failure_i > 0 :
            # video_writer.write(frame)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            # video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.append_data(frame)

    # Release the video writer
    # video_writer.release()
    writer.close()
    print("Saved Video!")
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True, episodes=None):
    directory = pathlib.Path(directory).expanduser()

    if episodes is None :
        episodes = collections.OrderedDict()
    else:
        episodes = episodes

    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f, allow_pickle=True)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            
            # extract only filename without extension
            episodes["expert_" + str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1

            print("loaded episode from {}, total len {} ({} episodes)".format(filename, total, len(episodes)))
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def _sample_n(self):
        # torch.distributions.sample expects sample_shape to be tuple/torch.Size.
        return self._dist.sample((self._samples,))

    def mean(self):
        samples = self._sample_n()
        return torch.mean(samples, 0)

    def mode(self):
        samples = self._sample_n()
        logprob = self._dist.log_prob(samples)

        # samples shape: (S, *batch, *event), logprob shape: (S, *batch)
        batch_shape = logprob.shape[1:]
        event_shape = samples.shape[1 + len(batch_shape) :]

        flat_logprob = logprob.reshape(self._samples, -1)
        flat_samples = samples.reshape(self._samples, -1, *event_shape)

        best_idx = torch.argmax(flat_logprob, dim=0)
        batch_idx = torch.arange(flat_samples.shape[1], device=samples.device)
        best_samples = flat_samples[best_idx, batch_idx]
        return best_samples.reshape(*batch_shape, *event_shape)

    def entropy(self):
        sample = self._sample_n()
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    # PyTorch Transform API compatibility (newer versions require these attrs).
    domain = torchd.constraints.real
    codomain = torchd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, validate_args=False, name="tanh"):
        super().__init__(cache_size=1)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        del y
        log2 = math.log(2.0)
        return 2.0 * (log2 - x - torch.nn.functional.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = to_np(norm)
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
