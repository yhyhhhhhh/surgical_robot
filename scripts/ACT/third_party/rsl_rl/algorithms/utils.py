import math
import torch
from torch import nn
import os
import pickle
from collections import defaultdict
import numpy as np
def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=nn.Sigmoid()):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


class SerializedBuffer:
    def __init__(self, folder_path= '/home/yhy/code/data/apple', device = 'cuda:0'):
        """
        从指定文件夹下加载所有 pickle 文件，
        读取其中的 observation 数据，并将其转换成 torch.Tensor，
        然后合并成一个大的 tensor。

        :param folder_path: 存放 pickle 文件的文件夹路径
        :param device: 目标设备，例如 "cpu" 或 "cuda"
        """
        self.device = device

        # 筛选文件夹下所有扩展名为 .pickle 的文件
        pickle_files = [os.path.join(folder_path, file_name)
                        for file_name in os.listdir(folder_path)
                        if file_name.endswith('.pickle')]

        observations_list = []

        # 遍历每个 pickle 文件
        for file_path in pickle_files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                continue

            # 判断读取的数据是否包含 'joint' 键
            if 'joint' in data:
                # 将 joint 数据转换为 torch.Tensor，并转移到指定设备上
                obs = torch.tensor(data['joint'], dtype=torch.float32).to(self.device)
                # 预处理操作：归一化，即除以 1000
                obs = obs
                # 对 obs 的前 3 个通道进行反转操作：1 - obs[:, 0:3]
                obs[:, 0:4] = 1 - obs[:, 0:4]
                obs[:,4] = obs[:,4]-0.9355
                # 对于 obs[:, 4:5] 保持原始值（这里无需额外处理）
                observations_list.append(obs*2)
                print("loaddddddddddddddddddddddddddd")
            else:
                raise KeyError(f"文件 {file_path} 中未找到键 'joint'.")
            
        if len(observations_list) == 0:
            raise ValueError("没有在任何文件中加载到 observation 数据！")

        # 按样本维度（第 0 维）将所有 observation 数据合并为一个大的 tensor
        self.observations = torch.cat(observations_list, dim=0)
        self.buffer_size = self.observations.size(0)

    def sample(self, batch_size):
        """
        从合并后的 observation tensor 中随机采样一批数据。

        :param batch_size: 需要采样的数据数量
        :return: 采样得到的 observation 数据，形状为 [batch_size, ...]
        """
        # 随机生成下标
        idxes = np.random.randint(0, self.buffer_size, size=batch_size)
        # 根据下标从合并的 observation tensor 中取出样本
        return self.observations[idxes]




class SerializedBuffer_Rl:
    def __init__(
        self,
        folder_path: str = "/home/yhy/code/rollouts",
        device: str = "cuda:0",
    ):
        """
        从指定文件夹下加载所有 .pkl 文件，
        读取 top-level 的 hand_pos, hand_ori, angle, torque（或嵌套的多-step 格式），
        并拼成一个大 Tensor: [总条数, 19]
        """
        self.device = device

        # 1) 找到所有 .pkl
        pickle_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.endswith(".pkl")
        ]
        if not pickle_files:
            raise FileNotFoundError(f"{folder_path} 下没有 .pkl 文件")

        all_obs = []

        for path in pickle_files:
            try:
                data = pickle.load(open(path, "rb"))
            except Exception as e:
                print(f"[WARN] {path} load failed: {e}")
                continue

            # 2) 统一成 steps 列表
            if isinstance(data, list):
                steps = data

            elif isinstance(data, dict) and "hand_pos" in data:
                # 单 step 格式
                steps = [data]

            elif isinstance(data, dict):
                # 旧版 {'timestep_0':{}, ...}
                steps = []
                for key, val in data.items():
                    if isinstance(val, dict) and all(k in val for k in ("hand_pos", "hand_ori", "angle", "torque")):
                        # 可选：把 key 里的 index 提取到 val["t"]
                        try:
                            val["t"] = int(key.split("_", 1)[1])
                        except:
                            pass
                        steps.append(val)
                if not steps:
                    print(f"[WARN] {path} 不含任何有效 timestep，跳过")
                    continue
            else:
                print(f"[WARN] {path} 格式不支持，跳过")
                continue

            # 3) 对每个 step 做相同处理
            file_buf = []
            for step in steps:
                try:
                    hp = torch.tensor(step["hand_pos"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    ho = torch.tensor(step["hand_ori"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    ag = torch.tensor(step["angle"],    dtype=torch.float32, device=self.device).unsqueeze(0)
                    tq = torch.tensor(step["torque"],   dtype=torch.float32, device=self.device).unsqueeze(0)
                except KeyError as missing:
                    print(f"[WARN] {path} step 缺字段 {missing}，跳过")
                    continue
                except Exception as e:
                    print(f"[WARN] {path} step 转 tensor 失败: {e}，跳过")
                    continue

                # [num_envs, 3]+[num_envs,4]+[num_envs,6]+[num_envs,6] -> [num_envs,19]
                file_buf.append(torch.cat([hp, ho, ag, tq],dim=1))

            if file_buf:
                all_obs.append(torch.cat(file_buf, dim=0))

        if not all_obs:
            raise RuntimeError("未解析到任何有效数据！")

        # 4) 拼成最终大 Tensor
        self.observations = torch.cat(all_obs, dim=0)
        self.buffer_size  = self.observations.size(0)
        print(f"[INFO] 从 {len(pickle_files)} 个文件中，加载到 {self.buffer_size} 条记录。")

    def sample(self, batch_size: int) -> torch.Tensor:
        idx = np.random.randint(0, self.buffer_size, size=batch_size)
        return self.observations[idx]

# 示例：如何使用该 SerializedBuffer 类
if __name__ == '__main__':
    folder_path = '/home/yhy/code/rollouts'   # 指定你的 pickle 文件所在的文件夹路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 初始化 SerializedBuffer 对象，加载所有 pickle 文件中的 observation 数据
    buffer = SerializedBuffer_Rl(folder_path, device)
    
    # 随机采样一批 observation 数据
    batch = buffer.sample(batch_size=32)
    print("采样的 observation 数据形状：", batch.shape)
