import os
import math
import time
import random
import zarr
import numpy as np
from typing import Union, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# TensorBoard（没有装的话：pip install tensorboard）
from torch.utils.tensorboard import SummaryWriter

# Matplotlib（没有装的话：pip install matplotlib）
import matplotlib.pyplot as plt


# ==============================================================================
# 1. 数据集与归一化工具
# ==============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    episode_ids: np.ndarray = None,   # 可选：只为指定 episode 构建 indices
) -> np.ndarray:
    """
    indices: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    episode_ends: cumulative end indices, e.g. [len(ep0), len(ep0)+len(ep1), ...]
    episode_ids: subset episode id list (e.g. [0,2,5])，用于 train/val/test split
    """
    indices = []
    n_eps = len(episode_ends)

    if episode_ids is None:
        episode_ids = np.arange(n_eps)

    for i in episode_ids:
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])

    return np.array(indices, dtype=np.int64)


def sample_sequence(
    data_dict: Dict[str, np.ndarray],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, np.ndarray]:
    result = {}
    for key, arr in data_dict.items():
        sample = arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    data = data.reshape(-1, data.shape[-1])
    return {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}


def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    ndata = ndata * 2 - 1
    return ndata


class SequenceDatasetFromArrays(Dataset):
    """
    用于 train/val/test：共享同一份 normalized data，但 indices 不同
    """
    def __init__(
        self,
        normalized_data: Dict[str, np.ndarray],
        indices: np.ndarray,
        pred_horizon: int,
        obs_horizon: int,
    ):
        self.normalized_data = normalized_data
        self.indices = indices
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        nsample = sample_sequence(
            data_dict=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=int(buffer_start_idx),
            buffer_end_idx=int(buffer_end_idx),
            sample_start_idx=int(sample_start_idx),
            sample_end_idx=int(sample_end_idx),
        )
        # obs 只取前 obs_horizon
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        # 转 torch
        return {
            "obs": torch.from_numpy(nsample["obs"]).float(),
            "action": torch.from_numpy(nsample["action"]).float(),
        }


def load_zarr_arrays(dataset_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    root = zarr.open(dataset_path, "r")
    data = {
        "action": root["data"]["action"][:],
        "obs": root["data"]["state"][:],  # 如果你的 key 不是 state，改这里
    }
    episode_ends = root["meta"]["episode_ends"][:]
    return data, episode_ends


def split_episodes(
    num_episodes: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)
    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]
    return train_ids, val_ids, test_ids


# ==============================================================================
# 2. 神经网络定义 (Conditional 1D U-Net)
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], global_cond=None):
        sample = sample.moveaxis(-1, -2)  # (B, T, C) -> (B, C, T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)  # (B, C, T) -> (B, T, C)
        return x


# ==============================================================================
# 3. 评估 / 可视化工具
# ==============================================================================
@torch.no_grad()
def evaluate_noise_mse(
    model: nn.Module,
    dataloader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    obs_horizon: int,
) -> float:
    model.eval()
    losses = []

    for batch in dataloader:
        nobs = batch["obs"].to(device)        # (B, obs_h, obs_dim)
        naction = batch["action"].to(device)  # (B, pred_h, action_dim)
        B = nobs.shape[0]

        obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)
        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = model(noisy_actions, timesteps, global_cond=obs_cond)
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses)) if len(losses) > 0 else float("nan")


def plot_curves(save_path: str, history: Dict[str, List[float]]):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    for k, v in history.items():
        if len(v) > 0:
            plt.plot(v, label=k)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==============================================================================
# 4. 训练主程序
# ==============================================================================
if __name__ == "__main__":
    # ---------------- 必须修改的地方 ----------------
    dataset_path = "/home/yhy/IsaacLabExtensionTemplate/expert_data_episodes/expert_data.zarr"
    obs_dim = 30
    action_dim = 5
    # ----------------------------------------------

    # 可调超参
    seed = 42
    pred_horizon = 16
    obs_horizon = 5
    action_horizon = 8

    batch_size = 256
    num_epochs = 100
    num_diffusion_iters = 100

    lr = 1e-4
    weight_decay = 1e-6
    grad_clip_norm = 1.0

    num_workers = 4
    use_amp = True  # 混合精度（CUDA 才有效）

    log_dir = "runs/custom_diffusion_policy"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print(f"Device: {device} 🧪")
    print("Loading zarr arrays...")
    raw_data, episode_ends = load_zarr_arrays(dataset_path)
    num_episodes = len(episode_ends)
    print(f"Episodes: {num_episodes}, Total steps: {episode_ends[-1]}")

    # episode split（按 episode 划分，避免序列跨集泄漏）
    train_ep, val_ep, test_ep = split_episodes(num_episodes, 0.8, 0.1, 0.1, seed=seed)
    print(f"Split episodes -> train {len(train_ep)}, val {len(val_ep)}, test {len(test_ep)}")

    # 为各 split 构建 indices
    train_indices = create_sample_indices(
        episode_ends=episode_ends,
        sequence_length=pred_horizon,
        pad_before=obs_horizon - 1,
        pad_after=action_horizon - 1,
        episode_ids=train_ep,
    )
    val_indices = create_sample_indices(
        episode_ends=episode_ends,
        sequence_length=pred_horizon,
        pad_before=obs_horizon - 1,
        pad_after=action_horizon - 1,
        episode_ids=val_ep,
    )
    test_indices = create_sample_indices(
        episode_ends=episode_ends,
        sequence_length=pred_horizon,
        pad_before=obs_horizon - 1,
        pad_after=action_horizon - 1,
        episode_ids=test_ep,
    )
    print(f"Sequences -> train {len(train_indices)}, val {len(val_indices)}, test {len(test_indices)}")

    # 只用 train 计算 stats（避免 val/test 泄漏）
    print("Computing stats on TRAIN split only...")
    train_action = raw_data["action"][:]  # (N, action_dim)
    train_obs = raw_data["obs"][:]        # (N, obs_dim)

    # 用 indices 对应到 train 的时间点集合来做 stats（更严谨）
    # 注意：indices 是 buffer 范围，这里近似用 buffer_start/end 覆盖到的点集合计算 stats
    train_time_ids = []
    for (bs, be, _, _) in train_indices:
        train_time_ids.append(np.arange(bs, be))
    train_time_ids = np.unique(np.concatenate(train_time_ids)) if len(train_time_ids) else np.array([], dtype=np.int64)

    stats = {}
    if len(train_time_ids) == 0:
        raise RuntimeError("TRAIN indices is empty. Check horizons and episode lengths.")

    stats["action"] = get_data_stats(train_action[train_time_ids])
    stats["obs"] = get_data_stats(train_obs[train_time_ids])

    # 用 train stats 归一化全量数据（train/val/test 共享一份 normalized 数组）
    print("Normalizing all data with TRAIN stats...")
    normalized_data = {
        "action": normalize_data(train_action, stats["action"]).astype(np.float32),
        "obs": normalize_data(train_obs, stats["obs"]).astype(np.float32),
    }

    # Dataset / DataLoader
    train_ds = SequenceDatasetFromArrays(normalized_data, train_indices, pred_horizon, obs_horizon)
    val_ds = SequenceDatasetFromArrays(normalized_data, val_indices, pred_horizon, obs_horizon)
    test_ds = SequenceDatasetFromArrays(normalized_data, test_indices, pred_horizon, obs_horizon)

    persistent_workers = (num_workers > 0)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    print("Initializing model...")
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # TensorBoard
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # 记录曲线
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best_ema.ckpt")

    print("Starting training... 📈")
    global_step = 0

    with tqdm(range(num_epochs), desc="Epoch") as tglobal:
        for epoch in tglobal:
            noise_pred_net.train()
            epoch_losses = []

            with tqdm(train_loader, desc="Batch", leave=False) as tepoch:
                for batch in tepoch:
                    nobs = batch["obs"].to(device, non_blocking=True)
                    naction = batch["action"].to(device, non_blocking=True)
                    B = nobs.shape[0]

                    obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)

                    noise = torch.randn_like(naction)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                        noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(noise_pred_net.parameters(), grad_clip_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()

                    ema.step(noise_pred_net.parameters())

                    loss_item = float(loss.item())
                    epoch_losses.append(loss_item)

                    # TB（batch级别）
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    writer.add_scalar("train/loss_step", loss_item, global_step)
                    writer.add_scalar("train/lr_step", lr_now, global_step)

                    tepoch.set_postfix(loss=loss_item, lr=lr_now)
                    global_step += 1

            train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else float("nan")

            # 用 EMA 权重做验证更靠谱
            ema_model = noise_pred_net
            ema.copy_to(ema_model.parameters())
            val_loss = evaluate_noise_mse(
                model=ema_model,
                dataloader=val_loader,
                noise_scheduler=noise_scheduler,
                device=device,
                obs_horizon=obs_horizon,
            )

            lr_epoch = float(optimizer.param_groups[0]["lr"])
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr_epoch)

            # TB（epoch级别）
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("train/lr_epoch", lr_epoch, epoch)

            tglobal.set_postfix(train_loss=train_loss, val_loss=val_loss, lr=lr_epoch)

            # 保存 best（按 val loss）
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ema_model.state_dict(), best_path)

    # 训练结束：保存最终 EMA、stats、曲线图
    print("Training done. Saving artifacts... 🧷")
    final_ema_model = noise_pred_net
    ema.copy_to(final_ema_model.parameters())

    final_ckpt_path = os.path.join(ckpt_dir, "custom_diffusion_policy_ema_final.ckpt")
    torch.save(final_ema_model.state_dict(), final_ckpt_path)

    stats_path = os.path.join(ckpt_dir, "dataset_stats.npy")
    np.save(stats_path, stats, allow_pickle=True)

    curve_png = os.path.join(ckpt_dir, "training_curves.png")
    plot_curves(curve_png, {"train_loss": history["train_loss"], "val_loss": history["val_loss"]})

    # 最终 test 评估（用 best EMA 或 final EMA 都行；这里用 best EMA）
    print("Evaluating on TEST with BEST EMA checkpoint... 🧾")
    best_model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss = evaluate_noise_mse(
        model=best_model,
        dataloader=test_loader,
        noise_scheduler=noise_scheduler,
        device=device,
        obs_horizon=obs_horizon,
    )

    writer.add_scalar("test/loss", test_loss, 0)
    writer.close()

    print(f"✅ Best val loss: {best_val:.6f}")
    print(f"✅ Test loss (epsilon MSE): {test_loss:.6f}")
    print(f"Saved best EMA ckpt: {best_path}")
    print(f"Saved final EMA ckpt: {final_ckpt_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved curves: {curve_png}")

    print("\nTensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")