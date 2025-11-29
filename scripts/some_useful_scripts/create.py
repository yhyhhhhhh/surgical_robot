#!/usr/bin/env python3
"""generate_command_buffer.py

一次性离线生成 object 位姿序列 (position + quaternion) 并写入文件，
供主程序在运行时直接读取，保证 **完全可复现**。

默认：
    * 采样圆心   = (0.0, -0.29)
    * 半径       = 0.004
    * z 坐标     = 取自命令行 --z (默认 0.0)
    * 总数量     = 2000
    * 随机种子   = 42  （固定随机，保障每次生成 identical）

生成格式：
    command_buffer.pt   →  torch.float32  Tensor, 形状 (N,7)
                       （列依次为 x y z qw qx qy qz）

使用示例：
    $ python generate_command_buffer.py                               # 默认参数
    $ python generate_command_buffer.py -n 5000 -o data/cmd.pt --z 0.12

在主程序中加载：
    cmd_buf = torch.load('command_buffer.pt', map_location=device)  # (N,7)
"""

import argparse
import math
import os
from pathlib import Path
import torch

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate deterministic command buffer")
    p.add_argument("-n", "--num", type=int, default=20,
                   help="number of poses to generate (default: 2000)")
    p.add_argument("-o", "--output", type=str, default="command_buffer.pt",
                   help="output file path (default: command_buffer.pt)")
    p.add_argument("--center-x", type=float, default=0.0,
                   help="circle center x")
    p.add_argument("--center-y", type=float, default=-0.29,
                   help="circle center y")
    p.add_argument("--radius", type=float, default=0.004,
                   help="circle radius")
    p.add_argument("--z", type=float, default=-0.2354,
                   help="fixed z value")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed (default: 42)")
    return p

def generate_buffer(num: int, center_x: float, center_y: float,
                     radius: float, z_val: float, seed: int, device: str = "cpu") -> torch.Tensor:
    """Return (num,7) tensor: x y z qw qx qy qz"""
    torch.manual_seed(seed)

    # 均匀采样圆面积
    u = torch.rand(num, device=device)
    r = radius * torch.sqrt(u)
    theta = torch.rand(num, device=device) * 2 * math.pi

    x = center_x + r * torch.cos(theta)
    y = center_y + r * torch.sin(theta)
    z = torch.full_like(x, z_val)
    pos = torch.stack([x, y, z], dim=1)

    # 随机绕 Z 轴旋转
    ang = torch.rand(num, device=device) * 2 * math.pi
    quat = torch.stack([
        torch.cos(ang / 2),
        torch.zeros_like(ang),
        torch.zeros_like(ang),
        torch.sin(ang / 2)
    ], dim=1)

    return torch.cat([pos, quat], dim=1).float()

def main():
    args = build_parser().parse_args()

    buf = generate_buffer(
        num=args.num,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        z_val=args.z,
        seed=args.seed,
    )

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buf.cpu(), out_path)
    print(f"Saved command buffer: {out_path} (shape {tuple(buf.shape)})")

if __name__ == "__main__":
    main()
