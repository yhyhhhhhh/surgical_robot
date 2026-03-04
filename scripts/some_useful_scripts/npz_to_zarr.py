#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import numpy as np
import zarr


def pick_key(npz, candidates, required=True, kind="array"):
    for k in candidates:
        if k in npz:
            return k
    if required:
        raise KeyError(f"Cannot find {kind} key. Tried: {candidates}. Available keys: {list(npz.keys())}")
    return None


def main():
    ap = argparse.ArgumentParser("Convert per-episode npz files to diffusion-policy-style zarr replay buffer.")
    ap.add_argument("--in_dir", type=str, default="/home/yhy/IsaacLabExtensionTemplate/expert_data_episodes/", help="Directory containing episode .npz files")
    ap.add_argument("--out_zarr", type=str, default="/home/yhy/IsaacLabExtensionTemplate/expert_data_episodes/expert_data.zarr", help="Output .zarr directory path")
    ap.add_argument("--state_key", type=str, default="policy", help="Explicit key for state (default: auto)")
    ap.add_argument("--action_key", type=str, default="action", help="Key for action (default: action)")
    ap.add_argument("--state_dim", type=int, default=30, help="Expected state dim (default: 30)")
    ap.add_argument("--sort", type=str, default="name", choices=["name", "mtime"], help="How to order episodes")
    ap.add_argument("--chunk_T", type=int, default=4096, help="Time chunk size for zarr datasets")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_zarr = Path(args.out_zarr)

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    files = list(in_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {in_dir}")

    if args.sort == "name":
        files = sorted(files, key=lambda p: p.name)
    else:
        files = sorted(files, key=lambda p: p.stat().st_mtime)

    # auto key candidates
    state_candidates = [k for k in [args.state_key] if k] + [
        "policy", "state", "obs", "observation", "observations"
    ]
    action_key = args.action_key

    all_states = []
    all_actions = []
    episode_ends = []

    total_T = 0
    act_dim = None

    print(f"Found {len(files)} episode files in {in_dir}")
    for i, fp in enumerate(files, 1):
        with np.load(fp, allow_pickle=False) as npz:
            sk = pick_key(npz, state_candidates, required=True, kind="state")
            if action_key not in npz:
                raise KeyError(f"Action key '{action_key}' not in {fp}. Available keys: {list(npz.keys())}")

            s = npz[sk].astype(np.float32)
            a = npz[action_key].astype(np.float32)

            # Ensure shape (T, D)
            if s.ndim != 2:
                raise ValueError(f"{fp}: state must be 2D (T, D), got {s.shape}")
            if a.ndim != 2:
                raise ValueError(f"{fp}: action must be 2D (T, A), got {a.shape}")
            if s.shape[0] != a.shape[0]:
                raise ValueError(f"{fp}: length mismatch, state T={s.shape[0]} vs action T={a.shape[0]}")

            if s.shape[1] != args.state_dim:
                raise ValueError(f"{fp}: state_dim mismatch, expected {args.state_dim}, got {s.shape[1]}")

            if act_dim is None:
                act_dim = a.shape[1]
            elif a.shape[1] != act_dim:
                raise ValueError(f"{fp}: act_dim mismatch, expected {act_dim}, got {a.shape[1]}")

            T = s.shape[0]
            total_T += T
            episode_ends.append(total_T)

            all_states.append(s)
            all_actions.append(a)

        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] processed, total transitions: {total_T}")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    assert states.shape[0] == actions.shape[0] == episode_ends[-1]

    # Remove old output if exists
    if out_zarr.exists():
        import shutil
        shutil.rmtree(out_zarr)

    # Write zarr
    root = zarr.open(out_zarr.as_posix(), mode="w")
    grp_data = root.create_group("data")
    grp_meta = root.create_group("meta")

    # chunking
    chunk_T = min(args.chunk_T, states.shape[0])
    grp_data.create_dataset(
        "state", data=states, dtype="f4",
        chunks=(chunk_T, states.shape[1]),
    )
    grp_data.create_dataset(
        "action", data=actions, dtype="f4",
        chunks=(chunk_T, actions.shape[1]),
    )
    grp_meta.create_dataset("episode_ends", data=episode_ends, dtype="i8")

    # Store a bit of metadata
    root.attrs["state_key_used"] = str(state_candidates[0] if args.state_key else "auto")
    root.attrs["action_key_used"] = action_key
    root.attrs["num_episodes"] = int(len(episode_ends))
    root.attrs["num_transitions"] = int(states.shape[0])
    root.attrs["state_dim"] = int(states.shape[1])
    root.attrs["action_dim"] = int(actions.shape[1])

    print("\n✅ Done!")
    print(f"Saved zarr: {out_zarr}")
    print(f"Episodes: {len(episode_ends)} | Transitions: {states.shape[0]} | state_dim: {states.shape[1]} | action_dim: {actions.shape[1]}")


if __name__ == "__main__":
    main()