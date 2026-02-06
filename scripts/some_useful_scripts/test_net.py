import torch
from collections.abc import Mapping, Sequence

def _brief(x):
    if torch.is_tensor(x):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    return f"{type(x).__name__}"

def dump_tree(obj, prefix="", max_items=40, max_depth=6, depth=0):
    if depth > max_depth:
        print(prefix + "…")
        return

    if isinstance(obj, Mapping):
        keys = list(obj.keys())
        print(prefix + f"dict[{len(keys)}]")
        for k in keys[:max_items]:
            v = obj[k]
            print(prefix + f"  ├─ {k}: {_brief(v)}")
            dump_tree(v, prefix + "  │   ", max_items=max_items, max_depth=max_depth, depth=depth+1)
        if len(keys) > max_items:
            print(prefix + f"  └─ … (+{len(keys)-max_items} keys)")
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        print(prefix + f"{type(obj).__name__}[{len(obj)}]")
        for i, v in enumerate(list(obj)[:min(len(obj), max_items)]):
            print(prefix + f"  ├─ [{i}]: {_brief(v)}")
            dump_tree(v, prefix + "  │   ", max_items=max_items, max_depth=max_depth, depth=depth+1)
        if len(obj) > max_items:
            print(prefix + f"  └─ … (+{len(obj)-max_items} items)")
        return

    # torch module?
    if isinstance(obj, torch.nn.Module):
        print(prefix + f"nn.Module: {obj.__class__.__name__}")
        print(prefix + "repr:")
        print(prefix + str(obj))
        return

    print(prefix + str(obj)[:200])


def list_tensors_in_state_dict(sd, grep=None, topk=80):
    items = list(sd.items())
    if grep:
        items = [(k,v) for k,v in items if grep in k]
    print(f"[state_dict] tensors={len(items)}")
    for k, v in items[:topk]:
        if torch.is_tensor(v):
            print(f"  {k:60s}  {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:60s}  {type(v)}")
    if len(items) > topk:
        print(f"  … (+{len(items)-topk} more)")


def guess_actor_value_keys(sd):
    # 适配常见命名：actor.*, value.*, critic.*, _task_behavior.actor.*, ...
    candidates = ["actor", "value", "critic", "_task_behavior.actor", "_task_behavior.value"]
    hits = {}
    for c in candidates:
        ks = [k for k in sd.keys() if c in k]
        if ks:
            hits[c] = ks[:10]
    return hits


def main(path):
    ckpt = torch.load(path, map_location="cpu")  # ⚠️ 只加载你信任的文件（pickle 有风险）
    print("=== CKPT TREE ===")
    dump_tree(ckpt)

    # case 1: 直接就是 state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd = ckpt
        print("\n=== Detected: plain state_dict ===")
        list_tensors_in_state_dict(sd)
        print("\nPossible actor/value key hints:", guess_actor_value_keys(sd))
        return

    # case 2: 常见 checkpoint dict 里嵌着 state_dict
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "agent_state_dict", "model", "agent"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                # 有些是嵌套 dict，里面再找 tensor
                maybe = ckpt[key]
                if any(torch.is_tensor(v) for v in maybe.values()):
                    sd = maybe
                    print(f"\n=== Found state_dict in ckpt['{key}'] ===")
                    list_tensors_in_state_dict(sd)
                    print("\nPossible actor/value key hints:", guess_actor_value_keys(sd))
                    return

    print("\n[WARN] No obvious state_dict found. It may have saved full objects or custom structure.")


if __name__ == "__main__":
    import sys
    main('/home/yhy/DVRK/IsaacLabExtensionTemplate/latent_safety/log/dreamerv3/1225/latest.pt')
