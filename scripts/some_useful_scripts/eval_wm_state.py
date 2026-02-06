# eval_wm_state.py
import argparse
import collections
import json
import os
import pathlib
import sys
import numpy as np
import torch
import gymnasium as gym
import ruamel.yaml as yaml
sys.path.append("scripts")
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools


# ----------------------------
# Config parsing (same pattern as train_dreamer.py)
# ----------------------------
def recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            recursive_update(base[k], v)
        else:
            base[k] = v


def load_config(config_path: str, config_names):
    yaml_parser = yaml.YAML(typ="safe", pure=True)
    configs = yaml_parser.load(pathlib.Path(config_path).read_text())
    name_list = ["defaults", *config_names] if config_names else ["defaults"]
    merged = {}
    for name in name_list:
        recursive_update(merged, configs[name])
    return merged


# ----------------------------
# Utilities
# ----------------------------
EXCLUDE_KEYS = {
    # "action", "reward", "discount", "cont", "failure",
    # "is_first", "is_terminal",
    # "logprob", "policy", "disagreement",
}

def first_npz_file(directory: str) -> pathlib.Path:
    d = pathlib.Path(directory).expanduser()
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz found in {directory}")
    return files[0]


def infer_spaces_from_episode(npz_path: pathlib.Path):
    with npz_path.open("rb") as f:
        ep = np.load(f)
        ep = {k: ep[k] for k in ep.keys()}

    if "action" not in ep:
        raise KeyError(f"{npz_path} does not contain key 'action'")

    # Infer action space
    act = ep["action"]
    if act.ndim == 1:
        act_dim = 1
    else:
        act_dim = act.shape[-1]
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    # Infer observation space from episode keys (everything except obvious non-obs keys)
    obs_spaces = {}
    for k, arr in ep.items():
        if k in {"action"}:
            continue
        # arr shape: (T, ...)
        if not hasattr(arr, "shape") or arr.shape[0] < 1:
            continue
        shape = arr[0].shape  # () or (dim,)
        # make everything float32 to match preprocess casting
        obs_spaces[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    obs_space = gym.spaces.Dict(obs_spaces)
    return obs_space, act_space, act_dim


def dist_mean(dist):
    m = getattr(dist, "mean", None)
    return m() if callable(m) else m


def safe_mask_is_first(data, start_idx=0):
    # data["is_first"] exists (preprocess asserts) :contentReference[oaicite:5]{index=5}
    is_first = data["is_first"]
    if is_first.ndim == 3 and is_first.shape[-1] == 1:
        is_first = is_first[..., 0]
    if start_idx > 0:
        is_first = is_first[:, start_idx:]
    not_first = (is_first < 0.5).float()

    # valid=1 until first reset, then 0 (prevents cross-episode rollout contamination)
    valid = torch.cumprod(not_first, dim=1)
    return valid


def choose_state_keys(pred_dict, data_dict):
    keys = []
    for k in pred_dict.keys():
        if k not in data_dict:
            continue
        if k in EXCLUDE_KEYS:
            continue
        lk = k.lower()
        if ("cam" in lk) or ("image" in lk):
            continue
        # ignore scalars that are really flags if they exist redundantly
        if lk.startswith("log_"):
            continue
        keys.append(k)
    return keys


@torch.no_grad()
def evaluate_one_dataset(agent, episodes, config, batches, warmup, horizons):
    wm = agent._wm
    ds = dreamer.make_dataset(episodes, config)

    # accumulators
    out = {
        "one_step": {"nll": {}, "rmse": {}},
        "open_loop": {f"k{h}": {"nll": {}, "rmse": {}} for h in horizons},
        "diag": {"kl": 0.0, "dyn_loss": 0.0, "rep_loss": 0.0},
        "meta": {"batches": batches, "warmup": warmup, "horizons": horizons},
    }

    counts_1 = collections.Counter()
    counts_k = {h: collections.Counter() for h in horizons}

    # figure out which keys to score on the first batch
    state_keys = None

    for _ in range(batches):
        batch = next(ds)
        data = wm.preprocess(batch)

        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])  # :contentReference[oaicite:6]{index=6}

        # diagnostics: KL/dyn/rep (mask out is_first)
        kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
            post, prior, config.kl_free, config.dyn_scale, config.rep_scale
        )
        valid_1 = safe_mask_is_first(data, start_idx=0)
        denom_1 = torch.clamp(valid_1.sum(), min=1.0)
        out["diag"]["kl"] += (kl_value * valid_1).sum().item() / denom_1.item()
        out["diag"]["dyn_loss"] += (dyn_loss * valid_1).sum().item() / denom_1.item()
        out["diag"]["rep_loss"] += (rep_loss * valid_1).sum().item() / denom_1.item()

        # 1-step prior prediction on obs/state keys
        feat_prior = wm.dynamics.get_feat(prior)
        pred_prior = wm.heads["decoder"](feat_prior)

        if state_keys is None:
            state_keys = choose_state_keys(pred_prior, data)
            if not state_keys:
                raise RuntimeError(
                    "No state keys found to evaluate. "
                    "Check your .npz keys and decoder outputs."
                )

        for k in state_keys:
            # NLL
            nll = -pred_prior[k].log_prob(data[k])
            if nll.ndim == 3 and nll.shape[-1] == 1:
                nll = nll[..., 0]
            nll_mean = (nll * valid_1).sum() / denom_1
            out["one_step"]["nll"][k] = out["one_step"]["nll"].get(k, 0.0) + nll_mean.item()
            counts_1[f"nll:{k}"] += 1

            # RMSE
            mu = dist_mean(pred_prior[k])
            err2 = (mu - data[k]) ** 2
            # reduce over feature dims
            if err2.ndim > 2:
                err2 = err2.mean(dim=list(range(2, err2.ndim)))
            rmse = torch.sqrt(err2 + 1e-8)
            rmse_mean = (rmse * valid_1).sum() / denom_1
            out["one_step"]["rmse"][k] = out["one_step"]["rmse"].get(k, 0.0) + rmse_mean.item()
            counts_1[f"rmse:{k}"] += 1

        # open-loop rollout
        W = warmup
        W = min(W, data["action"].shape[1] - 1)
        post_w, _ = wm.dynamics.observe(embed[:, :W], data["action"][:, :W], data["is_first"][:, :W])
        init = {kk: vv[:, -1] for kk, vv in post_w.items()}
        
        # 1. 获取完整的想象序列预测 (Shape: [Batch, T_imag, ...])
        prior_ol = wm.dynamics.imagine_with_action(data["action"][:, W:], init)
        feat_ol = wm.dynamics.get_feat(prior_ol)
        pred_ol = wm.heads["decoder"](feat_ol)

        # 2. 预先计算整个序列的 NLL 和 RMSE (Compute Full Sequence Metrics First)
        # 确保 target 长度与 pred_ol 的长度一致
        T_imag = data["action"].shape[1] - W
        
        full_nll = {}
        full_rmse = {}
        
        for k in state_keys:
            target = data[k][:, W : W + T_imag] # 取出对应的完整真实数据
            
            # --- 计算 NLL ---
            # 此时 target 长度等于 pred_ol[k] 长度，AssertionError 消失
            nll = -pred_ol[k].log_prob(target)
            if nll.ndim == 3 and nll.shape[-1] == 1:
                nll = nll[..., 0]
            full_nll[k] = nll
            
            # --- 计算 RMSE ---
            mu = dist_mean(pred_ol[k])
            err2 = (mu - target) ** 2
            if err2.ndim > 2:
                err2 = err2.mean(dim=list(range(2, err2.ndim)))
            rmse = torch.sqrt(err2 + 1e-8)
            full_rmse[k] = rmse

        # 3. 在循环中切片 (Slice the Metrics Loop)
        valid_ol = safe_mask_is_first(data, start_idx=W)
        
        for h in horizons:
            hh = min(h, valid_ol.shape[1])
            m = valid_ol[:, :hh]
            denom = torch.clamp(m.sum(), min=1.0)

            for k in state_keys:
                # 直接切片已经算好的 Tensor
                nll_seq = full_nll[k][:, :hh]
                nll_mean = (nll_seq * m).sum() / denom
                
                out["open_loop"][f"k{h}"]["nll"][k] = out["open_loop"][f"k{h}"]["nll"].get(k, 0.0) + nll_mean.item()
                counts_k[h][f"nll:{k}"] += 1

                rmse_seq = full_rmse[k][:, :hh]
                rmse_mean = (rmse_seq * m).sum() / denom
                
                out["open_loop"][f"k{h}"]["rmse"][k] = out["open_loop"][f"k{h}"]["rmse"].get(k, 0.0) + rmse_mean.item()
                counts_k[h][f"rmse:{k}"] += 1
    # average over batches
    out["diag"]["kl"] /= batches
    out["diag"]["dyn_loss"] /= batches
    out["diag"]["rep_loss"] /= batches

    for k in list(out["one_step"]["nll"].keys()):
        out["one_step"]["nll"][k] /= max(1, counts_1[f"nll:{k}"])
        out["one_step"]["rmse"][k] /= max(1, counts_1[f"rmse:{k}"])

    for h in horizons:
        block = out["open_loop"][f"k{h}"]
        for k in list(block["nll"].keys()):
            block["nll"][k] /= max(1, counts_k[h][f"nll:{k}"])
            block["rmse"][k] /= max(1, counts_k[h][f"rmse:{k}"])

    # also provide overall averages across state keys
    def avg_dict(d):
        if not d:
            return None
        return float(np.mean(list(d.values())))

    out["one_step"]["nll_avg"] = avg_dict(out["one_step"]["nll"])
    out["one_step"]["rmse_avg"] = avg_dict(out["one_step"]["rmse"])
    for h in horizons:
        out["open_loop"][f"k{h}"]["nll_avg"] = avg_dict(out["open_loop"][f"k{h}"]["nll"])
        out["open_loop"][f"k{h}"]["rmse_avg"] = avg_dict(out["open_loop"][f"k{h}"]["rmse"])

    out["scored_keys"] = state_keys
    return out


def print_summary(tag, res):
    print(f"\n=== {tag} ===")
    print(f"keys: {res.get('scored_keys', [])}")
    print(f"1-step: NLL(avg)={res['one_step'].get('nll_avg')}  RMSE(avg)={res['one_step'].get('rmse_avg')}")
    for k, v in res["one_step"]["rmse"].items():
        print(f"  1-step RMSE {k}: {v:.6f}")
    for h, block in res["open_loop"].items():
        print(f"{h}: NLL(avg)={block.get('nll_avg')}  RMSE(avg)={block.get('rmse_avg')}")
    print(f"diag: kl={res['diag']['kl']:.6f} dyn_loss={res['diag']['dyn_loss']:.6f} rep_loss={res['diag']['rep_loss']:.6f}")


def main():
    # stage-1 args (eval-specific)
    p1 = argparse.ArgumentParser()
    p1.add_argument("--configs", nargs="+")
    p1.add_argument("--config_path", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")

    p1.add_argument("--ckpt_base", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    p1.add_argument("--ckpt_ft", type=str, default="latent_safety/log/dreamerv3/world_model_only/0123/205246_test/latest.pt")

    p1.add_argument("--id_dir", type=str, default='latent_safety/log/dreamerv3/collect_data/')
    p1.add_argument("--ood_dir", type=str, default='latent_safety/log/dreamerv3/collect_data_ood/')

    p1.add_argument("--id_limit", type=int, default=0, help="total steps limit when loading episodes")
    p1.add_argument("--ood_limit", type=int, default=0)

    p1.add_argument("--eval_batches", type=int, default=200)
    p1.add_argument("--warmup", type=int, default=5)
    p1.add_argument("--horizons", type=str, default="1,5,10,20")
    p1.add_argument("--out_json", type=str, default="wm_eval_report.json")
    p1.add_argument("--seed", type=int, default=0)

    args1, remaining = p1.parse_known_args()
    horizons = [int(x) for x in args1.horizons.split(",") if x.strip()]

    # load config defaults + chosen blocks
    defaults = load_config(args1.config_path, args1.configs)

    # stage-2 parse (auto-generate all config keys like training)
    p2 = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        p2.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = p2.parse_args(remaining)

    # seeds
    tools.set_seed_everywhere(args1.seed)

    # load episodes (ID + OOD)
    id_eps = tools.load_episodes(args1.id_dir, limit=args1.id_limit or None)
    ood_eps = tools.load_episodes(args1.ood_dir, limit=args1.ood_limit or None)

    # infer spaces without launching env
    sample_npz = first_npz_file(args1.id_dir)
    obs_space, act_space, act_dim = infer_spaces_from_episode(sample_npz)
    config.num_actions = act_dim

    # minimal logger + dataset for agent init
    logdir = pathlib.Path("/home/yhy/IsaacLabExtensionTemplate/logs_wm_eval_tmp")
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)

    init_dataset = dreamer.make_dataset(id_eps, config)
    agent = dreamer.Dreamer(obs_space, act_space, config, logger, init_dataset).to(config.device)
    agent.requires_grad_(False)

    def run_for_ckpt(ckpt_path, tag):
        ckpt = torch.load(ckpt_path, map_location=config.device)
        agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
        del ckpt
        torch.cuda.empty_cache()

        res_id = evaluate_one_dataset(agent, id_eps, config, args1.eval_batches, args1.warmup, horizons)
        res_ood = evaluate_one_dataset(agent, ood_eps, config, args1.eval_batches, args1.warmup, horizons)
        print_summary(f"{tag} / ID", res_id)
        print_summary(f"{tag} / OOD", res_ood)
        return {"ID": res_id, "OOD": res_ood}

    report = {}
    report["baseline"] = run_for_ckpt(args1.ckpt_base, "baseline")
    report["finetuned"] = run_for_ckpt(args1.ckpt_ft, "finetuned")

    # deltas (finetuned - baseline) on avg metrics
    def delta(a, b):
        if a is None or b is None:
            return None
        return float(a - b)

    report["delta"] = {
        "ID": {
            "one_step_nll_avg": delta(report["finetuned"]["ID"]["one_step"]["nll_avg"], report["baseline"]["ID"]["one_step"]["nll_avg"]),
            "one_step_rmse_avg": delta(report["finetuned"]["ID"]["one_step"]["rmse_avg"], report["baseline"]["ID"]["one_step"]["rmse_avg"]),
        },
        "OOD": {
            "one_step_nll_avg": delta(report["finetuned"]["OOD"]["one_step"]["nll_avg"], report["baseline"]["OOD"]["one_step"]["nll_avg"]),
            "one_step_rmse_avg": delta(report["finetuned"]["OOD"]["one_step"]["rmse_avg"], report["baseline"]["OOD"]["one_step"]["rmse_avg"]),
        },
    }

    with open(args1.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {args1.out_json}")


if __name__ == "__main__":
    main()
