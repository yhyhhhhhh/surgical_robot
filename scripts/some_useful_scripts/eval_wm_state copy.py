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
# NOTE: 你原来把这块注释掉了，导致把 'policy' 之类非 state 的字段当作 state 去评估。
# 这会直接造成 keys=['policy']，并且 RMSE/NLL 形同虚设。
# EXCLUDE_KEYS = {
#     "action", "reward", "discount", "cont", "failure",
#     "is_first", "is_terminal",
#     "logprob", "policy", "disagreement",
# }


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
        if not hasattr(arr, "shape") or arr.shape[0] < 1:
            continue
        shape = arr[0].shape
        obs_spaces[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    obs_space = gym.spaces.Dict(obs_spaces)
    return obs_space, act_space, act_dim


def dist_mean(dist):
    m = getattr(dist, "mean", None)
    return m() if callable(m) else m


def _ensure_BT(x: torch.Tensor) -> torch.Tensor:
    # normalize (B,T,1) -> (B,T)
    if x is None:
        return None
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def safe_mask_is_first(data, start_idx=0, drop_first=True):
    """
    修正版 mask：
    - 避免你原来 cumprod(not_first) 导致整段全 0 的问题
    - 保护 batch 内跨 episode（出现第二次 reset 后置 0）
    - 默认 drop_first=True：把 is_first 那一帧丢掉（通常 action/prev_state 对齐不可靠）
    """
    is_first = data["is_first"]
    is_first = _ensure_BT(is_first)

    if start_idx > 0:
        is_first = is_first[:, start_idx:]

    resets = (is_first > 0.5).float()  # (B,T)

    # keep until (and including) the first reset marker; cut off after the 2nd reset within the same chunk
    # cumsum: first episode region -> 1, after next reset -> 2...
    csum = torch.cumsum(resets, dim=1)
    valid = (csum <= 1.0).float()

    if drop_first:
        # drop reset-marked step itself
        valid = valid * (1.0 - resets)

    return valid


def choose_state_keys(pred_dict, data_dict):
    keys = []
    for k in pred_dict.keys():
        if k not in data_dict:
            continue

        lk = k.lower()
        if ("cam" in lk) or ("image" in lk):
            continue
        if lk.startswith("log_"):
            continue
        keys.append(k)
    return keys


# ----------------------------
# Uncertainty evaluation helpers (ensemble disagreement)
# ----------------------------
def _fast_auc(scores: np.ndarray, labels01: np.ndarray):
    """
    AUC where higher score should indicate label==1.
    Returns None if degenerate.
    """
    scores = scores.astype(np.float64)
    labels01 = labels01.astype(np.int32)
    n = scores.shape[0]
    n_pos = int(labels01.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)  # low->high
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    pos_ranks = ranks[labels01 == 1]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class UncBuffer:
    """
    Store (uncertainty, error) pairs with a size cap, then summarize:
    - corr
    - AURC
    - risk@coverage
    - AUC of detecting high-error events
    """
    def __init__(self, max_store=200000):
        self.max_store = int(max_store)
        self.u = []
        self.e = []

    def add(self, u: torch.Tensor, e: torch.Tensor, m: torch.Tensor):
        u = _ensure_BT(u)
        e = _ensure_BT(e)
        m = _ensure_BT(m)

        # mask & flatten
        mask = (m > 0.5)
        if mask.numel() == 0:
            return
        u_sel = u[mask]
        e_sel = e[mask]
        if u_sel.numel() == 0:
            return

        u_np = u_sel.detach().cpu().numpy().reshape(-1).astype(np.float64)
        e_np = e_sel.detach().cpu().numpy().reshape(-1).astype(np.float64)

        # filter finite
        fin = np.isfinite(u_np) & np.isfinite(e_np)
        u_np = u_np[fin]
        e_np = e_np[fin]
        if u_np.size == 0:
            return

        # cap
        if self.max_store > 0:
            room = self.max_store - len(self.u)
            if room <= 0:
                return
            take = min(room, u_np.size)
            self.u.extend(u_np[:take].tolist())
            self.e.extend(e_np[:take].tolist())

    def finalize(self, keep_fracs=(0.9, 0.8, 0.5)):
        if len(self.u) == 0:
            return {"n": 0}

        u = np.asarray(self.u, dtype=np.float64)
        e = np.asarray(self.e, dtype=np.float64)

        out = {
            "n": int(u.size),
            "unc_mean": float(u.mean()),
            "unc_std": float(u.std()),
            "err_mean": float(e.mean()),
            "err_std": float(e.std()),
        }

        # Pearson corr
        if u.size >= 10:
            u0 = u - u.mean()
            e0 = e - e.mean()
            denom = (np.sqrt((u0 * u0).mean()) * np.sqrt((e0 * e0).mean()) + 1e-12)
            out["unc_err_corr"] = float((u0 * e0).mean() / denom)
        else:
            out["unc_err_corr"] = None

        # AURC: sort by uncertainty ascending, compute prefix mean error, take mean(prefix_mean)
        idx = np.argsort(u)  # low-unc first
        e_sorted = e[idx]
        prefix_mean = np.cumsum(e_sorted) / (np.arange(e_sorted.size) + 1.0)
        out["aurc"] = float(prefix_mean.mean())

        # risk@coverage: keep most certain fraction
        for frac in keep_fracs:
            k = max(1, int(frac * e_sorted.size))
            out[f"risk_keep_{int(frac * 100)}pct"] = float(e_sorted[:k].mean())

        # AUC: can uncertainty detect high-error events (top-10%) ?
        thr = np.quantile(e, 0.90)
        labels = (e >= thr).astype(np.int32)
        out["auc_err_p90"] = _fast_auc(u, labels)

        return out


def _mean_rmse_over_keys(rmse_dict, keys):
    # rmse_dict[k] is (B,T) tensor
    if not keys:
        return None
    arr = [rmse_dict[k] for k in keys if k in rmse_dict]
    if not arr:
        return None
    return torch.stack(arr, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_one_dataset(agent, episodes, config, batches, warmup, horizons):
    wm = agent._wm
    ds = dreamer.make_dataset(episodes, config)

    # ensemble (if exists)
    ensemble = getattr(agent, "_disag_ensemble", None)

    out = {
        "one_step": {"nll": {}, "rmse": {}},
        "open_loop": {f"k{h}": {"nll": {}, "rmse": {}} for h in horizons},
        "diag": {"kl": 0.0, "dyn_loss": 0.0, "rep_loss": 0.0},
        "meta": {"batches": batches, "warmup": warmup, "horizons": horizons},
    }

    # uncertainty outputs
    if ensemble is not None:
        out["uncertainty"] = {
            "one_step": None,
            "open_loop": {f"k{h}": None for h in horizons},
        }
        unc_one = UncBuffer(max_store=200000)
        unc_k = {h: UncBuffer(max_store=200000) for h in horizons}
    else:
        unc_one = None
        unc_k = None

    counts_1 = collections.Counter()
    counts_k = {h: collections.Counter() for h in horizons}

    state_keys = None

    for _ in range(batches):
        batch = next(ds)
        data = wm.preprocess(batch)

        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

        # diagnostics: KL/dyn/rep (mask out is_first + cross-episode)
        kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
            post, prior, config.kl_free, config.dyn_scale, config.rep_scale
        )

        valid_1 = safe_mask_is_first(data, start_idx=0, drop_first=True)
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

        # We will also keep per-step RMSE tensors for uncertainty correlation.
        rmse_1step_perkey = {}

        for k in state_keys:
            # NLL
            nll = -pred_prior[k].log_prob(data[k])
            nll = _ensure_BT(nll)
            nll_mean = (nll * valid_1).sum() / denom_1
            out["one_step"]["nll"][k] = out["one_step"]["nll"].get(k, 0.0) + nll_mean.item()
            counts_1[f"nll:{k}"] += 1

            # RMSE
            mu = dist_mean(pred_prior[k])
            err2 = (mu - data[k]) ** 2
            if err2.ndim > 2:
                err2 = err2.mean(dim=list(range(2, err2.ndim)))
            rmse = torch.sqrt(err2 + 1e-8)  # (B,T)
            rmse_1step_perkey[k] = rmse

            rmse_mean = (rmse * valid_1).sum() / denom_1
            out["one_step"]["rmse"][k] = out["one_step"]["rmse"].get(k, 0.0) + rmse_mean.item()
            counts_1[f"rmse:{k}"] += 1

        # ---- uncertainty for 1-step (if ensemble exists) ----
        # Align as: uncertainty for predicting obs[t] uses (state[t-1], action[t]).
        if ensemble is not None:
            # build prev-state sequence from post: prev_post[t] = post[t-1] (t>0), dummy for t=0 (masked out anyway)
            prev_post = {kk: torch.cat([vv[:, :1], vv[:, :-1]], dim=1) for kk, vv in post.items()}
            actions = {"action": data["action"]}
            dis_1 = tools.get_uncertainty(wm, ensemble, prev_post, actions)
            dis_1 = _ensure_BT(dis_1)

            err_1 = _mean_rmse_over_keys(rmse_1step_perkey, state_keys)
            if err_1 is not None:
                unc_one.add(dis_1, err_1, valid_1)

        # open-loop rollout
        W = warmup
        W = min(W, data["action"].shape[1] - 1)

        post_w, _ = wm.dynamics.observe(embed[:, :W], data["action"][:, :W], data["is_first"][:, :W])
        init = {kk: vv[:, -1] for kk, vv in post_w.items()}

        prior_ol = wm.dynamics.imagine_with_action(data["action"][:, W:], init)
        feat_ol = wm.dynamics.get_feat(prior_ol)
        pred_ol = wm.heads["decoder"](feat_ol)

        T_imag = data["action"].shape[1] - W

        full_nll = {}
        full_rmse = {}

        for k in state_keys:
            target = data[k][:, W: W + T_imag]

            # NLL
            nll = -pred_ol[k].log_prob(target)
            nll = _ensure_BT(nll)
            full_nll[k] = nll

            # RMSE
            mu = dist_mean(pred_ol[k])
            err2 = (mu - target) ** 2
            if err2.ndim > 2:
                err2 = err2.mean(dim=list(range(2, err2.ndim)))
            rmse = torch.sqrt(err2 + 1e-8)  # (B,T_imag)
            full_rmse[k] = rmse

        valid_ol = safe_mask_is_first(data, start_idx=W, drop_first=False)

        # ---- uncertainty for open-loop (if ensemble exists) ----
        # Align as: uncertainty for predicting obs[W+t] uses (state[W+t-1], action[W+t]).
        if ensemble is not None:
            # latent_prev sequence length T_imag:
            # t=0 uses init (post at W-1), t>0 uses prior_ol at t-1
            latent_prev = {
                kk: torch.cat([init[kk].unsqueeze(1), prior_ol[kk][:, :-1]], dim=1)
                for kk in prior_ol.keys()
            }
            act_seq = {"action": data["action"][:, W: W + T_imag]}
            dis_ol = tools.get_uncertainty(wm, ensemble, latent_prev, act_seq)
            dis_ol = _ensure_BT(dis_ol)

            err_ol = _mean_rmse_over_keys(full_rmse, state_keys)  # (B,T_imag)

        for h in horizons:
            hh = min(h, valid_ol.shape[1])
            m = valid_ol[:, :hh]
            denom = torch.clamp(m.sum(), min=1.0)

            for k in state_keys:
                nll_seq = full_nll[k][:, :hh]
                nll_mean = (nll_seq * m).sum() / denom
                out["open_loop"][f"k{h}"]["nll"][k] = out["open_loop"][f"k{h}"]["nll"].get(k, 0.0) + nll_mean.item()
                counts_k[h][f"nll:{k}"] += 1

                rmse_seq = full_rmse[k][:, :hh]
                rmse_mean = (rmse_seq * m).sum() / denom
                out["open_loop"][f"k{h}"]["rmse"][k] = out["open_loop"][f"k{h}"]["rmse"].get(k, 0.0) + rmse_mean.item()
                counts_k[h][f"rmse:{k}"] += 1

            # uncertainty stats per horizon
            if ensemble is not None and err_ol is not None:
                unc_k[h].add(dis_ol[:, :hh], err_ol[:, :hh], m)

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

    # finalize uncertainty
    if ensemble is not None:
        out["uncertainty"]["one_step"] = unc_one.finalize()
        for h in horizons:
            out["uncertainty"]["open_loop"][f"k{h}"] = unc_k[h].finalize()

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

    # uncertainty summary
    if "uncertainty" in res and res["uncertainty"] is not None:
        u1 = res["uncertainty"].get("one_step", None)
        if u1 and u1.get("n", 0) > 0:
            print(f"unc(1-step): mean={u1.get('unc_mean'):.6f} corr={u1.get('unc_err_corr')} aurc={u1.get('aurc')}")
        uol = res["uncertainty"].get("open_loop", {})
        for hk, uv in uol.items():
            if uv and uv.get("n", 0) > 0:
                print(f"unc({hk}): mean={uv.get('unc_mean'):.6f} corr={uv.get('unc_err_corr')} aurc={uv.get('aurc')}")


def main():
    # stage-1 args (eval-specific)
    p1 = argparse.ArgumentParser()
    p1.add_argument("--configs", nargs="+")
    p1.add_argument("--config_path", type=str, default="/home/yhy/IsaacLabExtensionTemplate/scripts/dreamerv3_torch/configs.yaml")

    p1.add_argument("--ckpt_base", type=str, default="latent_safety/log/dreamerv3/1225/latest.pt")
    p1.add_argument("--ckpt_ft", type=str, default="latent_safety/log/dreamerv3/world_model_only/0123/205246_test/latest.pt")

    p1.add_argument("--id_dir", type=str, default="latent_safety/log/dreamerv3/collect_data/")
    p1.add_argument("--ood_dir", type=str, default="latent_safety/log/dreamerv3/collect_data_ood/")

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
