"""Evaluate one trained grid checkpoint on the ordered test input + the 9
audit shuffle variants (or a subset).

Inputs are consumed EXCLUSIVELY through src.recsys.eval_loader.load_eval_input
(the machine-enforced consumer contract). Per (checkpoint, input): full-catalog
top-10 (no seen filtering), per-user HR@10 / NDCG@10, Jaccard@10 vs the SAME
checkpoint's ordered top-10 lists, Δabs / Δrel with paired user-bootstrap 95%
CIs (B=1000, numpy seed 4242), one fcntl-locked row appended to
results/recsys_audit/metrics.csv, and the top-10 lists saved to
results/recsys_audit/recs/ for offline recomputation.

CLI::

    python -m src.recsys.eval_grid --checkpoint checkpoints/recsys/SASRec_ordered_s17.pt \
        --audit-dir data/recsys/30music/audit --split-dir data/recsys/30music/split \
        --inputs all
"""

from __future__ import annotations

import argparse
import gzip
import os
import socket
import time

import numpy as np
import torch

from .eval_loader import load_eval_input
from .recbole_grid import (append_locked, build_model, full_sort_topk,
                           hr_ndcg_at_k, paired_bootstrap_ci)

METRICS_HEADER = (
    "model,train_seed,train_mode,eval_input,n_users,hr10,hr10_ci_lo,hr10_ci_hi,"
    "ndcg10,ndcg10_ci_lo,ndcg10_ci_hi,jaccard10,jaccard10_ci_lo,jaccard10_ci_hi,"
    "d_hr10_abs,d_hr10_rel,d_hr10_ci_lo,d_hr10_ci_hi,"
    "d_ndcg10_abs,d_ndcg10_rel,d_ndcg10_ci_lo,d_ndcg10_ci_hi,"
    "order_invariant,seconds,site,host,gpu,slurm_job_id,checkpoint")

ALL_INPUTS = ["ordered"] + [f"{v}_s{s}" for v in
                            ["full_shuffle", "keep_last", "early_half"]
                            for s in [101, 102, 103]]


def input_path(name: str, audit_dir: str, split_dir: str) -> str:
    if name == "ordered":
        return os.path.join(split_dir, "test_30Music.csv")
    return os.path.join(audit_dir, f"test_30Music_{name}.csv")


def evaluate_input(model, name: str, audit_dir: str, split_dir: str,
                   device, limit_users: int = 0) -> dict:
    """Returns {user_ids, topk, hr, ndcg} for one eval input, users sorted."""
    seqs = load_eval_input(input_path(name, audit_dir, split_dir), audit_dir)
    users = sorted(seqs)
    if limit_users:
        users = users[:limit_users]
    windows = [[i + 1 for i in seqs[u]["window"]] for u in users]  # +1 shift
    targets = np.array([seqs[u]["target"] for u in users])
    topk = full_sort_topk(model, windows, device)
    hr, ndcg = hr_ndcg_at_k(topk, targets)
    return {"users": np.array(users), "topk": topk, "hr": hr, "ndcg": ndcg}


def jaccard_per_user(topk_a: np.ndarray, topk_b: np.ndarray) -> np.ndarray:
    out = np.empty(len(topk_a))
    for i in range(len(topk_a)):
        a, b = set(topk_a[i].tolist()), set(topk_b[i].tolist())
        out[i] = len(a & b) / len(a | b)
    return out


def save_recs(path: str, users: np.ndarray, topk: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("user_id," + ",".join(f"rank{r}" for r in range(1, 11)) + "\n")
        for u, row in zip(users, topk):
            fh.write(str(u) + "," + ",".join(map(str, row)) + "\n")


def metric_row(label: dict, name: str, res: dict, ordered: dict,
               t0: float) -> str:
    """One metrics.csv row; Δ/Jaccard columns empty for the ordered input."""
    hr_ci = paired_bootstrap_ci(res["hr"])
    ndcg_ci = paired_bootstrap_ci(res["ndcg"])
    if name == "ordered":
        jac = jac_ci = d_hr = d_ndcg = ("", "")
        jac_mean = d_hr_abs = d_hr_rel = d_ndcg_abs = d_ndcg_rel = ""
    else:
        j = jaccard_per_user(res["topk"], ordered["topk"])
        jac_mean = f"{j.mean():.6f}"
        jac_ci = paired_bootstrap_ci(j)
        d_hr_vec = res["hr"] - ordered["hr"]
        d_ndcg_vec = res["ndcg"] - ordered["ndcg"]
        d_hr_abs = f"{d_hr_vec.mean():.6f}"
        d_hr_rel = (f"{d_hr_vec.mean() / ordered['hr'].mean():.6f}"
                    if ordered["hr"].mean() > 0 else "")
        d_ndcg_abs = f"{d_ndcg_vec.mean():.6f}"
        d_ndcg_rel = (f"{d_ndcg_vec.mean() / ordered['ndcg'].mean():.6f}"
                      if ordered["ndcg"].mean() > 0 else "")
        d_hr = paired_bootstrap_ci(d_hr_vec)
        d_ndcg = paired_bootstrap_ci(d_ndcg_vec)

    def ci(pair):
        return ("", "") if pair == ("", "") else (f"{pair[0]:.6f}", f"{pair[1]:.6f}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cells = [label["model"], label["seed"], label["mode"], name,
             len(res["hr"]), f"{res['hr'].mean():.6f}", *ci(hr_ci),
             f"{res['ndcg'].mean():.6f}", *ci(ndcg_ci),
             jac_mean, *ci(jac_ci if name != "ordered" else ("", "")),
             d_hr_abs, d_hr_rel, *ci(d_hr if name != "ordered" else ("", "")),
             d_ndcg_abs, d_ndcg_rel,
             *ci(d_ndcg if name != "ordered" else ("", "")),
             label.get("order_invariant", "false"),
             round(time.time() - t0, 1), label["site"], socket.gethostname(),
             gpu.replace(",", " "), os.environ.get("SLURM_JOB_ID", ""),
             label.get("checkpoint", "")]
    return ",".join(str(c) for c in cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--inputs", nargs="+", default=["all"])
    ap.add_argument("--results-csv", default="results/recsys_audit/metrics.csv")
    ap.add_argument("--recs-dir", default="results/recsys_audit/recs")
    ap.add_argument("--limit-users", type=int, default=0,
                    help="smoke-testing only")
    ap.add_argument("--site", default=os.environ.get("KIP_SITE", "local"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt["model_name"], ckpt["raw_vocab"], device)
    model.load_state_dict(ckpt["state_dict"])
    label = {"model": ckpt["model_name"], "seed": ckpt["seed"],
             "mode": ckpt["mode"], "site": args.site,
             "checkpoint": args.checkpoint}
    tag = f"{ckpt['model_name']}_{ckpt['mode']}_s{ckpt['seed']}"

    names = ALL_INPUTS if args.inputs == ["all"] else args.inputs
    # ordered must always be evaluated FIRST (Jaccard/Δ reference)
    names = ["ordered"] + [n for n in names if n != "ordered"]

    ordered = None
    for name in names:
        res = evaluate_input(model, name, args.audit_dir, args.split_dir,
                             device, args.limit_users)
        if name == "ordered":
            ordered = res
        save_recs(os.path.join(args.recs_dir, f"{tag}__{name}.csv.gz"),
                  res["users"], res["topk"])
        row = metric_row(label, name, res, ordered, t0)
        append_locked(args.results_csv, METRICS_HEADER, row)
        print(f"{tag} {name}: HR@10 {res['hr'].mean():.4f} "
              f"NDCG@10 {res['ndcg'].mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("EVAL DONE", tag, flush=True)


if __name__ == "__main__":
    main()
