"""Non-neural baselines for the 30Music audit grid: MostPopular, ItemKNN
(bag-of-items), Markov-1 and Markov-2 with deterministic backoff.

All baselines are fit on the ordered TRAIN split only and evaluated on the
same 10 inputs as the models, through the contract-enforcing eval_loader.

Order sensitivity by construction:
* MostPopular / ItemKNN score a user from the WINDOW MULTISET only — the audit
  variants preserve that multiset, so their scores (and top-10 lists) are
  provably identical across all 10 inputs. They are computed ONCE on the
  ordered input and re-emitted per variant with order_invariant=true
  (Jaccard@10 = 1, Δ = 0 by construction — that is the point of these rows).
* Markov-1 scores from the LAST window item; Markov-2 from the last TWO
  (backoff Markov-2 → Markov-1 → popularity, top-10 lists filled through the
  same ladder, ties broken by (count desc, popularity desc, item id asc)).
  These are order-sensitive and are evaluated per input for real.

CLI::

    python -m src.recsys.baselines --split-dir data/recsys/30music/split \
        --audit-dir data/recsys/30music/audit [--baselines all]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from .eval_loader import _assert_structure, _group_positions, _read, load_eval_input
from .eval_grid import (ALL_INPUTS, METRICS_HEADER, input_path, metric_row,
                        save_recs)
from .recbole_grid import append_locked, hr_ndcg_at_k

K = 10


def load_train(split_dir: str) -> pd.DataFrame:
    df = _read(os.path.join(split_dir, "train_30Music.csv"))
    _assert_structure(df, "train_30Music.csv")
    return df


def popularity_order(train: pd.DataFrame, n_items: int) -> tuple:
    """(counts [n_items], item ids sorted by count desc then id asc)."""
    counts = np.zeros(n_items, dtype=np.int64)
    vc = train["item_id"].value_counts()
    counts[vc.index.to_numpy()] = vc.to_numpy()
    order = np.lexsort((np.arange(n_items), -counts))
    return counts, order


def rank_key(counts: np.ndarray, pop_counts: np.ndarray, items: np.ndarray):
    """Sort items by (count desc, popularity desc, id asc)."""
    return np.lexsort((items, -pop_counts[items], -counts))


def build_markov1(train: pd.DataFrame, pop_counts: np.ndarray) -> dict:
    users = train["user_id"].to_numpy()
    items = train["item_id"].to_numpy()
    prev, nxt = items[:-1], items[1:]
    same = users[1:] == users[:-1]
    df = pd.DataFrame({"prev": prev[same], "next": nxt[same]})
    counts = df.groupby(["prev", "next"]).size().reset_index(name="c")
    counts = counts.sort_values("c", ascending=False, kind="stable")
    table = {}
    for p, grp in counts.groupby("prev", sort=False):
        it = grp["next"].to_numpy()
        order = rank_key(grp["c"].to_numpy(), pop_counts, it)
        table[int(p)] = it[order][:K]
    return table


def build_markov2(train: pd.DataFrame, pop_counts: np.ndarray,
                  n_items: int) -> dict:
    users = train["user_id"].to_numpy()
    items = train["item_id"].to_numpy().astype(np.int64)
    p2, p1, nxt = items[:-2], items[1:-1], items[2:]
    same = (users[2:] == users[:-2]) & (users[1:-1] == users[:-2])
    ctx = p2[same] * n_items + p1[same]
    df = pd.DataFrame({"ctx": ctx, "next": nxt[same]})
    counts = df.groupby(["ctx", "next"]).size().reset_index(name="c")
    counts = counts.sort_values("c", ascending=False, kind="stable")
    table = {}
    for c, grp in counts.groupby("ctx", sort=False):
        it = grp["next"].to_numpy()
        order = rank_key(grp["c"].to_numpy(), pop_counts, it)
        table[int(c)] = it[order][:K]
    return table


def fill_topk(primary: np.ndarray, *ladders) -> np.ndarray:
    """Dedup-fill a top-K list from primary then each backoff ladder."""
    out, seen = [], set()
    for src in (primary, *ladders):
        for item in src:
            if item not in seen:
                seen.add(item)
                out.append(item)
                if len(out) == K:
                    return np.array(out)
    return np.array(out)  # cannot happen with popularity as final ladder


def itemknn_topk(train: pd.DataFrame, windows: list, n_items: int,
                 device, pop_top: np.ndarray) -> np.ndarray:
    """score(i) = sum_{j in window bag} cos(i, j) over the binary train
    user-item matrix; computed as D^-1 X^T (X (D^-1 w)) in user batches."""
    ui = train[["user_id", "item_id"]].drop_duplicates()
    u_codes = ui["user_id"].astype("category").cat.codes.to_numpy()
    rows = torch.tensor(u_codes, dtype=torch.long)
    cols = torch.tensor(ui["item_id"].to_numpy(), dtype=torch.long)
    n_users = int(rows.max()) + 1
    X = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), torch.ones(len(rows)),
        (n_users, n_items)).coalesce().to(device)
    norms = torch.sparse.sum(X * 1.0, dim=0).to_dense().sqrt().clamp(min=1e-12)

    topk = []
    B = 128
    for start in range(0, len(windows), B):
        chunk = windows[start:start + B]
        W = torch.zeros((n_items, len(chunk)), device=device)
        for j, w in enumerate(chunk):
            it, cnt = np.unique(w, return_counts=True)
            W[torch.tensor(it, device=device), j] = \
                torch.tensor(cnt, dtype=torch.float32, device=device)
        W = W / norms[:, None]
        scores = torch.sparse.mm(X.t(), torch.sparse.mm(X, W)) / norms[:, None]
        top = torch.topk(scores, K, dim=0)
        topk.append(top.indices.T.cpu().numpy())
        # a window of fully train-unseen items scores 0 everywhere -> top-10
        # would be arbitrary; fall back to popularity (deterministic)
        dead = (top.values.max(dim=0).values <= 0).cpu().numpy()
        if dead.any():
            topk[-1][dead] = pop_top[:K]
    return np.concatenate(topk, axis=0)


def eval_baseline_on_input(name: str, input_name: str, seqs_sorted: dict,
                           state: dict, device) -> np.ndarray:
    """Top-10 lists [n_users, 10] for one baseline on one loaded input."""
    users = state["users"]
    windows = [seqs_sorted[u]["window"] for u in users]
    if name == "MostPopular":
        return np.tile(state["pop_top10"], (len(users), 1))
    if name == "ItemKNN":
        return itemknn_topk(state["train"], windows, state["n_items"], device,
                            state["pop_top10"])
    if name == "Markov1":
        return np.stack([
            fill_topk(state["m1"].get(w[-1], np.empty(0, np.int64)),
                      state["pop_top_pad"])
            for w in windows])
    if name == "Markov2":
        n = state["n_items"]
        return np.stack([
            fill_topk(state["m2"].get(w[-2] * n + w[-1], np.empty(0, np.int64))
                      if len(w) >= 2 else np.empty(0, np.int64),
                      state["m1"].get(w[-1], np.empty(0, np.int64)),
                      state["pop_top_pad"])
            for w in windows])
    raise ValueError(name)


ORDER_INVARIANT = {"MostPopular": True, "ItemKNN": True,
                   "Markov1": False, "Markov2": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--baselines", nargs="+", default=["all"])
    ap.add_argument("--results-csv", default="results/recsys_audit/metrics.csv")
    ap.add_argument("--recs-dir", default="results/recsys_audit/recs")
    ap.add_argument("--limit-users", type=int, default=0)
    ap.add_argument("--site", default=os.environ.get("KIP_SITE", "local"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = (["MostPopular", "ItemKNN", "Markov1", "Markov2"]
             if args.baselines == ["all"] else args.baselines)

    train = load_train(args.split_dir)
    n_items = int(max(train["item_id"].max(),
                      _read(os.path.join(args.split_dir,
                                         "test_30Music.csv"))["item_id"].max())) + 1
    pop_counts, pop_order = popularity_order(train, n_items)
    state = {"train": train, "n_items": n_items,
             "pop_top10": pop_order[:K],
             "pop_top_pad": pop_order[:2 * K]}  # ladder tail can dedup away
    if "Markov1" in names or "Markov2" in names:
        state["m1"] = build_markov1(train, pop_counts)
        print(f"Markov-1 contexts: {len(state['m1'])} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if "Markov2" in names:
        state["m2"] = build_markov2(train, pop_counts, n_items)
        print(f"Markov-2 contexts: {len(state['m2'])} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for name in names:
        results = {}
        for input_name in ALL_INPUTS:
            if ORDER_INVARIANT[name] and input_name != "ordered":
                res = {**results["ordered"]}  # provably identical scores
            else:
                seqs = load_eval_input(
                    input_path(input_name, args.audit_dir, args.split_dir),
                    args.audit_dir)
                users = sorted(seqs)
                if args.limit_users:
                    users = users[:args.limit_users]
                st = {**state, "users": users}
                topk = eval_baseline_on_input(name, input_name, seqs, st, device)
                targets = np.array([seqs[u]["target"] for u in users])
                hr, ndcg = hr_ndcg_at_k(topk, targets)
                res = {"users": np.array(users), "topk": topk,
                       "hr": hr, "ndcg": ndcg}
            results[input_name] = res
            label = {"model": name, "seed": "", "mode": "ordered",
                     "site": args.site, "checkpoint": "",
                     "order_invariant": str(ORDER_INVARIANT[name]).lower()}
            save_recs(os.path.join(args.recs_dir,
                                   f"{name}__{input_name}.csv.gz"),
                      res["users"], res["topk"])
            row = metric_row(label, input_name, res, results["ordered"], t0)
            append_locked(args.results_csv, METRICS_HEADER, row)
            print(f"{name} {input_name}: HR@10 {res['hr'].mean():.4f} "
                  f"NDCG@10 {res['ndcg'].mean():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("BASELINES DONE", flush=True)


if __name__ == "__main__":
    main()
