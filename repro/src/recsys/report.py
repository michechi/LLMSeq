"""Job-1 report for the 30Music audit grid: main table + baseline ladder.

Recomputes everything from the saved top-10 lists (results/recsys_audit/recs/)
plus the canonical targets, so every number is reproducible offline and CIs
are true user bootstraps of seed-averaged per-user metrics (B=1000, numpy
seed 4242 — HANDOFF spec).

Main table — rows = models (aggregated over train seeds {17,32,45}) and
baselines; columns = ordered | full_shuffle | keep_last | early_half (variant
cells = mean over the 3 shuffle seeds); each cell: HR@10 and NDCG@10 with 95%
CI, Δabs and Δrel vs ordered; Jaccard@10 vs the SAME model's ordered lists
for the sequence models.

Ladder summary per sequence model M and rung b in MostPopular → ItemKNN →
Markov-1 → Markov-2:

* reach(b)          = b_ordered / M_ordered            (fraction of M's ordered
                      performance available to rung b's mechanism)
* drop_explained(b) = b_drop / M_drop, where X_drop = X_ordered −
                      X_full_shuffle(mean over shuffle seeds)  (how much of
                      M's order sensitivity rung b's mechanism reproduces;
                      0 by construction for the permutation-invariant rungs)

CLI::

    python -m src.recsys.report --audit-dir data/recsys/30music/audit \
        --recs-dir results/recsys_audit/recs \
        --out results/recsys_audit/report_job1.md
"""

from __future__ import annotations

import argparse
import gzip
import os

import numpy as np
import pandas as pd

K = 10
TRAIN_SEEDS = (17, 32, 45)
MODELS = ("SASRec", "GRU4Rec", "BERT4Rec")
BASELINES = ("MostPopular", "ItemKNN", "Markov1", "Markov2")
VARIANTS = ("full_shuffle", "keep_last", "early_half")
SHUFFLE_SEEDS = (101, 102, 103)
BOOT_B, BOOT_SEED = 1000, 4242


def read_recs(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh).set_index("user_id").sort_index()


def per_user_metrics(recs: pd.DataFrame, targets: pd.Series) -> tuple:
    top = recs.to_numpy()
    tgt = targets.loc[recs.index].to_numpy()
    hits = top == tgt[:, None]
    hr = hits.any(axis=1).astype(np.float64)
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 1)
    return hr, hr * (1.0 / np.log2(ranks + 1))


def jaccard(recs_a: pd.DataFrame, recs_b: pd.DataFrame) -> np.ndarray:
    a, b = recs_a.to_numpy(), recs_b.loc[recs_a.index].to_numpy()
    out = np.empty(len(a))
    for i in range(len(a)):
        sa, sb = set(a[i].tolist()), set(b[i].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(values), size=(BOOT_B, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def tags_for(entity: str, input_name: str) -> list[str]:
    """recs-file tags contributing to one (entity, input) cell."""
    if entity in BASELINES:
        return [f"{entity}__{input_name}"]
    return [f"{entity}_ordered_s{s}__{input_name}" for s in TRAIN_SEEDS]


def collect(recs_dir: str, targets: pd.Series, entity: str,
            input_names: list[str]) -> dict | None:
    """Per-user metric vectors averaged over every (train seed x input) file
    contributing to a cell; None if any file is missing."""
    hrs, ndcgs = [], []
    for input_name in input_names:
        for tag in tags_for(entity, input_name):
            path = os.path.join(recs_dir, f"{tag}.csv.gz")
            if not os.path.exists(path):
                return None
            hr, nd = per_user_metrics(read_recs(path), targets)
            hrs.append(hr)
            ndcgs.append(nd)
    return {"hr": np.mean(hrs, axis=0), "ndcg": np.mean(ndcgs, axis=0)}


def collect_jaccard(recs_dir: str, entity: str, variant: str) -> np.ndarray | None:
    if entity in BASELINES:
        pairs = [(f"{entity}__{variant}_s{s}", f"{entity}__ordered")
                 for s in SHUFFLE_SEEDS]
    else:
        pairs = [(f"{entity}_ordered_s{t}__{variant}_s{s}",
                  f"{entity}_ordered_s{t}__ordered")
                 for t in TRAIN_SEEDS for s in SHUFFLE_SEEDS]
    vecs = []
    for var_tag, ord_tag in pairs:
        pv = os.path.join(recs_dir, f"{var_tag}.csv.gz")
        po = os.path.join(recs_dir, f"{ord_tag}.csv.gz")
        if not (os.path.exists(pv) and os.path.exists(po)):
            return None
        vecs.append(jaccard(read_recs(pv), read_recs(po)))
    return np.mean(vecs, axis=0)


def fmt(mean: float, ci: tuple[float, float]) -> str:
    return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--recs-dir", default="results/recsys_audit/recs")
    ap.add_argument("--out", default="results/recsys_audit/report_job1.md")
    args = ap.parse_args()

    tgt_df = pd.read_csv(os.path.join(args.audit_dir, "targets_30Music.csv"))
    targets = tgt_df.set_index("user_id")["item_id"]

    entities = list(MODELS) + list(BASELINES)
    cells: dict = {}
    for e in entities:
        cells[e] = {}
        ordered = collect(args.recs_dir, targets, e, ["ordered"])
        cells[e]["ordered"] = ordered
        for v in VARIANTS:
            names = [f"{v}_s{s}" for s in SHUFFLE_SEEDS]
            cells[e][v] = collect(args.recs_dir, targets, e, names)

    lines = ["# RecSys audit — Job 1 table (30Music, window shuffles)", ""]
    missing = [e for e in entities if cells[e]["ordered"] is None]
    if missing:
        lines += [f"> INCOMPLETE — missing runs for: {', '.join(missing)}", ""]

    for metric in ("hr", "ndcg"):
        label = "HR@10" if metric == "hr" else "NDCG@10"
        lines += [f"## {label}", "",
                  "| model | ordered | full_shuffle | keep_last | early_half |",
                  "|---|---|---|---|---|"]
        for e in entities:
            if cells[e]["ordered"] is None:
                lines.append(f"| {e} | (missing) | | | |")
                continue
            o = cells[e]["ordered"][metric]
            row = [e, fmt(o.mean(), boot_ci(o))]
            for v in VARIANTS:
                c = cells[e][v]
                if c is None:
                    row.append("(missing)")
                    continue
                s = c[metric]
                d = s - o
                drel = d.mean() / o.mean() if o.mean() else float("nan")
                row.append(f"{fmt(s.mean(), boot_ci(s))}<br>"
                           f"Δabs {d.mean():+.4f} {list(boot_ci(d))}<br>"
                           f"Δrel {drel:+.1%}")
            suffix = " *(reference line — permutation-invariant by construction)*" \
                if e in ("MostPopular", "ItemKNN") else ""
            lines.append("| " + " | ".join(row) + suffix + " |")
        lines.append("")

    lines += ["## Jaccard@10 vs own ordered top-10 (mean over seeds)", "",
              "| model | full_shuffle | keep_last | early_half |",
              "|---|---|---|---|"]
    for e in entities:
        row = [e]
        for v in VARIANTS:
            j = collect_jaccard(args.recs_dir, e, v)
            row.append("(missing)" if j is None
                       else fmt(j.mean(), boot_ci(j)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += ["## Baseline ladder", "",
              "reach(b) = b_ordered / model_ordered;  "
              "drop_explained(b) = (b_ordered − b_full_shuffle) / "
              "(model_ordered − model_full_shuffle); HR@10.", ""]
    for m in MODELS:
        mo = cells[m]["ordered"]
        mf = cells[m]["full_shuffle"]
        if mo is None or mf is None:
            lines.append(f"- {m}: (missing)")
            continue
        m_ord, m_drop = mo["hr"].mean(), mo["hr"].mean() - mf["hr"].mean()
        lines += [f"### {m} (ordered HR@10 {m_ord:.4f}, "
                  f"shuffle drop {m_drop:.4f})", "",
                  "| rung | ordered HR@10 | reach | drop_explained |",
                  "|---|---|---|---|"]
        for b in BASELINES:
            bo, bf = cells[b]["ordered"], cells[b]["full_shuffle"]
            if bo is None or bf is None:
                lines.append(f"| {b} | (missing) | | |")
                continue
            b_ord = bo["hr"].mean()
            b_drop = b_ord - bf["hr"].mean()
            reach = b_ord / m_ord if m_ord else float("nan")
            expl = b_drop / m_drop if m_drop else float("nan")
            lines.append(f"| {b} | {b_ord:.4f} | {reach:.1%} | {expl:.1%} |")
        lines.append("")

    lines += ["---", "",
              "Cells: mean over train seeds {17,32,45} (models) and shuffle "
              "seeds {101,102,103}; 95% CI = user bootstrap of the "
              "seed-averaged per-user metric (B=1000, numpy seed 4242); "
              "Δ CIs are paired (per-user differences). Recomputed from "
              "results/recsys_audit/recs/ + audit targets — independent of "
              "the per-run rows in metrics.csv."]

    out = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(out)
    print(out)
    print(f"[report] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
