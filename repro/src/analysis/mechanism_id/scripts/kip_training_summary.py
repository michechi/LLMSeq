"""Summarize results/kip_training.csv: AUC mean +/- std per (model, dataset, mode).

Emits a markdown table extending the KIP cross-task picture, keeping the two
reference lines from the audit/reveal work:

  precedence-bits + MLP oracle rung     AUC 1.000 at both m
  best audit-ladder baseline            1.000 at m4 (all-lag pairs + XGBoost),
                                        0.503 at m6 (nothing exceeds chance)

CLI::

    python -m src.analysis.mechanism_id.scripts.kip_training_summary
    python -m src.analysis.mechanism_id.scripts.kip_training_summary --wallclock
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

RESULTS_CSV = Path("/root/LLMSeq/results/kip_training.csv")

MODE_LABEL = {
    "ordered": "(a) ordered/ordered",
    "shuffled_train": "(b) shuffled-train",
    "shuffled_eval": "(c) shuffled-eval",
}

REFERENCE_ROWS = [
    ("precedence bits + MLP (oracle rung)", "kip_m4", "--", "1.000 (exact)"),
    ("precedence bits + MLP (oracle rung)", "kip_m6", "--", "1.000 (exact)"),
    ("best audit rung (all-lag pairs + XGBoost)", "kip_m4", "--", "1.000"),
    ("best audit rung (all baselines)", "kip_m6", "--", "0.503"),
]


def summarize(df: pd.DataFrame, wallclock: bool) -> str:
    df = df[df["smoke"].astype(str).str.lower() != "true"].copy()
    if df.empty:
        return "(no non-smoke rows yet)"
    df["test_auc"] = pd.to_numeric(df["test_auc"])
    df["test_f1"] = pd.to_numeric(df["test_f1"])
    df["wallclock_s"] = pd.to_numeric(df["wallclock_s"])

    g = (df.groupby(["model", "dataset", "mode"])
           .agg(auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
                f1_mean=("test_f1", "mean"),
                n_seeds=("seed", "nunique"),
                wall_mean_s=("wallclock_s", "mean"),
                epochs_mean=("epochs_done", "mean"))
           .reset_index())
    g["auc_std"] = g["auc_std"].fillna(0.0)

    header = ["model", "dataset", "mode", "AUC mean +/- std", "F1 mean", "seeds"]
    if wallclock:
        header += ["mean wall-clock", "mean epochs"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, r in g.sort_values(["dataset", "model", "mode"]).iterrows():
        row = [r["model"], r["dataset"], MODE_LABEL.get(r["mode"], r["mode"]),
               f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}",
               f"{r['f1_mean']:.4f}", str(int(r["n_seeds"]))]
        if wallclock:
            m, s = divmod(int(r["wall_mean_s"]), 60)
            row += [f"{m}m{s:02d}s", f"{r['epochs_mean']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")

    for name, ds, mode, auc in REFERENCE_ROWS:
        row = [f"*{name}*", ds, mode, auc, "--", "--"]
        if wallclock:
            row += ["--", "--"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize KIP training results")
    p.add_argument("--results_csv", type=Path, default=RESULTS_CSV)
    p.add_argument("--wallclock", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results_csv.exists():
        logger.error("no results yet at %s", args.results_csv)
        return 1
    table = summarize(pd.read_csv(args.results_csv), args.wallclock)
    print(table)
    if args.out:
        args.out.write_text(table + "\n")
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
