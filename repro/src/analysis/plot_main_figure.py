"""Aggregate per-experiment JSON results and produce the main 1x3 AUC figure.

Each training script in src/experiments/ writes one JSON per run with this shape:

    {
      "args": { "number_to_use": "6", "model_name": "...", ... },
      "results": [
        { "model": "...", "fraction": 0.01, "train_samples": 4000,
          "test_auc": 0.81, "val_auc": 0.79, ... },
        ...
      ]
    }

This script discovers every JSON in --results_dir, infers (model, dataset, fraction, seed, auc),
and plots AUC vs train_samples for each model family on each of the 3 datasets.

Usage:
    python -m src.analysis.plot_main_figure --results_dir results --output figures/auc_combined_main_1row.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

# The panel grid is derived from this list, so adding a tag adds a panel. KIP
# (kip_m4 / kip_m6) is deliberately absent: no model-training runs exist for it
# yet, and an empty panel would degrade the main figure. Add the two entries here
# once src/experiments results for those tags are present.
DATASETS = [
    ("6", "Tricky Deterministic"),
    ("9", "Tricky Random"),
    ("test_just_pair", "Parity"),
]

# pi = 0 and rho = 0.5 for both KIP tags, so AUC* = 1 - pi = 1.000.
ORACLE_AUC = {
    "6": 1.000,
    "9": 0.670,
    "test_just_pair": 1.000,
    "kip_m4": 1.000,
    "kip_m6": 1.000,
}


def iter_records(results_dir: Path) -> Iterable[dict]:
    for jf in sorted(results_dir.rglob("*.json")):
        try:
            payload = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        args = payload.get("args", {})
        ds = str(args.get("number_to_use") or args.get("csv_to_use") or "?")
        seed = args.get("seed", 0)
        for r in payload.get("results", []):
            auc = r.get("test_auc") or r.get("val_auc")
            if auc is None:
                continue
            yield {
                "model": r.get("model") or args.get("model_name") or jf.stem,
                "dataset": ds,
                "fraction": r.get("fraction"),
                "train_samples": r.get("train_samples"),
                "auc": float(auc),
                "seed": seed,
                "source": jf.name,
            }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("figures/auc_combined_main_1row.png"))
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated model names to include (default: all found)")
    args = p.parse_args()

    records: List[dict] = list(iter_records(args.results_dir))
    if not records:
        raise SystemExit(f"No JSON results found under {args.results_dir}. "
                         "Run scripts/reproduce_main.sh first.")
    df = pd.DataFrame.from_records(records)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        df = df[df["model"].isin(wanted)]

    # Derived from DATASETS rather than hardcoded to 3: with a fixed count, a
    # fourth entry would be silently dropped by the zip below and its panel would
    # never appear.
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.0 * len(DATASETS), 4.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ax, (tag, title) in zip(axes, DATASETS):
        sub = df[df["dataset"] == tag]
        if sub.empty:
            ax.set_title(f"{title}\n(no data)")
            continue
        for model, grp in sub.groupby("model"):
            agg = grp.groupby("train_samples")["auc"].mean().sort_index()
            ax.plot(agg.index, agg.values, marker="o", label=model)
        ax.axhline(ORACLE_AUC.get(tag, 1.0), color="grey", linestyle="--", alpha=0.6,
                   label=f"AUC* = {ORACLE_AUC.get(tag, 1.0):.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("Training samples")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test AUC")
    axes[-1].legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
