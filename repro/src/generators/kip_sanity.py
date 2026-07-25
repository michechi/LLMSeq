"""KIP sanity suite -- must pass before any audit or model run.

Five checks, each tied to a property that is *exactly* true of the KIP
construction rather than approximately true, so a failure means the generator is
wrong (or the theory statement is):

1. Structure     every key letter appears exactly once per sequence, and no
                 distractor position holds a key letter.
2. Balance       label mean = 0.5 +/- 0.01. sign(pi) is uniform for m >= 2.
3. Swap-flip     swapping the letters at two key positions flips the label every
                 time (a transposition negates sign(pi)); permuting only
                 distractor positions never changes it.
4. Shuffle       a uniform shuffle of the whole sequence preserves the label
                 50% +/- 2% of the time (it re-randomises pi).
5. Count purity  LogisticRegression *and* XGBoost on the order-invariant 26-dim
                 letter-count vector reach test AUC in [0.48, 0.52]. Key counts
                 are constant at 1 and distractors are independent of pi, so the
                 count vector carries exactly zero information about Y.

Checks 1-4 use only the standard library plus pandas for CSV reading; check 5
additionally needs scikit-learn and xgboost.

Exits non-zero if any check fails, so it can gate a pipeline (the same
convention as ``mimic_analysis/scripts/08_qc_reports.py``).

CLI::

    DATA_DIR=/path/to/data python -m src.generators.kip_sanity
    DATA_DIR=/path/to/data python -m src.generators.kip_sanity --m 6 --skip_counts
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common import SEP, TESTED_DIR
from src.generators.kip import (
    ALPHABET,
    M_VALUES,
    N_POSITIONS,
    SPLITS,
    label_sequence,
    load_rule,
    tag_for,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

BALANCE_ROWS = 100_000
BALANCE_TOL = 0.01
SWAP_TRIALS = 1_000
SHUFFLE_TRIALS = 10_000
SHUFFLE_TOL = 0.02
COUNT_AUC_BAND = (0.48, 0.52)
SEED = 9550


@dataclass
class Result:
    check: str
    m: int
    measured: str
    expected: str
    passed: bool
    notes: str = ""


@dataclass
class Dataset:
    tag: str
    m: int
    kappa: Dict[str, int]
    key_set: frozenset
    seqs: Dict[str, List[List[str]]] = field(default_factory=dict)
    labels: Dict[str, List[int]] = field(default_factory=dict)


def _tokens(raw: str) -> List[str]:
    if SEP in raw:
        return [t for t in raw.split(SEP) if t]
    return list(raw)


def load_dataset(m: int, data_dir: Path) -> Dataset:
    tag = tag_for(m)
    rule = load_rule(data_dir, tag)
    kappa = {k: int(v) for k, v in rule["kappa"].items()}
    ds = Dataset(tag=tag, m=m, kappa=kappa, key_set=frozenset(kappa))
    for split in SPLITS:
        X = pd.read_csv(data_dir / f"X_{split}_{tag}.csv")
        y = pd.read_csv(data_dir / f"y_{split}_{tag}.csv")
        ds.seqs[split] = [_tokens(s) for s in X["Sequences"].tolist()]
        ds.labels[split] = [int(v) for v in y["Outcome"].tolist()]
    return ds


# --------------------------------------------------------------------------- #
# 1. Structure                                                                #
# --------------------------------------------------------------------------- #
def check_structure(ds: Dataset) -> Result:
    want = sorted(ds.key_set)
    bad_len = bad_keys = 0
    total = 0
    for split in SPLITS:
        for toks in ds.seqs[split]:
            total += 1
            if len(toks) != N_POSITIONS:
                bad_len += 1
                continue
            if sorted(t for t in toks if t in ds.key_set) != want:
                bad_keys += 1
    passed = bad_len == 0 and bad_keys == 0
    return Result(
        check="1 structure",
        m=ds.m,
        measured=f"bad_length={bad_len} bad_keyset={bad_keys} of {total}",
        expected="0 / 0",
        passed=passed,
        notes="each key letter exactly once; distractors never in S",
    )


# --------------------------------------------------------------------------- #
# 2. Balance                                                                  #
# --------------------------------------------------------------------------- #
def check_balance(ds: Dataset, rows: int = BALANCE_ROWS) -> Result:
    labels = ds.labels["train"][:rows]
    mean = sum(labels) / len(labels)
    passed = abs(mean - 0.5) <= BALANCE_TOL
    return Result(
        check="2 balance",
        m=ds.m,
        measured=f"{mean:.5f} (n={len(labels)})",
        expected=f"0.5 +/- {BALANCE_TOL}",
        passed=passed,
    )


# --------------------------------------------------------------------------- #
# 3. Swap-flip                                                                #
# --------------------------------------------------------------------------- #
def check_swap_flip(ds: Dataset, trials: int = SWAP_TRIALS) -> Result:
    rng = random.Random(SEED)
    pool = ds.seqs["test"]
    key_flip_failures = 0
    distractor_change_failures = 0

    for _ in range(trials):
        toks = list(rng.choice(pool))
        base = label_sequence(toks, ds.kappa)

        key_pos = [i for i, t in enumerate(toks) if t in ds.key_set]
        i, j = rng.sample(key_pos, 2)
        swapped = list(toks)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        if label_sequence(swapped, ds.kappa) == base:
            key_flip_failures += 1

        dis_pos = [i for i, t in enumerate(toks) if t not in ds.key_set]
        shuffled = list(toks)
        vals = [toks[p] for p in dis_pos]
        rng.shuffle(vals)
        for p, v in zip(dis_pos, vals):
            shuffled[p] = v
        if label_sequence(shuffled, ds.kappa) != base:
            distractor_change_failures += 1

    passed = key_flip_failures == 0 and distractor_change_failures == 0
    return Result(
        check="3 swap-flip",
        m=ds.m,
        measured=f"key_swap_not_flipped={key_flip_failures} "
                 f"distractor_perm_changed={distractor_change_failures} of {trials}",
        expected="0 / 0",
        passed=passed,
        notes="transposing two key letters must flip Y; distractors must not matter",
    )


# --------------------------------------------------------------------------- #
# 4. Shuffle decorrelation                                                    #
# --------------------------------------------------------------------------- #
def check_shuffle(ds: Dataset, trials: int = SHUFFLE_TRIALS) -> Result:
    rng = random.Random(SEED + 1)
    pool = ds.seqs["test"]
    preserved = 0
    for _ in range(trials):
        toks = list(rng.choice(pool))
        base = label_sequence(toks, ds.kappa)
        rng.shuffle(toks)
        if label_sequence(toks, ds.kappa) == base:
            preserved += 1
    frac = preserved / trials
    passed = abs(frac - 0.5) <= SHUFFLE_TOL
    return Result(
        check="4 shuffle",
        m=ds.m,
        measured=f"{frac:.4f} (n={trials})",
        expected=f"0.50 +/- {SHUFFLE_TOL}",
        passed=passed,
        notes="uniform shuffle re-randomises pi",
    )


# --------------------------------------------------------------------------- #
# 5. Count purity                                                             #
# --------------------------------------------------------------------------- #
def _count_matrix(seqs: Sequence[Sequence[str]]) -> np.ndarray:
    idx = {a: i for i, a in enumerate(ALPHABET)}
    out = np.zeros((len(seqs), len(ALPHABET)), dtype=np.float32)
    for r, toks in enumerate(seqs):
        for t in toks:
            j = idx.get(t)
            if j is not None:
                out[r, j] += 1.0
    return out


def check_count_purity(ds: Dataset, rows_train: int, rows_eval: int) -> List[Result]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xtr = _count_matrix(ds.seqs["train"][:rows_train])
    ytr = np.asarray(ds.labels["train"][:rows_train])
    Xte = _count_matrix(ds.seqs["test"][:rows_eval])
    yte = np.asarray(ds.labels["test"][:rows_eval])

    results: List[Result] = []

    # Same settings as phase2_baseline_ladder.py's _fit_logreg, minus its
    # n_jobs=-1 (a no-op since sklearn 1.8; it only emits a FutureWarning).
    logreg = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    logreg.fit(Xtr, ytr)
    auc_lr = float(roc_auc_score(yte, logreg.predict_proba(Xte)[:, 1]))
    results.append(Result(
        check="5 count purity (LogReg)",
        m=ds.m,
        measured=f"{auc_lr:.4f}",
        expected=f"[{COUNT_AUC_BAND[0]}, {COUNT_AUC_BAND[1]}]",
        passed=COUNT_AUC_BAND[0] <= auc_lr <= COUNT_AUC_BAND[1],
        notes="26-dim order-invariant counts",
    ))

    try:
        from xgboost import XGBClassifier
    except ImportError:
        results.append(Result(
            check="5 count purity (XGBoost)", m=ds.m, measured="xgboost missing",
            expected=f"[{COUNT_AUC_BAND[0]}, {COUNT_AUC_BAND[1]}]", passed=False,
        ))
        return results

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    xgb.fit(Xtr, ytr)
    auc_xgb = float(roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1]))
    results.append(Result(
        check="5 count purity (XGBoost)",
        m=ds.m,
        measured=f"{auc_xgb:.4f}",
        expected=f"[{COUNT_AUC_BAND[0]}, {COUNT_AUC_BAND[1]}]",
        passed=COUNT_AUC_BAND[0] <= auc_xgb <= COUNT_AUC_BAND[1],
        notes="26-dim order-invariant counts",
    ))
    return results


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def format_report(results: Sequence[Result]) -> str:
    head = f"{'check':26s} {'m':>2s}  {'measured':46s} {'expected':22s} result"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.check:26s} {r.m:>2d}  {r.measured:46s} {r.expected:22s} "
            f"{'PASS' if r.passed else 'FAIL'}"
        )
    n_fail = sum(1 for r in results if not r.passed)
    lines.append("-" * len(head))
    lines.append(f"{len(results) - n_fail}/{len(results)} passed"
                 + ("" if n_fail == 0 else f"  -- {n_fail} FAILED"))
    return "\n".join(lines)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIP sanity suite")
    p.add_argument("--m", type=int, nargs="+", default=list(M_VALUES))
    p.add_argument("--data_dir", type=Path, default=TESTED_DIR)
    p.add_argument("--rows_train", type=int, default=400_000)
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--skip_counts", action="store_true",
                   help="skip check 5 (the only one needing sklearn/xgboost)")
    p.add_argument("--out", type=Path, default=None, help="also write the report here")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logger.info("data_dir = %s", args.data_dir)

    results: List[Result] = []
    for m in args.m:
        logger.info("loading %s ...", tag_for(m))
        ds = load_dataset(m, args.data_dir)
        logger.info("  S (kappa order) = %s",
                    ",".join(sorted(ds.kappa, key=lambda k: ds.kappa[k])))
        results.append(check_structure(ds))
        results.append(check_balance(ds))
        results.append(check_swap_flip(ds))
        results.append(check_shuffle(ds))
        if not args.skip_counts:
            results.extend(check_count_purity(ds, args.rows_train, args.rows_eval))

    report = format_report(results)
    print("\n" + report + "\n")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        logger.info("wrote %s", args.out)

    failed = [r for r in results if not r.passed]
    if failed:
        logger.error("SANITY FAILED: %d check(s)", len(failed))
        return 1
    logger.info("SANITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
