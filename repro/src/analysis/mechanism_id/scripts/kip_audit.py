"""KIP audit ladder -- do order-invariant and low-order features stay at chance?

Runs the baseline families a reviewer would reach for on the KIP datasets:

    unigram_counts     26-dim letter counts (order-invariant)
    kgram_k{2,3,4}     contiguous k-grams, the estimator baseline the paper cites
    lag_pair_l{lag}    aggregated lag-pair counts g_ab^(lag), 676-dim
    max_over_lag       max_lag g_ab^(lag), 676-dim          } lag-agnostic
    sum_over_lag       sum_lag g_ab^(lag), 676-dim          } pair families
    stacked_all_lag    G_{lag,a,b} for every lag, (n-1)*676-dim, built sparse

Theory (see src/generators/kip.py): any feature that is a function of the
relative order of at most m-2 key letters is *exactly* independent of the label,
because two key letters always remain outside such a subset and transposing them
flips sign(pi) while preserving the feature. Unigram counts are independent for a
stronger reason -- every key letter has count exactly 1 in every sequence.

That lemma is *marginal*: it constrains one coordinate at a time, not a
classifier fitted to a whole family. The distinction matters here, because every
key letter occurs exactly once, so the lag between an ordered key pair is unique
and

    max_lag G_ab^(lag)  =  sum_lag G_ab^(lag)  =  1{a occurs before b}

exactly. The lag-agnostic families therefore reconstruct the complete C(m,2)
precedence matrix, which determines Y. What is measured:

    fixed-lag pairs    at chance for every lag tested, at both m -- this is the
                       family that saturates the compliance tasks (linear AUC
                       0.997 on tag 6 and 0.671 = the noise ceiling on tag 9,
                       per src/analysis/mechanism_id/report.md), so the shortcut
                       that makes those results uninterpretable is closed off
    linear models      at chance everywhere; parity of the bit-sum is not a
                       linear threshold function
    max/sum over lag   XGBoost reaches AUC 1.000 at m=4 (6-bit parity) but sits
                       at ~0.502 at m=6 (15-bit parity), despite the information
                       being fully present

So m is a difficulty knob that separates information-theoretic accessibility
from computational accessibility, and KIP-m6 is the variant no baseline in this
ladder solves.

Classifier settings and the AUC protocol are those of phase2_baseline_ladder.py
so numbers are directly comparable with the existing ladder.

Anything theory calls chance but which lands outside [0.48, 0.52] is flagged in
the `outside_band` column rather than adjusted.

CLI::

    DATA_DIR=/path/to/data python -m src.analysis.mechanism_id.scripts.kip_audit
    DATA_DIR=/path/to/data python -m src.analysis.mechanism_id.scripts.kip_audit \
        --m 6 --families unigram_counts stacked_all_lag
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    extract_ngrams,
    feat_count26,
    feat_lag_pair,
    kgram_estimator_scores,
    load_split,
    tokens,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

SEED = 9550
N_DEFAULT = 20
CHANCE_BAND = (0.48, 0.52)
DEFAULT_LAGS: Tuple[int, ...] = (1, 2, 3, 5, 7)
DEFAULT_KS: Tuple[int, ...] = (2, 3, 4)

PAIR_FAMILIES = ("max_over_lag", "sum_over_lag", "stacked_all_lag")


# --------------------------------------------------------------------------- #
# Dense 676-dim lag-agnostic families (phase2c_lag_agnostic.py definitions)    #
# --------------------------------------------------------------------------- #
def feat_max_over_lag(toks: Sequence[str], n: int) -> np.ndarray:
    best = np.zeros((26, 26), dtype=np.float32)
    for lag in range(1, n):
        cur = np.zeros((26, 26), dtype=np.float32)
        for t in range(len(toks) - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai, bi = ord(a) - ord("A"), ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    cur[ai, bi] += 1.0
        np.maximum(best, cur, out=best)
    return best.reshape(-1)


def feat_sum_over_lag(toks: Sequence[str], n: int) -> np.ndarray:
    out = np.zeros((26, 26), dtype=np.float32)
    for lag in range(1, n):
        for t in range(len(toks) - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai, bi = ord(a) - ord("A"), ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    out[ai, bi] += 1.0
    return out.reshape(-1)


def stacked_all_lag_sparse_row(toks: Sequence[str], n: int):
    """(cols, data, D) for G_{lag,a,b}. ~190 nonzeros of 12844 at n=20, so the
    dense form would be 5.1GB at 100K rows against 0.15GB sparse."""
    cols: Dict[int, float] = {}
    for lag in range(1, n):
        base = (lag - 1) * 676
        for t in range(len(toks) - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai, bi = ord(a) - ord("A"), ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    key = base + ai * 26 + bi
                    cols[key] = cols.get(key, 0.0) + 1.0
    idx = sorted(cols)
    return idx, [cols[i] for i in idx], (n - 1) * 676


def _sparse_matrix(seqs: Sequence[Sequence[str]], n: int):
    from scipy.sparse import csr_matrix

    indptr = [0]
    indices: List[int] = []
    values: List[float] = []
    ncols = (n - 1) * 676
    for toks in seqs:
        cols, data, _ = stacked_all_lag_sparse_row(toks, n)
        indices.extend(cols)
        values.extend(data)
        indptr.append(len(indices))
    return csr_matrix(
        (np.asarray(values, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(seqs), ncols),
    )


# --------------------------------------------------------------------------- #
# Classifiers -- settings copied from phase2_baseline_ladder.py               #
# --------------------------------------------------------------------------- #
def _fit_logreg(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"test_auc": float(roc_auc_score(yte, p)),
            "test_f1": float(f1_score(yte, (p >= 0.5).astype(int)))}


def _fit_xgb(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"test_auc": float(roc_auc_score(yte, p)),
            "test_f1": float(f1_score(yte, (p >= 0.5).astype(int)))}


# --------------------------------------------------------------------------- #
# Theory                                                                      #
# --------------------------------------------------------------------------- #
def family_determines_y(family: str) -> bool:
    """Does this family provably contain enough information to fix Y exactly?

    True only for the lag-agnostic pair families, which reconstruct the whole
    precedence matrix. A k-gram family at small m merely *permits* leakage -- the
    m-2 lemma stops guaranteeing chance -- which is a much weaker statement and
    must not be conflated with determining Y.
    """
    return family in ("max_over_lag", "sum_over_lag", "stacked_all_lag")


def theory_expects_chance(family: str, m: int, model: str) -> Tuple[bool, str]:
    """Prediction for (family, model).

    The m-2 lemma is a statement about a *single* feature: any one coordinate
    that depends on the relative order of at most m-2 key letters is marginally
    independent of Y. It does NOT say a classifier over the whole family is at
    chance, because coordinates can be informative jointly while each is
    uninformative alone.

    That distinction decides this table, and it is where an earlier version of
    this script was wrong. Because every key letter occurs exactly once,
    lag = pos(b) - pos(a) is unique for each ordered key pair, so

        max_lag G_ab^(lag) = sum_lag G_ab^(lag) = 1{a occurs before b}

    exactly (verified: 0 mismatches over 24000 ordered key pairs). The
    lag-agnostic families therefore hand a learner the complete C(m,2)
    precedence matrix, which determines Y. Only the *linear* models are blocked,
    since parity of the bit-sum is not a linear threshold function.
    """
    linear = model in ("logreg_L2", "logreg")

    if family == "unigram_counts":
        return True, ("key counts are constant at 1 and distractors are drawn "
                      "independently of pi -> exact independence")

    if family.startswith("kgram_k"):
        k = int(family.split("kgram_k")[1])
        touched = min(k, m)
        if touched <= m - 2:
            return True, (f"each k-gram touches at most {touched} key letters "
                          f"<= m-2 = {m - 2}")
        return False, (f"a k-gram can touch {touched} key letters > m-2 = {m - 2}, "
                       f"so leakage is permitted")

    if family.startswith("lag_pair_l"):
        # Marginally each coordinate is at chance, and a single lag cannot
        # determine Y. But the coordinates jointly reveal the precedence of
        # whichever key pairs happen to sit exactly that far apart, which is a
        # larger fraction of the C(m,2) bits when m is small -- so a nonlinear
        # model can pick up mild leakage at m=4. Kept as a chance prediction so
        # that any such leakage is surfaced rather than absorbed.
        return True, ("single lag: each coordinate marginally at chance, and one "
                      "lag cannot determine Y; joint partial precedence may leak "
                      "slightly at small m")

    if family in ("max_over_lag", "sum_over_lag"):
        if linear:
            return True, ("recovers all precedence bits, but parity of their sum "
                          "is not linearly separable")
        return False, (f"equals the full C({m},2) precedence matrix, which "
                       f"determines Y exactly")

    if family == "stacked_all_lag":
        if linear:
            return True, ("same information as max/sum over lag; still not "
                          "linearly separable")
        return False, ("precedence is recoverable but spread across per-lag "
                       "blocks, so only partially exploitable")

    return False, "no theoretical prediction"


# --------------------------------------------------------------------------- #
# Runner                                                                      #
# --------------------------------------------------------------------------- #
def run_family(family: str, m: int, data, seqs, labels, n: int,
               rows_train: int, rows_eval: int, rows_kgram: int,
               rows_stacked: int | None = None) -> List[Dict]:
    if family == "stacked_all_lag" and rows_stacked is not None:
        # (n-1)*676 = 12844 columns; phase2c_lag_agnostic.py caps this family at
        # 100K rows for the same reason.
        rows_train = min(rows_train, rows_stacked)
    rows: List[Dict] = []
    t0 = time.time()

    if family.startswith("kgram_k"):
        k = int(family.split("kgram_k")[1])
        scores, coverage = kgram_estimator_scores(
            seqs["train"][:rows_kgram], labels["train"][:rows_kgram],
            seqs["test"][:rows_eval], k,
        )
        yte = np.asarray(labels["test"][:rows_eval])
        auc = float(roc_auc_score(yte, scores))
        rows.append({
            "family": family, "m": m, "model": "kgram_estimator",
            "dim": f"26^{k} keys (sparse dict)", "n_train": min(rows_kgram, len(seqs["train"])),
            "test_auc": auc, "test_f1": float("nan"), "coverage": coverage,
        })
    else:
        if family == "unigram_counts":
            ex = feat_count26
        elif family.startswith("lag_pair_l"):
            lag = int(family.split("lag_pair_l")[1])
            ex = lambda t: feat_lag_pair(t, lag)  # noqa: E731
        elif family == "max_over_lag":
            ex = lambda t: feat_max_over_lag(t, n)  # noqa: E731
        elif family == "sum_over_lag":
            ex = lambda t: feat_sum_over_lag(t, n)  # noqa: E731
        elif family == "stacked_all_lag":
            ex = None
        else:
            raise ValueError(f"unknown family {family}")

        ytr = np.asarray(labels["train"][:rows_train])
        yte = np.asarray(labels["test"][:rows_eval])
        if family == "stacked_all_lag":
            Xtr = _sparse_matrix(seqs["train"][:rows_train], n)
            ytr = np.asarray(labels["train"][:Xtr.shape[0]])
            Xte = _sparse_matrix(seqs["test"][:rows_eval], n)
            dim = Xtr.shape[1]
        else:
            Xtr = np.stack([ex(t) for t in seqs["train"][:rows_train]], axis=0)
            Xte = np.stack([ex(t) for t in seqs["test"][:rows_eval]], axis=0)
            dim = Xtr.shape[1]

        for model_name, fit in (("logreg_L2", _fit_logreg), ("xgboost", _fit_xgb)):
            try:
                res = fit(Xtr, ytr, Xte, yte)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  %s/%s failed: %s", family, model_name, exc)
                continue
            rows.append({
                "family": family, "m": m, "model": model_name, "dim": dim,
                "n_train": Xtr.shape[0], "coverage": float("nan"), **res,
            })

    for r in rows:
        auc = r["test_auc"]
        expects_chance, reason = theory_expects_chance(family, m, r["model"])
        in_band = CHANCE_BAND[0] <= auc <= CHANCE_BAND[1]
        r["theory_expects_chance"] = expects_chance
        r["theory_reason"] = reason
        r["at_chance"] = bool(in_band)
        # Two distinct kinds of deviation, both worth reporting:
        #   leakage      theory says chance, measurement says otherwise
        #   unexploited  theory says the family determines Y, yet a learner with
        #                that family in hand still sits at chance -- i.e. the
        #                information is present but computationally out of reach
        r["outside_band"] = bool(expects_chance and not in_band)
        r["unexploited"] = bool(family_determines_y(family) and not expects_chance
                                and in_band)
        flag = ""
        if r["outside_band"]:
            flag = "  <-- LEAKAGE (theory said chance)"
        elif r["unexploited"]:
            flag = "  <-- INFORMATION PRESENT BUT UNEXPLOITED"
        logger.info(
            "  %-16s %-15s dim=%-9s auc=%.4f%s%s",
            r["family"], r["model"], str(r["dim"]), auc,
            "" if np.isnan(r["coverage"]) else f" cov={r['coverage']:.3f}",
            flag,
        )
    logger.info("  (%s took %.1fs)", family, time.time() - t0)
    return rows


def load_kip(m: int, data_dir: Path, rows_train: int, rows_eval: int):
    tag = f"kip_m{m}"
    data = {}
    for split, rows in (("train", rows_train), ("val", rows_eval), ("test", rows_eval)):
        data[split] = load_split(tag, split, data_dir=data_dir, rows=rows)
    seqs = {s: [tokens(x) for x in data[s][0]["Sequences"].tolist()] for s in data}
    labels = {s: [int(v) for v in data[s][1].iloc[:, 0].tolist()] for s in data}
    rule = json.loads((data_dir / f"{tag}_rule.json").read_text())
    return data, seqs, labels, rule


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIP audit ladder")
    p.add_argument("--m", type=int, nargs="+", default=[4, 6])
    p.add_argument("--families", nargs="+", default=None,
                   help="default: unigram_counts, kgram_k2/3/4, lag_pair_l*, "
                        "max_over_lag, sum_over_lag, stacked_all_lag")
    p.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS))
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--rows_train", type=int, default=200_000,
                   help="rows for classifier families")
    p.add_argument("--rows_kgram_train", type=int, default=400_000,
                   help="rows for the k-gram estimator (cheap, benefits from coverage)")
    p.add_argument("--rows_train_stacked", type=int, default=100_000,
                   help="separate, lower cap for the 12844-dim stacked_all_lag family")
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "kip_audit.csv")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    for k, v in sorted(vars(args).items()):
        logger.info("arg %s = %s", k, v)

    families = args.families or (
        ["unigram_counts"]
        + [f"kgram_k{k}" for k in args.ks]
        + [f"lag_pair_l{lag}" for lag in args.lags]
        + ["max_over_lag", "sum_over_lag", "stacked_all_lag"]
    )

    all_rows: List[Dict] = []
    for m in args.m:
        logger.info("=== KIP m=%d ===", m)
        rows_needed = max(args.rows_train, args.rows_kgram_train)
        data, seqs, labels, rule = load_kip(m, args.data_dir, rows_needed, args.rows_eval)
        logger.info("S (kappa order) = %s", ",".join(rule["key_letters_in_kappa_order"]))
        for fam in families:
            all_rows.extend(run_family(
                fam, m, data, seqs, labels, args.n,
                args.rows_train, args.rows_eval, args.rows_kgram_train,
                args.rows_train_stacked,
            ))

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))

    leaks = df[df["outside_band"]]
    if len(leaks):
        logger.warning("%d LEAKAGE result(s) -- theory says chance, measurement "
                       "disagrees:", len(leaks))
        for _, r in leaks.iterrows():
            logger.warning("  m=%d %s/%s auc=%.4f", r["m"], r["family"],
                           r["model"], r["test_auc"])
    else:
        logger.info("no leakage: every theory-chance family stayed inside %s",
                    CHANCE_BAND)

    unexp = df[df["unexploited"]]
    if len(unexp):
        logger.info("%d family/model pair(s) had enough information to determine Y "
                    "but stayed at chance (computational, not statistical, limit):",
                    len(unexp))
        for _, r in unexp.iterrows():
            logger.info("  m=%d %s/%s auc=%.4f", r["m"], r["family"],
                        r["model"], r["test_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
