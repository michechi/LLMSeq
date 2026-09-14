"""KIP oracle + reveal ladder -- where exactly does the difficulty sit?

Mirrors the structure of phase5_parity.py, which decomposed the parity failure by
progressively revealing the hidden structure. The same question for KIP: the
label is the parity of inv(pi), so what has to be handed to a model before it can
compute it?

Because inv(pi) = C(m,2) - sum(precedence bits), the C(m,2) pairwise precedence
bits determine the label exactly. That gives a closed-form oracle needing no
training at all, and it makes the ladder interpretable:

    oracle_closed_form   Y computed directly from precedence bits -> AUC 1.0 exactly
    (a) raw_tokens       n x 26 one-hot; the model must first discover S and kappa
    (b) membership_bits  which positions hold key letters, without their identity
    (c) rank_sequence    kappa rank at each position, 0 at distractors
    (d) precedence_bits  C(m,2) bits
    (e) inversion_count  the scalar inv(pi)

Two predictions worth stating before looking at the numbers:

*   **(b) must be exactly at chance.** Those bits encode only the key *position
    set* P, and P is drawn independently of the letter-to-position bijection that
    determines pi. It is a built-in negative control: if (b) scores above chance,
    the generator leaks.
*   **LogReg on (d) and (e) must be at chance while the MLP succeeds.** The label
    is the parity of a sum taking m(m-1)/2 + 1 distinct values, which no linear
    threshold can express, but a 64-unit hidden layer can approximate as a
    staircase.

The hidden rule (S, kappa) is loaded from the persisted <TAG>_rule.json, never
hardcoded.

CLI::

    DATA_DIR=/path/to/data python -m src.analysis.mechanism_id.scripts.kip_oracle_reveal
    DATA_DIR=/path/to/data python -m src.analysis.mechanism_id.scripts.kip_oracle_reveal \
        --m 6 --rows_train 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DATA_DIR,
    RESULTS_DIR,
    feat_kip_inversion_count,
    feat_kip_precedence_bits,
    feat_kip_rank_sequence,
    feat_letter_onehot,
    feat_membership_bits,
    kip_oracle_label,
    load_split,
    tokens,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

SEED = 9550
N_DEFAULT = 20
CHANCE_BAND = (0.48, 0.52)

# What theory says each rung should reach, for the report's side-by-side column.
EXPECTED = {
    ("oracle_closed_form", "closed_form"): "1.0 exactly",
    ("a_raw_tokens", "logreg"): "chance",
    ("a_raw_tokens", "mlp_64"): "chance to partial (must discover S and kappa)",
    ("b_membership_bits", "logreg"): "chance (exact: P independent of pi)",
    ("b_membership_bits", "mlp_64"): "chance (exact: P independent of pi)",
    ("c_rank_sequence", "logreg"): "chance",
    ("c_rank_sequence", "mlp_64"): "high (pi is fully determined)",
    ("d_precedence_bits", "logreg"): "chance (parity is not linearly separable)",
    ("d_precedence_bits", "mlp_64"): "~1.0",
    ("e_inversion_count", "logreg"): "chance (parity of a scalar)",
    ("e_inversion_count", "mlp_64"): "~1.0",
}


def _eval(model, Xtr, ytr, Xv, yv, Xte, yte):
    """Identical harness to phase5_parity.py::_eval -- AUC on test, F1 threshold
    chosen on validation."""
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xte)[:, 1]
    else:
        pv = model.decision_function(Xv)
        pt = model.decision_function(Xte)
    auc = float(roc_auc_score(yte, pt)) if len(np.unique(yte)) >= 2 else float("nan")
    thrs = np.linspace(pt.min() - 1e-6, pt.max() + 1e-6, 201)
    if len(np.unique(yv)) >= 2:
        bt = thrs[int(np.argmax([f1_score(yv, pv >= t) for t in thrs]))]
        f1 = float(f1_score(yte, pt >= bt))
    else:
        f1 = float("nan")
    return auc, f1


def load_kip(m: int, data_dir: Path, rows_train: int, rows_eval: int):
    tag = f"kip_m{m}"
    data = {}
    for split, rows in (("train", rows_train), ("val", rows_eval), ("test", rows_eval)):
        data[split] = load_split(tag, split, data_dir=data_dir, rows=rows)
    toks = {s: [tokens(x) for x in data[s][0]["Sequences"].tolist()] for s in data}
    labels = {s: np.asarray([int(v) for v in data[s][1].iloc[:, 0].tolist()]) for s in data}
    rule = json.loads((data_dir / f"{tag}_rule.json").read_text())
    kappa = {k: int(v) for k, v in rule["kappa"].items()}
    return toks, labels, kappa, rule


def run_for_m(m: int, data_dir: Path, n: int, rows_train: int, rows_eval: int) -> List[Dict]:
    toks, labels, kappa, rule = load_kip(m, data_dir, rows_train, rows_eval)
    key_set = frozenset(kappa)
    logger.info("=== KIP m=%d  S(kappa order)=%s ===",
                m, ",".join(rule["key_letters_in_kappa_order"]))

    rows: List[Dict] = []

    # ---- closed-form oracle: no training ---------------------------------- #
    t0 = time.time()
    preds = np.asarray([kip_oracle_label(t, kappa) for t in toks["test"]], dtype=float)
    yte = labels["test"]
    auc = float(roc_auc_score(yte, preds))
    f1 = float(f1_score(yte, preds >= 0.5))
    agree = float(np.mean(preds == yte))
    rows.append({
        "m": m, "rung": "oracle_closed_form", "model": "closed_form",
        "feat_dim": m * (m - 1) // 2, "n_train": 0,
        "AUC": auc, "F1": f1, "agreement": agree, "train_sec": time.time() - t0,
    })
    logger.info("  oracle_closed_form            AUC=%.6f F1=%.6f agreement=%.6f",
                auc, f1, agree)
    if auc != 1.0:
        logger.error("  ORACLE IS NOT EXACT (AUC=%.6f) -- the label rule and the "
                     "precedence-bit derivation disagree", auc)

    # ---- reveal ladder ---------------------------------------------------- #
    variants = {
        "a_raw_tokens":      lambda t: feat_letter_onehot(t, n),
        "b_membership_bits": lambda t: feat_membership_bits(t, key_set, n),
        "c_rank_sequence":   lambda t: feat_kip_rank_sequence(t, kappa, n),
        "d_precedence_bits": lambda t: feat_kip_precedence_bits(t, kappa),
        "e_inversion_count": lambda t: feat_kip_inversion_count(t, kappa),
    }

    for vname, ex in variants.items():
        t0 = time.time()
        Xtr = np.stack([ex(t) for t in toks["train"]], 0)
        Xv = np.stack([ex(t) for t in toks["val"]], 0)
        Xte = np.stack([ex(t) for t in toks["test"]], 0)
        logger.info("  [%s] dim=%d build=%.1fs", vname, Xtr.shape[1], time.time() - t0)

        for mname, model in (
            ("logreg", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
            ("mlp_64", MLPClassifier(hidden_layer_sizes=(64,), random_state=SEED,
                                     max_iter=200, early_stopping=True)),
        ):
            t0 = time.time()
            try:
                auc, f1 = _eval(model, Xtr, labels["train"], Xv, labels["val"], Xte, yte)
            except Exception as exc:  # noqa: BLE001
                logger.warning("    %s failed: %s", mname, exc)
                continue
            dt = time.time() - t0
            rows.append({
                "m": m, "rung": vname, "model": mname,
                "feat_dim": int(Xtr.shape[1]), "n_train": int(Xtr.shape[0]),
                "AUC": auc, "F1": f1, "agreement": float("nan"), "train_sec": dt,
            })
            logger.info("    %-7s AUC=%.4f F1=%.4f (%.1fs)", mname, auc, f1, dt)

    for r in rows:
        r["expected"] = EXPECTED.get((r["rung"], r["model"]), "")
        r["at_chance"] = bool(CHANCE_BAND[0] <= r["AUC"] <= CHANCE_BAND[1])
    return rows


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIP oracle + reveal ladder")
    p.add_argument("--m", type=int, nargs="+", default=[4, 6])
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--rows_train", type=int, default=100_000,
                   help="capped below the full 400K because the MLP on the 520-dim "
                        "raw-token rung dominates runtime")
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "kip_oracle_reveal.csv")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    for k, v in sorted(vars(args).items()):
        logger.info("arg %s = %s", k, v)

    all_rows: List[Dict] = []
    for m in args.m:
        all_rows.extend(run_for_m(m, args.data_dir, args.n,
                                  args.rows_train, args.rows_eval))

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(df))

    # The negative control is the one result that would invalidate the dataset.
    bad = df[(df["rung"] == "b_membership_bits") & (~df["at_chance"])]
    if len(bad):
        logger.warning("NEGATIVE CONTROL FAILED: membership bits are above chance, "
                       "so key positions leak label information:")
        for _, r in bad.iterrows():
            logger.warning("  m=%d %s AUC=%.4f", r["m"], r["model"], r["AUC"])
    else:
        logger.info("negative control OK: membership bits at chance for all m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
