"""KIP (Key-Inversion Parity) -- synthetic task generator.

Each sequence contains every one of the m hidden key letters exactly once, at m
uniformly random positions; every other position is filled i.i.d. uniform from
the *distractor* alphabet A \\ S, so a key letter never appears more than once.
Reading the key letters in sequence order and mapping them through the hidden
ranking kappa yields a permutation pi of {1..m}, and

    Y = 1  iff  inv(pi) is even                (equivalently, iff pi is even)

Two properties make this task useful as a benchmark, and both are exact rather
than empirical:

*   Order-invariant features carry **zero** information. Every key letter has
    count exactly 1 in every sequence, and the distractors are drawn
    independently of pi, so the 26-dim letter-count vector is statistically
    independent of Y.
*   Any *single* feature that is a function of the relative order of at most
    m-2 key letters is **exactly** at chance. Condition on the relative order
    of such a subset T: at least two key letters lie outside T, and transposing
    their positions preserves both T's relative order and the generating
    distribution while flipping sign(pi).

Note carefully what the second property does and does not say. It is a
*marginal* statement about one coordinate, not about a classifier fitted to a
whole family of them. Because every key letter occurs exactly once, the lag
between an ordered key pair is unique, so

    max_lag G_ab^(lag)  =  sum_lag G_ab^(lag)  =  1{a occurs before b}

and a lag-agnostic pair-count family therefore reconstructs the entire C(m,2)
precedence matrix, which determines Y. Measured consequences (see
``src/analysis/mechanism_id/scripts/kip_audit.py``): *fixed*-lag pair counts --
the family that saturates the compliance tasks -- stay at chance for every lag,
and all linear models stay at chance everywhere, but XGBoost on max/sum-over-lag
reaches AUC 1.000 at m=4. At m=6 the same family is at 0.502, because the
implied 15-bit parity is beyond the learner even though the information is
present. m is thus a difficulty knob separating information-theoretic from
computational accessibility.

Unlike the compliance and parity generators, whose key set is a hardcoded
literal, KIP samples (S, kappa) once per dataset and persists them to
``<TAG>_rule.json``. Downstream audit and oracle scripts must *load* that file.
Hardcoding a key set would reintroduce the class of bug documented in
``src/analysis/mechanism_id/report.md``, where the paper states
S = {W,D,Q,J,X,U} but the parity generator actually used {W,D,Q,J,X,N}.

Sizes, splits and seed handling mirror Tricky Deterministic (tag ``6``): 500,000
sequences split 80/10/10 into 400K/50K/50K by two successive ``train_test_split``
calls with ``random_state=999``. The one deliberate difference is that no
class-balancing subsample is applied -- KIP has rho = 0.5 by construction,
because sign(pi) is uniform for m >= 2, so there is nothing to rebalance.

CLI::

    python -m src.generators.kip --verify
    python -m src.generators.kip --dry_run                 # 1K per tag, separate out dir
    python -m src.generators.kip --build                   # both m in {4, 6}
    python -m src.generators.kip --build --m 6
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import string
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import DATA_DIR, SEP, TESTED_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
ALPHABET: Tuple[str, ...] = tuple(string.ascii_uppercase)  # ell = 26
N_POSITIONS = 20  # n, matching every other task variant
M_VALUES: Tuple[int, ...] = (4, 6)

TOTAL_FULL = 500_000  # -> 400K train / 50K val / 50K test, as in tag "6"
TOTAL_DRY = 1_000

# Mirrors the module-level random.seed(959693) in test_simulation_det.py and
# properties_9.py. Each m gets a distinct but reproducible derived seed.
RULE_SEED = 959693
SPLIT_RANDOM_STATE = 999

SPLITS: Tuple[str, ...] = ("train", "val", "test")
DRY_RUN_DIR = DATA_DIR / "simulation" / "kip_dryrun"


def tag_for(m: int) -> str:
    return f"kip_m{m}"


# --------------------------------------------------------------------------- #
# Hidden rule                                                                 #
# --------------------------------------------------------------------------- #
def sample_rule(m: int, seed: int) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Sample (S, kappa): a uniform m-subset of A and a uniform bijection to 1..m.

    ``key_letters`` is returned in kappa order, so kappa[key_letters[i]] == i + 1.
    """
    if not 2 <= m <= len(ALPHABET):
        raise ValueError(f"m must be in [2, {len(ALPHABET)}], got {m}")
    rng = random.Random(seed)
    subset = rng.sample(ALPHABET, m)  # uniform m-subset
    rng.shuffle(subset)  # uniform bijection subset -> {1..m}
    key_letters = tuple(subset)
    kappa = {letter: rank for rank, letter in enumerate(key_letters, start=1)}
    return key_letters, kappa


def rule_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"{tag}_rule.json"


def save_rule(out_dir: Path, tag: str, m: int, key_letters: Sequence[str],
              kappa: Dict[str, int], seed: int) -> Path:
    path = rule_path(out_dir, tag)
    payload = {
        "tag": tag,
        "task": "KIP",
        "m": m,
        "n": N_POSITIONS,
        "alphabet": "".join(ALPHABET),
        "key_letters_in_kappa_order": list(key_letters),
        "kappa": kappa,
        "seed": seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_rule(data_dir: Path, tag: str) -> Dict:
    """Load the persisted (S, kappa). Never hardcode these downstream."""
    path = rule_path(data_dir, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"missing hidden-rule sidecar {path}; run "
            f"`python -m src.generators.kip --build` first"
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Label rule                                                                  #
# --------------------------------------------------------------------------- #
def inversions(ranks: Sequence[int]) -> int:
    """Number of pairs i < j with ranks[i] > ranks[j]. m <= 26 so O(m^2) is fine."""
    return sum(
        1
        for i in range(len(ranks))
        for j in range(i + 1, len(ranks))
        if ranks[i] > ranks[j]
    )


def label_from_ranks(ranks: Sequence[int]) -> int:
    """Y = 1 iff the number of inversions is even."""
    return int(inversions(ranks) % 2 == 0)


def key_ranks(tokens: Sequence[str], kappa: Dict[str, int]) -> List[int]:
    """kappa values of the key letters, in sequence order."""
    return [kappa[t] for t in tokens if t in kappa]


def label_sequence(tokens: Sequence[str], kappa: Dict[str, int]) -> int:
    """Recompute Y from a tokenised sequence. Used by --verify and the oracle."""
    return label_from_ranks(key_ranks(tokens, kappa))


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def generate_sequence(
    key_letters: Sequence[str],
    distractors: Sequence[str],
    kappa: Dict[str, int],
    n: int,
    rng: random.Random,
) -> Tuple[List[str], int]:
    """One sequence: every key letter exactly once, distractors never in S."""
    m = len(key_letters)
    positions = sorted(rng.sample(range(n), m))
    placed = list(key_letters)
    rng.shuffle(placed)  # uniform bijection S -> positions

    tokens = [rng.choice(distractors) for _ in range(n)]
    for pos, letter in zip(positions, placed):
        tokens[pos] = letter

    # placed is already in ascending position order, so this is pi directly.
    return tokens, label_from_ranks([kappa[letter] for letter in placed])


def generate_dataset(m: int, total: int, seed: int) -> Tuple[pd.DataFrame, Tuple[str, ...], Dict[str, int]]:
    key_letters, kappa = sample_rule(m, seed)
    key_set = set(key_letters)
    distractors = tuple(a for a in ALPHABET if a not in key_set)
    rng = random.Random(seed + 1)

    sequences: List[str] = []
    labels: List[float] = []
    for _ in range(total):
        tokens, y = generate_sequence(key_letters, distractors, kappa, N_POSITIONS, rng)
        sequences.append(SEP.join(tokens))
        labels.append(float(y))

    df = pd.DataFrame({"Sequences": sequences, "Outcome": labels})
    return df, key_letters, kappa


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def build(m: int, total: int, out_dir: Path, seed: int) -> Dict[str, int]:
    """Generate, split 80/10/10 exactly as tag "6" does, and write six CSVs."""
    tag = tag_for(m)
    derived_seed = seed + m
    logger.info("building %s: m=%d total=%d seed=%d", tag, m, total, derived_seed)

    df, key_letters, kappa = generate_dataset(m, total, derived_seed)
    logger.info("  S (in kappa order) = %s", ",".join(key_letters))
    logger.info("  label mean = %.6f", df["Outcome"].mean())

    # No sample_with_proportion: rho = 0.5 by construction.
    X, y = df["Sequences"], df["Outcome"]
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, train_size=0.80, random_state=SPLIT_RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, train_size=0.50, random_state=SPLIT_RANDOM_STATE
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [X_train, X_val, X_test, y_train, y_val, y_test]
    names = [f"X_{s}_{tag}" for s in SPLITS] + [f"y_{s}_{tag}" for s in SPLITS]
    for frame, name in zip(frames, names):
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    save_rule(out_dir, tag, m, key_letters, kappa, derived_seed)

    counts = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    logger.info("  wrote %s -> %s", counts, out_dir)
    return counts


# --------------------------------------------------------------------------- #
# Verify                                                                      #
# --------------------------------------------------------------------------- #
def _tokens(raw: str) -> List[str]:
    if SEP in raw:
        return [t for t in raw.split(SEP) if t]
    return list(raw)


def verify(m: int, data_dir: Path, n_check: int = 5_000) -> bool:
    """Re-derive labels from the stored CSVs and assert they match."""
    tag = tag_for(m)
    rule = load_rule(data_dir, tag)
    kappa = {k: int(v) for k, v in rule["kappa"].items()}
    key_set = set(kappa)
    ok = True

    for split in SPLITS:
        X = pd.read_csv(data_dir / f"X_{split}_{tag}.csv")
        y = pd.read_csv(data_dir / f"y_{split}_{tag}.csv")
        if len(X) != len(y):
            logger.error("  %s/%s: length mismatch X=%d y=%d", tag, split, len(X), len(y))
            ok = False
            continue

        head = min(n_check, len(X))
        seqs = X["Sequences"].tolist()[:head]
        labels = y["Outcome"].tolist()[:head]

        bad_label = bad_structure = 0
        for raw, stored in zip(seqs, labels):
            toks = _tokens(raw)
            if len(toks) != N_POSITIONS:
                bad_structure += 1
            elif sorted(t for t in toks if t in key_set) != sorted(key_set):
                # every key letter exactly once, and no extras
                bad_structure += 1
            if label_sequence(toks, kappa) != int(stored):
                bad_label += 1

        status = "OK" if (bad_label == 0 and bad_structure == 0) else "FAIL"
        logger.info(
            "  %-9s %-6s n=%-7d checked=%-6d label_mismatch=%d structure_bad=%d  [%s]",
            tag, split, len(X), head, bad_label, bad_structure, status,
        )
        ok = ok and bad_label == 0 and bad_structure == 0

    return ok


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the KIP task variant")
    p.add_argument("--build", action="store_true", help=f"build {TOTAL_FULL} sequences per m")
    p.add_argument("--dry_run", action="store_true", help=f"build only {TOTAL_DRY} per m")
    p.add_argument("--verify", action="store_true", help="re-derive labels from stored CSVs")
    p.add_argument("--m", type=int, nargs="+", default=list(M_VALUES), help="key-set sizes")
    p.add_argument("--seed", type=int, default=RULE_SEED)
    p.add_argument("--out_dir", type=Path, default=None,
                   help=f"default {TESTED_DIR} (or {DRY_RUN_DIR} with --dry_run)")
    p.add_argument("--n_check", type=int, default=5_000, help="rows per split for --verify")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    for k, v in sorted(vars(args).items()):
        logger.info("arg %s = %s", k, v)

    if not (args.build or args.dry_run or args.verify):
        logger.error("nothing to do: pass --build, --dry_run or --verify")
        return 2

    out_dir = args.out_dir or (DRY_RUN_DIR if args.dry_run else TESTED_DIR)

    if args.build or args.dry_run:
        total = TOTAL_DRY if args.dry_run else TOTAL_FULL
        for m in args.m:
            build(m, total, out_dir, args.seed)

    if args.verify:
        all_ok = True
        for m in args.m:
            all_ok = verify(m, out_dir, args.n_check) and all_ok
        if not all_ok:
            logger.error("VERIFY FAILED")
            return 1
        logger.info("VERIFY OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
