"""Contract-enforcing loader for the 30Music audit eval files.

THE ONLY SUPPORTED WAY to read eval sequences in the H200 grid. It enforces,
in code, the consumer contract that the audit files rely on (file row order is
the sequence order; never re-sort by timestamp alone), and it re-verifies the
shuffle invariants on every load:

* structural contract: each user's rows are contiguous in the file and
  timestamps are non-decreasing within the user block;
* per-user visible-window ITEM MULTISET of a variant == the ordered window's
  multiset (from ``ordered_windows_30Music.csv``);
* per-user TARGET of a variant == the canonical target
  (``targets_30Music.csv``);
* variant-specific extras (inferred from the filename): ``keep_last`` keeps
  the ordered window's last item in the last slot; ``early_half`` keeps the
  ordered window's second half identical (order included).

Any violation raises :class:`AuditContractViolation` listing offending users —
the grid must fail loudly, not silently score corrupted inputs.

Harness API::

    from src.recsys.eval_loader import load_eval_input
    seqs = load_eval_input(variant_csv, audit_dir)   # verifies, then returns
    # seqs: dict user_id -> {"window": [item_id,...], "target": item_id}

CLI (H200 bootstrap step — run once after ``sha256sum -c``)::

    python -m src.recsys.eval_loader --audit-dir .../audit \
        --split-dir .../split --check-all
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

MAX_LENGTH = 128
VARIANT_RE = re.compile(r"test_30Music_(full_shuffle|keep_last|early_half)_s(\d+)")


class AuditContractViolation(AssertionError):
    """The shipped eval files and the loaded data disagree — do not proceed."""


def _read(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"user_id", "item_id", "timestamp"} - set(df.columns)
    if missing:
        raise AuditContractViolation(f"{csv_path}: missing columns {missing}")
    return df[["user_id", "item_id", "timestamp"]].astype(np.int64)


def _assert_structure(df: pd.DataFrame, csv_path: str) -> None:
    """File-order contract: contiguous user blocks, non-decreasing timestamps."""
    users = df["user_id"].to_numpy()
    block_starts = int((users[1:] != users[:-1]).sum()) + 1
    if block_starts != df["user_id"].nunique():
        raise AuditContractViolation(
            f"{csv_path}: user rows are not contiguous "
            f"({block_starts} blocks vs {df['user_id'].nunique()} users) — "
            "file was re-sorted or corrupted")
    ts = df["timestamp"].to_numpy()
    decreasing = (ts[1:] < ts[:-1]) & (users[1:] == users[:-1])
    if decreasing.any():
        bad = df["user_id"].to_numpy()[1:][decreasing][:5].tolist()
        raise AuditContractViolation(
            f"{csv_path}: timestamps decrease within user blocks "
            f"(first offending users: {bad}) — file order violated")


def _group_positions(df: pd.DataFrame) -> dict:
    """user -> positional row indices, in file order (structure pre-validated)."""
    return df.groupby("user_id", sort=False).indices


def load_sequences(csv_path: str, max_length: int = MAX_LENGTH) -> dict:
    """Read a test-shaped file (inputs + target row per user) in file order."""
    df = _read(csv_path)
    _assert_structure(df, csv_path)
    items = df["item_id"].to_numpy()
    out = {}
    for user, idx in _group_positions(df).items():
        if len(idx) < 2:
            raise AuditContractViolation(
                f"{csv_path}: user {user} has {len(idx)} row(s); "
                "expected inputs + target")
        input_idx = idx[:-1]
        out[user] = {"window": items[input_idx[-max_length:]].tolist(),
                     "target": int(items[idx[-1]])}
    return out


def load_reference(audit_dir: str) -> tuple[dict, dict]:
    """Canonical ordered windows and targets shipped beside the variants."""
    win_df = _read(os.path.join(audit_dir, "ordered_windows_30Music.csv"))
    _assert_structure(win_df, "ordered_windows_30Music.csv")
    witems = win_df["item_id"].to_numpy()
    windows = {u: witems[idx].tolist()
               for u, idx in _group_positions(win_df).items()}
    tgt_df = pd.read_csv(os.path.join(audit_dir, "targets_30Music.csv"))
    targets = dict(zip(tgt_df["user_id"].astype(np.int64),
                       tgt_df["item_id"].astype(np.int64)))
    return windows, targets


def verify_variant(seqs: dict, ref_windows: dict, ref_targets: dict,
                   name: str, variant: str | None) -> None:
    """Raise AuditContractViolation unless every invariant holds for every user."""
    if set(seqs) != set(ref_windows) or set(seqs) != set(ref_targets):
        raise AuditContractViolation(
            f"{name}: user set differs from reference "
            f"({len(seqs)} vs {len(ref_windows)} windows / {len(ref_targets)} targets)")

    bad_multiset, bad_target, bad_variant = [], [], []
    for user, rec in seqs.items():
        ref_win = ref_windows[user]
        if sorted(rec["window"]) != sorted(ref_win):
            bad_multiset.append(user)
        if rec["target"] != ref_targets[user]:
            bad_target.append(user)
        if variant == "keep_last" and rec["window"][-1] != ref_win[-1]:
            bad_variant.append(user)
        elif variant == "early_half":
            half = len(ref_win) // 2
            if rec["window"][half:] != ref_win[half:]:
                bad_variant.append(user)
        elif variant == "ordered" and rec["window"] != ref_win:
            bad_variant.append(user)

    problems = []
    if bad_multiset:
        problems.append(f"window multiset differs for {len(bad_multiset)} users "
                        f"(e.g. {bad_multiset[:5]})")
    if bad_target:
        problems.append(f"target differs for {len(bad_target)} users "
                        f"(e.g. {bad_target[:5]})")
    if bad_variant:
        problems.append(f"{variant} invariant broken for {len(bad_variant)} users "
                        f"(e.g. {bad_variant[:5]})")
    if problems:
        raise AuditContractViolation(f"{name}: " + "; ".join(problems))


def load_eval_input(variant_csv: str, audit_dir: str,
                    max_length: int = MAX_LENGTH) -> dict:
    """Verify-then-return: the function the H200 harness must call."""
    seqs = load_sequences(variant_csv, max_length)
    ref_windows, ref_targets = load_reference(audit_dir)
    base = os.path.basename(variant_csv)
    m = VARIANT_RE.search(base)
    variant = m.group(1) if m else ("ordered" if base == "test_30Music.csv" else None)
    verify_variant(seqs, ref_windows, ref_targets, base, variant)
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--split-dir", help="required for --check-all (ordered test file)")
    ap.add_argument("--file", help="verify a single eval input file")
    ap.add_argument("--check-all", action="store_true")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args()

    targets = [args.file] if args.file else []
    if args.check_all:
        if not args.split_dir:
            ap.error("--check-all needs --split-dir")
        targets.append(os.path.join(args.split_dir, "test_30Music.csv"))
        for variant in ["full_shuffle", "keep_last", "early_half"]:
            for seed in [101, 102, 103]:
                targets.append(os.path.join(
                    args.audit_dir, f"test_30Music_{variant}_s{seed}.csv"))

    failures = 0
    for path in targets:
        try:
            seqs = load_eval_input(path, args.audit_dir, args.max_length)
            print(f"OK   {os.path.basename(path)}  ({len(seqs)} users)", flush=True)
        except AuditContractViolation as exc:
            failures += 1
            print(f"FAIL {os.path.basename(path)}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{failures} file(s) violated the audit contract")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
