"""Build the audit shuffle variants of the 30Music test split (main-result data).

This deliberately differs from [25]'s own shuffle control (which permutes the
FULL history before truncation, changing which items the model sees). Here the
ordered visible window is fixed first, and permutation happens strictly WITHIN
that window, so every variant shows the model the same item multiset and the
same target — only the order changes.

Per test user (sequence = test rows stable-sorted by (user_id, timestamp)):

* target        = last event; never touched by any variant.
* inputs        = sequence minus target.
* visible window = last min(128, len(inputs)) input rows — exactly the slice
  a max_length=128 model consumes at inference.
* Variants permute the ITEM values across window rows (timestamps keep their
  slots, so time-sorting reproduces the permuted order):
    - full_shuffle: all window items permuted.
    - keep_last:    last window item (the one just before the target) fixed,
                    the rest permuted.
    - early_half:   first floor(W/2) window items permuted, second half intact.
* Rows outside the window (older than last-128) and the target row are copied
  unchanged.

Determinism: each user's permutation comes from
``random.Random(f"{seed}:{variant}:{user_id}")`` — depends only on
(shuffle seed, variant, user id), so it is stable across machines, runs, and
library versions. One shuffle per user per variant×seed, saved to disk; every
downstream consumer reads the SAME files.

CONSUMER CONTRACT (load-bearing): files are stable-sorted by
(user_id, timestamp); within equal timestamps the FILE ROW ORDER is the
sequence order. Consumers must group consecutive rows per user (or stable-sort
by (user_id, timestamp)) — 30Music has tied timestamps, and re-sorting by
timestamp alone with an unstable sort (as ref [25]'s LMDataset does) can move
tied rows across the window boundary for a small set of users, silently
breaking the same-multiset / fixed-last-item invariants. Do NOT feed these
files through [25]'s unmodified loader; the manifest records the number of
tie-affected users for transparency.

Outputs (in --out-dir):
* ``test_30Music_<variant>_s<seed>.csv``  x 9 — drop-in replacements for
  ``test_30Music.csv`` (same schema: index, user_id, item_id, timestamp).
* ``ordered_windows_30Music.csv``  — the ordered visible windows (canonical
  record of what the ordered condition sees).
* ``targets_30Music.csv``          — user_id,item_id of each test target.
* ``audit_manifest.json`` + ``audit_files.sha256`` — see build_manifest().

CLI::

    python -m src.recsys.audit_shuffles --test-csv .../split/test_30Music.csv \
        --raw-csv .../30M.csv --out-dir .../audit [--max-length 128]
        [--seeds 101 102 103]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd

VARIANTS = ("full_shuffle", "keep_last", "early_half")


def permuted_window_items(items: list, variant: str, rng: random.Random) -> list:
    """Return the window's item list under the given variant's permutation."""
    items = list(items)
    w = len(items)
    if variant == "full_shuffle":
        rng.shuffle(items)
    elif variant == "keep_last":
        head = items[:-1]
        rng.shuffle(head)
        items = head + items[-1:]
    elif variant == "early_half":
        half = w // 2
        head = items[:half]
        rng.shuffle(head)
        items = head + items[half:]
    else:
        raise ValueError(variant)
    return items


def build_variant(test: pd.DataFrame, variant: str, seed: int,
                  max_length: int) -> pd.DataFrame:
    """test must be stable-sorted by (user_id, timestamp) with a fresh 0..n index."""
    out_items = test["item_id"].to_numpy().copy()
    for user_id, idx in test.groupby("user_id", sort=True).indices.items():
        # idx is positional and contiguous per user after the stable sort
        input_idx = idx[:-1]                      # drop target row
        window_idx = input_idx[-max_length:]      # ordered visible window
        rng = random.Random(f"{seed}:{variant}:{user_id}")
        out_items[window_idx] = permuted_window_items(
            out_items[window_idx].tolist(), variant, rng)
    out = test.copy()
    out["item_id"] = out_items
    return out


def build_full_sequence_shuffle(df: pd.DataFrame, seed: int, role: str) -> pd.DataFrame:
    """Mode-(b) training-side shuffle: permute EVERY item of each user's sequence
    (no target/window concept on the training side). df must be stable-sorted
    by (user_id, timestamp) with a fresh 0..n index."""
    out_items = df["item_id"].to_numpy().copy()
    for user_id, idx in df.groupby("user_id", sort=True).indices.items():
        rng = random.Random(f"{seed}:{role}_full_shuffle:{user_id}")
        items = out_items[idx].tolist()
        rng.shuffle(items)
        out_items[idx] = items
    out = df.copy()
    out["item_id"] = out_items
    return out


def verify_written_variant(path: str, test: pd.DataFrame, window_positions: np.ndarray,
                           targets: pd.DataFrame) -> list:
    """Validate the SHIPPED bytes of a window-variant file against the ordered test
    frame: targets identical, rows outside the window byte-identical, and each
    user's window item multiset preserved."""
    problems = []
    df = pd.read_csv(path)[["user_id", "item_id", "timestamp"]].astype(np.int64)
    if not df[["user_id", "timestamp"]].equals(test[["user_id", "timestamp"]]):
        problems.append("user/timestamp columns differ from ordered test")
    if not targets_of(df).equals(targets):
        problems.append("targets differ")
    outside = np.ones(len(test), dtype=bool)
    outside[window_positions] = False
    if not (df["item_id"].to_numpy()[outside]
            == test["item_id"].to_numpy()[outside]).all():
        problems.append("rows outside the visible window were modified")
    w_users = test["user_id"].to_numpy()[window_positions]
    w_ours = pd.Series(df["item_id"].to_numpy()[window_positions]).groupby(w_users).apply(
        lambda s: tuple(sorted(s)))
    w_ref = pd.Series(test["item_id"].to_numpy()[window_positions]).groupby(w_users).apply(
        lambda s: tuple(sorted(s)))
    if not w_ours.equals(w_ref):
        problems.append("per-user window item multiset changed")
    return problems


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def targets_of(df: pd.DataFrame) -> pd.DataFrame:
    """Last event per user of a (user_id, timestamp) stable-sorted frame."""
    return (df.groupby("user_id", sort=True).tail(1)
            [["user_id", "item_id"]].reset_index(drop=True))


def tie_risk_stats(test: pd.DataFrame, max_length: int) -> dict:
    """Count users whose window invariants would be tie-order-sensitive if a
    consumer re-sorted by timestamp alone (contract violation): a tied
    timestamp straddling the window boundary, or a tie on the last window slot."""
    ts = test["timestamp"].to_numpy()
    boundary_risk = last_slot_risk = tied_users = 0
    for _, idx in test.groupby("user_id", sort=True).indices.items():
        input_idx = idx[:-1]
        window_idx = input_idx[-max_length:]
        user_ts = ts[idx]
        if len(np.unique(user_ts)) < len(user_ts):
            tied_users += 1
        if len(input_idx) > max_length and ts[window_idx[0]] == ts[input_idx[-max_length - 1]]:
            boundary_risk += 1
        if len(window_idx) >= 2 and ts[window_idx[-1]] == ts[window_idx[-2]]:
            last_slot_risk += 1
    return {"users_with_tied_timestamps": tied_users,
            "users_window_boundary_tie": boundary_risk,
            "users_last_slot_tie": last_slot_risk}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-csv", required=True,
                    help="raw 30M.csv, checksummed into the manifest")
    ap.add_argument("--prep-csv", required=True,
                    help="preprocessed CSV, checksummed into the manifest")
    ap.add_argument("--split-dir", required=True,
                    help="directory with train/validation/test CSVs; variants are "
                         "built from test_30Music.csv HERE (single source of truth)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--train-csv", help="also build mode-(b) full-sequence "
                    "shuffles of train (+ validation) at --train-shuffle-seed")
    ap.add_argument("--validation-csv")
    ap.add_argument("--train-shuffle-seed", type=int, default=101)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_csv = os.path.join(args.split_dir, "test_30Music.csv")
    test = pd.read_csv(test_csv)[["user_id", "item_id", "timestamp"]].astype(np.int64)
    test = test.sort_values(["user_id", "timestamp"],
                            kind="stable").reset_index(drop=True)

    targets = targets_of(test)
    targets_path = os.path.join(args.out_dir, "targets_30Music.csv")
    targets.to_csv(targets_path, index=False)

    # canonical ordered windows (what the ordered condition sees)
    window_rows = []
    for user_id, idx in test.groupby("user_id", sort=True).indices.items():
        window_rows.append(test.iloc[idx[:-1]][-args.max_length:])
    windows = pd.concat(window_rows, ignore_index=True)
    windows_path = os.path.join(args.out_dir, "ordered_windows_30Music.csv")
    windows.to_csv(windows_path, index=False)

    files = {"targets_30Music.csv": sha256_file(targets_path),
             "ordered_windows_30Music.csv": sha256_file(windows_path)}

    # positional index of every window row (used by written-file verification)
    window_positions = np.concatenate(
        [idx[:-1][-args.max_length:]
         for _, idx in test.groupby("user_id", sort=True).indices.items()])

    target_identity = True
    for variant in VARIANTS:
        for seed in args.seeds:
            variant_df = build_variant(test, variant, seed, args.max_length)
            name = f"test_30Music_{variant}_s{seed}.csv"
            path = os.path.join(args.out_dir, name)
            variant_df.to_csv(path)
            files[name] = sha256_file(path)

            problems = verify_written_variant(path, test, window_positions, targets)
            if problems:
                target_identity = False
                print(f"IDENTITY VIOLATION in {name}: {problems}")
            print(f"built + verified {name}", flush=True)

    if args.train_csv:
        for role, src in [("train", args.train_csv),
                          ("validation", args.validation_csv)]:
            if src is None:
                continue
            df = pd.read_csv(src)[["user_id", "item_id", "timestamp"]].astype(np.int64)
            df = df.sort_values(["user_id", "timestamp"],
                                kind="stable").reset_index(drop=True)
            shuf = build_full_sequence_shuffle(df, args.train_shuffle_seed, role)
            name = f"{role}_30Music_full_shuffle_s{args.train_shuffle_seed}.csv"
            path = os.path.join(args.out_dir, name)
            shuf.to_csv(path)
            files[name] = sha256_file(path)
            written = pd.read_csv(path)[["user_id", "item_id", "timestamp"]].astype(np.int64)
            multiset_ok = (written.groupby("user_id")["item_id"].apply(
                lambda s: tuple(sorted(s))).equals(
                df.groupby("user_id")["item_id"].apply(lambda s: tuple(sorted(s)))))
            slots_ok = written[["user_id", "timestamp"]].equals(df[["user_id", "timestamp"]])
            if not (multiset_ok and slots_ok):
                target_identity = False
                print(f"IDENTITY VIOLATION in {name}: "
                      f"multiset_ok={multiset_ok} slots_ok={slots_ok}")
            print(f"built + verified {name}", flush=True)

    for name in ["train_30Music.csv", "validation_30Music.csv", "test_30Music.csv"]:
        files[f"split/{name}"] = sha256_file(os.path.join(args.split_dir, name))

    try:
        proc = subprocess.run(["git", "describe", "--always", "--dirty"],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        git_rev = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() \
            else "unknown"
    except OSError:
        git_rev = "unknown"

    convert_meta_path = args.raw_csv + ".convert_meta.json"
    convert_meta = None
    if os.path.exists(convert_meta_path):
        with open(convert_meta_path) as fh:
            convert_meta = json.load(fh)

    manifest = {
        "dataset": "30Music",
        "source": "ThirtyMusic.tar.gz (Politecnico di Milano ReMAP lab SharePoint), "
                  "relations/events.idomaar -> 30M.csv via src.recsys.convert_30music",
        "raw_30M_csv_sha256": sha256_file(args.raw_csv),
        "raw_convert_meta": convert_meta,
        "prep_csv_sha256": sha256_file(args.prep_csv),
        "preprocessing": {
            "protocol": "Klenitskiy et al. 2024 (ref [25]): iterative 5-core + "
                        "consecutive-repeat removal, LabelEncoder ids, 90% global "
                        "temporal split, 500 validation users",
            "config": {"min_seq_len": 5, "min_item_count": 5, "drop_repeats": True,
                       "core": True, "encoding": True, "boundary_quantile": 0.9,
                       "validation_size": 500, "val_sampling_np_seed": 17},
            "executed_with": "their code (scratch clone, run_prepr_split.py driver); "
                             "independently re-implemented + verified IDENTICAL in "
                             "src.recsys.preprocess",
        },
        "consumer_contract": "sequence order = file row order per user (files are "
                             "stable-sorted by (user_id,timestamp)); do NOT re-sort "
                             "by timestamp alone / do NOT use [25]'s LMDataset loader",
        "tie_risk_stats": tie_risk_stats(test, args.max_length),
        "window_rule": f"per test user: drop target, window = last "
                       f"min({args.max_length}, len) inputs; permute WITHIN window",
        "variants": list(VARIANTS),
        "shuffle_seeds": args.seeds,
        "mode_b_train_shuffle": (
            {"seed": args.train_shuffle_seed,
             "rule": "full-sequence per-user shuffle of train (+validation), "
                     "rng random.Random(f'{seed}:{role}_full_shuffle:{user_id}')"}
            if args.train_csv else None),
        "rng": "random.Random(f'{seed}:{variant}:{user_id}')",
        "target_identity_check": "PASS" if target_identity else "FAIL",
        "code_git_rev": git_rev,
        "files_sha256": files,
    }
    manifest_path = os.path.join(args.out_dir, "audit_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    with open(os.path.join(args.out_dir, "audit_files.sha256"), "w") as fh:
        fh.write(f"# sha256sum -c compatible; paths relative to {args.out_dir}\n")
        for name, digest in sorted(files.items()):
            if not name.startswith("split/"):
                fh.write(f"{digest}  {name}\n")

    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"},
                     indent=2))
    print(f"target_identity_check: {'PASS' if target_identity else 'FAIL'}")
    assert target_identity, "target identity violated — do not ship these files"


if __name__ == "__main__":
    main()
