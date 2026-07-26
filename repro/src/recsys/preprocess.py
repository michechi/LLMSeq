"""Independent re-implementation of the preprocessing + split protocol of
Klenitskiy et al. 2024 (ref [25], "Does It Look Sequential?", RecSys '24).

Written from the protocol description, NOT copied from their (un-licensed)
repository. Behavioral equivalence is checked by ``--verify`` against the
outputs their code produced on the same raw CSV (see run log in the audit
manifest).

Protocol being reproduced
-------------------------
1. Iterative core filtering until fixed point, per-iteration op order:
   (a) drop users with < 5 events, (b) drop items with < 5 occurrences,
   (c) collapse consecutive repeats of the same item within a user's
   time-sorted sequence (i-i-j -> i-j). Loop re-checks (a)+(b) conditions
   after each pass.
2. Encode users and items to consecutive ids by sorted original id
   (sklearn LabelEncoder semantics, reproduced with numpy).
3. Global temporal split at the 90% quantile of all timestamps
   (pandas linear-interpolated quantile). Train users: users whose SECOND
   event is at or before the boundary; their events after the boundary are
   dropped. Test users: users whose LAST event is after the boundary; their
   FULL history is kept (train and test user sets overlap by design — this
   is their protocol). Validation: 500 users sampled uniformly without
   replacement from the train users (np.random.seed(VAL_SEED), recorded),
   moved out of train entirely.

CLI::

    python -m src.recsys.preprocess --raw .../30M.csv --out-dir .../own \
        [--val-seed 17]
    python -m src.recsys.preprocess --verify --out-dir .../own \
        --theirs-prep .../prep_30Music.csv --theirs-split-dir .../split
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

MIN_SEQ_LEN = 5
MIN_ITEM_COUNT = 5
BOUNDARY_QUANTILE = 0.9
VALIDATION_USERS = 500


def collapse_consecutive_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse runs of the same (user, item) in time-sorted order to one event.

    A run of length k is collapsed to its FIRST event, matching iterative
    removal of rows equal to their predecessor.
    """
    df = df.sort_values(["user_id", "timestamp"], kind="quicksort")
    same_as_prev = (df["user_id"].eq(df["user_id"].shift())
                    & df["item_id"].eq(df["item_id"].shift()))
    return df.loc[~same_as_prev]


def core_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Iterative 5-core + consecutive-repeat removal until fixed point."""
    iterations = 0
    while True:
        if df.empty:  # reference loop also exits here (NaN comparisons are False)
            break
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        if user_counts.min() >= MIN_SEQ_LEN and item_counts.min() >= MIN_ITEM_COUNT:
            break
        iterations += 1
        df = df[df["user_id"].isin(user_counts.index[user_counts >= MIN_SEQ_LEN])]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts.index[item_counts >= MIN_ITEM_COUNT])]
        df = collapse_consecutive_repeats(df)
    return df, iterations


def encode_sorted(series: pd.Series) -> pd.Series:
    """LabelEncoder semantics: consecutive codes assigned by sorted unique value."""
    uniques = np.sort(series.unique())
    mapping = pd.Series(np.arange(len(uniques)), index=uniques)
    return series.map(mapping)


def temporal_split(df: pd.DataFrame, val_seed: int):
    boundary = df["timestamp"].quantile(BOUNDARY_QUANTILE)

    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    # per-user SECOND event timestamp (GroupBy.nth is a row filter in modern
    # pandas, so extract it via cumcount)
    position = df.groupby("user_id").cumcount()
    second_event = (df.loc[position == 1, ["user_id", "timestamp"]]
                    .set_index("user_id")["timestamp"])
    last_event = df.groupby("user_id")["timestamp"].max()

    train_users = second_event.index[second_event <= boundary]
    test_users = last_event.index[last_event > boundary]

    train = df[df["user_id"].isin(train_users) & (df["timestamp"] <= boundary)]
    test = df[df["user_id"].isin(test_users)]

    np.random.seed(val_seed)
    val_users = np.random.choice(train["user_id"].unique(),
                                 size=VALIDATION_USERS, replace=False)
    validation = train[train["user_id"].isin(val_users)]
    train = train[~train["user_id"].isin(val_users)]

    return train, validation, test, boundary


def build(raw_csv: str, out_dir: str, val_seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    df = df.rename(columns={"user": "user_id", "item": "item_id"})
    raw_stats = summarize(df)

    df, iterations = core_filter(df)
    df = df.copy()
    df["item_id"] = encode_sorted(df["item_id"])
    df["user_id"] = encode_sorted(df["user_id"])
    prep_stats = summarize(df)

    train, validation, test, boundary = temporal_split(df, val_seed)

    df.to_csv(os.path.join(out_dir, "prep_30Music.csv"))
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part = part[["user_id", "item_id", "timestamp"]].astype(int)
        part.to_csv(os.path.join(out_dir, f"{name}_30Music.csv"))

    meta = {
        "raw": raw_stats,
        "preprocessed": prep_stats,
        "core_filter_iterations": iterations,
        "boundary_quantile": BOUNDARY_QUANTILE,
        "boundary_timestamp": float(boundary),
        "val_seed": val_seed,
        "validation_users": VALIDATION_USERS,
        "min_seq_len": MIN_SEQ_LEN,
        "min_item_count": MIN_ITEM_COUNT,
        "splits": {name: summarize(part) for name, part in
                   [("train", train), ("validation", validation), ("test", test)]},
    }
    with open(os.path.join(out_dir, "preprocess_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


def summarize(df: pd.DataFrame) -> dict:
    lens = df.groupby("user_id")["timestamp"].count()
    return {
        "interactions": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "mean_length": float(lens.mean()),
        "median_length": float(lens.median()),
    }


def load_canonical(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["user_id", "item_id", "timestamp"]].astype(np.int64)
    return df.sort_values(["user_id", "timestamp", "item_id"],
                          kind="stable").reset_index(drop=True)


def verify(out_dir: str, theirs_prep: str, theirs_split_dir: str) -> None:
    ok = True
    pairs = [("prep", os.path.join(out_dir, "prep_30Music.csv"), theirs_prep)]
    for name in ["train", "validation", "test"]:
        pairs.append((name, os.path.join(out_dir, f"{name}_30Music.csv"),
                      os.path.join(theirs_split_dir, f"{name}_30Music.csv")))
    for name, ours_path, theirs_path in pairs:
        ours, theirs = load_canonical(ours_path), load_canonical(theirs_path)
        identical = ours.equals(theirs)
        status = "IDENTICAL" if identical else "DIFF"
        print(f"{name}: ours {len(ours)} rows / theirs {len(theirs)} rows -> {status}")
        if not identical:
            ok = False
            merged = ours.merge(theirs, how="outer", indicator=True,
                                on=list(ours.columns))
            only_ours = (merged["_merge"] == "left_only").sum()
            only_theirs = (merged["_merge"] == "right_only").sum()
            print(f"  rows only in ours: {only_ours}, only in theirs: {only_theirs}")
            print(f"  ours stats: {summarize(ours)}")
            print(f"  theirs stats: {summarize(theirs)}")
    print("VERIFY:", "PASS (all splits identical)" if ok else "MISMATCH — see above")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-seed", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--theirs-prep")
    ap.add_argument("--theirs-split-dir")
    args = ap.parse_args()
    if args.verify:
        verify(args.out_dir, args.theirs_prep, args.theirs_split_dir)
    else:
        build(args.raw, args.out_dir, args.val_seed)


if __name__ == "__main__":
    main()
