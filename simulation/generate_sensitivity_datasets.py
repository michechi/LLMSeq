"""
Generate synthetic datasets for parameter sensitivity analysis (Q1 response).

Produces tricky-deterministic datasets with varied (n, m, λ) configurations
in the same format as the existing experiment datasets.

Uses the SAME labeling logic as the original datasets (check_lag + assign_outcome_positional)
to ensure consistency.

Usage:
    # Generate all ablation configs
    python -m simulation.generate_sensitivity_datasets --all --output_dir data/simulation/sensitivity/

    # Generate a single config
    python -m simulation.generate_sensitivity_datasets --output_dir data/simulation/sensitivity/ \
        --seq_length 20 --key_size 6 --lag 7 --dataset_id 102

    # List all planned configurations
    python -m simulation.generate_sensitivity_datasets --list
"""

import random
import string
import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from simulation.do_assign_outcome_pos import assign_outcome_positional, efficient_check_parallel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALPHABET = list(string.ascii_uppercase)
SEP = "\x1f"
SEED = 42
TOTAL_SAMPLES = 400_000

# Ablation configurations: (dataset_id, n, m, λ, description)
# Baseline: n=20, m=6, λ=7
ABLATION_CONFIGS = [
    # Vary n (sequence length), m=6, λ=7
    (100, 10, 6, 7, "n=10 (short sequences)"),
    (101, 15, 6, 7, "n=15"),
    (102, 20, 6, 7, "n=20 (baseline)"),
    (103, 30, 6, 7, "n=30 (longer sequences)"),
    # 2D grid: m × λ (all with n=20)
    # m=3 row
    (130, 20, 3, 1, "m=3, lambda=1"),
    (131, 20, 3, 3, "m=3, lambda=3"),
    (132, 20, 3, 5, "m=3, lambda=5"),
    (110, 20, 3, 7, "m=3, lambda=7"),
    (133, 20, 3, 9, "m=3, lambda=9"),
    (134, 20, 3, 10, "m=3, lambda=10 (pairs only)"),
    # m=6 row
    (120, 20, 6, 1, "m=6, lambda=1"),
    (121, 20, 6, 3, "m=6, lambda=3"),
    (122, 20, 6, 5, "m=6, lambda=5"),
    # m=6, λ=7 is already 102
    (123, 20, 6, 9, "m=6, lambda=9"),
    (124, 20, 6, 10, "m=6, lambda=10 (pairs only)"),
    # m=10 row
    (140, 20, 10, 1, "m=10, lambda=1"),
    (141, 20, 10, 3, "m=10, lambda=3"),
    (142, 20, 10, 5, "m=10, lambda=5"),
    (112, 20, 10, 7, "m=10, lambda=7"),
    (143, 20, 10, 9, "m=10, lambda=9"),
    (144, 20, 10, 10, "m=10, lambda=10 (pairs only)"),
]


def generate_random_sequences(alphabet, seq_length, n_sequences, seed=SEED):
    """Generate n_sequences random letter sequences of given length (with replacement)."""
    rng = np.random.default_rng(seed)
    L = np.array(alphabet)
    indices = rng.integers(0, len(L), size=(n_sequences, seq_length))
    seqs = L[indices]
    # Join each row with \x1f separator (same format as existing datasets)
    return [SEP.join(row) for row in seqs]


# Fixed keys matching dataset 9 (nested: m=3 ⊂ m=6 ⊂ m=10)
FIXED_KEYS = {
    3:  ["W", "Q", "X"],
    6:  ["W", "D", "Q", "J", "X", "N"],
    10: ["W", "D", "Q", "J", "X", "N", "A", "P", "G", "Z"],
}


def generate_random_key(alphabet, key_size, seed=SEED):
    """Return the fixed key for this key_size, matching dataset 9."""
    if key_size in FIXED_KEYS:
        key_letters = FIXED_KEYS[key_size]
    else:
        # Fallback for non-standard key sizes
        rng = random.Random(seed)
        key_letters = rng.sample(alphabet, key_size)
    c_vocab = {letter: rank for rank, letter in enumerate(key_letters)}
    return key_letters, c_vocab


def generate_dataset(seq_length, key_size, lag, total_samples=TOTAL_SAMPLES,
                     seed=SEED, tolerance=False, rnd=False, pr_1=0.7):
    """
    Generate a dataset using the SAME labeling logic as the original experiments.

    Uses assign_outcome_positional() from do_assign_outcome_pos.py which calls
    check_lag() from do_check_lag.py — ensuring consistency with existing datasets.

    Args:
        seq_length: n, length of each sequence
        key_size: m, number of key letters
        lag: λ, required spacing between consecutive key letters
        total_samples: number of sequences to generate
        seed: random seed
        tolerance: allow one ordering violation (False for tricky-deterministic)
        rnd: stochastic labeling (False for deterministic, True for random with pr_1)
        pr_1: probability of label=1 when ordered (only used if rnd=True)
    """
    key_letters, c_vocab = generate_random_key(ALPHABET, key_size, seed)

    logger.info(f"Parameters: n={seq_length}, m={key_size}, λ={lag}, ℓ={len(ALPHABET)}")
    logger.info(f"Key: {' -> '.join(key_letters)}")
    logger.info(f"Tolerance: {tolerance}, Stochastic: {rnd}")

    max_chain = (seq_length - 1) // lag + 1 if lag > 0 else seq_length
    max_chain = min(max_chain, key_size)
    logger.info(f"Max possible chain length: {max_chain}")

    if max_chain < 2 and lag > 0:
        logger.warning(f"WARNING: max chain length < 2, no ordered sequences possible! "
                       f"(n={seq_length}, λ={lag})")

    # Generate random sequences
    logger.info(f"Generating {total_samples:,} random sequences...")
    sequences = generate_random_sequences(ALPHABET, seq_length, total_samples, seed)

    # Label using the existing pipeline (parallel for speed)
    logger.info("Labeling sequences using assign_outcome_positional (parallel)...")
    check, df_monitor = efficient_check_parallel(
        sequences=sequences,
        c_vocab=c_vocab,
        assign_outcome_positional=assign_outcome_positional,
        tolerance=tolerance,
        lags=lag,
        rnd=rnd,
        debugging=False,
        min_chain_length=2,
    )

    df = pd.DataFrame({
        'Sequences': df_monitor['seq'],
        'Outcome': df_monitor['outcome'].astype(int),
    })

    # Report class balance
    rho = df['Outcome'].mean()
    logger.info(f"Class balance: ρ = {rho:.4f} (ordered={df['Outcome'].sum():,}, "
                f"unordered={len(df) - df['Outcome'].sum():,})")

    return df, key_letters, c_vocab


def split_and_save(df, output_dir, dataset_id, seed=999):
    """Split dataset into train/val/test (80/10/10) and save as CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    X = df[['Sequences']]
    y = df[['Outcome']]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, train_size=0.80, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, train_size=0.50, random_state=seed, stratify=y_val_test
    )

    for data, name in [
        (X_train, f"X_train_{dataset_id}"),
        (X_val, f"X_val_{dataset_id}"),
        (X_test, f"X_test_{dataset_id}"),
        (y_train, f"y_train_{dataset_id}"),
        (y_val, f"y_val_{dataset_id}"),
        (y_test, f"y_test_{dataset_id}"),
    ]:
        path = os.path.join(output_dir, f"{name}.csv")
        data.to_csv(path, index=False)

    logger.info(f"Saved: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
                f"→ {output_dir}/*_{dataset_id}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default="data/simulation/sensitivity/")
    parser.add_argument("--total_samples", type=int, default=TOTAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tolerance", action="store_true",
                        help="Allow one ordering violation (default: False for deterministic)")
    parser.add_argument("--rnd", action="store_true",
                        help="Stochastic labeling (default: False for deterministic)")
    parser.add_argument("--pr_1", type=float, default=0.7,
                        help="P(label=1 | ordered) for stochastic labeling")

    # Single config mode
    parser.add_argument("--dataset_id", type=int, default=None,
                        help="Generate a single dataset with this ID")
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--key_size", type=int, default=6)
    parser.add_argument("--lag", type=int, default=7)

    # Utility
    parser.add_argument("--list", action="store_true",
                        help="List all planned ablation configurations and exit")
    parser.add_argument("--all", action="store_true",
                        help="Generate all ablation configurations")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only these IDs from the config table (comma-separated, e.g. '133,134,123')")

    args = parser.parse_args()

    if args.list:
        print("\nPlanned ablation configurations (deterministic):")
        print(f"{'ID':<6} {'n':<5} {'m':<5} {'λ':<5} {'Description'}")
        print("-" * 55)
        for did, n, m, lag, desc in ABLATION_CONFIGS:
            baseline = " ← baseline" if (n == 20 and m == 6 and lag == 7) else ""
            print(f"{did:<6} {n:<5} {m:<5} {lag:<5} {desc}{baseline}")

        print("\n2D grid (m × λ) for Q5 difficulty decomposition:")
        print(f"{'':>8} {'λ=1':<8} {'λ=3':<8} {'λ=5':<8} {'λ=7':<8}")
        print(f"{'m=3':>8} {'130':<8} {'131':<8} {'132':<8} {'110':<8}")
        print(f"{'m=6':>8} {'120':<8} {'121':<8} {'122':<8} {'102':<8}")
        print(f"{'m=10':>8} {'140':<8} {'141':<8} {'142':<8} {'112':<8}")
        return

    if args.all or args.only:
        # Filter configs if --only is specified
        if args.only:
            only_ids = set(int(x) for x in args.only.split(','))
            configs = [c for c in ABLATION_CONFIGS if c[0] in only_ids]
            if not configs:
                parser.error(f"No configs found for IDs: {only_ids}")
        else:
            configs = ABLATION_CONFIGS

        # Generate configurations
        for did, n, m, lag, desc in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Config {did}: {desc}")
            logger.info(f"{'='*60}")
            df, key, c_vocab = generate_dataset(
                n, m, lag, args.total_samples, args.seed,
                tolerance=args.tolerance, rnd=args.rnd, pr_1=args.pr_1
            )
            split_and_save(df, args.output_dir, did)

        # Save config metadata
        meta = pd.DataFrame(ABLATION_CONFIGS,
                            columns=['dataset_id', 'n', 'm', 'lag', 'description'])
        meta_path = os.path.join(args.output_dir, "sensitivity_configs.csv")
        meta.to_csv(meta_path, index=False)
        logger.info(f"\nConfig metadata saved to {meta_path}")
        return

    # Single config mode
    if args.dataset_id is None:
        parser.error("Specify --dataset_id for single config, or use --all or --list")

    df, key, c_vocab = generate_dataset(
        args.seq_length, args.key_size, args.lag,
        args.total_samples, args.seed,
        tolerance=args.tolerance, rnd=args.rnd, pr_1=args.pr_1
    )
    split_and_save(df, args.output_dir, args.dataset_id)


if __name__ == "__main__":
    main()
