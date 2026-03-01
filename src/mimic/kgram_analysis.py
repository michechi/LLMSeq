"""
K-gram baseline analysis on MIMIC CKD→ESRD cohort.
Adapted from simulation/stat_test_ICML.py.

For each k in 1..5: learns P(Y=1|g) for every k-gram g on the training set,
predicts each test sequence by averaging its k-gram probabilities,
and reports AUC, F1, Precision, Recall.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "ckd_cohort_ccs.csv"
OUTPUT = ROOT / "results" / "mimic" / "kgram_analysis.csv"

SEED = 42
MAX_K = 5

np.random.seed(SEED)


# ── utilities ─────────────────────────────────────────────────────────

def extract_kgrams(tokens, k):
    """Extract k-grams from a list of tokens."""
    return [
        "-".join(tokens[i:i + k])
        for i in range(len(tokens) - k + 1)
    ] if len(tokens) >= k else []


def kgram_predict(tokens, probs, k):
    """Predict P(Y=1) for a sequence by averaging its k-gram probabilities."""
    kgrams = extract_kgrams(tokens, k)
    if not kgrams:
        return 0.5
    return np.mean([probs.get(g, 0.5) for g in kgrams])


# ── main ──────────────────────────────────────────────────────────────

def main():
    # Load data
    df = pd.read_csv(COHORT)
    df["tokens"] = df["codes"].apply(json.loads)

    # Stratified 80/20 split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  (Y=1: {train_df['label'].sum()})")
    print(f"Test:  {len(test_df)}  (Y=1: {test_df['label'].sum()})")

    results = []

    for k in range(1, MAX_K + 1):
        # Learn P(Y=1|g) from training set
        stats = defaultdict(lambda: [0, 0])
        for tokens, label in zip(train_df["tokens"], train_df["label"]):
            for g in extract_kgrams(tokens, k):
                stats[g][int(label)] += 1

        probs = {
            g: c[1] / (c[0] + c[1]) if (c[0] + c[1]) > 0 else 0.5
            for g, c in stats.items()
        }

        # Predict on test set
        preds = [
            kgram_predict(tokens, probs, k)
            for tokens in test_df["tokens"]
        ]
        y_true = test_df["label"].values

        auc = roc_auc_score(y_true, preds)

        # Find best F1 threshold
        thresholds = np.linspace(0, 1, 101)
        f1s = [f1_score(y_true, (np.array(preds) >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1s))
        best_thr = thresholds[best_idx]
        preds_bin = (np.array(preds) >= best_thr).astype(int)

        row = {
            "k": k,
            "n_unique_kgrams": len(probs),
            "AUC": round(auc, 4),
            "F1": round(f1s[best_idx], 4),
            "Precision": round(precision_score(y_true, preds_bin, zero_division=0), 4),
            "Recall": round(recall_score(y_true, preds_bin, zero_division=0), 4),
            "Best_threshold": round(best_thr, 2),
        }
        results.append(row)
        print(f"k={k}  |  AUC={row['AUC']:.4f}  F1={row['F1']:.4f}  "
              f"P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
              f"thr={row['Best_threshold']:.2f}  "
              f"({row['n_unique_kgrams']:,} unique {k}-grams)")

    results_df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
