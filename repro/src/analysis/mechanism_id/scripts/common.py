"""Shared utilities: data loading, label rules, feature extraction.

Single source of truth used by every mechanism-ID phase.
"""
from __future__ import annotations

import os
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEP = "\x1f"
ALPHABET = list(string.ascii_uppercase)

# From src/data/parity_variants.py and simulation/signal_decomposition.py:
KEY_LETTERS_IMPL: Tuple[str, ...] = ("W", "D", "Q", "J", "X", "N")
# κ(W)=0, κ(D)=1, κ(Q)=2, κ(J)=3, κ(X)=4, κ(N)=5
KAPPA_IMPL: Dict[str, int] = {k: p for p, k in enumerate(KEY_LETTERS_IMPL)}
KEY_SET_IMPL = frozenset(KEY_LETTERS_IMPL)

# As stated in the paper (Section 4.1) — note the discrepancy with code:
KEY_LETTERS_PAPER: Tuple[str, ...] = ("W", "D", "Q", "J", "X", "U")
KAPPA_PAPER: Dict[str, int] = {k: p for p, k in enumerate(KEY_LETTERS_PAPER)}

REPO_ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parents[4]))
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "tested"
PARITY_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "parity_decomp"
ANALYSIS_ROOT = REPO_ROOT / "src" / "analysis" / "mechanism_id"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ANALYSIS_ROOT / "results"))
PLOTS_DIR = ANALYSIS_ROOT / "plots"

for d in (RESULTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def tokens(seq: str) -> List[str]:
    if SEP in seq:
        return [t for t in seq.split(SEP) if t]
    return list(seq)


def load_split(
    tag: str,
    split: str,
    data_dir: Path = DATA_DIR,
    rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(data_dir / f"X_{split}_{tag}.csv").fillna("")
    y = pd.read_csv(data_dir / f"y_{split}_{tag}.csv").fillna(0)
    if rows is not None:
        X = X.iloc[:rows].reset_index(drop=True)
        y = y.iloc[:rows].reset_index(drop=True)
    return X, y


def load_tricky_det(**kwargs) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    return {split: load_split("6", split, **kwargs) for split in ("train", "val", "test")}


def load_tricky_rnd(**kwargs) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    return {split: load_split("9", split, **kwargs) for split in ("train", "val", "test")}


def load_parity(**kwargs) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    return {split: load_split("test_just_pair", split, **kwargs)
            for split in ("train", "val", "test")}


# --------------------------------------------------------------------------
# Candidate label rules (all operate on tokenised letter lists).
# Useful for verifying Q1: which rule matches the stored labels?
# --------------------------------------------------------------------------
def rule_pair_any_lag(
    toks: Sequence[str], lag: int, key_set: frozenset, kappa: Dict[str, int],
) -> int:
    """Paper k=2: exists t with toks[t], toks[t+lag] in S and κ(toks[t]) <= κ(toks[t+lag])."""
    n = len(toks)
    for t in range(n - lag):
        a, b = toks[t], toks[t + lag]
        if a in key_set and b in key_set and kappa[a] <= kappa[b]:
            return 1
    return 0


def rule_pair_strict(
    toks: Sequence[str], lag: int, key_set: frozenset, kappa: Dict[str, int],
) -> int:
    """Exists t with toks[t], toks[t+lag] in S and κ(toks[t]) < κ(toks[t+lag]) (strict)."""
    n = len(toks)
    for t in range(n - lag):
        a, b = toks[t], toks[t + lag]
        if a in key_set and b in key_set and kappa[a] < kappa[b]:
            return 1
    return 0


def rule_paper_exists_monotone_chain(
    toks: Sequence[str], lag: int, key_set: frozenset, kappa: Dict[str, int],
    min_len: int = 2,
) -> int:
    """Paper Def.1 exactly: exists chain of length>=min_len at exact lag spacing,
    all tokens in S, κ non-decreasing along the chain."""
    n = len(toks)
    for t_start in range(n - (min_len - 1) * lag):
        pos = t_start
        chain_tokens = [toks[pos]]
        # Greedy extend by lag as long as still in S AND non-decreasing
        while True:
            next_pos = pos + lag
            if next_pos >= n:
                break
            a, b = chain_tokens[-1], toks[next_pos]
            if a in key_set and b in key_set and kappa[a] <= kappa[b]:
                chain_tokens.append(b)
                pos = next_pos
            else:
                break
        if len(chain_tokens) >= min_len and all(x in key_set for x in chain_tokens):
            return 1
    # Also search for sub-chains: need to try *all* subsets of positions at
    # fixed lag; easier to enumerate all (start, len) pairs and check.
    for t_start in range(n):
        for L in range(min_len, n // lag + 2):
            if t_start + (L - 1) * lag >= n:
                break
            positions = [t_start + j * lag for j in range(L)]
            letters = [toks[p] for p in positions]
            if all(x in key_set for x in letters):
                if all(kappa[letters[j]] <= kappa[letters[j + 1]] for j in range(L - 1)):
                    return 1
    return 0


def rule_greedy_monotone_impl(
    toks: Sequence[str], lag: int, key_set: frozenset, kappa: Dict[str, int],
    tolerance: bool = False, min_chain_length: int = 2,
) -> int:
    """Reproduce simulation/do_check_lag.check_lag + is_ordered_chain logic
    with a given tolerance and min_chain_length. Extracts greedy non-
    overlapping chains of key letters at exact-lag spacing, then checks
    each chain for nondecreasing κ (with optional one-violation tolerance)."""
    n = len(toks)
    key_positions = [(i, t) for i, t in enumerate(toks) if t in key_set]
    if not key_positions:
        return 0

    chains: List[List[Tuple[int, str]]] = []
    used: set = set()

    for start_pos, start_tok in key_positions:
        if start_pos in used:
            continue
        chain: List[Tuple[int, str]] = [(start_pos, start_tok)]
        cur = start_pos
        while True:
            nxt = cur + lag
            if nxt >= n:
                break
            found = None
            for pos, t in key_positions:
                if pos == nxt:
                    found = (pos, t)
                    break
            if found is None:
                break
            chain.append(found)
            cur = nxt

        if len(chain) < 2:
            continue

        chain_pos = [p for p, _ in chain]
        is_subset = False
        for i, existing in enumerate(chains):
            ep = [p for p, _ in existing]
            if set(chain_pos).issubset(set(ep)):
                is_subset = True
                break
            if set(ep).issubset(set(chain_pos)):
                chains[i] = chain
                used.update(chain_pos)
                is_subset = True
                break
        if not is_subset:
            chains.append(chain)
            used.update(chain_pos)

    # Now check each chain for nondecreasing κ
    for chain in chains:
        if len(chain) < min_chain_length:
            continue
        tol = tolerance
        ok = True
        toks_chain = [t for _, t in chain]
        for a, b in zip(toks_chain[:-1], toks_chain[1:]):
            if kappa[a] > kappa[b]:
                if tol:
                    tol = False
                else:
                    ok = False
                    break
        if ok:
            return 1
    return 0


def rule_parity_total_count(
    toks: Sequence[str], key_set: frozenset,
) -> int:
    """Implemented rule in src/data/parity_variants.py: Y=1 iff total
    count of key letters is even."""
    c = sum(1 for t in toks if t in key_set)
    return int(c % 2 == 0)


def rule_parity_paper(
    toks: Sequence[str], key_set: frozenset,
) -> int:
    """Paper Def.4: Y=1 iff an even number of key letters have even count."""
    counts = Counter(t for t in toks if t in key_set)
    even_count_keys = sum(1 for k in key_set if counts.get(k, 0) % 2 == 0)
    return int(even_count_keys % 2 == 0)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def feat_count26(toks: Sequence[str]) -> np.ndarray:
    """A.1 — 26-dim full letter counts."""
    c = Counter(toks)
    return np.array([c.get(l, 0) for l in ALPHABET], dtype=np.float32)


def feat_count_key(toks: Sequence[str], keys: Sequence[str]) -> np.ndarray:
    c = Counter(toks)
    return np.array([c.get(k, 0) for k in keys], dtype=np.float32)


def feat_residue(toks: Sequence[str], lag: int) -> np.ndarray:
    """B — residue-class counts: 26 * lag features."""
    feats = np.zeros((26, lag), dtype=np.float32)
    for i, t in enumerate(toks):
        if len(t) != 1:
            continue
        a = ord(t) - ord("A")
        if 0 <= a < 26:
            feats[a, i % lag] += 1
    return feats.reshape(-1)


def feat_lag_pair(toks: Sequence[str], lag: int) -> np.ndarray:
    """C.1 — aggregated lag-pair counts g_ab^(λ)."""
    n = len(toks)
    feats = np.zeros((26, 26), dtype=np.float32)
    for t in range(n - lag):
        a, b = toks[t], toks[t + lag]
        if len(a) == 1 and len(b) == 1:
            ai = ord(a) - ord("A")
            bi = ord(b) - ord("A")
            if 0 <= ai < 26 and 0 <= bi < 26:
                feats[ai, bi] += 1
    return feats.reshape(-1)


def feat_lag_pair_position(toks: Sequence[str], lag: int, n: int) -> np.ndarray:
    """C.2 — position-aware lag-pair indicators, dim = (n-lag) * 26 * 26."""
    seq_len = len(toks)
    nstart = n - lag
    feats = np.zeros((nstart, 26, 26), dtype=np.float32)
    for t in range(min(nstart, seq_len - lag)):
        a, b = toks[t], toks[t + lag]
        if len(a) == 1 and len(b) == 1:
            ai = ord(a) - ord("A")
            bi = ord(b) - ord("A")
            if 0 <= ai < 26 and 0 <= bi < 26:
                feats[t, ai, bi] = 1.0
    return feats.reshape(-1)


def feat_lag_pair_position_sparse_row(toks: Sequence[str], lag: int, n: int):
    """C.2 sparse: returns (cols, data, n_features) for a single sequence,
    used to assemble a CSR matrix without the dense-matrix blow-up."""
    seq_len = len(toks)
    nstart = n - lag
    cols = []
    data = []
    for t in range(min(nstart, seq_len - lag)):
        a, b = toks[t], toks[t + lag]
        if len(a) == 1 and len(b) == 1:
            ai = ord(a) - ord("A")
            bi = ord(b) - ord("A")
            if 0 <= ai < 26 and 0 <= bi < 26:
                col = t * 26 * 26 + ai * 26 + bi
                cols.append(col)
                data.append(1.0)
    return cols, data, nstart * 26 * 26


def stack_sparse_bundle(
    data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    row_fn,
):
    """Build a CSR matrix per split using `row_fn` that returns (cols, data, D)."""
    from scipy.sparse import csr_matrix

    def _mat(split):
        seqs = data[split][0]["Sequences"].tolist()
        indptr = [0]
        indices = []
        values = []
        ncols = 0
        for s in seqs:
            cols, d, D = row_fn(tokens(s))
            if D > ncols:
                ncols = D
            indices.extend(cols)
            values.extend(d)
            indptr.append(len(indices))
        return csr_matrix(
            (np.asarray(values, dtype=np.float32),
             np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int32)),
            shape=(len(seqs), ncols),
        )

    return {split: _mat(split) for split in ("train", "val", "test")}


def feat_lag_pair_key_only(toks: Sequence[str], lag: int, keys: Sequence[str]) -> np.ndarray:
    """C — aggregated lag-pair counts restricted to key letters (6x6)."""
    key_idx = {k: i for i, k in enumerate(keys)}
    n = len(toks)
    m = len(keys)
    feats = np.zeros((m, m), dtype=np.float32)
    for t in range(n - lag):
        a, b = toks[t], toks[t + lag]
        if a in key_idx and b in key_idx:
            feats[key_idx[a], key_idx[b]] += 1
    return feats.reshape(-1)


def feat_lag_trigram(toks: Sequence[str], lag: int) -> np.ndarray:
    """D.1 — aggregated lag-trigram counts. 26^3 dim — large."""
    n = len(toks)
    feats = np.zeros(26 ** 3, dtype=np.float32)
    for t in range(n - 2 * lag):
        a, b, c = toks[t], toks[t + lag], toks[t + 2 * lag]
        if len(a) == 1 and len(b) == 1 and len(c) == 1:
            ai = ord(a) - ord("A")
            bi = ord(b) - ord("A")
            ci = ord(c) - ord("A")
            if 0 <= ai < 26 and 0 <= bi < 26 and 0 <= ci < 26:
                feats[ai * 26 * 26 + bi * 26 + ci] += 1
    return feats


def feat_lag_trigram_key_only(
    toks: Sequence[str], lag: int, keys: Sequence[str],
) -> np.ndarray:
    """D.1' — aggregated lag-trigram counts, restricted to key letters (6^3)."""
    key_idx = {k: i for i, k in enumerate(keys)}
    m = len(keys)
    n = len(toks)
    feats = np.zeros((m, m, m), dtype=np.float32)
    for t in range(n - 2 * lag):
        a, b, c = toks[t], toks[t + lag], toks[t + 2 * lag]
        if a in key_idx and b in key_idx and c in key_idx:
            feats[key_idx[a], key_idx[b], key_idx[c]] += 1
    return feats.reshape(-1)


@dataclass
class FeatureBundle:
    name: str
    Xtr: np.ndarray
    Xv: np.ndarray
    Xte: np.ndarray


def bundle_features(
    data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    extractor,
    name: str,
) -> FeatureBundle:
    def _mat(split):
        seqs = data[split][0]["Sequences"].tolist()
        return np.stack([extractor(tokens(s)) for s in seqs], axis=0)

    return FeatureBundle(
        name=name,
        Xtr=_mat("train"),
        Xv=_mat("val"),
        Xte=_mat("test"),
    )


# --------------------------------------------------------------------------
# Contiguous k-gram baseline (the estimator the paper cites, Section 4.2:
# P(Y=1|g) = #(g, Y=1) / #(g), averaged over the k-grams of a sequence).
#
# Ported from simulation/stat_test_ICML.py:48-80 so the repro tree does not
# depend on simulation/. Contiguous k-grams are deliberately absent from
# phase2_baseline_ladder.py, whose families are all count- or lag-based; for
# k >= 3 a dense count matrix is infeasible (26^3 = 17576, 26^4 = 456976
# columns), which is why an estimator rather than a feature matrix is used.
# --------------------------------------------------------------------------
def extract_ngrams(toks: Sequence[str], k: int) -> List[str]:
    """Contiguous k-grams of a token list, joined with '-' as in the original."""
    if len(toks) < k:
        return []
    return ["-".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]


def kgram_estimator_scores(
    train_seqs: Sequence[Sequence[str]],
    train_labels: Sequence[int],
    eval_seqs: Sequence[Sequence[str]],
    k: int,
    default: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Fit P(Y=1|g) on train, score eval sequences by averaging over their k-grams.

    Returns (scores, coverage) where coverage is the fraction of k-gram
    occurrences in eval that were seen during training -- worth reporting,
    because a low-coverage k-gram model scores mostly `default` and its AUC is
    then uninformative rather than genuinely at chance.
    """
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for toks, label in zip(train_seqs, train_labels):
        li = int(label)
        for g in extract_ngrams(toks, k):
            counts[g][li] += 1

    probs = {g: (c[1] / (c[0] + c[1]) if (c[0] + c[1]) > 0 else default)
             for g, c in counts.items()}

    scores = np.empty(len(eval_seqs), dtype=np.float64)
    seen = total = 0
    for r, toks in enumerate(eval_seqs):
        grams = extract_ngrams(toks, k)
        if not grams:
            scores[r] = default
            continue
        vals = []
        for g in grams:
            total += 1
            p = probs.get(g)
            if p is None:
                vals.append(default)
            else:
                seen += 1
                vals.append(p)
        scores[r] = float(np.mean(vals))

    coverage = (seen / total) if total else 0.0
    return scores, coverage


# --------------------------------------------------------------------------
# KIP (Key-Inversion Parity) feature families.
#
# The label is the parity of inv(pi), where pi is the permutation obtained by
# reading the key letters in sequence order and mapping through kappa. Because
# inv(pi) = C(m,2) - sum(precedence bits), the precedence bits determine the
# label exactly, which is what the reveal ladder exploits.
#
# All of these take the hidden rule explicitly. Never hardcode a KIP key set:
# it is sampled per dataset and persisted to <TAG>_rule.json.
# --------------------------------------------------------------------------
def kip_key_ranks(toks: Sequence[str], kappa: Dict[str, int]) -> List[int]:
    """kappa values of the key letters, in sequence order."""
    return [kappa[t] for t in toks if t in kappa]


def kip_precedence_pairs(m: int) -> List[Tuple[int, int]]:
    """Canonical pair order: (r_a, r_b) over kappa ranks with r_a < r_b."""
    return [(a, b) for a in range(1, m + 1) for b in range(a + 1, m + 1)]


def feat_kip_precedence_bits(toks: Sequence[str], kappa: Dict[str, int]) -> np.ndarray:
    """C(m,2) bits: 1 iff the letter of rank r_a occurs before that of rank r_b."""
    m = len(kappa)
    pos = {kappa[t]: i for i, t in enumerate(toks) if t in kappa}
    pairs = kip_precedence_pairs(m)
    out = np.zeros(len(pairs), dtype=np.float32)
    for idx, (ra, rb) in enumerate(pairs):
        pa, pb = pos.get(ra), pos.get(rb)
        if pa is not None and pb is not None and pa < pb:
            out[idx] = 1.0
    return out


def feat_kip_rank_sequence(toks: Sequence[str], kappa: Dict[str, int], n: int) -> np.ndarray:
    """Length-n vector: kappa rank at each position, 0 at distractor positions."""
    out = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(toks[:n]):
        r = kappa.get(t)
        if r is not None:
            out[i] = float(r)
    return out


def feat_kip_inversion_count(toks: Sequence[str], kappa: Dict[str, int]) -> np.ndarray:
    """Scalar: the number of inversions of pi."""
    ranks = kip_key_ranks(toks, kappa)
    inv = sum(
        1
        for i in range(len(ranks))
        for j in range(i + 1, len(ranks))
        if ranks[i] > ranks[j]
    )
    return np.array([float(inv)], dtype=np.float32)


def feat_membership_bits(toks: Sequence[str], key_set, n: int) -> np.ndarray:
    """Length-n 0/1 indicator of key-letter positions (no letter identity)."""
    out = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(toks[:n]):
        if t in key_set:
            out[i] = 1.0
    return out


def feat_letter_onehot(toks: Sequence[str], n: int) -> np.ndarray:
    """Flattened n x 26 one-hot encoding of the raw token sequence."""
    out = np.zeros((n, 26), dtype=np.float32)
    for i, t in enumerate(toks[:n]):
        if len(t) == 1:
            a = ord(t) - ord("A")
            if 0 <= a < 26:
                out[i, a] = 1.0
    return out.reshape(-1)


def kip_oracle_label(toks: Sequence[str], kappa: Dict[str, int]) -> int:
    """Closed-form label from the precedence bits: Y = 1 iff inv(pi) is even.

    inv(pi) = C(m,2) - sum(precedence bits), so this is exactly the parity of
    that difference -- no training involved.
    """
    m = len(kappa)
    bits = feat_kip_precedence_bits(toks, kappa)
    inv = int(round(m * (m - 1) / 2 - float(bits.sum())))
    return int(inv % 2 == 0)
