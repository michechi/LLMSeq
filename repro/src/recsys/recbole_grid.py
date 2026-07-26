"""Shared plumbing for the 30Music audit grid built on RecBole model classes.

Models come from recbole.model.sequential_recommender (recbole==1.2.1, MIT,
PINNED — the mock config below matches that version's constructor contract).
We use RecBole's models and objectives (CE over the full catalog; BERT4Rec's
cloze masking replicated from recbole.data.transform.MaskItemSequence) with a
thin training/eval loop of our own so that data ingestion obeys the audit
consumer contract (src.recsys.eval_loader) instead of RecBole's re-sorting
dataset pipeline. Scale-matched hyperparameters per HANDOFF.md.

Item-id convention: raw ids from [25]'s LabelEncoder are 0-based; RecBole
reserves token 0 for padding, so ALL ids are shifted +1 on the way into a
model and -1 on the way out. `n_items` passed to models = raw vocab + 1 (pad).
BERT4Rec additionally appends its own internal mask token (= n_items).
"""

from __future__ import annotations

import random

import numpy as np
import torch

MAX_LEN = 128
PAD = 0

MODEL_CONFIGS = {
    # scale-matched to ref [25] (HANDOFF.md grid table)
    "SASRec": dict(n_layers=2, n_heads=2, hidden_size=64, inner_size=64,
                   hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
                   hidden_act="gelu", layer_norm_eps=1e-12,
                   initializer_range=0.02, loss_type="CE"),
    "GRU4Rec": dict(embedding_size=64, hidden_size=64, num_layers=1,
                    dropout_prob=0.1, loss_type="CE"),
    "BERT4Rec": dict(n_layers=2, n_heads=2, hidden_size=64, inner_size=64,
                     hidden_dropout_prob=0.1, attn_dropout_prob=0.1,
                     hidden_act="gelu", layer_norm_eps=1e-12,
                     initializer_range=0.02, mask_ratio=0.2, ft_ratio=0.5,
                     loss_type="CE"),
}

# field names used by the mock config (SequentialRecommender contract)
F_ITEM = "item_id"
F_SEQ = "item_id_list"
F_LEN = "item_length"
F_MASK_SEQ = "Mask_item_id_list"
F_POS = "Pos_item_id"
F_NEG = "Neg_item_id"
F_MASK_IDX = "MASK_INDEX"


class MockConfig(dict):
    """Just enough of recbole.config.Config for model construction."""

    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        self.update({
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": F_ITEM,
            "LIST_SUFFIX": "_list",
            "ITEM_LIST_LENGTH_FIELD": F_LEN,
            "NEG_PREFIX": "Neg_",
            "MAX_ITEM_LIST_LENGTH": MAX_LEN,
            "device": device,
            "MASK_ITEM_SEQ": F_MASK_SEQ,
            "POS_ITEMS": F_POS,
            "NEG_ITEMS": F_NEG,
            "MASK_INDEX": F_MASK_IDX,
        })
        self.update(MODEL_CONFIGS[model_name])


class MockDataset:
    """Just enough of recbole.data.Dataset: models only call .num(ITEM_ID)."""

    def __init__(self, n_item_tokens: int):
        self.n_item_tokens = n_item_tokens

    def num(self, field: str) -> int:
        assert field == F_ITEM, field
        return self.n_item_tokens


def build_model(name: str, raw_vocab_size: int, device: torch.device):
    from recbole.model.sequential_recommender import SASRec, GRU4Rec, BERT4Rec
    cls = {"SASRec": SASRec, "GRU4Rec": GRU4Rec, "BERT4Rec": BERT4Rec}[name]
    config = MockConfig(name, device)
    dataset = MockDataset(raw_vocab_size + 1)  # +1 for padding token 0
    model = cls(config, dataset).to(device)
    return model, config


def pad_batch(seqs: list, max_len: int = MAX_LEN):
    """Right-pad already-shifted (+1) sequences; returns (seq [B,L], len [B])."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        out[i, :len(s)] = s
        lens[i] = len(s)
    return torch.from_numpy(out), torch.from_numpy(lens)


class Interaction(dict):
    """Duck-typed stand-in for recbole.data.interaction.Interaction (models
    only index it by field name)."""


def train_batch_causal(model, item_seqs: torch.Tensor, lens: torch.Tensor,
                       targets: torch.Tensor):
    """SASRec / GRU4Rec: CE over the full catalog at the last position."""
    inter = Interaction({F_SEQ: item_seqs, F_LEN: lens, F_ITEM: targets})
    return model.calculate_loss(inter)


def cloze_mask(item_seqs: torch.Tensor, lens: torch.Tensor, n_item_tokens: int,
               mask_ratio: float, rng: np.random.Generator):
    """Replicates the pure-cloze branch of recbole MaskItemSequence
    (recbole==1.2.1): per real position, mask with prob mask_ratio (token =
    n_item_tokens); keep the LAST mask_len masked positions; 0-padded (left)
    index/pos lists. Draws vectorized for speed."""
    mask_len = int(mask_ratio * item_seqs.size(1))
    seqs = item_seqs.clone()
    B, L = seqs.shape
    lens_np = lens.numpy()
    valid = np.arange(L)[None, :] < lens_np[:, None]
    hit = (rng.random((B, L)) < mask_ratio) & valid
    pos_items = torch.zeros((B, mask_len), dtype=torch.long)
    masked_index = torch.zeros((B, mask_len), dtype=torch.long)
    for b in range(B):
        idx = np.flatnonzero(hit[b])[-mask_len:]
        if len(idx):
            pos_items[b, -len(idx):] = seqs[b, idx]
            masked_index[b, -len(idx):] = torch.from_numpy(idx)
            seqs[b, idx] = n_item_tokens
    return seqs, pos_items, masked_index


def mask_last(item_seqs: torch.Tensor, lens: torch.Tensor, n_item_tokens: int,
              mask_ratio: float):
    """The ft branch of recbole MaskItemSequence (_append_mask_last): mask ONLY
    the last real position — the objective aligned with BERT4Rec's eval-time
    reconstruct_test_data. RecBole's BERT4Rec.yaml sets ft_ratio 0.5, so the
    standard objective is a per-batch coin flip between this and cloze_mask."""
    mask_len = int(mask_ratio * item_seqs.size(1))
    seqs = item_seqs.clone()
    B = seqs.size(0)
    last = (lens - 1).clamp(min=0)
    rows = torch.arange(B)
    pos_items = torch.zeros((B, mask_len), dtype=torch.long)
    masked_index = torch.zeros((B, mask_len), dtype=torch.long)
    pos_items[:, -1] = seqs[rows, last]
    masked_index[:, -1] = last
    seqs[rows, last] = n_item_tokens
    return seqs, pos_items, masked_index


def train_batch_cloze(model, item_seqs: torch.Tensor, lens: torch.Tensor,
                      n_item_tokens: int, mask_ratio: float,
                      rng: np.random.Generator, device: torch.device,
                      ft_ratio: float = 0.5):
    """BERT4Rec: masked-item CE, RecBole-standard objective — per batch, with
    prob ft_ratio mask only the last position, else random cloze. NEG_ITEMS is
    required by the interface but unused under CE; zeros suffice."""
    if rng.random() < ft_ratio:
        masked, pos_items, masked_index = mask_last(
            item_seqs, lens, n_item_tokens, mask_ratio)
    else:
        masked, pos_items, masked_index = cloze_mask(
            item_seqs, lens, n_item_tokens, mask_ratio, rng)
    if (masked_index > 0).sum() == 0:
        # nothing the loss would count (recbole's masked_index>0 validity
        # quirk, replicated) — skip to avoid a 0/0 NaN loss
        return None
    inter = Interaction({
        F_MASK_SEQ: masked.to(device),
        F_POS: pos_items.to(device),
        F_NEG: torch.zeros_like(pos_items).to(device),
        F_MASK_IDX: masked_index.to(device),
    })
    return model.calculate_loss(inter)


@torch.no_grad()
def full_sort_topk(model, seqs: list, device: torch.device, k: int = 10,
                   batch_size: int = 256) -> np.ndarray:
    """Top-k RAW item ids for each (already +1-shifted) input sequence.
    No seen-item filtering (matches [25] filter_seen=False); padding token
    masked out. Returns [N, k] raw ids."""
    model.eval()
    out = []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        item_seqs, lens = pad_batch(chunk)
        inter = Interaction({F_SEQ: item_seqs.to(device),
                             F_LEN: lens.to(device)})
        scores = model.full_sort_predict(inter)
        scores[:, PAD] = -np.inf
        top = torch.topk(scores, k, dim=1).indices.cpu().numpy()
        out.append(top - 1)  # shift back to raw ids
    return np.concatenate(out, axis=0)


def hr_ndcg_at_k(topk: np.ndarray, targets: np.ndarray, k: int = 10):
    """Per-user HR@k and NDCG@k (single target -> hit indicator and
    1/log2(rank+1))."""
    hits = topk[:, :k] == targets[:, None]
    hr = hits.any(axis=1).astype(np.float64)
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 1)
    ndcg = hr * (1.0 / np.log2(ranks + 1))
    return hr, ndcg


def paired_bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                        seed: int = 4242) -> tuple:
    """95% CI of the mean via user bootstrap. `values` [n_users] (may be a
    difference vector for Δ CIs — pairing happens by resampling user indices)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def append_locked(csv_path: str, header: str, row: str) -> None:
    """fcntl-locked append (same convention as results/kip_training.csv).
    Size is re-checked and the write flushed UNDER the lock, so a concurrent
    first write cannot produce a duplicate header."""
    import fcntl
    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(header + "\n")
            fh.write(row + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
