"""Train one (model, seed) cell of the 30Music audit grid.

RecBole model classes + RecBole-standard training objective (prefix
augmentation; CE over the full catalog at the last position for
SASRec/GRU4Rec, cloze masking for BERT4Rec), with [25]-protocol model
selection: early stop on validation NDCG@10 (input = all-but-last capped at
128, target = last), patience 5, best checkpoint restored. Data ingestion
honors the audit consumer contract (contiguous user blocks in file order).

One invocation = one row appended (fcntl-locked) to
results/recsys_audit/training.csv + checkpoint + curve.json.

CLI::

    python -m src.recsys.train_grid --model SASRec --seed 17 \
        --split-dir data/recsys/30music/split --out-dir checkpoints/recsys
    # mode-(b) shuffled-train (optional final block):
    python -m src.recsys.train_grid --model GRU4Rec --seed 17 \
        --split-dir data/recsys/30music/split \
        --train-csv .../audit/train_30Music_full_shuffle_s101.csv \
        --val-csv .../audit/validation_30Music_full_shuffle_s101.csv \
        --mode-label shuffled_train_s101
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time

import numpy as np
import torch

from .eval_loader import _assert_structure, _group_positions, _read
from .recbole_grid import (MAX_LEN, append_locked, build_model, full_sort_topk,
                           hr_ndcg_at_k, pad_batch, train_batch_causal,
                           train_batch_cloze)

TRAIN_HEADER = ("model,seed,mode,best_epoch,epochs_run,best_val_ndcg10,"
                "val_hr10_at_best,batch_size,lr,patience,max_len,raw_vocab,"
                "train_users,train_samples,seconds,site,host,gpu,slurm_job_id,"
                "torch_version,recbole_version,checkpoint")


def load_user_sequences(csv_path: str) -> list:
    """All users' full sequences (+1-shifted), file order, contract-checked."""
    df = _read(csv_path)
    _assert_structure(df, csv_path)
    items = df["item_id"].to_numpy() + 1  # shift for RecBole padding token 0
    seqs = [items[idx] for idx in _group_positions(df).values()]
    shortest = min(len(s) for s in seqs)
    if shortest < 2:
        raise ValueError(f"{csv_path}: shortest user sequence has {shortest} "
                         "event(s); need >=2 (input + target)")
    return seqs


def build_samples(seqs: list) -> tuple:
    """Prefix augmentation: one sample per (sequence, target position t>=1)."""
    seq_ref, pos = [], []
    for i, s in enumerate(seqs):
        n = len(s)
        seq_ref.append(np.full(n - 1, i, dtype=np.int32))
        pos.append(np.arange(1, n, dtype=np.int32))
    return np.concatenate(seq_ref), np.concatenate(pos)


def val_ndcg(model, val_seqs: list, device) -> tuple:
    inputs = [s[:-1] for s in val_seqs]
    targets = np.array([s[-1] - 1 for s in val_seqs])
    topk = full_sort_topk(model, inputs, device)
    hr, ndcg = hr_ndcg_at_k(topk, targets)
    return float(ndcg.mean()), float(hr.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    choices=["SASRec", "GRU4Rec", "BERT4Rec"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--train-csv", help="override (mode-b); default split-dir train")
    ap.add_argument("--val-csv", help="override (mode-b); default split-dir validation")
    ap.add_argument("--mode-label", default="ordered")
    ap.add_argument("--out-dir", default="checkpoints/recsys")
    ap.add_argument("--results-csv", default="results/recsys_audit/training.csv")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0,
                    help="smoke-testing only; 0 = full epoch")
    ap.add_argument("--site", default=os.environ.get("KIP_SITE", "local"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_csv = args.train_csv or os.path.join(args.split_dir, "train_30Music.csv")
    val_csv = args.val_csv or os.path.join(args.split_dir, "validation_30Music.csv")
    train_seqs = load_user_sequences(train_csv)
    val_seqs = load_user_sequences(val_csv)
    test_max = _read(os.path.join(args.split_dir, "test_30Music.csv"))["item_id"].max()
    raw_vocab = int(max(max(s.max() for s in train_seqs) - 1,
                        max(s.max() for s in val_seqs) - 1, test_max)) + 1

    seq_ref, pos = build_samples(train_seqs)
    n_samples = len(pos)
    print(f"{args.model} seed {args.seed} mode {args.mode_label}: "
          f"{len(train_seqs)} users, {n_samples} samples, vocab {raw_vocab}",
          flush=True)

    model, config = build_model(args.model, raw_vocab, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    cloze = args.model == "BERT4Rec"

    best = {"ndcg": -1.0, "hr": 0.0, "epoch": -1, "state": None}
    curve = []
    epochs_run = 0
    for epoch in range(args.max_epochs):
        epochs_run = epoch + 1
        model.train()
        order = np.random.permutation(n_samples)
        steps = len(order) // args.batch_size + 1
        if args.max_steps_per_epoch:
            steps = min(steps, args.max_steps_per_epoch)
        mask_rng = np.random.default_rng([args.seed, epoch])
        losses = []
        for step in range(steps):
            sel = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(sel) == 0:
                continue
            ctx = [train_seqs[seq_ref[i]][max(0, pos[i] - MAX_LEN):pos[i]]
                   for i in sel]
            tgt = torch.tensor(
                [int(train_seqs[seq_ref[i]][pos[i]]) for i in sel],
                dtype=torch.long, device=device)
            item_seqs, lens = pad_batch(ctx)
            opt.zero_grad()
            if cloze:
                loss = train_batch_cloze(model, item_seqs, lens, raw_vocab + 1,
                                         config["mask_ratio"], mask_rng, device,
                                         ft_ratio=config["ft_ratio"])
                if loss is None:
                    continue
            else:
                loss = train_batch_causal(model, item_seqs.to(device),
                                          lens.to(device), tgt)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ndcg, hr = val_ndcg(model, val_seqs, device)
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                      "val_ndcg10": ndcg, "val_hr10": hr,
                      "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} "
              f"val_ndcg10 {ndcg:.4f} val_hr10 {hr:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ndcg > best["ndcg"]:
            best = {"ndcg": ndcg, "hr": hr, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch} (best {best['epoch']})",
                  flush=True)
            break

    model.load_state_dict(best["state"])
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.out_dir, f"{args.model}_{args.mode_label}_s{args.seed}.pt")
    torch.save({"model_name": args.model, "state_dict": best["state"],
                "raw_vocab": raw_vocab, "seed": args.seed,
                "mode": args.mode_label, "best_epoch": best["epoch"],
                "best_val_ndcg10": best["ndcg"],
                "model_config": {k: v for k, v in config.items()
                                 if k != "device"}}, ckpt_path)
    with open(ckpt_path.replace(".pt", "_curve.json"), "w") as fh:
        json.dump(curve, fh, indent=2)

    import recbole
    gpu = (torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu")
    row = ",".join(str(x) for x in [
        args.model, args.seed, args.mode_label, best["epoch"], epochs_run,
        f"{best['ndcg']:.6f}", f"{best['hr']:.6f}", args.batch_size, args.lr,
        args.patience, MAX_LEN, raw_vocab, len(train_seqs), n_samples,
        round(time.time() - t0, 1), args.site, socket.gethostname(),
        gpu.replace(",", " "), os.environ.get("SLURM_JOB_ID", ""),
        torch.__version__, recbole.__version__, ckpt_path])
    append_locked(args.results_csv, TRAIN_HEADER, row)
    print(f"DONE {ckpt_path} best val_ndcg10 {best['ndcg']:.4f} "
          f"@epoch {best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
