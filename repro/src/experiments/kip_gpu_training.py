"""KIP GPU training driver -- one (model, dataset, mode, seed) run per invocation.

Recipe fidelity: this file trains NOTHING itself. All model classes, configs and
training loops are imported from the repo's experiment scripts and invoked with
the exact settings used for Tricky Deterministic (tag "6"):

  LSTM / Transformer  src/experiments/DL_TR_baselines_experiment.py
                      OPTIMAL_CONFIGS + OPTIMAL_LR (LSTM 5e-4, Transformer 1e-3),
                      Adam, CrossEntropyLoss, batch 64, max 30 epochs,
                      early stop on val F1 with patience 3.
                      NOTE the paper's Appendix (tab:dl_training) documents
                      patience 5, but every committed invocation -- the argparse
                      default and the sensitivity SLURM runs -- executed
                      patience 3. This driver runs patience 3 (as-run) and
                      records it in the results row.

  BERT                src/experiments/BERT_fraction_experiment.py with --peft:
                      bert-base-uncased + LoRA(r=8, alpha=16, dropout 0.1,
                      target qkv, bias none, SEQ_CLS), AdamW lr 2e-5, batch 16,
                      max_length 512 (pad to max, as documented in
                      tab:bert_config), max 20 epochs, early stop on val loss
                      with patience 3, 6% linear warmup, threshold tuned on val
                      by find_optimal_threshold. Matches reproduce_main.sh and
                      the paper appendix exactly.

Modes:
  ordered        (a) train on ordered data, test on ordered test.
  shuffled_train (b) train/val/test on per-sequence-shuffled data with ORIGINAL
                     labels (built by src/data/kip_shuffles.py, fixed seed).
  shuffled_eval  (c) NO training: load the mode-(a) checkpoint for the same
                     (model, tag, seed) and evaluate it on the shuffled test
                     set. BERT reuses the threshold tuned during run (a); the DL
                     scripts' evaluate_on_test uses argmax and needs none.

Each run saves its checkpoint and appends one row to the results CSV the moment
it finishes (fcntl-locked append, so the row is durable even if a later run in
the block dies).

CLI::

    python -m src.experiments.kip_gpu_training \
        --model LSTM --tag kip_m4 --mode ordered --seed 9550
    python -m src.experiments.kip_gpu_training \
        --model BERT --tag kip_m4 --mode shuffled_eval --seed 9550
    python -m src.experiments.kip_gpu_training --model LSTM --tag kip_m4 \
        --mode ordered --seed 9550 --smoke     # 2K rows, 1 epoch, smoke outputs
"""

from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.experiments import BERT_fraction_experiment as bfx
from src.experiments import DL_TR_baselines_experiment as dlx
from src.experiments import LLM_fraction_experiment as lfx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

REPO_ROOT = Path("/root/LLMSeq")
ORDERED_CSV = REPO_ROOT / "data" / "simulation" / "tested"
SHUFFLED_CSV = REPO_ROOT / "data" / "simulation" / "kip_shuffled"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "kip"
RESULTS_CSV = REPO_ROOT / "results" / "kip_training.csv"
HF_CACHE = Path(os.environ.get("HF_HOME", "/root/hf_cache"))

DL_MODELS = ("LSTM", "Transformer")
MODES = ("ordered", "shuffled_train", "shuffled_eval")

RESULT_COLUMNS = [
    "timestamp", "model", "dataset", "mode", "seed", "train_rows", "epochs_done",
    "val_auc", "val_f1", "test_auc", "test_f1", "test_precision", "test_recall",
    "threshold", "n_params", "recipe", "wallclock_s", "checkpoint", "smoke",
]


def append_result(row: dict, results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    line = pd.DataFrame([{c: row.get(c, "") for c in RESULT_COLUMNS}])
    with open(results_csv, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            write_header = f.tell() == 0
            f.write(line.to_csv(index=False, header=write_header))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    logger.info("result row appended to %s", results_csv)


def data_dir_for(mode: str, args) -> Path:
    """ordered trains/tests on ordered data; both shuffled modes read the
    shuffled directory (mode c only uses its test split)."""
    return args.ordered_csv if mode == "ordered" else args.shuffled_csv


def ckpt_dir_for(model: str, tag: str, mode: str, seed: int, args) -> Path:
    root = args.checkpoint_root / ("smoke" if args.smoke else "")
    return root / f"{model}_{tag}_{mode}_{seed}"


def _maybe_truncate(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return df if n is None else df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# DL family (LSTM / Transformer)                                              #
# --------------------------------------------------------------------------- #
def run_dl(args) -> dict:
    dlx.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    csv_dir = data_dir_for(args.mode, args)
    ns = argparse.Namespace(
        path_csv=str(csv_dir) + "/", number_to_use=args.tag,
        sequence_column="Sequences", label_column="Outcome",
        sequence_separator="\x1f",
    )
    (tr_s, tr_l, va_s, va_l, te_s, te_l) = dlx.load_data(ns)

    smoke_rows = 2000 if args.smoke else None
    smoke_eval = 1000 if args.smoke else None
    if smoke_rows:
        tr_s, tr_l = tr_s[:smoke_rows], tr_l[:smoke_rows]
        va_s, va_l = va_s[:smoke_eval], va_l[:smoke_eval]
        te_s, te_l = te_s[:smoke_eval], te_l[:smoke_eval]

    max_seq_length = max(len(str(s)) for s in list(tr_s) + list(va_s) + list(te_s))
    config = copy.deepcopy(dlx.OPTIMAL_CONFIGS[args.model])
    if args.model in ("MLP", "Transformer", "RNNTransformer", "Mamba"):
        config["max_seq_length"] = max_seq_length
    lr = dlx.OPTIMAL_LR[args.model]
    epochs = 1 if args.smoke else 30
    patience = args.dl_patience
    batch_size = 64
    recipe = (f"DL_TR_baselines_experiment OPTIMAL: Adam lr={lr} bs={batch_size} "
              f"epochs<={epochs} patience={patience} early=val_f1 config={config}")
    logger.info("recipe: %s", recipe)

    test_loader = DataLoader(dlx.LetterSequenceDataset(te_s, te_l), batch_size=batch_size,
                             shuffle=False, collate_fn=dlx.collate_fn, num_workers=0)

    t0 = time.time()
    if args.mode == "shuffled_eval":
        src = ckpt_dir_for(args.model, args.tag, "ordered", args.seed, args) / "ckpt.pt"
        if not src.exists():
            raise FileNotFoundError(f"mode shuffled_eval needs the ordered checkpoint first: {src}")
        payload = torch.load(src, map_location="cpu", weights_only=False)
        model = dlx.create_model(args.model, payload["config"])
        model.load_state_dict(payload["state_dict"])
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        test_metrics = dlx.evaluate_on_test(model, test_loader, device=device)
        row = {
            "train_rows": 0, "epochs_done": 0, "val_auc": "", "val_f1": "",
            "threshold": "argmax", "n_params": n_params,
            "checkpoint": str(src),
        }
    else:
        val_loader = DataLoader(dlx.LetterSequenceDataset(va_s, va_l), batch_size=batch_size,
                                shuffle=False, collate_fn=dlx.collate_fn, num_workers=0)
        train_loader = DataLoader(dlx.LetterSequenceDataset(tr_s, tr_l), batch_size=batch_size,
                                  shuffle=True, collate_fn=dlx.collate_fn, num_workers=0)
        model = dlx.create_model(args.model, config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        (model, best_val_f1, best_val_auc, _p, _r, epochs_done, _h) = \
            dlx.train_model_with_early_stopping(
                model, train_loader, val_loader,
                num_epochs=epochs, lr=lr, patience=patience, device=device,
            )
        test_metrics = dlx.evaluate_on_test(model, test_loader, device=device)

        ckpt_dir = ckpt_dir_for(args.model, args.tag, args.mode, args.seed, args)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "model": args.model, "seed": args.seed, "mode": args.mode,
                    "tag": args.tag, "lr": lr}, ckpt_dir / "ckpt.pt")
        logger.info("checkpoint saved: %s", ckpt_dir / "ckpt.pt")
        row = {
            "train_rows": len(tr_s), "epochs_done": epochs_done,
            "val_auc": best_val_auc, "val_f1": best_val_f1,
            "threshold": "argmax", "n_params": n_params,
            "checkpoint": str(ckpt_dir / "ckpt.pt"),
        }

    row.update({
        "test_auc": test_metrics["auc"], "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
        "recipe": recipe, "wallclock_s": round(time.time() - t0, 1),
    })
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return row


# --------------------------------------------------------------------------- #
# BERT family                                                                 #
# --------------------------------------------------------------------------- #
def bert_args(args, csv_dir: Path) -> argparse.Namespace:
    """The exact reproduce_main.sh invocation: defaults + --peft.

    --bert_max_length below 512 is a user-approved deviation from the paper
    appendix (pad-to-512): the longest real KIP prompt is ~43 tokens, so with
    attention masking a shorter pad changes only wasted compute. The value used
    is recorded in the results row's recipe string either way.
    """
    argv = [
        "--number_to_use", args.tag,
        "--path_csv", str(csv_dir) + "/",
        "--model_name", "bert-base-uncased",
        "--peft",
        "--seed", str(args.seed),
        "--cache_dir", str(HF_CACHE),
        "--output_dir", str(REPO_ROOT / "results" / "kip_bert_raw"),
        "--fractions", "1.0",
        "--max_length", str(args.bert_max_length),
    ]
    bargs = bfx.parse_args(argv)
    if args.smoke:
        bargs.epochs = 1
    return bargs


def _bert_loaders(csv_dir: Path, tag: str, tokenizer, bargs, smoke: bool):
    def _read(name):
        return pd.read_csv(csv_dir / f"{name}_{tag}.csv",
                           na_values=["", "None", "NaN", "na", "nan"]).fillna("")

    X_train, y_train = _read("X_train"), _read("y_train")
    X_val, y_val = _read("X_val"), _read("y_val")
    X_test, y_test = _read("X_test"), _read("y_test")
    if smoke:
        X_train, y_train = _maybe_truncate(X_train, 2000), _maybe_truncate(y_train, 2000)
        X_val, y_val = _maybe_truncate(X_val, 1000), _maybe_truncate(y_val, 1000)
        X_test, y_test = _maybe_truncate(X_test, 1000), _maybe_truncate(y_test, 1000)

    def _ds(X, y):
        texts = X.apply(bfx.standard_narrative_prompt, axis=1).tolist()
        labels = y["Outcome"].tolist()
        return bfx.SequenceClassificationDataset(texts, labels, tokenizer,
                                                 max_length=bargs.max_length)

    train_loader = DataLoader(_ds(X_train, y_train), batch_size=bargs.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(_ds(X_val, y_val), batch_size=bargs.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(_ds(X_test, y_test), batch_size=bargs.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0)
    return train_loader, val_loader, test_loader, len(X_train)


def run_bert(args) -> dict:
    bfx.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    csv_dir = data_dir_for(args.mode, args)
    bargs = bert_args(args, csv_dir)
    recipe = (f"BERT_fraction_experiment defaults + --peft: bert-base-uncased "
              f"LoRA(r=8,a=16,qkv,drop=.1) AdamW lr={bargs.lr} bs={bargs.batch_size} "
              f"max_len={bargs.max_length} epochs<={bargs.epochs} "
              f"patience={bargs.patience} early={bargs.early} warmup=6%")
    logger.info("recipe: %s", recipe)

    tokenizer = bfx.load_tokenizer(bargs.model_name, bargs.cache_dir)
    train_loader, val_loader, test_loader, n_train = _bert_loaders(
        csv_dir, args.tag, tokenizer, bargs, args.smoke)

    t0 = time.time()
    if args.mode == "shuffled_eval":
        src_dir = ckpt_dir_for(args.model, args.tag, "ordered", args.seed, args)
        ckpt, meta_p = src_dir / "ckpt.pt", src_dir / "meta.json"
        if not ckpt.exists():
            raise FileNotFoundError(f"mode shuffled_eval needs the ordered checkpoint first: {ckpt}")
        meta = json.loads(meta_p.read_text())
        model = bfx.load_model(bargs, tokenizer)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model = model.to(device)
        threshold = float(meta["optimal_threshold"])
        test_metrics = bfx.evaluate_on_test(model, test_loader, device, threshold=threshold)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {"train_rows": 0, "epochs_done": 0, "val_auc": "", "val_f1": "",
               "threshold": threshold, "n_params": n_params, "checkpoint": str(ckpt)}
    else:
        model = bfx.load_model(bargs, tokenizer)
        model = model.to(device)
        torch.cuda.empty_cache()
        gc.collect()

        best_auc, best_f1, best_val_loss, epochs_done, val_preds, val_labels = \
            bfx.train_and_evaluate(model, train_loader, val_loader, bargs, device)
        threshold, val_f1_opt = bfx.find_optimal_threshold(val_labels, val_preds)
        logger.info("optimal threshold: %.2f (val F1 %.4f)", threshold, val_f1_opt)
        test_metrics = bfx.evaluate_on_test(model, test_loader, device, threshold=threshold)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ckpt_dir = ckpt_dir_for(args.model, args.tag, args.mode, args.seed, args)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "ckpt.pt")
        (ckpt_dir / "meta.json").write_text(json.dumps({
            "optimal_threshold": threshold, "val_f1_optimal": val_f1_opt,
            "val_auc": best_auc, "val_loss": best_val_loss,
            "epochs_done": epochs_done, "seed": args.seed, "mode": args.mode,
            "tag": args.tag, "model_name": bargs.model_name, "peft": True,
        }, indent=2) + "\n")
        logger.info("checkpoint saved: %s", ckpt_dir / "ckpt.pt")
        row = {"train_rows": n_train, "epochs_done": epochs_done,
               "val_auc": best_auc, "val_f1": best_f1, "threshold": threshold,
               "n_params": n_params, "checkpoint": str(ckpt_dir / "ckpt.pt")}

    row.update({
        "test_auc": test_metrics["auc"], "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
        "recipe": recipe, "wallclock_s": round(time.time() - t0, 1),
    })
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return row


# --------------------------------------------------------------------------- #
# Llama family (decoder + LoRA via LLM_fraction_experiment)                   #
# --------------------------------------------------------------------------- #
def llm_args(args, csv_dir: Path) -> argparse.Namespace:
    """The canonical REPRODUCING.md invocation: defaults + --peft --epochs 20.

    HF_TOKEN is read from the environment by this driver and passed only as a
    Python function argument, exactly as LLM_fraction_experiment.run_experiment
    does. It is never placed in argv, logs, results rows, or checkpoints.
    """
    argv = [
        "--number_to_use", args.tag,
        "--path_csv", str(csv_dir) + "/",
        "--model_name", args.llm_model_name,
        "--peft",
        "--epochs", "20",
        "--seed", str(args.seed),
        "--cache_dir", str(HF_CACHE),
        "--output_dir", str(REPO_ROOT / "results" / "kip_llm_raw"),
        "--fractions", "1.0",
        "--max_length", str(args.llm_max_length),
        "--batch_size", str(args.llm_batch_size),
    ]
    largs = lfx.parse_args(argv)
    if args.smoke:
        largs.epochs = 1
    return largs


def run_llm(args) -> dict:
    lfx.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    csv_dir = data_dir_for(args.mode, args)
    largs = llm_args(args, csv_dir)
    recipe = (f"LLM_fraction_experiment defaults + --peft: {largs.model_name} "
              f"LoRA(r=8,a=16,qkvo,drop=.1,CAUSAL_LM) AdamW lr={largs.lr} "
              f"bs={largs.batch_size} max_len={largs.max_length} "
              f"epochs<={largs.epochs} patience={largs.patience} "
              f"early={largs.early} warmup=6% bf16")
    logger.info("recipe: %s", recipe)

    hf_token = os.environ.get("HF_TOKEN", None)  # never logged
    tokenizer = lfx.load_tokenizer(largs.model_name, largs.model_type, hf_token,
                                   largs.cache_dir)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token. Adding [PAD].")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _read(name):
        return pd.read_csv(csv_dir / f"{name}_{args.tag}.csv",
                           na_values=["", "None", "NaN", "na", "nan"]).fillna("")

    X_train, y_train = _read("X_train"), _read("y_train")
    X_val, y_val = _read("X_val"), _read("y_val")
    X_test, y_test = _read("X_test"), _read("y_test")
    if args.smoke:
        X_train, y_train = _maybe_truncate(X_train, 2000), _maybe_truncate(y_train, 2000)
        X_val, y_val = _maybe_truncate(X_val, 1000), _maybe_truncate(y_val, 1000)
        X_test, y_test = _maybe_truncate(X_test, 1000), _maybe_truncate(y_test, 1000)

    def _ds(X, y):
        texts = X.apply(lfx.standard_narrative_prompt, axis=1).tolist()
        labels = y["Outcome"].tolist()
        return lfx.TemporalCausalDataset(texts, labels, tokenizer,
                                         max_length=largs.max_length)

    train_loader = DataLoader(_ds(X_train, y_train), batch_size=largs.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(_ds(X_val, y_val), batch_size=largs.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(_ds(X_test, y_test), batch_size=largs.batch_size,
                             shuffle=False, pin_memory=True, num_workers=0)

    t0 = time.time()
    if args.mode == "shuffled_eval":
        src_dir = ckpt_dir_for(args.model, args.tag, "ordered", args.seed, args)
        ckpt, meta_p = src_dir / "ckpt.pt", src_dir / "meta.json"
        if not ckpt.exists():
            raise FileNotFoundError(f"mode shuffled_eval needs the ordered checkpoint first: {ckpt}")
        meta = json.loads(meta_p.read_text())
        model = lfx.load_model_causal(largs, tokenizer, hf_token)
        if tokenizer.pad_token is not None:
            model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        model = model.to(device, dtype=torch.bfloat16)
        threshold = float(meta["optimal_threshold"])
        test_metrics = lfx.evaluate_on_test(model, test_loader, device, threshold=threshold)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row = {"train_rows": 0, "epochs_done": 0, "val_auc": "", "val_f1": "",
               "threshold": threshold, "n_params": n_params, "checkpoint": str(ckpt)}
    else:
        model = lfx.load_model_causal(largs, tokenizer, hf_token)
        if tokenizer.pad_token is not None:
            model.resize_token_embeddings(len(tokenizer))
        model = model.to(device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        gc.collect()

        best_auc, best_f1, best_val_loss, epochs_done, val_preds, val_labels = \
            lfx.train_and_evaluate_causal(model, train_loader, val_loader, largs,
                                          device, use_quantization=False)
        threshold, val_f1_opt = lfx.find_optimal_threshold(val_labels, val_preds)
        logger.info("optimal threshold: %.2f (val F1 %.4f)", threshold, val_f1_opt)
        test_metrics = lfx.evaluate_on_test(model, test_loader, device, threshold=threshold)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ckpt_dir = ckpt_dir_for(args.model, args.tag, args.mode, args.seed, args)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "ckpt.pt")
        (ckpt_dir / "meta.json").write_text(json.dumps({
            "optimal_threshold": threshold, "val_f1_optimal": val_f1_opt,
            "val_auc": best_auc, "val_loss": best_val_loss,
            "epochs_done": epochs_done, "seed": args.seed, "mode": args.mode,
            "tag": args.tag, "model_name": largs.model_name, "peft": True,
        }, indent=2) + "\n")
        logger.info("checkpoint saved: %s", ckpt_dir / "ckpt.pt")
        row = {"train_rows": len(X_train), "epochs_done": epochs_done,
               "val_auc": best_auc, "val_f1": best_f1, "threshold": threshold,
               "n_params": n_params, "checkpoint": str(ckpt_dir / "ckpt.pt")}

    row.update({
        "test_auc": test_metrics["auc"], "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"], "test_recall": test_metrics["recall"],
        "recipe": recipe, "wallclock_s": round(time.time() - t0, 1),
    })
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return row


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIP GPU training driver (one run)")
    p.add_argument("--model", required=True, choices=list(DL_MODELS) + ["BERT", "Llama1B"])
    p.add_argument("--tag", required=True, choices=["kip_m4", "kip_m6"])
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ordered_csv", type=Path, default=ORDERED_CSV)
    p.add_argument("--shuffled_csv", type=Path, default=SHUFFLED_CSV)
    p.add_argument("--checkpoint_root", type=Path, default=CHECKPOINT_ROOT)
    p.add_argument("--results_csv", type=Path, default=RESULTS_CSV)
    p.add_argument("--smoke", action="store_true",
                   help="2K train rows, 1 epoch, separate smoke results file")
    p.add_argument("--llm_model_name", type=str, default="meta-llama/Llama-3.2-1B",
                   help="Decoder model for --model Llama1B. Smoke tests may pass an "
                        "ungated stand-in (e.g. HuggingFaceTB/SmolLM2-135M).")
    p.add_argument("--llm_max_length", type=int, default=512,
                   help="Decoder pad/truncate length. 512 = script default; smaller "
                        "values are a recorded deviation.")
    p.add_argument("--llm_batch_size", type=int, default=8,
                   help="Decoder batch size. 8 = script default.")
    p.add_argument("--bert_max_length", type=int, default=512,
                   help="BERT tokenizer pad/truncate length. 512 = paper appendix; "
                        "smaller values are a recorded deviation (see bert_args).")
    p.add_argument("--dl_patience", type=int, default=3,
                   help="DL early-stopping patience. 3 = as-run default (every "
                        "committed invocation); 5 = the paper appendix value. "
                        "When overriding, also point --results_csv and "
                        "--checkpoint_root elsewhere so variant rows and "
                        "checkpoints never mix with the as-run results.")
    return p.parse_args(list(args) if args is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    # One-shot abort sentinel: while this file exists, invocations exit
    # immediately without training. Used to drain an already-running block
    # runner's remaining queue after a mid-block plan change, without touching
    # the run currently on the GPU (its python process already loaded this
    # module and never re-reads it).
    skip_sentinel = Path("/root/LLMSeq/logs/kip/SKIP_BLOCKB_QUEUE")
    if skip_sentinel.exists():
        logger.info("skip sentinel %s present -- exiting without running "
                    "(%s %s %s seed=%d)", skip_sentinel, args.model, args.tag,
                    args.mode, args.seed)
        return 0

    if args.smoke:
        args.results_csv = args.results_csv.with_name("kip_training_smoke.csv")
    for k, v in sorted(vars(args).items()):
        logger.info("arg %s = %s", k, v)

    if args.model == "BERT":
        run_fn = run_bert
    elif args.model == "Llama1B":
        run_fn = run_llm
    else:
        run_fn = run_dl
    row = run_fn(args)
    row.update({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model, "dataset": args.tag, "mode": args.mode,
        "seed": args.seed, "smoke": args.smoke,
    })
    append_result(row, args.results_csv)
    logger.info("DONE %s %s %s seed=%d: test_auc=%.4f test_f1=%.4f (%.1fs)",
                args.model, args.tag, args.mode, args.seed,
                row["test_auc"], row["test_f1"], row["wallclock_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
