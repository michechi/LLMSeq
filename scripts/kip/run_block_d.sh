#!/bin/bash
# Block D: Llama-3.2-1B + LoRA on KIP-m4, mode (a) ordered, ONE seed (9550).
#
# Usage: bash scripts/kip/run_block_d.sh [MAX_LENGTH] [BATCH_SIZE]
#
# Defaults (50 / 64) reproduce the AS-RUN cluster recipe for the paper's
# letter-dataset Llama jobs (scripts/slurm/LLM_fraction_experiment.sh:16-26:
# Llama-3.2-1B, bs 64, max_length 50, LoRA, lr 2e-5, epochs 20, patience 3,
# fractions 1.0) rather than the python script's generic defaults (8 / 512).
# KIP prompts are ~26 Llama tokens, so max_length 50 truncates nothing.
#
# HF_TOKEN must be present in the environment (injected by the user into the
# tmux session environment; new windows inherit it). This script never echoes,
# logs, or writes the token; the python driver reads it from the environment
# and passes it only as an in-process function argument.

set -uo pipefail
MAXLEN="${1:-50}"
BS="${2:-64}"
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
mkdir -p "$LOGDIR"
cd "$REPO/repro"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[$(date '+%F %T')] ERROR: HF_TOKEN not set in environment; aborting."
    exit 1
fi
echo "[$(date '+%F %T')] HF_TOKEN present (not shown)."

log="$LOGDIR/Llama1B_kip_m4_ordered_9550.log"
echo "[$(date '+%F %T')] START Llama1B kip_m4 ordered seed=9550 maxlen=$MAXLEN bs=$BS -> $log"
if $PY -m src.experiments.kip_gpu_training \
    --model Llama1B --tag kip_m4 --mode ordered --seed 9550 \
    --llm_max_length "$MAXLEN" --llm_batch_size "$BS" > "$log" 2>&1; then
    echo "[$(date '+%F %T')] OK    Llama1B kip_m4 ordered seed=9550"
else
    echo "[$(date '+%F %T')] FAIL  Llama1B kip_m4 ordered seed=9550 (see $log)"
fi
echo "[$(date '+%F %T')] BLOCK D COMPLETE"
