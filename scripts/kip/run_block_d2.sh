#!/bin/bash
# Block D2 (user request 2026-07-26): FULL fine-tuning (no LoRA) of
# Llama-3.2-1B and Llama-3.1-8B on KIP-m4 mode (a) ordered, seed 9550,
# 120K training rows (stratified fraction 0.3 of the 400K), sequential.
# Other settings stay the as-run letters recipe: bs 64, max_length 50,
# AdamW 2e-5, <=20 epochs, patience 3 on val loss. Deviations (full FT,
# fraction) are recorded in the results row's recipe string.
#
# The 8B full FT is memory-tight on one A100-80GB (bf16 params+grads+AdamW
# ~64GB static): if a run fails with CUDA OOM, retry ONCE at batch 32
# (still recorded in the recipe string via bs=).
#
# HF_TOKEN must be in the environment (inherited from the kip_blockD tmux
# session env). Never echoed or logged.

set -uo pipefail
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
mkdir -p "$LOGDIR"
cd "$REPO/repro"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[$(date '+%F %T')] ERROR: HF_TOKEN not set; aborting."
    exit 1
fi
echo "[$(date '+%F %T')] HF_TOKEN present (not shown)."

run_llm () {
    local model="$1" bs="$2"
    local log="$LOGDIR/${model}_kip_m4_ordered_fullft_9550.log"
    echo "[$(date '+%F %T')] START $model full-FT kip_m4 ordered seed=9550 bs=$bs fraction=0.3 -> $log"
    $PY -m src.experiments.kip_gpu_training \
        --model "$model" --tag kip_m4 --mode ordered --seed 9550 \
        --llm_full_ft --llm_fraction 0.3 \
        --llm_max_length 50 --llm_batch_size "$bs" \
        --results_csv "$REPO/results/kip_training_fullft.csv" \
        --checkpoint_root "$REPO/checkpoints/kip_fullft" \
        > "$log" 2>&1
}

run_with_oom_fallback () {
    local model="$1"
    if run_llm "$model" 64; then
        echo "[$(date '+%F %T')] OK    $model full-FT (bs 64)"
    elif grep -q "CUDA out of memory" "$LOGDIR/${model}_kip_m4_ordered_fullft_9550.log"; then
        echo "[$(date '+%F %T')] OOM at bs 64 -- retrying $model at bs 32"
        if run_llm "$model" 32; then
            echo "[$(date '+%F %T')] OK    $model full-FT (bs 32 after OOM)"
        else
            echo "[$(date '+%F %T')] FAIL  $model full-FT (bs 32 retry also failed)"
        fi
    else
        echo "[$(date '+%F %T')] FAIL  $model full-FT (non-OOM error, see log)"
    fi
}

run_with_oom_fallback Llama1B
run_with_oom_fallback Llama8B

echo "[$(date '+%F %T')] BLOCK D2 COMPLETE"
