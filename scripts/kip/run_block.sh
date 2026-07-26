#!/bin/bash
# KIP training block runner -- sequential (model x dataset x mode x seed) jobs
# on the single local A100. One log per run in logs/kip/, one results row
# appended to results/kip_training.csv per run by the driver itself.
#
# Usage:
#   bash scripts/kip/run_block.sh A          # LSTM + Transformer on kip_m4
#   bash scripts/kip/run_block.sh B          # BERT on kip_m4
#   bash scripts/kip/run_block.sh C          # all three on kip_m6 (mode a first)
#   bash scripts/kip/run_block.sh SMOKE     # 1-epoch pipeline check, all families
#
# Never echoes or logs HF_TOKEN. bert-base-uncased is ungated; no token needed
# for blocks A-C.

set -uo pipefail

BLOCK="${1:?usage: run_block.sh A|B|C|SMOKE}"
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
mkdir -p "$LOGDIR"
cd "$REPO/repro"

SEEDS=(9550 9551 9552)

run_one () {
    local model="$1" tag="$2" mode="$3" seed="$4" extra="${5:-}"
    local log="$LOGDIR/${model}_${tag}_${mode}_${seed}.log"
    echo "[$(date '+%F %T')] START $model $tag $mode seed=$seed -> $log"
    if $PY -m src.experiments.kip_gpu_training \
        --model "$model" --tag "$tag" --mode "$mode" --seed "$seed" $extra \
        > "$log" 2>&1; then
        echo "[$(date '+%F %T')] OK    $model $tag $mode seed=$seed"
    else
        echo "[$(date '+%F %T')] FAIL  $model $tag $mode seed=$seed (see $log)"
    fi
}

case "$BLOCK" in
  SMOKE)
    # One tiny run per family and mode to validate the whole pipeline,
    # including checkpoint reuse for shuffled_eval.
    run_one LSTM kip_m4 ordered       9550 --smoke
    run_one LSTM kip_m4 shuffled_train 9550 --smoke
    run_one LSTM kip_m4 shuffled_eval 9550 --smoke
    run_one BERT kip_m4 ordered       9550 --smoke
    run_one BERT kip_m4 shuffled_eval 9550 --smoke
    ;;
  A)
    for model in LSTM Transformer; do
      for seed in "${SEEDS[@]}"; do
        run_one "$model" kip_m4 ordered        "$seed"
        run_one "$model" kip_m4 shuffled_train "$seed"
        run_one "$model" kip_m4 shuffled_eval  "$seed"
      done
    done
    ;;
  B)
    for seed in "${SEEDS[@]}"; do
      run_one BERT kip_m4 ordered        "$seed"
      run_one BERT kip_m4 shuffled_train "$seed"
      run_one BERT kip_m4 shuffled_eval  "$seed"
    done
    ;;
  C)
    # Mode (a) for all models first, controls after (per instruction).
    # BERT arm: ONE seed at max_length 64 (user decision after Block B's
    # 7.3h full-fidelity run; deviation recorded in the results recipe column).
    for model in LSTM Transformer; do
      for seed in "${SEEDS[@]}"; do
        run_one "$model" kip_m6 ordered "$seed"
      done
    done
    run_one BERT kip_m6 ordered 9550 "--bert_max_length 64"
    for model in LSTM Transformer; do
      for seed in "${SEEDS[@]}"; do
        run_one "$model" kip_m6 shuffled_train "$seed"
        run_one "$model" kip_m6 shuffled_eval  "$seed"
      done
    done
    run_one BERT kip_m6 shuffled_train 9550 "--bert_max_length 64"
    run_one BERT kip_m6 shuffled_eval  9550 "--bert_max_length 64"
    ;;
  *)
    echo "unknown block: $BLOCK" >&2; exit 2 ;;
esac

echo "[$(date '+%F %T')] BLOCK $BLOCK COMPLETE"
