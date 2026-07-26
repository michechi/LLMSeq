#!/bin/bash
# Patience-5 addendum (user-approved recipe VARIANT, paper-appendix value):
# mode (a) only, LSTM + Transformer on kip_m4, 3 seeds. Fully isolated from the
# as-run results: separate results CSV and separate checkpoint root, so nothing
# overwrites Block A's patience-3 checkpoints or pollutes its mean/std.

set -uo pipefail
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
mkdir -p "$LOGDIR"
cd "$REPO/repro"

for model in LSTM Transformer; do
  for seed in 9550 9551 9552; do
    log="$LOGDIR/${model}_kip_m4_ordered_p5_${seed}.log"
    echo "[$(date '+%F %T')] START $model kip_m4 ordered(p5) seed=$seed -> $log"
    if $PY -m src.experiments.kip_gpu_training \
        --model "$model" --tag kip_m4 --mode ordered --seed "$seed" \
        --dl_patience 5 \
        --results_csv "$REPO/results/kip_training_p5.csv" \
        --checkpoint_root "$REPO/checkpoints/kip_p5" \
        > "$log" 2>&1; then
      echo "[$(date '+%F %T')] OK    $model kip_m4 ordered(p5) seed=$seed"
    else
      echo "[$(date '+%F %T')] FAIL  $model kip_m4 ordered(p5) seed=$seed (see $log)"
    fi
  done
done
echo "[$(date '+%F %T')] P5 ADDENDUM COMPLETE"
