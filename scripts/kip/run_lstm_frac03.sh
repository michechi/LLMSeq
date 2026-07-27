#!/bin/bash
# Fraction-matched control for the Llama full-FT arm (user request 2026-07-27):
# LSTM on kip_m4 ordered, stratified fraction 0.3 (120K rows), 3 seeds, paper
# recipe (patience 5). If LSTM still hits 1.0 at 120K, the fraction mismatch
# between the LSTM headline (400K) and the full-FT Llama arm (120K) becomes a
# controlled variable instead of a caveat.
#
# Rows go to results/kip_training_fullft.csv (the full-FT comparison file,
# merge=union) so the whole fraction-matched comparison lives in one CSV;
# checkpoints isolated in checkpoints/kip_fullft/.

set -uo pipefail
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
mkdir -p "$LOGDIR"
cd "$REPO/repro"

for seed in 9550 9551 9552; do
    log="$LOGDIR/LSTM_kip_m4_ordered_frac03_${seed}.log"
    echo "[$(date '+%F %T')] START LSTM kip_m4 ordered frac=0.3 p5 seed=$seed -> $log"
    if $PY -m src.experiments.kip_gpu_training \
        --model LSTM --tag kip_m4 --mode ordered --seed "$seed" \
        --dl_fraction 0.3 --dl_patience 5 \
        --results_csv "$REPO/results/kip_training_fullft.csv" \
        --checkpoint_root "$REPO/checkpoints/kip_fullft" \
        > "$log" 2>&1; then
        echo "[$(date '+%F %T')] OK    LSTM frac03 seed=$seed"
    else
        echo "[$(date '+%F %T')] FAIL  LSTM frac03 seed=$seed (see $log)"
    fi
done
echo "[$(date '+%F %T')] LSTM FRAC03 COMPLETE"
