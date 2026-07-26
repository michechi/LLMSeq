#!/bin/bash
# Block B remainder after the mid-block plan change (user decision 2026-07-25):
# keep run 1 (BERT ordered seed 9550, max_length 512, full fidelity) untouched,
# then run ONE seed only for the controls, with max_length 64 for the training
# run (user-approved deviation; longest real prompt is ~43 tokens):
#
#   (b) BERT kip_m4 shuffled_train seed 9550  @ max_length 64
#   (c) BERT kip_m4 shuffled_eval  seed 9550  @ max_length 512 (checkpoint-
#       consistent eval of the run-1 checkpoint; costs only minutes)
#
# Waits for the original block runner to drain (runs 2-9 exit instantly via the
# SKIP_BLOCKB_QUEUE sentinel), then removes the sentinel and runs the two jobs.

set -uo pipefail
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
LOGDIR="$REPO/logs/kip"
SENTINEL="$LOGDIR/SKIP_BLOCKB_QUEUE"
cd "$REPO/repro"

echo "[$(date '+%F %T')] waiting for original Block B runner to drain..."
until grep -q "BLOCK B COMPLETE" "$LOGDIR/blockB_driver.log" 2>/dev/null; do
    sleep 60
done
rm -f "$SENTINEL"
echo "[$(date '+%F %T')] original runner drained; sentinel removed"

run_one () {
    local mode="$1" maxlen="$2"
    local log="$LOGDIR/BERT_kip_m4_${mode}_9550.log"
    echo "[$(date '+%F %T')] START BERT kip_m4 $mode seed=9550 maxlen=$maxlen -> $log"
    if $PY -m src.experiments.kip_gpu_training \
        --model BERT --tag kip_m4 --mode "$mode" --seed 9550 \
        --bert_max_length "$maxlen" > "$log" 2>&1; then
        echo "[$(date '+%F %T')] OK    BERT kip_m4 $mode seed=9550"
    else
        echo "[$(date '+%F %T')] FAIL  BERT kip_m4 $mode seed=9550 (see $log)"
    fi
}

run_one shuffled_train 64
run_one shuffled_eval 512

echo "[$(date '+%F %T')] BLOCK B2 COMPLETE"
