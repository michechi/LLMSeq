#!/bin/bash
# RecSys audit mode-(b) arm — retrain on SHUFFLED sequences (local H200 box).
#
# Counterpart of the MIMIC shuffle test's "retrain on shuffled" condition
# (src/mimic/shuffle_test.py) and of KIP mode (b): train AND early-stop on the
# full-sequence-shuffled train/validation files (audit shuffle seed 101), then
# evaluate the checkpoint on all 10 eval inputs. Comparison rows: the FOX
# ordered-train grid already in results/recsys_audit/{training,metrics}.csv.
#
# Run order: seed 17 for both models first (earliest complete comparison),
# then seeds 32/45. One training at a time (single GPU).
#
# Usage (ALWAYS inside a detached tmux):
#   tmux new-session -d -s recsys_shuftrain \
#     "bash /root/LLMSeq/scripts/recsys/local_shuffled_train.sh 2>&1 | tee -a /root/LLMSeq/logs/recsys/shuftrain_driver.log"

set -u
REPO=/root/LLMSeq
PY=/root/kip-venv/bin/python
DATA30=$REPO/data/recsys/30music
AUDIT=$DATA30/audit
LOGDIR=$REPO/logs/recsys
mkdir -p "$LOGDIR"
cd "$REPO/repro"

run_cell () {
  local MODEL=$1 SEED=$2
  local TAG="${MODEL}_shuffled_train_s101_s${SEED}"
  echo "=== [$(date -u +%H:%M:%S)] TRAIN $TAG ==="
  $PY -m src.recsys.train_grid --model "$MODEL" --seed "$SEED" \
      --split-dir "$DATA30/split" \
      --train-csv "$AUDIT/train_30Music_full_shuffle_s101.csv" \
      --val-csv "$AUDIT/validation_30Music_full_shuffle_s101.csv" \
      --mode-label shuffled_train_s101 \
      --out-dir "$REPO/checkpoints/recsys" \
      --results-csv "$REPO/results/recsys_audit/training.csv" \
      --batch-size 2048 --site local \
      > "$LOGDIR/train_${TAG}.log" 2>&1 \
    || { echo "TRAIN FAILED: $TAG (see $LOGDIR/train_${TAG}.log)"; return 1; }
  echo "=== [$(date -u +%H:%M:%S)] EVAL $TAG ==="
  $PY -m src.recsys.eval_grid \
      --checkpoint "$REPO/checkpoints/recsys/${TAG}.pt" \
      --audit-dir "$AUDIT" --split-dir "$DATA30/split" \
      --inputs all \
      --results-csv "$REPO/results/recsys_audit/metrics.csv" \
      --recs-dir "$REPO/results/recsys_audit/recs" --site local \
      > "$LOGDIR/eval_${TAG}.log" 2>&1 \
    || { echo "EVAL FAILED: $TAG (see $LOGDIR/eval_${TAG}.log)"; return 1; }
  echo "=== [$(date -u +%H:%M:%S)] DONE $TAG ==="
}

FAILURES=0
for CELL in SASRec:17 GRU4Rec:17 SASRec:32 GRU4Rec:32 SASRec:45 GRU4Rec:45; do
  run_cell "${CELL%%:*}" "${CELL##*:}" || FAILURES=$((FAILURES+1))
done
echo "=== ALL CELLS FINISHED (failures: $FAILURES) ==="
