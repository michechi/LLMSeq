#!/bin/bash
#
# RECSYS AUDIT — Job 2 (OPTIONAL; DO NOT SUBMIT until the Job-1 table in
# results/recsys_audit/report_job1.md has been reviewed).
#
# GRU4Rec shuffled-train (mode-b uniformity block), 1 seed (17): train AND
# early-stop on the full_shuffle s101 files, then eval on full_shuffle_s101
# (the harness auto-adds the ordered input as the Jaccard/Δ reference).
#
# Submit with:
#   cd scripts/slurm/FOX/recsys && mkdir -p logs && sbatch recsys_job2_shuffled_train.slurm
#
#SBATCH --job-name=recsys_job2
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH --output=logs/recsys_job2_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

RESULTS="$REPO_ROOT/results/recsys_audit"
CKPT="$REPO_ROOT/checkpoints/recsys"
mkdir -p "$RESULTS/recs" "$CKPT"

CK="$CKPT/GRU4Rec_shuffled_train_s101_s17.pt"
if [ -f "$CK" ]; then
    echo "=== skip training (checkpoint exists) ==="
else
    python -m src.recsys.train_grid --model GRU4Rec --seed 17 \
        --split-dir "$DATA30/split" \
        --train-csv "$DATA30/audit/train_30Music_full_shuffle_s101.csv" \
        --val-csv "$DATA30/audit/validation_30Music_full_shuffle_s101.csv" \
        --mode-label shuffled_train_s101 \
        --out-dir "$CKPT" --results-csv "$RESULTS/training.csv" \
        --batch-size 2048 --site fox
fi

python -m src.recsys.eval_grid --checkpoint "$CK" \
    --audit-dir "$DATA30/audit" --split-dir "$DATA30/split" \
    --inputs full_shuffle_s101 \
    --results-csv "$RESULTS/metrics.csv" \
    --recs-dir "$RESULTS/recs" --site fox

echo "=== JOB FINISHED ==="
