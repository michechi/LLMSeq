#!/bin/bash
#
# RECSYS AUDIT — Job 2 (OPTIONAL; DO NOT SUBMIT until the Job-1 table in
# results/recsys_audit/report_job1.md has been reviewed).
#
# GRU4Rec shuffled-train, 1 seed (17): train on
# audit/train_30Music_full_shuffle_s101.csv, early-stop on the matching
# shuffled validation file, evaluate on ordered + full_shuffle_s101 —
# protocol uniformity with the synthetic mode (b).
#
# Submit with:
#   cd scripts/slurm/FOX/recsys && mkdir -p logs && sbatch recsys_job2_shuffled_train.slurm
#
#SBATCH --job-name=recsys_job2
#SBATCH --account=ec12
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/recsys_job2_%j.out

set -o errexit
set -o nounset

# ============== MODULES (same 2024a stack as the KIP jobs) ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export KIP_SITE=fox
DATA="$REPO_ROOT/data/recsys/30music"
CKPT="$REPO_ROOT/checkpoints/recsys"
RESULTS="$REPO_ROOT/results/recsys_audit"
mkdir -p "$CKPT" "$RESULTS/recs"

python3 -c "import recbole" 2>/dev/null || pip install --user "recbole==1.2.0"

cd "$REPO_ROOT/repro"
python3 -m src.recsys.eval_loader --audit-dir "$DATA/audit" \
    --split-dir "$DATA/split" --check-all

CK="$CKPT/GRU4Rec_shuffled_train_s101_s17.pt"
if [ -f "$CK" ]; then
    echo "=== skip training (checkpoint exists) ==="
else
    echo "=== TRAIN GRU4Rec shuffled-train seed 17 ==="
    python3 -m src.recsys.train_grid --model GRU4Rec --seed 17 \
        --split-dir "$DATA/split" \
        --train-csv "$DATA/audit/train_30Music_full_shuffle_s101.csv" \
        --val-csv "$DATA/audit/validation_30Music_full_shuffle_s101.csv" \
        --mode-label shuffled_train_s101 \
        --out-dir "$CKPT" --results-csv "$RESULTS/training.csv" \
        --batch-size 512
fi

echo "=== EVAL (ordered + full_shuffle_s101) ==="
python3 -m src.recsys.eval_grid --checkpoint "$CK" \
    --audit-dir "$DATA/audit" --split-dir "$DATA/split" \
    --inputs full_shuffle_s101 \
    --results-csv "$RESULTS/metrics.csv" --recs-dir "$RESULTS/recs"

echo "=== JOB FINISHED ==="
