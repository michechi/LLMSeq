#!/bin/bash
#
# RecSys audit (b) EVAL -- each trained checkpoint on ordered + 9 variants.
# Same array mapping as recsys_a_train. Submit AFTER (a) finishes:
#   sbatch --dependency=afterok:<jobid_of_a> recsys_b_eval.slurm
#
#SBATCH --job-name=recsys_b_eval
#SBATCH --account=ec12
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8
#SBATCH --output=logs/recsys_b_eval_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

MODELS=(SASRec GRU4Rec BERT4Rec)
SEEDS=(17 32 45)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

python -m src.recsys.eval_grid \
  --checkpoint "$REPO_ROOT/checkpoints/recsys/${MODEL}_ordered_s${SEED}.pt" \
  --audit-dir "$DATA30/audit" --split-dir "$DATA30/split" \
  --inputs all \
  --results-csv "$REPO_ROOT/results/recsys_audit/metrics.csv" \
  --recs-dir "$REPO_ROOT/results/recsys_audit/recs" --site fox
