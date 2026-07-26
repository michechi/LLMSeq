#!/bin/bash
#
# RecSys audit (d) OPTIONAL -- GRU4Rec shuffled-train (mode-b uniformity
# block): train AND early-stop on full_shuffle s101 files, then eval
# (harness auto-adds the ordered input as Jaccard/delta reference).
#
#SBATCH --job-name=recsys_d_gru_shufftrain
#SBATCH --account=ec12
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH --output=logs/recsys_d_gru_shufftrain_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

python -m src.recsys.train_grid --model GRU4Rec --seed 17 \
  --split-dir "$DATA30/split" \
  --train-csv "$DATA30/audit/train_30Music_full_shuffle_s101.csv" \
  --val-csv "$DATA30/audit/validation_30Music_full_shuffle_s101.csv" \
  --mode-label shuffled_train_s101 \
  --out-dir "$REPO_ROOT/checkpoints/recsys" \
  --results-csv "$REPO_ROOT/results/recsys_audit/training.csv" \
  --batch-size 2048 --site fox

python -m src.recsys.eval_grid \
  --checkpoint "$REPO_ROOT/checkpoints/recsys/GRU4Rec_shuffled_train_s101_s17.pt" \
  --audit-dir "$DATA30/audit" --split-dir "$DATA30/split" \
  --inputs full_shuffle_s101 \
  --results-csv "$REPO_ROOT/results/recsys_audit/metrics.csv" \
  --recs-dir "$REPO_ROOT/results/recsys_audit/recs" --site fox
