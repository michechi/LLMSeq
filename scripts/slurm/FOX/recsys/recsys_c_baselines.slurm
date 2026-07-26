#!/bin/bash
#
# RecSys audit (c) BASELINES -- MostPopular, ItemKNN (bag), Markov-1/2.
# GPU used only by ItemKNN scoring; independent of (a)/(b).
#
#SBATCH --job-name=recsys_c_baselines
#SBATCH --account=ec12
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH --output=logs/recsys_c_baselines_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

python -m src.recsys.baselines \
  --split-dir "$DATA30/split" --audit-dir "$DATA30/audit" \
  --results-csv "$REPO_ROOT/results/recsys_audit/metrics.csv" \
  --recs-dir "$REPO_ROOT/results/recsys_audit/recs" --site fox
