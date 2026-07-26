#!/bin/bash
#
# KIP (d) PRIORITY -- LSTM on kip_m6, mode (a) ordered, seeds 9550-9552,
# patience 5. Submit this FIRST.
#
# The open question for the rebuttal: LSTM solves m=4 (test_auc 1.0000, 3/3
# seeds at patience 5); pairwise-precedence counts stop sufficing at m=6
# (XGBoost AUC ~0.502) while an MLP on precedence bits still reaches 1.000.
# Does the LSTM find the m=6 rule from raw sequences?
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs && sbatch kip_d_lstm_m6.slurm
#
#SBATCH --job-name=kip_d_lstm_m6
#SBATCH --account=ec12
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/kip_d_lstm_m6_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

# ============== GRID ==============
MODEL=LSTM
TAG=kip_m6
MODE=ordered
SEEDS=(9550 9551 9552)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED dl_patience=5"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    --dl_patience 5 \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

# Preserve the per-epoch val curve + full training log in the repo (tracked).
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_d_lstm_m6_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
