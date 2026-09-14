#!/bin/bash
#
# KIP (b) -- LSTM on kip_m4, mode (a) ordered, NEW seeds 9570-9574, patience 5.
#
# Extends the local 3-seed patience-5 result (3/3 seeds at test_auc 1.0000)
# to 5 additional seeds for per-seed reporting in the rebuttal.
# 5 array tasks, one seed each.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs && sbatch kip_b_lstm_m4_seeds.slurm
#
#SBATCH --job-name=kip_b_lstm_m4
#SBATCH --account=ec12
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=logs/kip_b_lstm_m4_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack: torch 2.6.0 + cuda 12.6 has H200 (sm_90) kernels. The 2022b
# stack (torch 2.1.2 + cuda 12.0) predates the H200 nodes and dies with
# "CUDA error: no kernel image is available" on them. transformers 4.55.4
# instead of the paper's 4.47.1 is a recorded, accepted deviation.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
module load nlpl-bitsandbytes/0.46.1-foss-2024a-Python-3.12.3

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
TAG=kip_m4
MODE=ordered
SEEDS=(9570 9571 9572 9573 9574)

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
cp -f "$SLURM_SUBMIT_DIR/logs/kip_b_lstm_m4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
