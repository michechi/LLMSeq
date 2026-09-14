#!/bin/bash
#
# KIP (a) ANCHOR -- LSTM on kip_m4, mode (a) ordered, seed 9550, patience 5.
#
# Cross-site validation run: the local A100 produced test_auc = 1.0000
# (epochs_done 4, results/kip_training_p5.csv row 1). This job must reproduce
# test_auc = 1.0000 on FOX before the other KIP arrays are trusted.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs && sbatch kip_a_anchor.slurm
#
#SBATCH --job-name=kip_a_anchor
#SBATCH --account=ec12
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH --output=logs/kip_a_anchor_%A_%a.out

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
SEEDS=(9550)

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
# FOX merges stdout+stderr into the single .out file, so it holds the python
# logging (recipe, per-epoch lines, DONE summary) too.
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_a_anchor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
