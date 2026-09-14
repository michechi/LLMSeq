#!/bin/bash
#
# KIP (j) -- LSTM on kip_m4, mode (a) ordered, stratified fraction 0.3
# (120K rows), paper recipe (patience 5), seeds 9550-9552 (array 0-2).
#
# Purpose: fraction-matched control for the kip_i full-FT Llama arm, which
# trains on 120K rows while the LSTM headline (AUC 1.0, 9/9 seeds) used the
# full 400K. If the LSTM still hits 1.0 at 120K, the fraction mismatch
# becomes a controlled variable instead of a caveat; if a seed sits at
# chance, 120K is below the grokking threshold and the comparison must be
# reframed. Cheap either way (minutes per seed on H200).
#
# No HF token needed (pure-torch LSTM). Results join the full-FT comparison
# file results/kip_training_fullft.csv (merge=union); checkpoints (~2 MB
# each, no quota concern) in checkpoints/kip_fullft/.
#
# Requires the checkout to include the --dl_fraction driver support
# (committed together with this file).
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   sbatch kip_j_lstm_frac03.slurm
#
#SBATCH --job-name=kip_j_frac03
#SBATCH --account=ec12
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/kip_j_frac03_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack (H200/sm_90 kernels). The full kip_i module set, NOT a subset:
# the training driver imports the BERT and LLM experiment modules at top level
# regardless of which model runs, so peft (nlpl-llmtools) must be present even
# for a pure-torch LSTM job (verified: job 4012771 died with
# ModuleNotFoundError: peft when llmtools was omitted, 2026-07-27).
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
RESULTS_CSV="$REPO_ROOT/results/kip_training_fullft.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip_fullft}"
mkdir -p "$CKPT_ROOT" "$CURVES_DIR"

# ============== GRID ==============
SEEDS=(9550 9551 9552)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=LSTM
TAG=kip_m4
MODE=ordered

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED fraction=0.3 patience=5"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    --dl_fraction 0.3 --dl_patience 5 \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

# Ship the per-epoch curve + this log home via the tracked curves dir.
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/fox_frac03_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_j_frac03_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_frac03_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
