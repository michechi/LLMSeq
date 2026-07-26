#!/bin/bash
#
# KIP (f) FOLLOW-UP -- kip_m6 order controls, modes (b) + (c), for ONE model
# family whose mode-(a) runs converged (check test_auc in
# results/kip_training.csv first; only worth running if mode (a) beat chance).
#
#   mode (b) shuffled_train: fresh training on per-sequence-shuffled data with
#            original labels (expect chance -- labels are ~50% preserved).
#   mode (c) shuffled_eval:  NO training; loads THIS SITE's mode-(a) checkpoint
#            for the same (model, tag, seed) from $CKPT_ROOT and evaluates it
#            on the shuffled test split (expect chance).
#
# 6 array tasks: 2 modes x 3 seeds. Pick the family at submit time:
#
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   sbatch --export=ALL,KIP_MODEL=LSTM        kip_f_m6_controls.slurm
#   sbatch --export=ALL,KIP_MODEL=Transformer kip_f_m6_controls.slurm
#   sbatch --export=ALL,KIP_MODEL=BERT        kip_f_m6_controls.slurm
#
# Mode (c) tasks fail fast (FileNotFoundError) if the mode-(a) checkpoint for
# that (model, seed) is not in $CKPT_ROOT -- run kip_d/e1/e2 first, and keep
# CKPT_ROOT identical between the runs (default: $REPO_ROOT/checkpoints/kip).
# BERT runs at max_length 64, checkpoint-consistent with kip_e2; DL runs at
# patience 5 (site convention).
#
#SBATCH --job-name=kip_f_m6_ctrl
#SBATCH --account=ec12
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=logs/kip_f_m6_ctrl_%A_%a.out

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
KIP_MODEL="${KIP_MODEL:-LSTM}"
case "$KIP_MODEL" in
    LSTM|Transformer|BERT) ;;
    *) echo "ERROR: KIP_MODEL must be LSTM, Transformer or BERT (got '$KIP_MODEL')"; exit 2 ;;
esac
TAG=kip_m6
MODES=(shuffled_train shuffled_eval)
SEEDS=(9550 9551 9552)

N_SEEDS=${#SEEDS[@]}
TOTAL=$(( ${#MODES[@]} * N_SEEDS ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL -- exiting."
    exit 0
fi
MODE=${MODES[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}

if [ "$KIP_MODEL" = "BERT" ]; then
    EXTRA_FLAGS=(--bert_max_length 64)
else
    EXTRA_FLAGS=(--dl_patience 5)
fi

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$KIP_MODEL tag=$TAG mode=$MODE seed=$SEED extra=${EXTRA_FLAGS[*]}"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$KIP_MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    "${EXTRA_FLAGS[@]}" \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

# Preserve curves/log (curve.json exists only for DL shuffled_train runs).
RUN_NAME="${KIP_MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_f_m6_ctrl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
