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
#   cd scripts/slurm/Olivia/kip && mkdir -p logs
#   sbatch --export=ALL,KIP_MODEL=LSTM        kip_f_m6_controls.slurm
#   sbatch --export=ALL,KIP_MODEL=Transformer kip_f_m6_controls.slurm
#   sbatch --export=ALL,KIP_MODEL=BERT        kip_f_m6_controls.slurm
#
# Mode (c) tasks fail fast (FileNotFoundError) if the mode-(a) checkpoint for
# that (model, seed) is not in $CKPT_ROOT -- run kip_d/e1/e2 first.
# BERT runs at max_length 64, checkpoint-consistent with kip_e2; DL runs at
# patience 5 (site convention). BERT needs the pre-fetched HF cache (RUNBOOK
# step 4); HF_HUB_OFFLINE=1 is set for all families (harmless for DL).
#
#SBATCH --job-name=kip_f_m6_ctrl
#SBATCH --account=NN12048K
#SBATCH --time=08:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-5
#SBATCH --output=logs/kip_f_m6_ctrl_%A_%a.out
#SBATCH --error=logs/kip_f_m6_ctrl_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE (offline HF: pre-fetched, ungated, no token) ==============
export PERMANENT_CACHE="/cluster/work/projects/nn12048k/michechi/cache"
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p logs "$PERMANENT_CACHE"

# ============== MODULES ==============
module purge
module load NRIS/GPU
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-/cluster/home/michechi/MIMICIV}"
DATA_DIR="$REPO_ROOT/data"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="/cluster/work/projects/nn12048k/michechi/results/kip/checkpoints"
CONTAINER="/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"
mkdir -p "$CKPT_ROOT" "$CURVES_DIR"

CONTAINER_ENV="HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_HUB_OFFLINE=1"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,DATA_DIR=$DATA_DIR"
CONTAINER_ENV="$CONTAINER_ENV,LLMSEQ_ROOT=$REPO_ROOT"

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
echo "Container: $CONTAINER"

cd "$REPO_ROOT/repro"

srun apptainer exec --nv \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k:rw \
    --bind /cluster/home/michechi:/cluster/home/michechi:rw \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.kip_gpu_training \
        --model "$KIP_MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
        "${EXTRA_FLAGS[@]}" \
        --site olivia \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT_ROOT" \
        --results_csv "$RESULTS_CSV"

# Preserve curves/log (curve.json exists only for DL shuffled_train runs).
RUN_NAME="${KIP_MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/olivia_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/olivia_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_f_m6_ctrl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
      "$CURVES_DIR/olivia_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
