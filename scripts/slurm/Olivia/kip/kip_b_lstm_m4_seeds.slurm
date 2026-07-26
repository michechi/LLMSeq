#!/bin/bash
#
# KIP (b) -- LSTM on kip_m4, mode (a) ordered, NEW seeds 9570-9574, patience 5.
#
# Extends the local 3-seed patience-5 result (3/3 seeds at test_auc 1.0000)
# to 5 additional seeds for per-seed reporting in the rebuttal.
# 5 array tasks, one seed each.
#
# Submit with:
#   cd scripts/slurm/Olivia/kip && mkdir -p logs && sbatch kip_b_lstm_m4_seeds.slurm
#
#SBATCH --job-name=kip_b_lstm_m4
#SBATCH --account=NN12048K
#SBATCH --time=01:30:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=logs/kip_b_lstm_m4_%A_%a.out
#SBATCH --error=logs/kip_b_lstm_m4_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE ==============
export PERMANENT_CACHE="/cluster/work/projects/nn12048k/michechi/cache"
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
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
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,DATA_DIR=$DATA_DIR"
CONTAINER_ENV="$CONTAINER_ENV,LLMSEQ_ROOT=$REPO_ROOT"

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
echo "Container: $CONTAINER"

cd "$REPO_ROOT/repro"

srun apptainer exec --nv \
    --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
    --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k:rw \
    --bind /cluster/home/michechi:/cluster/home/michechi:rw \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.kip_gpu_training \
        --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
        --dl_patience 5 \
        --site olivia \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT_ROOT" \
        --results_csv "$RESULTS_CSV"

# Preserve the per-epoch val curve + full training log in the repo (tracked).
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/curve.json" \
      "$CURVES_DIR/olivia_${RUN_NAME}.curve.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_b_lstm_m4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
      "$CURVES_DIR/olivia_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
