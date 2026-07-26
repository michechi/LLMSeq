#!/bin/bash
#
# KIP (c) -- BERT on kip_m4, mode (a) ordered, seeds 9551 + 9552.
#
# SEED NOTE: the task list says "seeds 2-3", read as the 2nd and 3rd seeds of
# the canonical triplet (9550 9551 9552); seed 9550 already ran locally
# (test_auc 0.5002, chance, at full fidelity max_length=512, 7.3 h on A100).
# If literal seeds 2 and 3 were intended, edit SEEDS below before submitting.
#
# max_length=512 matches the local seed-9550 row so the three seeds form one
# homogeneous triplet (paper-appendix fidelity). That makes runs LONG:
# ~45 min/epoch, early stop typically ~10 epochs, hard cap 20 -> 20 h wall.
#
# Requires bert-base-uncased pre-fetched into $PERMANENT_CACHE from the login
# node (RUNBOOK step 4): jobs run with HF_HUB_OFFLINE=1, no token needed.
#
# Submit with:
#   cd scripts/slurm/Olivia/kip && mkdir -p logs && sbatch kip_c_bert_m4.slurm
#
#SBATCH --job-name=kip_c_bert_m4
#SBATCH --account=NN12048K
#SBATCH --time=20:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-1
#SBATCH --output=logs/kip_c_bert_m4_%A_%a.out
#SBATCH --error=logs/kip_c_bert_m4_%A_%a.err

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
MODEL=BERT
TAG=kip_m4
MODE=ordered
SEEDS=(9551 9552)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED bert_max_length=512"
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
        --bert_max_length 512 \
        --site olivia \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT_ROOT" \
        --results_csv "$RESULTS_CSV"

# BERT has no curve.json (per-epoch metrics are in the log); keep log + meta.
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/olivia_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_c_bert_m4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" \
      "$CURVES_DIR/olivia_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
