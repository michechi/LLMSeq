#!/bin/bash
# ==============================================================
# MIMIC Ordered vs Shuffled Experiment - Parametric SLURM script
# ==============================================================
# Submit via submit_all_mimic.sh (which sets --job-name, --time, --mem
# and --export on the sbatch command line), or manually:
#
#   sbatch --job-name=mimic_transformer_ordered \
#          --output=./logs/mimic_transformer_ordered_%j.out \
#          --time=04:00:00 --mem=32GB \
#          --export=ALL,MODEL=transformer,DATA=ordered,EXTRA_ARGS="" \
#          scripts/slurm/mimic_experiment.slurm
# ==============================================================

#SBATCH --account=NN12048K
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

set -o errexit
set -o nounset

# Validate required variables
: "${MODEL:?ERROR: MODEL not set (transformer|lstm|bilstm|bert)}"
: "${DATA:?ERROR: DATA not set (ordered|shuffled)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ============== SETUP CACHE & TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export HUGGING_FACE_HUB_TOKEN="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"

echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: ${MODEL}"
echo "Data: ${DATA}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Scratch Cache: $SCRATCH_CACHE"
echo "Free space in SCRATCH: $(df -h $SCRATCH | tail -1 | awk '{print $4}')"

# ============== LOAD MODULES ==============
module purge
module load NRIS/GPU

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PREPARE DIRECTORIES ==============
PROJECT_DIR="/cluster/home/michechi/MIMICIV"
CONTAINER="/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"

mkdir -p "${PROJECT_DIR}/results/mimic"
mkdir -p "${PROJECT_DIR}/logs"

# ============== CONTAINER ENVIRONMENT ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

echo "=== STARTING CONTAINER ==="
echo "Container: ${CONTAINER}"

# ============== RUN EXPERIMENT ==============
srun apptainer exec --nv \
    --bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
    --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --pwd "${PROJECT_DIR}" \
    --env "$CONTAINER_ENV" \
    "${CONTAINER}" \
    python3 -u src/mimic/train_mimic.py \
        --model "${MODEL}" \
        --data "${DATA}" \
        ${EXTRA_ARGS}

echo "=== JOB COMPLETE ==="
