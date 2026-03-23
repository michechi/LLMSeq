#!/bin/bash
# ==============================================================
# Visit-level shuffle test — neural models (LSTM + Transformer)
# ==============================================================
# Submit:
#   sbatch scripts/slurm/visit_shuffle.slurm

#SBATCH --account=NN12048K
#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=visit_shuffle
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=./logs/visit_shuffle_%j.out

set -o errexit
set -o nounset

# ============== SETUP ==============
PROJECT_DIR="/cluster/home/michechi/MIMICIV"
CONTAINER="/cluster/work/support/container/pytorch_nvidia_24.12_extended.sif"

export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export PYTHONUNBUFFERED=1

echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="

# ============== LOAD MODULES ==============
module purge
module load NRIS/GPU

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    module load Apptainer
fi

mkdir -p "${PROJECT_DIR}/results/mimic"
mkdir -p "${PROJECT_DIR}/logs"

# ============== RUN ==============
srun apptainer exec --nv \
    --bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
    --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --pwd "${PROJECT_DIR}" \
    --env "PYTHONUNBUFFERED=1" \
    "${CONTAINER}" \
    python3 -u src/mimic/visit_shuffle_test.py --mode all

echo "=== JOB COMPLETE ==="
