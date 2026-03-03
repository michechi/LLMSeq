#!/bin/bash
#SBATCH --job-name=test_speed
#SBATCH --account=NN12048K
#SBATCH --time=20:00:00
#SBATCH --partition=accel
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
set -o errexit
set -o nounset

# ============== PARTE 1: SETUP CACHE E TOKEN ==============
# 1. Cache PERMANENTE su /cluster/work (necessita bind read-write)
export PERMANENT_CACHE="/cluster/work/projects/nn12048k/michechi/cache"

# 2. Variabili d'ambiente per Hugging Face
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"

# 3. TOKEN DI AUTENTICAZIONE
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN environment variable before submitting}"

# 4. Log per debugging
echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Permanent Cache: $PERMANENT_CACHE"

# ============== PARTE 2: CARICA MODULI ==============
module purge
module load NRIS/GPU

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PARTE 3: COMANDO DI ESECUZIONE ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"

echo "=== STARTING CONTAINER ==="
echo "Container: /cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"
echo "Environment: $CONTAINER_ENV"

srun apptainer exec --nv \
--bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
--bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k:rw \
--env "$CONTAINER_ENV" \
/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif \
python3 -u ./Finetuning_FOX_time_simulation_tiny.py \
--cache_dir "$PERMANENT_CACHE" \
        ${@}

# ============== PARTE 4: NESSUNA PULIZIA ==============
JOB_EXIT_CODE=$?
echo "=== JOB FINISHED ==="
echo "Exit code: $JOB_EXIT_CODE"
echo "Cache permanente mantenuta in: $PERMANENT_CACHE"