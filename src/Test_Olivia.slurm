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
# 1. Cache su SCRATCH (filesystem scrivibile sui nodi)
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

# 2. Variabili d'ambiente per Hugging Face
export HF_HOME="$SCRATCH_CACHE"               # Sostituisce TRANSFORMERS_CACHE (deprecato)
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

# 3. TOKEN DI AUTENTICAZIONE (MODIFICA QUI!)
#    Sostituisci 'hf_tuo_TOKEN_qui' con il tuo token reale di Hugging Face
export HUGGING_FACE_HUB_TOKEN="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"

# 4. Log per debugging
echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Scratch Cache: $SCRATCH_CACHE"
echo "Free space in SCRATCH: $(df -h $SCRATCH | tail -1 | awk '{print $4}')"

# ============== PARTE 2: CARICA MODULI ==============
# Su Olivia, prova questi moduli nell'ordine
module purge
module load NRIS/GPU  # Oppure NRIS/CPU se GPU non funziona

# Verifica se Apptainer/Singularity è disponibile
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer  # Prova a caricare esplicitamente
fi

# ============== PARTE 3: COMANDO DI ESECUZIONE ==============
# Prepara le variabili d'ambiente da passare al container
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"

echo "=== STARTING CONTAINER ==="
echo "Container: /cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"
echo "Environment: $CONTAINER_ENV"

srun apptainer exec --nv \
    --bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
    --env "$CONTAINER_ENV" \
    /cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif \
    python3 -u ./Finetuning_FOX_time_simulation_tiny.py \
        --cache_dir "$SCRATCH_CACHE" \
        ${@}  # Passa altri argomenti dalla riga di comando

# ============== PARTE 4: PULIZIA (OPZIONALE) ==============
JOB_EXIT_CODE=$?
echo "=== JOB FINISHED ==="
echo "Exit code: $JOB_EXIT_CODE"

# Mantieni la cache se il job fallisce per debugging
if [[ $JOB_EXIT_CODE -eq 0 ]]; then
    echo "Cleaning up cache: $SCRATCH_CACHE"
    rm -rf "$SCRATCH_CACHE"
else
    echo "Job failed. Cache kept at: $SCRATCH_CACHE"
    echo "For debugging, you can check this directory."
fi