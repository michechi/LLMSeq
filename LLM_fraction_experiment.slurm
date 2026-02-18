#!/bin/bash
#SBATCH --job-name=LLM_fraction
#SBATCH --account=NN12048K
#SBATCH --time=24:00:00
#SBATCH --partition=accel
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

set -o errexit
set -o nounset

# ============== PARTE 1: SETUP CACHE E TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export HUGGING_FACE_HUB_TOKEN="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"
export PYTHONUNBUFFERED=1

echo "=== JOB CONFIGURATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Scratch Cache: $SCRATCH_CACHE"
echo "Free space in SCRATCH: $(df -h $SCRATCH | tail -1 | awk '{print $4}')"

# ============== PARTE 2: CARICA MODULI ==============
module purge
module load NRIS/GPU

if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "ERROR: Apptainer/Singularity not found. Trying to load module..."
    module load Apptainer
fi

# ============== PARTE 3: PREPARA DIRECTORY ==============
mkdir -p /cluster/work/projects/nn12048k/michechi/results/llm_fraction/

# ============== PARTE 4: COMANDO DI ESECUZIONE ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

echo "=== STARTING CONTAINER ==="
echo "Container: /cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"
echo "Environment: $CONTAINER_ENV"

# Run the LLM fraction experiment
srun apptainer exec --nv \
--bind "$SCRATCH_CACHE:$SCRATCH_CACHE" \
--bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k \
--env "$CONTAINER_ENV" \
/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif \
python3 -u ./LLM_fraction_experiment.py \
--number_to_use 9 \
--path_csv /cluster/home/michechi/MIMICIV/data/simulation/ \
--cache_dir "$SCRATCH_CACHE" \
--output_dir /cluster/work/projects/nn12048k/michechi/results/llm_fraction/ \
--model_name meta-llama/Llama-3.1-8B \
--tiny \
--tiny_type 1M \
--batch_size 16 \
--max_length 512 \
--epochs 20 \
--patience 3 \
--lr 2e-5 \
--seed 9550 \
--fractions "0.01,0.10,0.30,0.50,0.75,1.0"
