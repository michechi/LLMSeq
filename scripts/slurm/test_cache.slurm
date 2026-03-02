#!/bin/bash
#SBATCH --job-name=cache_test
#SBATCH --account NN12048K
#SBATCH --time=00:05:00
#SBATCH --partition=accel --gres=gpu:1 --mem=32G
exec 2>&1
export SCRATCH_CACHE="$SCRATCH/hf_test_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"
export TRANSFORMERS_CACHE="$SCRATCH_CACHE"
module load NRIS/CPU hpc-container-wrapper
# Test: scarica SOLO il tokenizer (pochi MB)
apptainer exec --nv --bind $SCRATCH_CACHE:$SCRATCH_CACHE \
    --env "TRANSFORMERS_CACHE=$SCRATCH_CACHE" \
    /cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif \
    python -c "from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-70B', cache_dir='$SCRATCH_CACHE'); print('Tokenizer caricato. Cache in:', '$SCRATCH_CACHE')"