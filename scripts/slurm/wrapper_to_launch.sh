#!/bin/bash
set -o errexit
set -o nounset

NAME=${1:?Usage: bash submit.sh <run_name>}

mkdir -p ./logs

sbatch \
    --job-name="LLM_${NAME}" \
    --output="./logs/LLM_${NAME}_%j.out" \
    --account=NN12048K \
    --time=12:00:00 \
    --partition=accel \
    --mem=90GB \
    --ntasks=1 \
    --nodes=1 \
    --gpus=1 \
    --export="ALL,RUN_NAME=${NAME}" \
    scripts/slurm/LLM_fraction_experiment.sh

echo "Submitted: LLM_${NAME}"
