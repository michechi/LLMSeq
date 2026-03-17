#!/bin/bash
set -o errexit
set -o nounset

NAME=${1:?Usage: bash wrapper_to_launch.sh <run_name> [--model MODEL] [--seed SEED] [--gpus N] [--batch_size BS] [--max_length ML] [--number_to_use N] [--fractions F] [--lr LR] [--epochs E] [--patience P] [--time T] [--no-peft] [--quantization]}
shift

# Defaults (match current LLM_fraction_experiment.sh)
MODEL_NAME="meta-llama/Llama-3.2-1B"
SEED=8888
GPUS=1
BATCH_SIZE=64
MAX_LENGTH=50
NUMBER_TO_USE=9
FRACTIONS="1.0"
PEFT=true
QUANTIZATION=false
LR="2e-5"
EPOCHS=20
PATIENCE=3
TIME="12:00:00"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --max_length) MAX_LENGTH="$2"; shift 2;;
        --number_to_use) NUMBER_TO_USE="$2"; shift 2;;
        --fractions) FRACTIONS="$2"; shift 2;;
        --no-peft) PEFT=false; shift;;
        --quantization) QUANTIZATION=true; shift;;
        --lr) LR="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --patience) PATIENCE="$2"; shift 2;;
        --time) TIME="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

mkdir -p ./logs

sbatch \
    --job-name="LLM_${NAME}" \
    --output="./logs/LLM_${NAME}_%j.out" \
    --account=NN12048K \
    --time="${TIME}" \
    --partition=accel \
    --mem=90GB \
    --ntasks=1 \
    --nodes=1 \
    --gpus="${GPUS}" \
    --export="ALL,RUN_NAME=${NAME},EXP_MODEL=${MODEL_NAME},EXP_SEED=${SEED},EXP_BATCH_SIZE=${BATCH_SIZE},EXP_MAX_LENGTH=${MAX_LENGTH},EXP_NUMBER=${NUMBER_TO_USE},EXP_FRACTIONS=${FRACTIONS},EXP_PEFT=${PEFT},EXP_QUANTIZATION=${QUANTIZATION},EXP_LR=${LR},EXP_EPOCHS=${EPOCHS},EXP_PATIENCE=${PATIENCE}" \
    scripts/slurm/LLM_fraction_experiment.sh

echo "Submitted: LLM_${NAME} (model=${MODEL_NAME}, seed=${SEED}, gpus=${GPUS})"
