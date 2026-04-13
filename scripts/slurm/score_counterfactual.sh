#!/bin/bash
#SBATCH --account=NN12048K
#SBATCH --time=0-12:00:00
#SBATCH --partition=accel
#SBATCH --mem=90GB
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1

set -o errexit
set -o nounset

# ──────────────────────────────────────────────────────────────────────
# Score counterfactual pairs — trains from scratch, then scores 47K pairs
#
# Usage (submit one per model):
#   RUN_NAME=cf_DL        sbatch scripts/slurm/score_counterfactual.sh DL
#   RUN_NAME=cf_BERT      sbatch scripts/slurm/score_counterfactual.sh BERT
#   RUN_NAME=cf_Llama1B   sbatch scripts/slurm/score_counterfactual.sh Llama1B
#   RUN_NAME=cf_Llama8B   sbatch scripts/slurm/score_counterfactual.sh Llama8B
#   RUN_NAME=cf_Qwen4B    sbatch scripts/slurm/score_counterfactual.sh Qwen4B
#   RUN_NAME=cf_Qwen14B   sbatch scripts/slurm/score_counterfactual.sh Qwen14B
# ──────────────────────────────────────────────────────────────────────

MODE=${1:-DL}
echo "=== RUN: ${RUN_NAME:?ERROR: RUN_NAME not set} — mode: $MODE ==="

# ============== PATHS ==============
DATA_PATH="/cluster/home/michechi/MIMICIV/data/simulation/tested/"
PAIRS_PATH="paper_tables/counterfactual_pairs_tricky_rnd.csv"
OUTPUT_DIR="/cluster/work/projects/nn12048k/michechi/results/counterfactual/"
SEED=9950

# ============== CACHE & PROXY ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE"

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

# ============== MODULES ==============
module purge
module load NRIS/GPU
if ! command -v apptainer &> /dev/null; then
    module load Apptainer
fi

mkdir -p "$OUTPUT_DIR"

# ============== CONTAINER ENV ==============
CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"

CONTAINER="/cluster/work/support/container/pytorch_nvidia_24.12_extended.sif"
BINDS="--bind $SCRATCH_CACHE:$SCRATCH_CACHE --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k --bind /cluster/home/michechi:/cluster/home/michechi"

# ============== BUILD COMMAND ==============
case "$MODE" in
    DL)
        echo "Mode: DL baselines (LSTM, Transformer, RNNTransformer)"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type DL \
            --models LSTM,Transformer,RNNTransformer \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    BERT)
        echo "Mode: BERT"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type LLM \
            --model_name google-bert/bert-base-uncased \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    Llama1B)
        echo "Mode: Llama-3.2-1B (LoRA)"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type LLM \
            --model_name meta-llama/Llama-3.2-1B \
            --peft \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    Llama8B)
        echo "Mode: Llama-3.1-8B (LoRA + 4bit)"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type LLM \
            --model_name meta-llama/Llama-3.1-8B \
            --peft --use_quantization \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    Qwen4B)
        echo "Mode: Qwen3-4B-Think (LoRA)"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type LLM \
            --model_name Qwen/Qwen3-4B \
            --peft \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    Qwen14B)
        echo "Mode: Qwen2.5-14B (LoRA + 4bit)"
        CMD="python3 -u ./src/experiments/score_counterfactual_pairs.py \
            --model_type LLM \
            --model_name Qwen/Qwen2.5-14B \
            --peft --use_quantization \
            --number_to_use 9 \
            --path_csv $DATA_PATH \
            --pairs_path $PAIRS_PATH \
            --output_dir $OUTPUT_DIR \
            --seed $SEED"
        ;;
    *)
        echo "Unknown mode: $MODE. Use: DL, BERT, Llama1B, Llama8B, Qwen4B, Qwen14B"
        exit 1
        ;;
esac

echo "=== STARTING ==="
srun apptainer exec --nv \
    $BINDS \
    --env "$CONTAINER_ENV" \
    $CONTAINER \
    $CMD
