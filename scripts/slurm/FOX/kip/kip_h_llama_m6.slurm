#!/bin/bash
#
# KIP (h) -- Llama-3.2-1B + LoRA on kip_m6, mode (a) ordered, seeds 9550-9552.
#
# Completes the per-model m6 coverage (LSTM kip_d, Transformer kip_e1, BERT
# kip_e2). Same AS-RUN Block D recipe as kip_g: LoRA, AdamW 2e-5, bs 64,
# max_length 50 (KIP prompts are ~26 Llama tokens regardless of m),
# epochs<=20, patience 3 on val loss. ~7-10 min/epoch expected on H200,
# 20-epoch cap ~3.5 h -> 8 h wall.
#
# TOKEN: meta-llama is gated. Export HF_TOKEN before sbatch, or have a cached
# `huggingface-cli login` at ~/.cache/huggingface/token -- the script falls
# back to it. Never echoed or logged.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # unless ~/.cache/huggingface/token exists
#   sbatch kip_h_llama_m6.slurm
#
#SBATCH --job-name=kip_h_llama_m6
#SBATCH --account=ec12
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/kip_h_llama_m6_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack: torch 2.6.0 + cuda 12.6 has H200 (sm_90) kernels. The 2022b
# stack (torch 2.1.2 + cuda 12.0) predates the H200 nodes and dies with
# "CUDA error: no kernel image is available" on them. transformers 4.55.4
# instead of the paper's 4.47.1 is a recorded, accepted deviation.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
module load nlpl-bitsandbytes/0.46.1-foss-2024a-Python-3.12.3

# ============== HF TOKEN (gated model; never echoed or logged) ==============
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
: "${HF_TOKEN:?ERROR: meta-llama is gated -- export HF_TOKEN before sbatch, or huggingface-cli login first}"
echo "HF token present (not shown)."

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

# ============== GRID ==============
MODEL=Llama1B
TAG=kip_m6
MODE=ordered
SEEDS=(9550 9551 9552)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED maxlen=50 bs=64"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    --llm_max_length 50 --llm_batch_size 64 \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

# Llama has no curve.json (per-epoch metrics are in the log); keep log + meta.
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_h_llama_m6_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
