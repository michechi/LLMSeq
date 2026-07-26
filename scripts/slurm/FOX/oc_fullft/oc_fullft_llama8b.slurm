#!/bin/bash
#
# OC FULL-FT (8B) -- Does the OC decoder conclusion hold under full
# fine-tuning instead of LoRA?
#
# Llama-3.1-8B (the paper's 8B decoder), OC NOISY (tricky_rnd, tag "9"),
# 100% training data, FULL fine-tuning. Recipe identical to the 1B script
# (paper as-run tag-9 recipe verbatim: bs 64, max_length 50, AdamW 2e-5,
# epochs<=20, patience 3, fractions 1.0, seed 8888); changes vs the paper's
# 8B runs: NO --peft (the experiment) and NO --use_quantization (4-bit was
# an A100-40GB memory workaround and full FT cannot train quantized
# weights; the H200's 141 GB makes it unnecessary).
#
# Memory: 8B in pure bf16 full FT = ~16G weights + ~16G grads + ~32G AdamW
# states + activations (bs 64 x 50 tokens) ~= 80-90 GB on-GPU -> fits the
# H200 NVL (141 GB) without gradient checkpointing.
#
# Wall time: FOX site limit is 24 h. At the expected ~1-1.5 h/epoch on H200
# that covers ~16-24 epochs; patience 3 on val loss typically stops near 10.
# CAVEAT: the paper pipeline has no mid-run checkpointing — a run killed at
# the wall is lost and restarts from scratch on resubmit. If a seed dies at
# the wall twice, tell the assistant; options then are a recorded deviation
# (e.g. --epochs 15) rather than silently losing compute.
#
# TOKEN: meta-llama is gated. Export HF_TOKEN before sbatch or have a cached
# `huggingface-cli login`. Never echoed or logged. ~16 GB model download on
# first use; set KIP_HF_CACHE to a persistent dir to avoid re-downloads.
#
# Submit with:
#   cd scripts/slurm/FOX/oc_fullft && mkdir -p logs
#   sbatch oc_fullft_llama8b.slurm            # all 3 seeds
#   sbatch --array=0 oc_fullft_llama8b.slurm  # just the paper-anchored seed 8888
#
#SBATCH --job-name=oc_fullft_8b
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --output=logs/oc_fullft_8b_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack: torch 2.6.0 + cuda 12.6 has H200 (sm_90) kernels; transformers
# 4.55.4 instead of the paper's 4.47.1 is a recorded, accepted deviation.
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
OUTPUT_DIR="$REPO_ROOT/results/oc_fullft"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

MODEL_NAME="meta-llama/Llama-3.1-8B"
# Task 0 = seed 8888, the paper's as-run wrapper default (the anchored
# comparison); 8889/8890 are extension seeds for per-seed reporting.
SEEDS=(8888 8889 8890)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL_NAME tag=9 (OC noisy) FULL-FT seed=$SEED bs=64 maxlen=50"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.LLM_fraction_experiment \
    --number_to_use 9 \
    --path_csv "$DATA_DIR/simulation/tested/" \
    --cache_dir "$HF_HOME" \
    --output_dir "$OUTPUT_DIR/" \
    --model_name "$MODEL_NAME" \
    --batch_size 64 \
    --max_length 50 \
    --epochs 20 \
    --patience 3 \
    --lr 2e-5 \
    --seed "$SEED" \
    --fractions 0.3
    # NO --peft (that is the experiment), NO --use_quantization

# Preserve the full training log (per-epoch metrics; merged stdout+stderr).
cp -f "$SLURM_SUBMIT_DIR/logs/oc_fullft_8b_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$OUTPUT_DIR/fox_oc_fullft_llama8b_seed${SEED}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
