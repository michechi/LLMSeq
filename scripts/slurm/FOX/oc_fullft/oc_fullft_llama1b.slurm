#!/bin/bash
#
# OC FULL-FT (1B) -- Does the OC decoder conclusion hold under full
# fine-tuning instead of LoRA?
#
# Llama-3.2-1B, OC NOISY (tricky_rnd, tag "9"), 100% training data, FULL
# fine-tuning. Recipe is the paper's as-run tag-9 letter recipe VERBATIM
# (scripts/slurm/wrapper_to_launch.sh defaults: bs 64, max_length 50,
# AdamW lr 2e-5, epochs<=20, patience 3 on val loss, fractions 1.0,
# seed 8888) -- the ONLY change is dropping --peft, so every parameter
# trains. Runs src/experiments/LLM_fraction_experiment.py, the exact paper
# pipeline; its results CSV (no "_lora" suffix in the filename = full FT)
# lands in results/oc_fullft/.
#
# TOKEN: meta-llama is gated. Export HF_TOKEN before sbatch or have a cached
# `huggingface-cli login`. Never echoed or logged.
#
# ~7-12 min/epoch expected on H200 (bf16, ~1B params) -> 20-epoch cap ~4 h.
#
# Submit with:
#   cd scripts/slurm/FOX/oc_fullft && mkdir -p logs
#   sbatch oc_fullft_llama1b.slurm          # seed override: OC_SEED=... sbatch ...
#
#SBATCH --job-name=oc_fullft_1b
#SBATCH --account=ec12
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/oc_fullft_1b_%j.out

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

MODEL_NAME="meta-llama/Llama-3.2-1B"
SEED="${OC_SEED:-8888}"

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
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
    --fractions 1.0
    # NO --peft (that is the experiment), NO --use_quantization

# Preserve the full training log (per-epoch metrics; merged stdout+stderr).
cp -f "$SLURM_SUBMIT_DIR/logs/oc_fullft_1b_${SLURM_JOB_ID}.out" \
      "$OUTPUT_DIR/fox_oc_fullft_llama1b_seed${SEED}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
