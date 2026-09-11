#!/bin/bash
#
# KIP (p) OPT-IN -- Llama-3.1-8B FULL fine-tuning on KIP, three seeds, both
# tags. July's kip_i planned a single 8B task (seed 9550) that never landed a
# row; this array completes the 8B arm to full triplets, mirroring the OC
# full-FT protocol (does the decoder-at-chance conclusion survive 8B scale?).
#
#   tasks 0-2: kip_m4 seeds 9550-9552, fraction 1.0
#   tasks 3-5: kip_m6 seeds 9550-9552, fraction 1.0
#
# Cost note: ~85 GB VRAM (bf16 full FT, bs 64 x 50 tok -- comfortable on the
# H200 NVL's 141 GB) and roughly 4-8 h per task; submit only if the ec12
# GPU budget allows ~30-50 GPU-hours. Rows append to
# results/kip_training_fullft.csv; checkpoints under checkpoints/kip_fullft/.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # unless ~/.cache/huggingface/token exists
#   sbatch kip_p_llama8b_fullft.slurm
#
#SBATCH --job-name=kip_p_8b_fullft
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --output=logs/kip_p_8b_fullft_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack (torch 2.6.0/cu12.6): the only stack with H200 sm_90 kernels;
# transformers 4.55.4 vs the paper's 4.47.1 is a recorded, accepted deviation.
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

# ============== GRID ==============
GRID=(
  "kip_m4 9550"
  "kip_m4 9551"
  "kip_m4 9552"
  "kip_m6 9550"
  "kip_m6 9551"
  "kip_m6 9552"
)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#GRID[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
read -r TAG SEED <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"
MODEL=Llama8B
MODE=ordered

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training_fullft.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip_fullft}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED FULL-FT fraction=1.0 maxlen=50 bs=64"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    --llm_full_ft --llm_fraction 1 \
    --llm_max_length 50 --llm_batch_size 64 \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/fox_fullft_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_p_8b_fullft_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_fullft_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
