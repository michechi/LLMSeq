#!/bin/bash
#
# KIP (l) COMPLETION -- FOX re-runs of the two seed-9550 mode-(a) cells that
# only ever trained locally, so that (1) kip_m's shuffled_eval tasks have a
# SAME-SITE mode-(a) checkpoint for every (model, seed), and (2) the ledger
# gains two more cross-site replicates (the site column distinguishes rows;
# both models are expected at chance, matching the local rows).
#
#   task 0: BERT   kip_m4 ordered seed 9550, max_length 512 (paper fidelity,
#           consistent with the FOX kip_c runs for seeds 9551/9552; ~2.5 h)
#   task 1: Llama1B LoRA kip_m4 ordered seed 9550 (Block-D as-run recipe;
#           needs the gated-model HF token; ~2-3 h)
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # unless ~/.cache/huggingface/token exists
#   sbatch kip_l_ordered_9550.slurm
# then submit kip_m with --dependency=afterok:<this job id>.
#
#SBATCH --job-name=kip_l_ord9550
#SBATCH --account=ec12
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --output=logs/kip_l_ord9550_%A_%a.out

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

# ============== GRID ==============
GRID=(
  "BERT    kip_m4 9550"
  "Llama1B kip_m4 9550"
)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#GRID[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
read -r MODEL TAG SEED <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"
MODE=ordered

# ============== HF TOKEN (Llama task only; never echoed or logged) ==============
if [ "$MODEL" = "Llama1B" ]; then
    if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
        export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    fi
    : "${HF_TOKEN:?ERROR: meta-llama is gated -- export HF_TOKEN before sbatch, or huggingface-cli login first}"
    echo "HF token present (not shown)."
fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

MODEL_FLAGS=()
if [ "$MODEL" = "BERT" ]; then
    MODEL_FLAGS=(--bert_max_length 512)   # ordered m4 = paper fidelity length
elif [ "$MODEL" = "Llama1B" ]; then
    MODEL_FLAGS=(--llm_max_length 50 --llm_batch_size 64)
fi

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED flags=${MODEL_FLAGS[*]:-none}"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    "${MODEL_FLAGS[@]}" \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
# BERT/Llama write meta.json (threshold + val stats), not curve.json
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_l_ord9550_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
