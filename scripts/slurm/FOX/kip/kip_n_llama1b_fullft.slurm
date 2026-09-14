#!/bin/bash
#
# KIP (n) COMPLETION -- Llama-3.2-1B FULL fine-tuning, the complete grid:
# every (tag, fraction, seed) full-FT cell without a row in
# results/kip_training_fullft.csv as of 2026-09-11 (seed 9550 ran kip_m4 at
# both fractions on FOX in July; the September local attempts at 9551/9552
# were interrupted by a node swap before any row landed).
#
#   cells 0-1: kip_m4 fraction 1.0, seeds 9551, 9552   (completes the triplet)
#   cells 2-3: kip_m4 fraction 0.3, seeds 9551, 9552   (completes the triplet)
#   cells 4-6: kip_m6 fraction 1.0, seeds 9550-9552    (never run anywhere)
#   cells 7-9: kip_m6 fraction 0.3, seeds 9550-9552    (never run anywhere)
#
# 2-GPU jobs (site working set): each array task takes TWO H200s and runs
# TWO consecutive cells concurrently, one per GPU (each training run stays
# single-GPU -- recipe fidelity). 10 cells -> 5 tasks, ~1.5-3.5 h per cell.
#
# Settings = Block-D as-run recipe minus --peft (bs 64, max_length 50,
# AdamW 2e-5, epochs<=20, patience 3 on val loss, bf16). Rows append to
# results/kip_training_fullft.csv (merge=union); checkpoints under
# checkpoints/kip_fullft/ (fraction 0.3 in a frac03/ subtree so the two
# fraction variants of one cell never collide while running concurrently).
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # unless ~/.cache/huggingface/token exists
#   sbatch kip_n_llama1b_fullft.slurm
#
#SBATCH --job-name=kip_n_1b_fullft
#SBATCH --account=ec12
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:2
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --output=logs/kip_n_1b_fullft_%A_%a.out

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
#      tag     fraction seed
GRID=(
  "kip_m4 1.0 9551"
  "kip_m4 1.0 9552"
  "kip_m4 0.3 9551"
  "kip_m4 0.3 9552"
  "kip_m6 1.0 9550"
  "kip_m6 1.0 9551"
  "kip_m6 1.0 9552"
  "kip_m6 0.3 9550"
  "kip_m6 0.3 9551"
  "kip_m6 0.3 9552"
)
MODEL=Llama1B
MODE=ordered

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training_fullft.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_BASE="${FULLFT_CKPT_ROOT:-$REPO_ROOT/checkpoints/kip_fullft}"   # distinct override name: a stale CKPT_ROOT from a controls shell must not aim full-FT at the LoRA tree
mkdir -p "$HF_HOME" "$CURVES_DIR"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

run_cell() {  # $1 = grid index, $2 = gpu slot (0|1)
    local IDX=$1 SLOT=$2 TAG FRACTION SEED
    read -r TAG FRACTION SEED <<< "${GRID[$IDX]}"
    # Checkpoint dirs are keyed by (model, tag, mode, seed) only -- the two
    # fraction variants of one cell would collide; 0.3 gets its own subtree.
    local CKPT="$CKPT_BASE"
    if [ "$FRACTION" != "1.0" ] && [ "$FRACTION" != "1" ]; then
        CKPT="$CKPT_BASE/frac03"
    fi
    mkdir -p "$CKPT"
    local LOG="$SLURM_SUBMIT_DIR/logs/kip_n_1b_fullft_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cell${IDX}.log"
    echo "cell $IDX: $MODEL $TAG $MODE $SEED fraction=$FRACTION on GPU slot $SLOT (device ${GPUS[$SLOT]:-$SLOT})"
    CUDA_VISIBLE_DEVICES="${GPUS[$SLOT]:-$SLOT}" \
    python3 -u -m src.experiments.kip_gpu_training \
        --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
        --llm_full_ft --llm_fraction "$FRACTION" \
        --llm_max_length 50 --llm_batch_size 64 \
        --site fox \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT" \
        --results_csv "$RESULTS_CSV" >"$LOG" 2>&1
    local RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
    cp -f "$CKPT/$RUN_NAME/meta.json" \
          "$CURVES_DIR/fox_fullft_${RUN_NAME}_f${FRACTION}.meta.json" 2>/dev/null || true
    cp -f "$LOG" "$CURVES_DIR/fox_fullft_${RUN_NAME}_f${FRACTION}.trainlog.txt" 2>/dev/null || true
    echo "cell $IDX: $MODEL $TAG $MODE $SEED fraction=$FRACTION DONE"
}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID / Node: $(hostname)"

cd "$REPO_ROOT/repro"

IDX0=$((SLURM_ARRAY_TASK_ID * 2))
IDX1=$((IDX0 + 1))
if [ "$IDX0" -ge "${#GRID[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi

RC=0
run_cell "$IDX0" 0 & P0=$!
P1=""
if [ "$IDX1" -lt "${#GRID[@]}" ]; then
    run_cell "$IDX1" 1 & P1=$!
fi
wait "$P0" || RC=1
if [ -n "$P1" ]; then wait "$P1" || RC=1; fi

echo "=== JOB FINISHED (rc=$RC) ==="
exit "$RC"
