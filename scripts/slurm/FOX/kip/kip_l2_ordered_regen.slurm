#!/bin/bash
#
# KIP (l2) -- FRESH-SITE ordered-checkpoint regeneration: use INSTEAD of
# kip_l when the July FOX checkpoints under $REPO_ROOT/checkpoints/kip are
# gone (e.g. a new checkout / cleaned scratch, as on 2026-09-14). kip_m's
# mode-(c) shuffled_eval cells load THIS SITE's mode-(a) checkpoint for
# every (model, tag, seed); this array retrains all 12 of them.
#
#   cells 0-2:  BERT    kip_m4 seeds 9550-9552 (max_length 512, ~2.5 h each)
#   cells 3-5:  Llama1B kip_m4 seeds 9550-9552 (LoRA, ~2-3 h each)
#   cells 6-8:  Llama1B kip_m6 seeds 9550-9552 (LoRA, ~2-3.5 h each)
#   cells 9-11: BERT    kip_m6 seeds 9550-9552 (max_length 64, ~0.5-1 h each)
#
# 2-GPU jobs (site working set): 12 cells -> 6 array tasks, two cells per
# task, one per GPU. Rows append to results/kip_training.csv as fresh
# site=fox re-runs (the July rows persist; slurm_job_id/timestamp
# distinguish them -- expect the same chance-level numbers, which doubles
# as a replication check). Then submit kip_m with
# --dependency=afterok:<this job id>.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export REPO_ROOT=$HOME/LLMSeq     # if the checkout is not ~/MIMICIV
#   export HF_TOKEN=...               # Llama cells; BERT cells run without
#   sbatch kip_l2_ordered_regen.slurm
#
#SBATCH --job-name=kip_l2_regen
#SBATCH --account=ec12
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:2
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --output=logs/kip_l2_regen_%A_%a.out

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

# ============== HF TOKEN (Llama cells only; never echoed or logged) ==============
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ============== GRID ==============
#            model    tag     seed  bert_len
GRID=(
  "BERT    kip_m4 9550 512"
  "BERT    kip_m4 9551 512"
  "BERT    kip_m4 9552 512"
  "Llama1B kip_m4 9550 -"
  "Llama1B kip_m4 9551 -"
  "Llama1B kip_m4 9552 -"
  "Llama1B kip_m6 9550 -"
  "Llama1B kip_m6 9551 -"
  "Llama1B kip_m6 9552 -"
  "BERT    kip_m6 9550 64"
  "BERT    kip_m6 9551 64"
  "BERT    kip_m6 9552 64"
)
MODE=ordered

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

run_cell() {  # $1 = grid index, $2 = gpu slot (0|1)
    local IDX=$1 SLOT=$2 MODEL TAG SEED BLEN
    read -r MODEL TAG SEED BLEN <<< "${GRID[$IDX]}"
    local -a MODEL_FLAGS=()
    if [ "$MODEL" = "BERT" ]; then
        MODEL_FLAGS=(--bert_max_length "$BLEN")
    else
        MODEL_FLAGS=(--llm_max_length 50 --llm_batch_size 64)
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "cell $IDX ($MODEL $TAG $SEED): HF_TOKEN missing -- FAILED"
            return 1
        fi
    fi
    local LOG="$SLURM_SUBMIT_DIR/logs/kip_l2_regen_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cell${IDX}.log"
    echo "cell $IDX: $MODEL $TAG $MODE $SEED on GPU slot $SLOT (device ${GPUS[$SLOT]:-$SLOT})"
    CUDA_VISIBLE_DEVICES="${GPUS[$SLOT]:-$SLOT}" \
    python3 -u -m src.experiments.kip_gpu_training \
        --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
        "${MODEL_FLAGS[@]}" \
        --site fox \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT_ROOT" \
        --results_csv "$RESULTS_CSV" >"$LOG" 2>&1
    local RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
    # BERT/Llama write meta.json (threshold + val stats), not curve.json
    cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
          "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
    cp -f "$LOG" "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true
    echo "cell $IDX: $MODEL $TAG $MODE $SEED DONE"
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
