#!/bin/bash
#
# KIP (l) COMPLETION -- FOX re-runs of the two seed-9550 mode-(a) cells that
# only ever trained locally, so that (1) kip_m's shuffled_eval tasks have a
# SAME-SITE mode-(a) checkpoint for every (model, seed), and (2) the ledger
# gains two more cross-site replicates (the site column distinguishes rows;
# both models are expected at chance, matching the local rows).
#
#   cell 0: BERT   kip_m4 ordered seed 9550, max_length 512 (paper fidelity,
#           consistent with the FOX kip_c runs for seeds 9551/9552; ~2.5 h)
#   cell 1: Llama1B LoRA kip_m4 ordered seed 9550 (Block-D as-run recipe;
#           needs the gated-model HF token; ~2-3 h)
#
# 2-GPU job (site working set): ONE job, both cells concurrently, one per
# GPU. Per-cell driver logs land in logs/ as ..._cell<N>.log.
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
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:2
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/kip_l_ord9550_%j.out

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

# ============== HF TOKEN (Llama cell only; never echoed or logged) ==============
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ============== GRID ==============
GRID=(
  "BERT    kip_m4 9550"
  "Llama1B kip_m4 9550"
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
    local IDX=$1 SLOT=$2 MODEL TAG SEED
    read -r MODEL TAG SEED <<< "${GRID[$IDX]}"
    local -a MODEL_FLAGS=()
    if [ "$MODEL" = "BERT" ]; then
        MODEL_FLAGS=(--bert_max_length 512)   # ordered m4 = paper fidelity length
    else
        MODEL_FLAGS=(--llm_max_length 50 --llm_batch_size 64)
        if [ -z "${HF_TOKEN:-}" ]; then
            echo "cell $IDX ($MODEL $TAG $SEED): HF_TOKEN missing -- FAILED"
            return 1
        fi
    fi
    local LOG="$SLURM_SUBMIT_DIR/logs/kip_l_ord9550_${SLURM_JOB_ID}_cell${IDX}.log"
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
echo "Job: $SLURM_JOB_ID / Node: $(hostname)"

cd "$REPO_ROOT/repro"

RC=0
run_cell 0 0 & P0=$!
run_cell 1 1 & P1=$!
wait "$P0" || RC=1
wait "$P1" || RC=1

echo "=== JOB FINISHED (rc=$RC) ==="
exit "$RC"
