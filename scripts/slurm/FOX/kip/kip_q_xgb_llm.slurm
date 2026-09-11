#!/bin/bash
#
# KIP (q) COMPLETION -- the paper's fourth XGBoost encoding on KIP:
# Llama-3.1-8B narrative-prompt embeddings -> XGBoost, via
# repro/src/experiments/XGBoost_fraction_experiment.py --run_llm.
# (The other three encodings -- basic / categorical / tfidf -- completed
# locally on CPU, 2026-09-10: all 216 cells at chance.)
#
# 2-GPU job (site working set): ONE job; kip_m4 runs on GPU 0 and kip_m6 on
# GPU 1 concurrently, each looping seeds 9550 -> 9551 -> 9552 SEQUENTIALLY.
# Seeds are sequential per tag on purpose: the embedding cache is per-tag
# (val/test and the fraction-1.0 train embeddings are seed-invariant and
# reused; sub-fraction caches are seed-keyed after the 2026-09-11 script
# fix), and concurrent writers to one cache directory would race. The two
# tags have disjoint caches, so they parallelize safely.
# Requires the 2026-09-11 script fix (seed-keyed subset cache + explicit
# val/test set_type -- the old length heuristic collided val and test for
# 50K/50K splits): pull the branch before submitting.
#
# xgboost is NOT in the NLPL 2024a stack (the sanity suite's two known FAILs)
# -- this job pip-installs it --user on first run (network is available on
# FOX compute nodes). Embeddings (~40 GB per tag) go under $SCRATCH -- NOT
# under the repo, so nothing large can be committed. Only the small result
# CSVs land in results/kip_xgb/.
#
# Cost: embedding extraction dominates, roughly 4-8 h per tag.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # Llama-3.1-8B is gated
#   sbatch kip_q_xgb_llm.slurm
#
#SBATCH --job-name=kip_q_xgb_llm
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:2
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/kip_q_xgb_llm_%j.out

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

# xgboost: missing from the stack; user-site install (recorded deviation).
python3 -c "import xgboost" 2>/dev/null || \
    python3 -m pip install --user --quiet xgboost
python3 -c "import xgboost; print('xgboost', xgboost.__version__)"

# ============== HF TOKEN (gated model; never echoed or logged) ==============
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
: "${HF_TOKEN:?ERROR: meta-llama is gated -- export HF_TOKEN before sbatch, or huggingface-cli login first}"
echo "HF token present (not shown)."

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
# Embeddings are ~40 GB/tag -- refuse to fall back to node-local /tmp.
: "${SCRATCH:?SCRATCH must be set (embeddings are ~40 GB per tag; do not use /tmp)}"
mkdir -p "$HF_HOME"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

run_tag() {  # $1 = tag, $2 = gpu slot (0|1)
    local TAG=$1 SLOT=$2 SEED
    # trailing slashes REQUIRED: the script concatenates dir + filename raw
    local EMB_DIR="${SCRATCH}/kip_xgb_embeddings/${TAG}/"
    mkdir -p "$EMB_DIR"
    for SEED in 9550 9551 9552; do
        local OUT_DIR="$REPO_ROOT/results/kip_xgb/${TAG}_ordered_llm_${SEED}"
        if [ -s "$OUT_DIR/xgb_fraction_experiment_results.csv" ]; then
            echo "$TAG seed $SEED already done -- skipping."
            continue
        fi
        mkdir -p "$OUT_DIR"
        local LOG="$SLURM_SUBMIT_DIR/logs/kip_q_xgb_llm_${SLURM_JOB_ID}_${TAG}_${SEED}.log"
        echo "$TAG seed $SEED START $(date -u +%FT%TZ) on GPU slot $SLOT (device ${GPUS[$SLOT]:-$SLOT})"
        CUDA_VISIBLE_DEVICES="${GPUS[$SLOT]:-$SLOT}" \
        python3 -u -m src.experiments.XGBoost_fraction_experiment \
            --csv_to_use "$TAG" \
            --path_csv "$DATA_DIR/simulation/tested/" \
            --output_dir "$OUT_DIR" \
            --embedding_dir "$EMB_DIR" \
            --cache_dir "$HF_HOME" \
            --model_name meta-llama/Llama-3.1-8B \
            --batch_size 64 \
            --seed "$SEED" \
            --run_llm >"$LOG" 2>&1
        echo "$TAG seed $SEED DONE $(date -u +%FT%TZ)"
    done
}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Node: $(hostname)"

cd "$REPO_ROOT/repro"

RC=0
run_tag kip_m4 0 & P0=$!
run_tag kip_m6 1 & P1=$!
wait "$P0" || RC=1
wait "$P1" || RC=1

echo "=== JOB FINISHED (rc=$RC) ==="
exit "$RC"
