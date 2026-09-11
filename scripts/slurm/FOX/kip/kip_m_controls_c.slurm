#!/bin/bash
#
# KIP (m) COMPLETION -- full mode-(c) shuffled_eval grid for the pretrained
# families: NO training; each task loads THIS SITE's mode-(a) checkpoint for
# the same (model, tag, seed) from $CKPT_ROOT and evaluates it on the
# shuffled test split (expect chance). Minutes per task.
#
#   tasks 0-2:  BERT    kip_m4 seeds 9550-9552 (max_length 512, matching the
#               kip_c / kip_l ordered checkpoints; threshold reused from (a))
#   tasks 3-5:  BERT    kip_m6 seeds 9550-9552 (max_length 64, matching kip_e2)
#   tasks 6-8:  Llama1B kip_m4 seeds 9550-9552 (ckpts: kip_g + kip_l)
#   tasks 9-11: Llama1B kip_m6 seeds 9550-9552 (ckpts: kip_h)
#
# Checkpoint provenance: every cell except (BERT m4 9550) and (Llama1B m4
# 9550) has its FOX ordered checkpoint from the July runs; those two come
# from kip_l. A task fails fast (FileNotFoundError) if its checkpoint is
# missing -- submit AFTER kip_l finishes:
#
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   export HF_TOKEN=...   # Llama tasks; unless ~/.cache/huggingface/token exists
#   sbatch --dependency=afterok:<kip_l job id> kip_m_controls_c.slurm
#
#SBATCH --job-name=kip_m_ctrl_c
#SBATCH --account=ec12
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
#SBATCH --output=logs/kip_m_ctrl_c_%A_%a.out

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
#            model    tag     seed  bert_len
GRID=(
  "BERT    kip_m4 9550 512"
  "BERT    kip_m4 9551 512"
  "BERT    kip_m4 9552 512"
  "BERT    kip_m6 9550 64"
  "BERT    kip_m6 9551 64"
  "BERT    kip_m6 9552 64"
  "Llama1B kip_m4 9550 -"
  "Llama1B kip_m4 9551 -"
  "Llama1B kip_m4 9552 -"
  "Llama1B kip_m6 9550 -"
  "Llama1B kip_m6 9551 -"
  "Llama1B kip_m6 9552 -"
)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#GRID[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
read -r MODEL TAG SEED BLEN <<< "${GRID[$SLURM_ARRAY_TASK_ID]}"
MODE=shuffled_eval

# ============== HF TOKEN (Llama tasks only; never echoed or logged) ==============
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
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME"

MODEL_FLAGS=()
if [ "$MODEL" = "BERT" ]; then
    MODEL_FLAGS=(--bert_max_length "$BLEN")
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

echo "=== JOB FINISHED ==="
