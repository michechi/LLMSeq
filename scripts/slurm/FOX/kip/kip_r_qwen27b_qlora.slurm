#!/bin/bash
#
# KIP (r) EXTENSION -- Qwen3.8-27B QLoRA on KIP: the paper's big-decoder
# recipe (4-bit NF4 double-quant base, bf16 compute, LoRA r=8 a=16 on
# q/k/v/o, AdamW 2e-5, bs 64, max_length 50, epochs<=20, patience 3 on val
# loss) applied to a 27.8B 2026-generation decoder -- the scale data point
# above the 1B/8B arms.
#
# Qwen/Qwen3.8-27B specifics (all verified on CPU with transformers 5.14.1
# before this file was written):
#  - the repo is multimodal, but AutoModelForCausalLM (what the paper script
#    calls) maps its config to the TEXT-ONLY Qwen3_5ForCausalLM backbone;
#  - the backbone is hybrid linear/full attention (full attention every 4th
#    of 64 layers): the recipe's q/k/v/o LoRA targets exist on the 16
#    full-attention layers only -- RECORDED DEVIATION from dense-model LoRA;
#    peft attaches and trains cleanly (forward+backward verified);
#  - it needs transformers >= 5.8: this job pip-installs pinned user-site
#    packages OVER the 2024a module stack (recorded deviation) and asserts
#    the version before training. The FLA/causal-conv1d fast-path kernels
#    are NOT installed; at max_length 50 the torch fallback is fine.
#
#   cells 0-2: kip_m4 seeds 9550-9552
#   cells 3-5: kip_m6 seeds 9550-9552
#
# 2-GPU jobs (site working set): each array task takes TWO H200s and runs
# TWO consecutive cells concurrently, one per GPU (a 4-bit 27.8B run needs
# ~25-35 GB VRAM, so one H200 per cell has ample headroom). 6 cells ->
# 3 tasks, roughly 5-10 h per cell; the first cells also download ~56 GB of
# weights into the shared HF cache (hub file locks make that safe).
#
# Qwen is UNGATED -- no HF token required (exporting one is harmless).
#
# Mode override (kip_f pattern): default is ordered; after the ordered rows
# land, controls can reuse this file --
#   sbatch --export=ALL,KIP_MODE=shuffled_train kip_r_qwen27b_qlora.slurm
#   sbatch --export=ALL,KIP_MODE=shuffled_eval  kip_r_qwen27b_qlora.slurm
# (shuffled_eval loads this site's ordered checkpoints -- run ordered first.)
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs
#   sbatch kip_r_qwen27b_qlora.slurm
#
#SBATCH --job-name=kip_r_qwen27b
#SBATCH --account=ec12
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:2
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2
#SBATCH --output=logs/kip_r_qwen27b_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack (torch 2.6.0/cu12.6): the only stack with H200 sm_90 kernels;
# transformers 4.55.4 (Qwen3 supported since 4.51) + bitsandbytes 0.46.1
# for the 4-bit NF4 base. 4.55.4 vs the paper's 4.47.1 is a recorded,
# accepted deviation.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
module load nlpl-bitsandbytes/0.46.1-foss-2024a-Python-3.12.3

# ============== USER-SITE OVERRIDES (recorded deviation) ==============
# Qwen3.8-27B's qwen3_5 architecture needs transformers >= 5.8; the 2024a
# stack ships 4.55.4. Install the locally-verified pins into the user site
# (which shadows the module stack) and FAIL FAST if the override is inert
# (e.g. PYTHONNOUSERSITE) or incompatible.
python3 -m pip install --user --quiet "transformers==5.14.1" "peft==0.19.1" -U bitsandbytes
python3 - <<'PYCHECK'
import sys
import transformers, peft, bitsandbytes
from packaging.version import Version
assert Version(transformers.__version__) >= Version("5.8"),     f"user-site transformers override inert: got {transformers.__version__}"
from transformers import BitsAndBytesConfig
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_use_double_quant=True)
print("override OK:", transformers.__version__, peft.__version__, bitsandbytes.__version__)
PYCHECK

# ============== GRID ==============
GRID=(
  "kip_m4 9550"
  "kip_m4 9551"
  "kip_m4 9552"
  "kip_m6 9550"
  "kip_m6 9551"
  "kip_m6 9552"
)
MODEL=Qwen27B
MODE="${KIP_MODE:-ordered}"

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
    local IDX=$1 SLOT=$2 TAG SEED
    read -r TAG SEED <<< "${GRID[$IDX]}"
    local LOG="$SLURM_SUBMIT_DIR/logs/kip_r_qwen27b_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_cell${IDX}.log"
    echo "cell $IDX: $MODEL $TAG $MODE $SEED on GPU slot $SLOT (device ${GPUS[$SLOT]:-$SLOT})"
    CUDA_VISIBLE_DEVICES="${GPUS[$SLOT]:-$SLOT}" \
    python3 -u -m src.experiments.kip_gpu_training \
        --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
        --llm_quant --llm_max_length 50 --llm_batch_size 64 \
        --site fox \
        --ordered_csv "$DATA_DIR/simulation/tested" \
        --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
        --checkpoint_root "$CKPT_ROOT" \
        --results_csv "$RESULTS_CSV" >"$LOG" 2>&1
    local RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
    # decoder runs write meta.json (threshold + val stats), not curve.json
    cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
          "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
    cp -f "$LOG" "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true
    echo "cell $IDX: $MODEL $TAG $MODE $SEED DONE"
}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID / Node: $(hostname) / mode=$MODE"

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
