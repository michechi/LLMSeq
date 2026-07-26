#!/bin/bash
#
# KIP (c) -- BERT on kip_m4, mode (a) ordered, seeds 9551 + 9552.
#
# SEED NOTE: the task list says "seeds 2-3", read as the 2nd and 3rd seeds of
# the canonical triplet (9550 9551 9552); seed 9550 already ran locally
# (test_auc 0.5002, chance, at full fidelity max_length=512, 7.3 h on A100).
# If literal seeds 2 and 3 were intended, edit SEEDS below before submitting.
#
# max_length=512 matches the local seed-9550 row so the three seeds form one
# homogeneous triplet (paper-appendix fidelity). That makes runs LONG:
# ~45 min/epoch on A100 (H200 should be faster), early stop typically ~10
# epochs, hard cap 20 -> 20 h wall to be safe.
#
# bert-base-uncased is ungated: the job downloads it into $HF_HOME on first
# use (FOX compute nodes have internet). No HF token needed. Export
# KIP_HF_CACHE to a persistent dir before sbatch to avoid re-downloads.
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs && sbatch kip_c_bert_m4.slurm
#
#SBATCH --job-name=kip_c_bert_m4
#SBATCH --account=ec12
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --output=logs/kip_c_bert_m4_%A_%a.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"
export HF_HOME="${KIP_HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RESULTS_CSV="$REPO_ROOT/results/kip_training.csv"
CURVES_DIR="$REPO_ROOT/results/kip_curves"
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints/kip}"
mkdir -p "$HF_HOME" "$CKPT_ROOT" "$CURVES_DIR"

# ============== GRID ==============
MODEL=BERT
TAG=kip_m4
MODE=ordered
SEEDS=(9551 9552)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SEEDS[@]}" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range -- exiting."
    exit 0
fi
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: model=$MODEL tag=$TAG mode=$MODE seed=$SEED bert_max_length=512"

cd "$REPO_ROOT/repro"

python3 -u -m src.experiments.kip_gpu_training \
    --model "$MODEL" --tag "$TAG" --mode "$MODE" --seed "$SEED" \
    --bert_max_length 512 \
    --site fox \
    --ordered_csv "$DATA_DIR/simulation/tested" \
    --shuffled_csv "$DATA_DIR/simulation/kip_shuffled" \
    --checkpoint_root "$CKPT_ROOT" \
    --results_csv "$RESULTS_CSV"

# BERT has no curve.json (per-epoch metrics are in the log); keep log + meta.
RUN_NAME="${MODEL}_${TAG}_${MODE}_${SEED}"
cp -f "$CKPT_ROOT/$RUN_NAME/meta.json" \
      "$CURVES_DIR/fox_${RUN_NAME}.meta.json" 2>/dev/null || true
cp -f "$SLURM_SUBMIT_DIR/logs/kip_c_bert_m4_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
      "$CURVES_DIR/fox_${RUN_NAME}.trainlog.txt" 2>/dev/null || true

echo "=== JOB FINISHED ==="
