#!/bin/bash
#
# RecSys audit (a) TRAINING -- 3 models x 3 seeds on ordered 30Music.
# Array cell = model_idx*3 + seed_idx; models (SASRec,GRU4Rec,BERT4Rec),
# seeds (17,32,45).
# Batch size is PER MODEL (recorded in the results row): 2048 for
# SASRec/GRU4Rec (last-position logits ~7 GB), 256 for BERT4Rec — its cloze
# CE materializes [B, 25, 839100] logits; measured peak ~61 GiB at B=256 on
# an A100-80GB run of the real vocab, while B=2048 would need >170 GB and
# OOM even an H200 NVL.
#
# Submit: cd scripts/slurm/FOX/recsys && mkdir -p logs && sbatch recsys_a_train.slurm
#
#SBATCH --job-name=recsys_a_train
#SBATCH --account=ec12
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8
#SBATCH --output=logs/recsys_a_train_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

MODELS=(SASRec GRU4Rec BERT4Rec)
SEEDS=(17 32 45)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
if [ "$MODEL" = "BERT4Rec" ]; then BATCH=256; else BATCH=2048; fi

python -m src.recsys.train_grid --model "$MODEL" --seed "$SEED" \
  --split-dir "$DATA30/split" \
  --out-dir "$REPO_ROOT/checkpoints/recsys" \
  --results-csv "$REPO_ROOT/results/recsys_audit/training.csv" \
  --batch-size "$BATCH" --site fox
