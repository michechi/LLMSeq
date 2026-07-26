#!/bin/bash
#
# RECSYS AUDIT — Job 1: the whole main grid as ONE array job (≤2-job packing).
#
# Task t in 0-8: model = (SASRec,GRU4Rec,BERT4Rec)[t/3], seed = (17,32,45)[t%3]
#   1. train on ordered 30Music ([25] protocol: full-catalog CE, Adam 1e-3,
#      max 100 epochs, early stop patience 5 on val NDCG@10);
#   2. evaluate the checkpoint on ordered + all 9 audit shuffle files
#      (every input consumed through the contract-enforcing eval_loader,
#      which re-verifies the window/target invariants ON EVERY LOAD);
#   3. task 0 additionally runs the four baselines (MostPopular / ItemKNN /
#      Markov-1 / Markov-2) on the same 10 inputs — MostPopular + ItemKNN
#      are permutation-invariant by construction and are emitted as
#      reference lines (order_invariant=true).
#
# Batch size is PER MODEL (recorded in the results row): 2048 for
# SASRec/GRU4Rec (last-position logits ~7 GB), 256 for BERT4Rec — its cloze
# CE materializes [B, 25, 839100] logits; measured peak ~61 GiB VRAM at
# B=256; B=2048 would need >170 GB and OOM even an H200 NVL.
#
# After the array completes, build the report on the LOGIN node (CPU-only,
# ~2 min — RUNBOOK step 3), then STOP: Job 2 waits for the table review.
#
# Every result row is appended (fcntl-locked) to
# results/recsys_audit/{training,metrics}.csv the moment it lands, with
# site=fox + SLURM job id; per-epoch val curves live next to each checkpoint
# (*_curve.json) and in the task log. Resumable: finished work is skipped
# via checkpoint/recs-file markers — resubmit (whole array or one index)
# after a wall-time kill and it continues.
#
# Models are a few M params: 9 single-GPU tasks. If the H200s are saturated
# by the Q1 full-FT jobs, run identically on A100s:
#   sbatch --gpus=a100_80:1 recsys_job1_grid.slurm
#
# Submit with:
#   cd scripts/slurm/FOX/recsys && mkdir -p logs
#   ./recsys_setup.sh          # one-time, login node (pip --no-deps recbole)
#   sbatch recsys_job1_grid.slurm
#
#SBATCH --job-name=recsys_job1
#SBATCH --account=ec12
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8
#SBATCH --output=logs/recsys_job1_%A_%a.out

set -o errexit
set -o nounset
source "$SLURM_SUBMIT_DIR/modules.sh"

RESULTS="$REPO_ROOT/results/recsys_audit"
CKPT="$REPO_ROOT/checkpoints/recsys"
mkdir -p "$RESULTS/recs" "$CKPT"

# ============== GATES ==============
echo "=== GATE: split + audit bytes ==="
( cd "$DATA30/split" && sha256sum -c split_files.sha256 )
( cd "$DATA30/audit" && sha256sum -c audit_files.sha256 )
cd "$REPO_ROOT/repro"
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    echo "=== GATE (task 0): full consumer-contract check, 10 files ==="
    python -m src.recsys.eval_loader --audit-dir "$DATA30/audit" \
        --split-dir "$DATA30/split" --check-all
fi
# (other tasks rely on load_eval_input re-verifying every file it reads)

# ============== CELL ==============
MODELS=(SASRec GRU4Rec BERT4Rec)
SEEDS=(17 32 45)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
if [ "$MODEL" = "BERT4Rec" ]; then BATCH=256; else BATCH=2048; fi
echo "=== CELL: $MODEL seed $SEED batch $BATCH ==="

CK="$CKPT/${MODEL}_ordered_s${SEED}.pt"
if [ -f "$CK" ]; then
    echo "=== skip training (checkpoint exists) ==="
else
    # 20 h training budget: graceful stop with the best checkpoint so far,
    # leaving ~4 h inside the 24 h site wall limit for the 10 evals (+2 h
    # baselines on task 0). A budget-stopped cell still yields a valid
    # early-stopped model; epochs_run in training.csv shows what happened.
    python -m src.recsys.train_grid --model "$MODEL" --seed "$SEED" \
        --split-dir "$DATA30/split" --out-dir "$CKPT" \
        --results-csv "$RESULTS/training.csv" \
        --batch-size "$BATCH" --site fox --time-budget-hours 20
fi

MARK="$RESULTS/recs/${MODEL}_ordered_s${SEED}__early_half_s103.csv.gz"
if [ -f "$MARK" ]; then
    echo "=== skip evals (recs complete) ==="
else
    python -m src.recsys.eval_grid --checkpoint "$CK" \
        --audit-dir "$DATA30/audit" --split-dir "$DATA30/split" \
        --inputs all --results-csv "$RESULTS/metrics.csv" \
        --recs-dir "$RESULTS/recs" --site fox
fi

# ============== BASELINES (task 0 only) ==============
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    if [ -f "$RESULTS/recs/Markov2__early_half_s103.csv.gz" ]; then
        echo "=== skip baselines (recs complete) ==="
    else
        echo "=== BASELINES ==="
        python -m src.recsys.baselines --split-dir "$DATA30/split" \
            --audit-dir "$DATA30/audit" \
            --results-csv "$RESULTS/metrics.csv" \
            --recs-dir "$RESULTS/recs" --site fox
    fi
fi

echo "=== TASK FINISHED ==="
