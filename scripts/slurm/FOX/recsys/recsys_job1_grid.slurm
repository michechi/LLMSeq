#!/bin/bash
#
# RECSYS AUDIT — Job 1 (the whole main grid, one sequential driver).
#
#   * SASRec + GRU4Rec + BERT4Rec, training seeds {17,32,45}, ordered
#     training ([25] protocol: early stop patience 5 on val NDCG@10,
#     full-catalog CE, max 100 epochs) — 9 cells;
#   * every checkpoint evaluated on ordered + all 9 audit shuffle files
#     (through the contract-enforcing eval_loader);
#   * baselines MostPopular / ItemKNN / Markov-1 / Markov-2 on the same 10
#     inputs (MostPopular + ItemKNN are permutation-invariant by
#     construction — emitted as reference lines, order_invariant=true);
#   * final report table -> results/recsys_audit/report_job1.md.
#
# Every result row is appended (fcntl-locked) to
# results/recsys_audit/{training,metrics}.csv the moment it lands, with
# site=fox + SLURM job id; per-epoch val curves live next to each checkpoint
# (*_curve.json) and in this job's log. Resumable: finished cells are skipped
# via checkpoint/recs-file markers — requeue after a wall-time kill and it
# continues.
#
# The models are a few M params — this deliberately requests ONE GPU and does
# not compete with the Q1 full-FT jobs beyond that. If the H200s are
# saturated, submit on an A100 instead (same everything):
#   sbatch --gpus=a100_80:1 recsys_job1_grid.slurm
#
# Batch size 512 (single-target prefix samples) ~ [25]'s batch 4 x 128
# positions of per-position CE — recorded per run in training.csv.
#
# Submit with:
#   cd scripts/slurm/FOX/recsys && mkdir -p logs && sbatch recsys_job1_grid.slurm
#
#SBATCH --job-name=recsys_job1
#SBATCH --account=ec12
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/recsys_job1_%j.out

set -o errexit
set -o nounset

# ============== MODULES (same 2024a stack as the KIP jobs) ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export KIP_SITE=fox
DATA="$REPO_ROOT/data/recsys/30music"
CKPT="$REPO_ROOT/checkpoints/recsys"
RESULTS="$REPO_ROOT/results/recsys_audit"
mkdir -p "$CKPT" "$RESULTS/recs"

# ============== PREFLIGHT GATES (all must pass before anything trains) ====
python3 -c "import recbole" 2>/dev/null || pip install --user "recbole==1.2.0"
python3 - <<'PY'
import recbole, torch
print(f"preflight: recbole {recbole.__version__} torch {torch.__version__} "
      f"cuda={torch.cuda.is_available()}")
PY

echo "=== GATE 1/3: split bytes ==="
cd "$DATA/split" && sha256sum -c split_files.sha256
echo "=== GATE 2/3: audit bytes ==="
cd "$DATA/audit" && sha256sum -c audit_files.sha256
echo "=== GATE 3/3: consumer contract (incl. target-identity, all 10 files) ==="
cd "$REPO_ROOT/repro"
python3 -m src.recsys.eval_loader --audit-dir "$DATA/audit" \
    --split-dir "$DATA/split" --check-all

# ============== GRID: 3 models x 3 seeds, train -> 10 evals ==============
for MODEL in SASRec GRU4Rec BERT4Rec; do
  for SEED in 17 32 45; do
    CK="$CKPT/${MODEL}_ordered_s${SEED}.pt"
    if [ -f "$CK" ]; then
        echo "=== skip training $MODEL s$SEED (checkpoint exists) ==="
    else
        echo "=== TRAIN $MODEL seed $SEED ==="
        python3 -m src.recsys.train_grid --model "$MODEL" --seed "$SEED" \
            --split-dir "$DATA/split" --out-dir "$CKPT" \
            --results-csv "$RESULTS/training.csv" --batch-size 512
    fi
    # last of the 10 eval artifacts doubles as the done-marker
    MARK="$RESULTS/recs/${MODEL}_ordered_s${SEED}__early_half_s103.csv.gz"
    if [ -f "$MARK" ]; then
        echo "=== skip evals $MODEL s$SEED (recs complete) ==="
    else
        echo "=== EVAL $MODEL seed $SEED on 10 inputs ==="
        python3 -m src.recsys.eval_grid --checkpoint "$CK" \
            --audit-dir "$DATA/audit" --split-dir "$DATA/split" \
            --inputs all --results-csv "$RESULTS/metrics.csv" \
            --recs-dir "$RESULTS/recs"
    fi
  done
done

# ============== BASELINES (reference lines + Markov ladder) ==============
if [ -f "$RESULTS/recs/Markov2__early_half_s103.csv.gz" ]; then
    echo "=== skip baselines (recs complete) ==="
else
    echo "=== BASELINES ==="
    python3 -m src.recsys.baselines --split-dir "$DATA/split" \
        --audit-dir "$DATA/audit" --results-csv "$RESULTS/metrics.csv" \
        --recs-dir "$RESULTS/recs"
fi

# ============== REPORT ==============
echo "=== REPORT ==="
python3 -m src.recsys.report --audit-dir "$DATA/audit" \
    --recs-dir "$RESULTS/recs" --out "$RESULTS/report_job1.md"

echo "=== JOB FINISHED ==="
