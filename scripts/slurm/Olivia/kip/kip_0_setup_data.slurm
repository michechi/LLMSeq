#!/bin/bash
#
# KIP setup (fallback job) -- regenerate + verify the KIP datasets on Olivia.
#
# Prefer running these steps on the login node (see RUNBOOK.md step 2); submit
# this job only if apptainer is unavailable there. Runs, in order:
#   1. src.generators.kip --build --m 4 6      (deterministic, seed 959693)
#   2. src.generators.kip --verify --m 4 6     (re-derive labels from CSVs)
#   3. src.generators.kip_sanity --skip_counts (8/8 must pass; count-purity
#      check 5 needs xgboost, which the container lacks -- it was validated
#      locally, see repro/src/analysis/mechanism_id/results/kip_sanity.txt)
#   4. src.data.kip_shuffles                   (build + verify, seed 777)
#   5. sha256sum -c against kip_data.sha256    (byte-identical cross-site proof)
#   6. git status of data/simulation/tested    (must stay clean)
#
# Submit with:
#   cd scripts/slurm/Olivia/kip && mkdir -p logs && sbatch kip_0_setup_data.slurm
#
#SBATCH --job-name=kip_0_setup
#SBATCH --account=NN12048K
#SBATCH --time=02:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/kip_0_setup_%j.out
#SBATCH --error=logs/kip_0_setup_%j.err

set -o errexit
set -o nounset

# ============== CACHE ==============
export PERMANENT_CACHE="/cluster/work/projects/nn12048k/michechi/cache"
export PYTHONUNBUFFERED=1
mkdir -p logs "$PERMANENT_CACHE"

# ============== MODULES ==============
module purge
module load NRIS/GPU
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-/cluster/home/michechi/MIMICIV}"
DATA_DIR="$REPO_ROOT/data"
CONTAINER="/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif"

CONTAINER_ENV="PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,DATA_DIR=$DATA_DIR"
CONTAINER_ENV="$CONTAINER_ENV,LLMSEQ_ROOT=$REPO_ROOT"

cd "$REPO_ROOT/repro"

run_py () {
    srun apptainer exec --nv \
        --bind "$PERMANENT_CACHE:$PERMANENT_CACHE:rw" \
        --bind /cluster/work/projects/nn12048k:/cluster/work/projects/nn12048k:rw \
        --bind /cluster/home/michechi:/cluster/home/michechi:rw \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -u -m "$@"
}

echo "=== 1/6 build ==="
run_py src.generators.kip --build --m 4 6
echo "=== 2/6 verify labels ==="
run_py src.generators.kip --verify --m 4 6
echo "=== 3/6 sanity suite (--skip_counts: container has no xgboost) ==="
run_py src.generators.kip_sanity --skip_counts
echo "=== 4/6 shuffled controls (build + verify) ==="
run_py src.data.kip_shuffles
echo "=== 5/6 sha256 manifest check ==="
cd "$REPO_ROOT"
sha256sum -c scripts/slurm/Olivia/kip/kip_data.sha256
echo "=== 6/6 git cleanliness of tracked data ==="
git status --short data/simulation/tested
if [ -n "$(git status --short data/simulation/tested)" ]; then
    echo "ERROR: regeneration changed tracked KIP data -- DO NOT train. See RUNBOOK."
    exit 1
fi

echo "=== SETUP OK -- data byte-identical to the local site ==="
