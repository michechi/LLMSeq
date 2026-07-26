#!/bin/bash
#
# KIP setup (fallback job) -- regenerate + verify the KIP datasets on FOX.
#
# Prefer running these steps on the login node (see RUNBOOK.md step 2; the
# NLPL module stack is available there and no GPU is needed). Submit this job
# only if login-node execution is not possible. Runs, in order:
#   1. src.generators.kip --build --m 4 6      (deterministic, seed 959693)
#   2. src.generators.kip --verify --m 4 6     (re-derive labels from CSVs)
#   3. src.generators.kip_sanity               (12/12 if xgboost is available;
#      otherwise 10/12 with BOTH failures reading "xgboost missing" -- the
#      only acceptable failures; reference: repro/src/analysis/mechanism_id/
#      results/kip_sanity.txt, the tracked local 12/12 run)
#   4. src.data.kip_shuffles                   (build + verify, seed 777)
#   5. sha256sum -c against kip_data.sha256    (byte-identical cross-site proof)
#   6. git status of data/simulation/tested    (must stay clean)
#
# Submit with:
#   cd scripts/slurm/FOX/kip && mkdir -p logs && sbatch kip_0_setup_data.slurm
#
#SBATCH --job-name=kip_0_setup
#SBATCH --account=ec12
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/kip_0_setup_%j.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# 2024a stack: torch 2.6.0 + cuda 12.6 has H200 (sm_90) kernels. The 2022b
# stack (torch 2.1.2 + cuda 12.0) predates the H200 nodes and dies with
# "CUDA error: no kernel image is available" on them. transformers 4.55.4
# instead of the paper's 4.47.1 is a recorded, accepted deviation.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
module load nlpl-bitsandbytes/0.46.1-foss-2024a-Python-3.12.3

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"
export DATA_DIR="$REPO_ROOT/data"

cd "$REPO_ROOT/repro"

echo "=== 1/6 build ==="
python3 -u -m src.generators.kip --build --m 4 6
echo "=== 2/6 verify labels ==="
python3 -u -m src.generators.kip --verify --m 4 6
echo "=== 3/6 sanity suite (12/12, or 10/12 with only 'xgboost missing' FAILs) ==="
SANITY_LOG=$(mktemp)
set +o errexit
python3 -u -m src.generators.kip_sanity > "$SANITY_LOG" 2>&1
SANITY_RC=$?
set -o errexit
cat "$SANITY_LOG"
if [ "$SANITY_RC" -ne 0 ]; then
    # Report rows end in PASS/FAIL; the only acceptable FAILs are the two
    # xgboost-missing count-purity rows (validated locally at 12/12, see
    # repro/src/analysis/mechanism_id/results/kip_sanity.txt).
    if grep -E 'FAIL$' "$SANITY_LOG" | grep -v 'xgboost missing' | grep -q .; then
        echo "ERROR: sanity suite has non-xgboost failures -- DO NOT train."
        exit 1
    fi
    echo "NOTE: only 'xgboost missing' failures -- acceptable, continuing."
fi
echo "=== 4/6 shuffled controls (build + verify) ==="
python3 -u -m src.data.kip_shuffles
echo "=== 5/6 sha256 manifest check ==="
cd "$REPO_ROOT"
sha256sum -c scripts/slurm/FOX/kip/kip_data.sha256
echo "=== 6/6 git cleanliness of tracked data ==="
git status --short data/simulation/tested
if [ -n "$(git status --short data/simulation/tested)" ]; then
    echo "ERROR: regeneration changed tracked KIP data -- DO NOT train. See RUNBOOK."
    exit 1
fi

echo "=== SETUP OK -- data byte-identical to the local site ==="
