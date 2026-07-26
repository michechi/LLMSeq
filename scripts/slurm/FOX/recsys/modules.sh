# Shared module + environment setup for the RecSys audit grid on FOX.
# Sourced by every recsys_*.slurm job. Mirrors the KIP package: the 2024a
# stack has H200 (sm_90) kernels; 2022b does not ("no kernel image" crash).
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

export REPO_ROOT="${REPO_ROOT:-$HOME/MIMICIV}"
export KIP_SITE=fox
export DATA30="$REPO_ROOT/data/recsys/30music"
cd "$REPO_ROOT/repro"
