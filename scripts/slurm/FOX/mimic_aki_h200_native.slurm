#!/bin/bash -l
#SBATCH --job-name=mimic_aki_h200_v3
#SBATCH --account=ec12
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --output=logs/mimic_aki_h200_v3_%j.out

# Submit from REPO_ROOT after creating a private scheduler-log directory:
#   umask 077; mkdir -p logs; sbatch scripts/slurm/FOX/mimic_aki_h200_native.slurm

set -o errexit
set -o nounset
set -o pipefail
umask 077

# This is deliberately not an array job. One AKI train invocation executes the
# complete locked matrix with all three model and permutation seeds and writes
# one internally consistent experiment directory.

readonly CONFIG_NAME="mimic_aki_fox_h200_smd015_amendment_v3.yaml"
readonly EXPECTED_CONFIG_BYTE_SHA256="75d259be65637112bf6e50f31e54c63b572812f5bc69cd24d5facf712d000bdc"
readonly EXPECTED_CONFIG_SEMANTIC_SHA256="d27681de235eaa6d0421f3720b6253c2dc138a046be5c536d6ba7f498edf9345"
readonly EXPECTED_SOURCE_MANIFEST_SHA256="40e0042f3a43bc1fac47adfad618e78fd647c0ac6499dbd5dd29dadb0c2db486"
readonly EXPECTED_TRANSFER_MANIFEST_SHA256="0ffa6c00e39cb662c9c5907a7d472c4065268a354e593a5321b97942f5a2de04"

# ============== MODULES ==============
if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    echo "stage=workflow status=FAIL reason=submit-from-a-clean-non-virtualenv-shell" >&2
    exit 2
fi
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
# The 2024a PyTorch/CUDA stack includes H200 (sm_90) kernels. The datasets and
# scikit bundles provide the tabular AKI dependencies; the import gate below
# fails before opening prepared patient-level artifacts if anything is absent.
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
hash -r

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
export LLMSEQ_ROOT="$REPO_ROOT"

# Transfer the complete frozen CPU package to this path. Override AKI_RUN_ROOT
# with sbatch --export if it is stored elsewhere on private project storage.
AKI_RUN_ROOT="${AKI_RUN_ROOT:-$REPO_ROOT/data/processed/mimic_aki/aki-audit-20260812-fox-h200-v3}"
CONFIG="$AKI_RUN_ROOT/protocol/$CONFIG_NAME"
COHORT_DIR="$AKI_RUN_ROOT/cohort"
PREPARED_DIR="$AKI_RUN_ROOT/prepared"
EXPERIMENT_DIR="$AKI_RUN_ROOT/experiment"
RUN_LOG_DIR="$AKI_RUN_ROOT/logs"
TRANSFER_MANIFEST="$AKI_RUN_ROOT/protocol/cpu-to-fox-transfer.sha256"
SOURCE_MANIFEST="$REPO_ROOT/configs/mimic_aki_fox_h200_smd015_amendment_v3.sources.sha256"
NATIVE_JOB_SOURCE="$REPO_ROOT/scripts/slurm/FOX/mimic_aki_h200_native.slurm"
PREFLIGHT_LOG_DIR="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/logs"
JOB_TAG="fox_native_${SLURM_JOB_ID}"
PREFLIGHT_TMP_DIR="$PREFLIGHT_LOG_DIR/${JOB_TAG}_tmp"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

fail() {
    echo "stage=workflow status=FAIL reason=$1" >&2
    exit 2
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

run_stage() {
    local stage_name="$1"
    local stage_log="$2"
    shift 2
    echo "stage=$stage_name status=STARTED"
    if "$@" >"$stage_log" 2>&1; then
        chmod 0600 "$stage_log"
        echo "stage=$stage_name status=PASS"
    else
        local exit_code=$?
        chmod 0600 "$stage_log" 2>/dev/null || true
        echo "stage=$stage_name status=FAIL exit_code=$exit_code" >&2
        return "$exit_code"
    fi
}

verify_locked_state() {
    [[ "$(sha256_of "$CONFIG")" = "$EXPECTED_CONFIG_BYTE_SHA256" ]] \
        || fail "frozen-config-byte-hash-mismatch"
    [[ "$(sha256_of "$SOURCE_MANIFEST")" \
        = "$EXPECTED_SOURCE_MANIFEST_SHA256" ]] \
        || fail "source-manifest-hash-mismatch"
    [[ "$(sha256_of "$TRANSFER_MANIFEST")" \
        = "$EXPECTED_TRANSFER_MANIFEST_SHA256" ]] \
        || fail "transfer-manifest-hash-mismatch"
    (cd "$REPO_ROOT" && sha256sum --quiet -c "$SOURCE_MANIFEST") \
        >/dev/null 2>&1 || fail "source-checksum-verification-failed"
    (cd "$AKI_RUN_ROOT" && sha256sum --quiet -c "$TRANSFER_MANIFEST") \
        >/dev/null 2>&1 || fail "transfer-checksum-verification-failed"
    python3 -B "$REPO_ROOT/scripts/mimic_aki_verify_source_manifest.py" \
        --repo-root "$REPO_ROOT" --manifest "$SOURCE_MANIFEST" \
        >/dev/null 2>&1 || fail "exact-source-manifest-verification-failed"
    cmp -s "$REPO_ROOT/scripts/mimic_aki_integrity_gate.py" \
        "$AKI_RUN_ROOT/protocol/mimic_aki_integrity_gate.py" \
        || fail "integrity-gate-differs-from-frozen-copy"
    cmp -s "${BASH_SOURCE[0]}" "$NATIVE_JOB_SOURCE" \
        || fail "submitted-job-differs-from-repository-native-job"
}

[[ -d "$REPO_ROOT" && ! -L "$REPO_ROOT" ]] \
    || fail "repository-is-missing-or-a-symlink"
[[ "$REPO_ROOT" = /* && "$AKI_RUN_ROOT" = /* ]] \
    || fail "repository-and-run-root-paths-must-be-absolute"
[[ -d "$AKI_RUN_ROOT" && ! -L "$AKI_RUN_ROOT" ]] \
    || fail "transferred-run-root-is-missing-or-a-symlink"
[[ -d "$AKI_RUN_ROOT/protocol" && -d "$COHORT_DIR" \
    && -d "$PREPARED_DIR" && -d "$RUN_LOG_DIR" \
    && -d "$AKI_RUN_ROOT/tmp" ]] \
    || fail "transferred-run-root-is-incomplete"
[[ -f "$CONFIG" && -f "$TRANSFER_MANIFEST" && -f "$SOURCE_MANIFEST" ]] \
    || fail "frozen-config-or-checksum-manifest-is-missing"
[[ -f "$NATIVE_JOB_SOURCE" && ! -L "$NATIVE_JOB_SOURCE" ]] \
    || fail "native-job-source-is-missing-or-a-symlink"
[[ -d "$PREFLIGHT_LOG_DIR" && ! -L "$PREFLIGHT_LOG_DIR" ]] \
    || fail "private-submit-log-directory-is-missing-or-a-symlink"
if find "$PREFLIGHT_LOG_DIR" -maxdepth 0 -perm /0077 -print -quit \
    | grep -q .; then
    fail "submit-log-directory-has-group-or-world-permissions"
fi
[[ ! -e "$REPO_ROOT/.venv" && ! -e "$REPO_ROOT/sitecustomize.py" \
    && ! -e "$REPO_ROOT/usercustomize.py" ]] \
    || fail "source-tree-contains-a-local-environment-or-Python-customizer"
if find "$REPO_ROOT/src" "$REPO_ROOT/tests" \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
    -print -quit | grep -q .; then
    fail "source-tree-contains-Python-bytecode-cache"
fi
[[ ! -e "$EXPERIMENT_DIR" ]] \
    || fail "experiment-directory-already-exists-refusing-overwrite"
[[ -z "$(find "$AKI_RUN_ROOT/tmp" -mindepth 1 -print -quit)" ]] \
    || fail "run-temporary-directory-is-not-empty"
if find "$AKI_RUN_ROOT" -type l -print -quit | grep -q .; then
    fail "transferred-run-tree-contains-a-symlink"
fi
if find "$AKI_RUN_ROOT" -type f \( \
    -name labevents.csv -o -name labevents.csv.gz -o \
    -name admissions.csv -o -name admissions.csv.gz -o \
    -name patients.csv -o -name patients.csv.gz -o \
    -name d_labitems.csv -o -name d_labitems.csv.gz \
    \) -print -quit | grep -q .; then
    fail "transferred-run-tree-contains-a-raw-MIMIC-source-table"
fi
if find "$AKI_RUN_ROOT" -perm /0077 -print -quit | grep -q .; then
    fail "transferred-run-tree-has-group-or-world-permissions"
fi

# Verify the immutable CPU handoff before reading any Parquet artifact.
verify_locked_state
python3 -B "$REPO_ROOT/scripts/mimic_aki_verify_transfer_manifest.py" \
    --run-root "$AKI_RUN_ROOT" --manifest "$TRANSFER_MANIFEST" \
    >/dev/null 2>&1 || fail "exact-transfer-manifest-verification-failed"

GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "$GPU_ID" && "$GPU_ID" != *,* ]] \
    || fail "expected-exactly-one-Slurm-visible-GPU"
nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader \
    >/dev/null 2>&1 || fail "allocated-GPU-query-failed"

exec 9<"$TRANSFER_MANIFEST"
flock -n 9 || fail "run-root-is-locked-by-another-FOX-job"
[[ ! -e "$PREFLIGHT_TMP_DIR" ]] \
    || fail "preflight-temporary-directory-already-exists"
install -d -m 0700 "$PREFLIGHT_TMP_DIR"
export TMPDIR="$PREFLIGHT_TMP_DIR"

NATIVE_JOB_SHA256="$(sha256_of "${BASH_SOURCE[0]}")"

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Protocol: v3 $EXPECTED_CONFIG_SEMANTIC_SHA256"
echo "Models: logistic_regression xgboost lstm transformer"
echo "Native launcher SHA-256: $NATIVE_JOB_SHA256"

# ============== PREFLIGHT AND TESTS ==============
run_stage environment_and_cuda \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_environment_and_cuda.log" \
    python3 -B -c '
import importlib.metadata as metadata
import platform
import numpy
import pandas
import pyarrow
import pytest
import scipy
import sklearn
import torch
import xgboost
import yaml

packages = (
    "torch", "pandas", "numpy", "pyarrow", "PyYAML",
    "scikit-learn", "xgboost", "scipy", "pytest",
)
for package in packages:
    print(f"{package}={metadata.version(package)}")
print(f"python={platform.python_version()}")
print(f"machine={platform.machine()}")
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"
name = torch.cuda.get_device_name(0)
print(f"gpu={name}")
assert "H200" in name.upper(), "allocated GPU is not an H200/GH200"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)
x = torch.randn(256, 256, device="cuda:0")
_ = x @ x.T
lstm = torch.nn.LSTM(8, 16, batch_first=True).to("cuda:0")
_ = lstm(torch.randn(4, 32, 8, device="cuda:0"))
layer = torch.nn.TransformerEncoderLayer(
    d_model=16, nhead=4, batch_first=True
).to("cuda:0")
_ = layer(torch.randn(4, 32, 16, device="cuda:0"))
torch.cuda.synchronize()
print("cuda_smoke=PASS")
'

cd "$REPO_ROOT"

run_stage validate_config \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_validate_config.log" \
    python3 -B -m src.mimic.aki validate-config --config "$CONFIG"
grep -q "$EXPECTED_CONFIG_SEMANTIC_SHA256" \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_validate_config.log" \
    || fail "validated-semantic-hash-is-unexpected"

run_stage unit_tests \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_unit_tests.log" \
    python3 -B -m pytest -q tests/mimic/aki tests/test_aki_cohort.py \
    --ignore=tests/mimic/aki/test_end_to_end_synthetic.py \
    -p no:cacheprovider

run_stage synthetic_end_to_end_tests \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_synthetic_e2e.log" \
    python3 -B -m pytest -q tests/mimic/aki/test_end_to_end_synthetic.py \
    -p no:cacheprovider

run_stage pretraining_integrity_gate \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_pretraining_gate.log" \
    python3 -B scripts/mimic_aki_integrity_gate.py pretrain \
    --config "$CONFIG" \
    --cohort-dir "$COHORT_DIR" \
    --prepared-dir "$PREPARED_DIR"
grep -q '"training_authorized": true' \
    "$PREFLIGHT_LOG_DIR/${JOB_TAG}_pretraining_gate.log" \
    || fail "pretraining-gate-did-not-authorize-training"

# ============== COMPLETE MODEL MATRIX ==============
verify_locked_state

# The pre-training checks passed. Preserve their private logs and the exact
# native launcher before creating any training output.
JOB_SCRIPT_COPY="$AKI_RUN_ROOT/protocol/mimic_aki_h200_native_${SLURM_JOB_ID}.slurm"
[[ ! -e "$JOB_SCRIPT_COPY" ]] || fail "job-script-provenance-copy-exists"
for preflight_name in \
    environment_and_cuda \
    validate_config \
    unit_tests \
    synthetic_e2e \
    pretraining_gate; do
    source_log="$PREFLIGHT_LOG_DIR/${JOB_TAG}_${preflight_name}.log"
    target_log="$RUN_LOG_DIR/${JOB_TAG}_${preflight_name}.log"
    [[ -f "$source_log" && ! -e "$target_log" ]] \
        || fail "preflight-log-is-missing-or-target-already-exists"
done
install -m 0400 "${BASH_SOURCE[0]}" "$JOB_SCRIPT_COPY"
for preflight_name in \
    environment_and_cuda \
    validate_config \
    unit_tests \
    synthetic_e2e \
    pretraining_gate; do
    source_log="$PREFLIGHT_LOG_DIR/${JOB_TAG}_${preflight_name}.log"
    target_log="$RUN_LOG_DIR/${JOB_TAG}_${preflight_name}.log"
    install -m 0600 "$source_log" "$target_log"
done

export TMPDIR="$AKI_RUN_ROOT/tmp"
mkdir "$EXPERIMENT_DIR"
chmod 0700 "$EXPERIMENT_DIR"

TELEMETRY_LOG="$RUN_LOG_DIR/${JOB_TAG}_gpu_telemetry.csv"
TELEMETRY_ERROR_LOG="$RUN_LOG_DIR/${JOB_TAG}_gpu_telemetry.stderr.log"
printf 'uuid,name,utilization_gpu_percent,memory_used_mib\n' > "$TELEMETRY_LOG"
nvidia-smi -i "$GPU_ID" \
    --query-gpu=uuid,name,utilization.gpu,memory.used \
    --format=csv,noheader,nounits --loop=1 \
    >> "$TELEMETRY_LOG" 2> "$TELEMETRY_ERROR_LOG" &
TELEMETRY_PID=$!

cleanup_telemetry() {
    if [[ -n "${TELEMETRY_PID:-}" ]]; then
        kill "$TELEMETRY_PID" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
        TELEMETRY_PID=""
    fi
}
trap cleanup_telemetry EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

TRAIN_LOG="$RUN_LOG_DIR/${JOB_TAG}_train.log"
run_stage train_complete_matrix "$TRAIN_LOG" \
    python3 -u -m src.mimic.aki train \
    --config "$CONFIG" \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$EXPERIMENT_DIR" \
    --models logistic_regression xgboost lstm transformer

cleanup_telemetry
chmod 0600 "$TELEMETRY_LOG" "$TELEMETRY_ERROR_LOG"

verify_locked_state
run_stage posttraining_integrity_gate \
    "$RUN_LOG_DIR/${JOB_TAG}_posttraining_gate.log" \
    python3 -B scripts/mimic_aki_integrity_gate.py posttrain \
    --config "$CONFIG" \
    --prepared-dir "$PREPARED_DIR" \
    --experiment-dir "$EXPERIMENT_DIR" \
    --train-log "$TRAIN_LOG" \
    --gpu-telemetry "$TELEMETRY_LOG"
grep -q '"overall_gate": "PASS"' \
    "$RUN_LOG_DIR/${JOB_TAG}_posttraining_gate.log" \
    || fail "posttraining-gate-did-not-pass"

OUTPUT_MANIFEST="$AKI_RUN_ROOT/protocol/${JOB_TAG}_training_output.sha256"
(cd "$AKI_RUN_ROOT" && {
    find experiment logs -type f -print0
    printf '%s\0' "protocol/$(basename -- "$JOB_SCRIPT_COPY")"
} | sort -z | xargs -0 sha256sum) > "$OUTPUT_MANIFEST"
chmod 0400 "$OUTPUT_MANIFEST"
chmod -R go-rwx "$AKI_RUN_ROOT"
if find "$AKI_RUN_ROOT" -type l -print -quit | grep -q .; then
    fail "completed-run-tree-contains-a-symlink"
fi
if find "$AKI_RUN_ROOT" -perm /0077 -print -quit | grep -q .; then
    fail "completed-run-tree-has-group-or-world-permissions"
fi

echo "=== JOB FINISHED ==="
echo "stage=workflow status=PASS"
