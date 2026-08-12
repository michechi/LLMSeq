# CPU preparation and FOX H200 training for the MIMIC-IV AKI audit

The v3 audit is deliberately split across two machines:

1. On the private CPU VM, use the frozen v3 YAML to extract, prepare, and run
   the complete pre-training integrity gate.
2. Transfer the complete private v3 run root through the institution-approved
   encrypted route to approved FOX project storage.
3. On FOX, submit a train-only Slurm job to one H200. The job reruns tests,
   configuration validation, and the pre-training gate before it trains.

Raw MIMIC tables stay on the CPU VM. The local x86 virtual environment and old
v1/v2 outputs are not transferred.

## Why CPU preparation can use the CUDA v3 protocol

The v2 cohort and prepared tables are scientifically valid, but their frozen
semantic hash specifies CPU training. They cannot be relabeled or used with an
unrecorded device override.

The v3 amendment changes only:

- `models.sequence_training.device`: `cpu` to `cuda:0`
- `models.sequence_training.pin_memory`: `false` to `true`

Extraction and preparation do not initialize the configured training device,
so both stages can run on the CPU VM with this exact CUDA-enabled YAML. FOX
training then consumes artifacts bearing the same v3 semantic hash.

Frozen v3 hashes:

- YAML bytes:
  `75d259be65637112bf6e50f31e54c63b572812f5bc69cd24d5facf712d000bdc`
- Semantic configuration:
  `d27681de235eaa6d0421f3720b6253c2dc138a046be5c536d6ba7f498edf9345`

## CPU VM stages

Use a new private run root and the frozen copy under its `protocol/` directory:

```bash
umask 077
RUN_ROOT=/approved/private/path/aki-audit-20260812-fox-h200-v3
FROZEN_CONFIG="$RUN_ROOT/protocol/mimic_aki_fox_h200_smd015_amendment_v3.yaml"

python -u -m src.mimic.aki extract \
  --config "$FROZEN_CONFIG" \
  --raw-dir /approved/private/path/mimic-iv \
  --output-dir "$RUN_ROOT/cohort"

python -u -m src.mimic.aki prepare \
  --config "$FROZEN_CONFIG" \
  --measurements "$RUN_ROOT/cohort/aki_creatinine_measurements.parquet" \
  --admissions "$RUN_ROOT/cohort/aki_admission_audit.parquet" \
  --output-dir "$RUN_ROOT/prepared" \
  --workers 16 \
  --patient-chunk-size 128

python -B scripts/mimic_aki_integrity_gate.py pretrain \
  --config "$FROZEN_CONFIG" \
  --cohort-dir "$RUN_ROOT/cohort" \
  --prepared-dir "$RUN_ROOT/prepared" \
  > "$RUN_ROOT/logs/pretraining-integrity-gate.console.log" 2>&1
chmod 0600 "$RUN_ROOT/logs/pretraining-integrity-gate.console.log"
```

The production workflow saves full output to private mode-`0600` log files.
Training is not permitted unless the gate exits zero and reports both
`"overall_gate": "PASS"` and `"training_authorized": true`.

After the gate passes, create the immutable transfer package once:

```bash
bash scripts/mimic_aki_finalize_fox_transfer.sh "$RUN_ROOT"
```

The finalizer verifies protocol/source hashes and the passing gate, copies the
frozen source/gate/Slurm provenance into `protocol/`, enforces private modes,
and creates `protocol/cpu-to-fox-transfer.sha256`. It reports only the file
count, total bytes, semantic hash, and independent manifest SHA-256.

Transfer the entire run root—`protocol/`, `cohort/`, `prepared/`, `logs/`, and
the empty `tmp/`—through the institution-approved encrypted channel. This is
about 250 MB based on the prior v2 run. Do not transfer raw MIMIC, `.venv`, old
failed training logs, or prior predictions. Preserve modes, hierarchy, and
filenames. Transfer a clean source repository separately, excluding `data/`,
`.venv/`, every `__pycache__/`, every `*.pyc`, and any root-level
`sitecustomize.py` or `usercustomize.py`. The wrapper fails closed if these are
present. On FOX, verify the exact source manifest before submission; the job
repeats that check before every material stage.

## Required FOX inputs

Set these values to approved persistent FOX storage:

- `FOX_ACCOUNT`: active FOX Slurm project/account.
- `AKI_REPO`: the exact verified source tree on FOX.
- `AKI_RUN_ROOT`: the existing, transferred private v3 run root.
- `AKI_SLURM_LOG_DIR`: a private persistent directory outside the repository
  and run root for fixed scheduler status output.
- `AKI_CONTAINER`: an immutable ARM64 Apptainer image containing CUDA-enabled
  PyTorch, pandas, NumPy, PyArrow, PyYAML, scikit-learn, XGBoost, SciPy, and
  pytest.
- `AKI_CONTAINER_SHA256`: the image's full SHA-256.
- `AKI_SOURCE_MANIFEST_SHA256`: the independent source-manifest hash printed
  by the CPU finalizer.
- `AKI_TRANSFER_MANIFEST`: the transferred
  `protocol/cpu-to-fox-transfer.sha256` file.
- `AKI_TRANSFER_MANIFEST_SHA256`: the independent hash printed by the CPU
  finalizer.

Do not copy the CPU VM's x86 virtual environment. Do not install from the
network inside the restricted-data job. Build or obtain the ARM64 image
separately, then pin it by checksum.

## FOX submission

From a FOX login node:

```bash
export FOX_ACCOUNT=<active-fox-account>
export AKI_REPO=/absolute/path/to/LLMSeq
export AKI_RUN_ROOT=/approved/private/path/aki-audit-20260812-fox-h200-v3
export AKI_SLURM_LOG_DIR=/approved/private/path/slurm-logs
export AKI_CONTAINER=/absolute/path/to/arm64-pytorch-cuda.sif
export AKI_CONTAINER_SHA256="$(sha256sum "$AKI_CONTAINER" | awk '{print $1}')"
export AKI_SOURCE_MANIFEST_SHA256=<value-printed-by-CPU-finalizer>
export AKI_TRANSFER_MANIFEST="$AKI_RUN_ROOT/protocol/cpu-to-fox-transfer.sha256"
export AKI_TRANSFER_MANIFEST_SHA256=<value-printed-by-CPU-finalizer>

bash scripts/slurm/FOX/submit_mimic_aki_h200_v3.sh
```

The account is passed on the `sbatch` command line because `#SBATCH`
directives do not expand shell variables. The wrapper exports only the eight
AKI path/hash variables required by the job and requests private scheduler
output permissions.

The job requests:

- partition `accel`
- one GPU, one node, and one task
- 16 CPU cores
- 128 GiB CPU memory
- 48 hours
- no automatic requeue

If FOX's current H200 resource syntax differs, update only the scheduler
resource request after checking FOX's local documentation. Do not change the
scientific YAML.

## Train-only Slurm gate

Before opening prepared Parquet files, the job:

- rejects public/group permissions, symlinks, an existing `experiment/`, or
  an incomplete transferred run root;
- verifies every transferred file against the independently pinned transfer
  manifest;
- verifies the source manifest, frozen and repository YAML hashes, source/gate
  copies, container checksum, ARM64 host, Slurm allocation, and H200/GH200;
- runs CUDA import and deterministic LSTM/Transformer smoke checks in an
  isolated `--cleanenv --no-home` Apptainer process.

It then reruns unit tests, synthetic end-to-end tests, strict configuration
validation, and the aggregate-only pre-training gate. Only after those pass
does it create the new `experiment/` artifacts and train logistic regression,
XGBoost, LSTM, and Transformer across the locked full matrix.

One-second telemetry targets only the Slurm-allocated GPU by allocation ID and
UUID. The post-training gate privately validates schemas, exact run counts,
identities, labels, finite/bounded probabilities, probability sums, recomputed
metrics, deterministic oracle outputs, aggregate tables, contrasts, and 24
LSTM/Transformer H200 start/completion markers. It emits aggregate violation
counts only—never identifiers, timestamps, or prediction values.

All detailed command output remains in mode-`0600` files under the private run
root. Slurm stdout/stderr receives fixed stage `STARTED`, `PASS`, or `FAIL`
messages only. A failed or partial experiment is preserved and never
overwritten.

## Monitoring

Use aggregate scheduler information:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode,AllocTRES,MaxRSS
```
