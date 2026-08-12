# MIMIC-IV AKI trajectory audit: FOX/H200 execution amendment v3

Amendment approved: 2026-08-12 UTC

Parent protocol: configs/mimic_aki_cpu_smd015_amendment_v2.yaml

Parent YAML byte SHA-256:
34a03dd8828460800bd232e7f1c2b40f54239bc080d05087449243613702a1f6

Parent semantic configuration SHA-256:
39a30570d1ee49928f72563cb909f95fed7d4a8ecfa30360ea62af22ae9f36e8

Amended protocol:
configs/mimic_aki_fox_h200_smd015_amendment_v3.yaml

Amended YAML byte SHA-256:
75d259be65637112bf6e50f31e54c63b572812f5bc69cd24d5facf712d000bdc

Amended semantic configuration SHA-256:
d27681de235eaa6d0421f3720b6253c2dc138a046be5c536d6ba7f498edf9345

## Trigger and status

The v2 preparation and aggregate pre-training integrity audit completed and
passed. CPU training then began, but the process was externally interrupted
before it created an experiment directory, predictions, aggregate metrics, or
a training manifest. No model-performance result was available when this
execution amendment was requested.

The v2 cohort, preparation, protocol, and logs remain preserved. The incomplete
v2 execution must not be relabeled as a completed analysis.

The v3 analysis is an amended confirmatory analysis. It retains the documented
v2 duration-only SMD limit of 0.15; the original v1 0.10 failure remains a
reported sensitivity result and limitation.

## Approved changes

1. Execute the audit on FOX through SLURM using one allocated NVIDIA H200 GPU.
2. Change models.sequence_training.device from cpu to cuda:0.
3. Change models.sequence_training.pin_memory from false to true, restoring the
   already reviewed v1 CUDA data-loader setting.
4. Record safe LSTM and Transformer device-start/device-completion markers in
   the private training log so GPU execution can be audited without exposing
   patient-level data.

No cohort, cleaning, episode, label, matching, split, representation,
architecture, optimization, seed, control, bootstrap, or metric definition
changes.

## Hash and rerun consequence

The execution settings are part of the strict protocol, so v3 has a new
semantic configuration hash. Existing v2 cohort/prepared artifacts cannot be
used for v3 training, and their embedded hash must not be edited or bypassed.
V3 extraction, preparation, pre-training inspection, and training must all use
the same frozen v3 YAML from a new run root.

## FOX requirements

- Submit all compute with sbatch; do not run the audit on a FOX login node.
- Request partition accel and one GPU, then fail before data access unless the
  allocated accelerator identifies as NVIDIA H200 and CUDA is usable.
- Use an ARM64-compatible CUDA/PyTorch Apptainer image with AKI dependencies
  already installed. Do not install dependencies from the network in the job.
- Place raw MIMIC-IV, derived Parquet files, predictions, manifests, and logs
  only in approved private persistent FOX project storage.
- Use umask 077, a new run root, read-only raw-data and repository binds, and
  end-to-end checksums for any institution-approved encrypted transfer.
- Never put patient identifiers, timestamps, or prediction values in SLURM
  output. Only aggregate audit summaries may be printed.
- Preserve the frozen YAML, byte and semantic hashes, amendment, source hashes,
  tests, console logs, GPU telemetry, manifests, audit tables, predictions, and
  aggregate metrics.
