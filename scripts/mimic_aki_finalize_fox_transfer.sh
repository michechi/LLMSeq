#!/bin/bash

# Freeze and checksum a completed CPU-prepared AKI v3 run for private transfer.

set -o errexit
set -o nounset
set -o pipefail
umask 077

readonly CONFIG_NAME="mimic_aki_fox_h200_smd015_amendment_v3.yaml"
readonly SOURCE_MANIFEST_NAME="mimic_aki_fox_h200_smd015_amendment_v3.sources.sha256"
readonly TRANSFER_MANIFEST_NAME="cpu-to-fox-transfer.sha256"
readonly EXPECTED_CONFIG_BYTE_SHA256="75d259be65637112bf6e50f31e54c63b572812f5bc69cd24d5facf712d000bdc"
readonly EXPECTED_CONFIG_SEMANTIC_SHA256="d27681de235eaa6d0421f3720b6253c2dc138a046be5c536d6ba7f498edf9345"

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 /absolute/private/v3-run-root" >&2
    exit 2
fi

run_root="$1"
[[ "$run_root" = /* && -d "$run_root" && ! -L "$run_root" ]] || {
    echo "ERROR: run root must be an absolute, existing, non-symlink directory" >&2
    exit 2
}
run_root="$(realpath -e -- "$run_root")"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

[[ ! -e "$run_root/experiment" ]] || {
    echo "ERROR: experiment output exists; this finalizer is for prepared-only transfer" >&2
    exit 2
}
for required_directory in protocol cohort prepared logs tmp; do
    [[ -d "$run_root/$required_directory" ]] || {
        echo "ERROR: run root is missing $required_directory/" >&2
        exit 2
    }
done
[[ -z "$(find "$run_root/tmp" -mindepth 1 -print -quit)" ]] || {
    echo "ERROR: run tmp/ must be empty before transfer" >&2
    exit 2
}
if find "$run_root" -type l -print -quit | grep -q .; then
    echo "ERROR: run root contains a symbolic link" >&2
    exit 2
fi
if find "$run_root" -type f \( \
    -name labevents.csv -o -name labevents.csv.gz -o \
    -name admissions.csv -o -name admissions.csv.gz -o \
    -name patients.csv -o -name patients.csv.gz -o \
    -name d_labitems.csv -o -name d_labitems.csv.gz \
    \) -print -quit | grep -q .; then
    echo "ERROR: run root contains a raw MIMIC source-table basename" >&2
    exit 2
fi

frozen_config="$run_root/protocol/$CONFIG_NAME"
source_config="$repo_root/configs/$CONFIG_NAME"
source_manifest="$repo_root/configs/$SOURCE_MANIFEST_NAME"
frozen_amendment="$run_root/protocol/mimic_aki_protocol_amendment_v3.md"
source_amendment="$repo_root/docs/mimic_aki_protocol_amendment_v3.md"
transfer_manifest="$run_root/protocol/$TRANSFER_MANIFEST_NAME"
[[ -f "$frozen_config" && -f "$source_config" && -f "$source_manifest" \
    && -f "$frozen_amendment" && -f "$source_amendment" \
    && ! -L "$frozen_amendment" && ! -L "$source_amendment" ]] || {
    echo "ERROR: frozen protocol or source manifest is missing" >&2
    exit 2
}
[[ ! -e "$transfer_manifest" ]] || {
    echo "ERROR: transfer manifest exists; refusing overwrite" >&2
    exit 2
}
[[ "$(sha256sum "$frozen_config" | awk '{print $1}')" \
    = "$EXPECTED_CONFIG_BYTE_SHA256" ]] || {
    echo "ERROR: frozen protocol byte hash mismatch" >&2
    exit 2
}
cmp -s "$source_config" "$frozen_config" || {
    echo "ERROR: source and frozen protocols differ" >&2
    exit 2
}
cmp -s "$source_amendment" "$frozen_amendment" || {
    echo "ERROR: source and frozen protocol amendments differ" >&2
    exit 2
}
(cd "$repo_root" && sha256sum --quiet -c "$source_manifest") \
    >/dev/null 2>&1 || {
    echo "ERROR: source files do not match the pinned manifest" >&2
    exit 2
}
python3 -B "$repo_root/scripts/mimic_aki_verify_source_manifest.py" \
    --repo-root "$repo_root" --manifest "$source_manifest" \
    >/dev/null 2>&1 || {
    echo "ERROR: source manifest verification failed" >&2
    exit 2
}

pretrain_log="$run_root/logs/pretraining-integrity-gate.console.log"
[[ -f "$pretrain_log" ]] || {
    echo "ERROR: CPU pre-training integrity-gate log is missing" >&2
    exit 2
}
grep -q '"overall_gate": "PASS"' "$pretrain_log" || {
    echo "ERROR: CPU pre-training integrity gate did not pass" >&2
    exit 2
}
grep -q '"training_authorized": true' "$pretrain_log" || {
    echo "ERROR: CPU pre-training gate did not authorize training" >&2
    exit 2
}
grep -q "$EXPECTED_CONFIG_SEMANTIC_SHA256" "$pretrain_log" || {
    echo "ERROR: CPU pre-training gate used an unexpected configuration hash" >&2
    exit 2
}

declare -a provenance_sources=(
    "$source_manifest"
    "$repo_root/scripts/mimic_aki_integrity_gate.py"
    "$repo_root/scripts/mimic_aki_finalize_fox_transfer.sh"
    "$repo_root/scripts/mimic_aki_verify_source_manifest.py"
    "$repo_root/scripts/mimic_aki_verify_transfer_manifest.py"
    "$repo_root/scripts/slurm/FOX/mimic_aki_h200_v3.slurm"
    "$repo_root/scripts/slurm/FOX/submit_mimic_aki_h200_v3.sh"
    "$repo_root/docs/mimic_aki_fox_h200.md"
)
declare -a provenance_targets=(
    "$run_root/protocol/$SOURCE_MANIFEST_NAME"
    "$run_root/protocol/mimic_aki_integrity_gate.py"
    "$run_root/protocol/mimic_aki_finalize_fox_transfer.sh"
    "$run_root/protocol/mimic_aki_verify_source_manifest.py"
    "$run_root/protocol/mimic_aki_verify_transfer_manifest.py"
    "$run_root/protocol/mimic_aki_h200_v3.slurm"
    "$run_root/protocol/submit_mimic_aki_h200_v3.sh"
    "$run_root/protocol/mimic_aki_fox_h200.md"
)
for target in "${provenance_targets[@]}"; do
    [[ ! -e "$target" ]] || {
        echo "ERROR: frozen provenance target exists; refusing overwrite" >&2
        exit 2
    }
done
for source in "${provenance_sources[@]}"; do
    [[ -f "$source" && ! -L "$source" ]] || {
        echo "ERROR: provenance source is missing or a symbolic link" >&2
        exit 2
    }
done
for index in "${!provenance_sources[@]}"; do
    install -m 0400 "${provenance_sources[$index]}" \
        "${provenance_targets[$index]}"
done

chmod -R go-rwx "$run_root"
(cd "$run_root" && find . -type f \
    ! -path "./protocol/$TRANSFER_MANIFEST_NAME" -print0 \
    | sort -z | xargs -0 sha256sum) > "$transfer_manifest"
chmod 0400 "$transfer_manifest"
(cd "$run_root" && sha256sum --quiet -c "$transfer_manifest") \
    >/dev/null 2>&1 || {
    echo "ERROR: transfer files do not match the checksum manifest" >&2
    exit 2
}
python3 -B "$repo_root/scripts/mimic_aki_verify_transfer_manifest.py" \
    --run-root "$run_root" --manifest "$transfer_manifest" >/dev/null || {
    echo "ERROR: exact transfer manifest verification failed" >&2
    exit 2
}

file_count="$(wc -l < "$transfer_manifest")"
total_bytes="$(du -sb "$run_root" | awk '{print $1}')"
manifest_sha256="$(sha256sum "$transfer_manifest" | awk '{print $1}')"
source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"

printf 'transfer_package_status=PASS\n'
printf 'config_semantic_sha256=%s\n' "$EXPECTED_CONFIG_SEMANTIC_SHA256"
printf 'transfer_file_count=%s\n' "$file_count"
printf 'transfer_total_bytes=%s\n' "$total_bytes"
printf 'source_manifest_sha256=%s\n' "$source_manifest_sha256"
printf 'transfer_manifest_sha256=%s\n' "$manifest_sha256"
