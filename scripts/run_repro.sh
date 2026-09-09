#!/usr/bin/env bash
# Select repro/src without changing the historical experiment implementations.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT="$REPO_DIR/repro"
export LLMSEQ_ROOT="${LLMSEQ_ROOT:-$REPO_DIR}"
export DATA_DIR="${DATA_DIR:-$LLMSEQ_ROOT/data}"
export RESULTS_DIR="${RESULTS_DIR:-$LLMSEQ_ROOT/results}"
export CACHE_DIR="${CACHE_DIR:-$LLMSEQ_ROOT/cache}"
export PYTHONPATH="$REPRO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPRO_ROOT"
exec "${PYTHON:-python}" "$@"
