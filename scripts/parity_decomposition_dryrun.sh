#!/usr/bin/env bash
#
# Parity decomposition — local dry run.
#
# Builds the 1K-sample variant CSVs (if not already there) and runs 6 DL
# combinations (3 variants × 2 models) with seed=0, short training, to
# smoke-test the pipeline end-to-end. Writes JSONs and prediction CSVs under
# results/parity_decomp_dryrun/.
#
# Usage:
#   bash scripts/parity_decomposition_dryrun.sh
#   PYBIN=/opt/anaconda3/envs/clibench/bin/python bash scripts/parity_decomposition_dryrun.sh
#
# No GPU required. Runtime ~5–10 minutes on a Mac CPU.

set -o errexit
set -o nounset
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-python3}"
DATA_DIR="${DATA_DIR:-data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-results/parity_decomp_dryrun}"
EPOCHS="${EPOCHS:-2}"
PATIENCE="${PATIENCE:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EVAL="${MAX_EVAL:-2000}"   # cap val/test for the dry run

echo "=== step 1: verify parity rule on existing CSV ==="
"$PYBIN" -m src.data.parity_variants --verify

if [ ! -f "$DATA_DIR/X_train_parity_raw_1K.csv" ]; then
    echo "=== step 2: build 1K variants ==="
    "$PYBIN" -m src.data.parity_variants --dry_run --data_dir data/simulation/tested --output_dir "$DATA_DIR"
else
    echo "=== step 2: 1K variants already exist, skipping ==="
fi

mkdir -p "$OUTPUT_DIR"

VARIANTS=("raw" "masked" "bitonly")
MODELS=("Transformer" "LSTM")
SEED=0
SIZE="1K"

echo "=== step 3: DL dry-run (6 combos) ==="
for VARIANT in "${VARIANTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "--- $VARIANT / $MODEL ---"
        "$PYBIN" -m src.experiments.parity_decomposition_dl \
            --variant "$VARIANT" --model "$MODEL" --size "$SIZE" --seed "$SEED" \
            --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --batch_size "$BATCH_SIZE" \
            --max_eval_samples "$MAX_EVAL"
    done
done

echo ""
echo "=== step 4: sanity checks on outputs ==="
"$PYBIN" - <<'PY'
import json, glob, os, sys
import pandas as pd

root = os.environ.get("OUTPUT_DIR", "results/parity_decomp_dryrun")
jsons = sorted(glob.glob(os.path.join(root, "*.json")))
print(f"found {len(jsons)} per-run JSONs under {root}")
assert len(jsons) >= 6, f"expected >=6, got {len(jsons)}"

preds = sorted(glob.glob(os.path.join(root, "predictions", "*.csv")))
print(f"found {len(preds)} prediction CSVs")
assert len(preds) >= 6

print()
print("Per-run test metrics (AUC≈0.5 expected at 1K samples):")
print(f"{'run_key':<35} {'n_params':>10} {'AUC':>7} {'F1':>7} {'P':>7} {'R':>7}")
print("-" * 75)
for p in jsons:
    with open(p) as f:
        r = json.load(f)
    m = r["test_metrics"]
    print(f"{r['run_key']:<35} {r['n_params']:>10d} "
          f"{m['auc']:>7.4f} {m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f}")

print()
print("Bit-only token-id sanity check:")
for p in preds:
    if "bitonly" not in p:
        continue
    df = pd.read_csv(p)
    print(f"  {os.path.basename(p)}: rows={len(df)}, "
          f"P(y_true=1)={df['y_true'].mean():.3f}, "
          f"mean y_prob={df['y_prob'].mean():.3f}")
PY

echo ""
echo "=== step 5: summary aggregator ==="
"$PYBIN" -m src.experiments.parity_decomposition_summary \
    --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" --plot_size "$SIZE"

echo ""
echo "Dry run complete. Inspect:"
echo "  $OUTPUT_DIR/*.json"
echo "  $OUTPUT_DIR/predictions/*.csv"
echo "  $OUTPUT_DIR/summary_table.csv"
