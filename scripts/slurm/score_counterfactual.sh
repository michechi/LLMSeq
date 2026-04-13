#!/bin/bash
#SBATCH --job-name=cf_pairs
#SBATCH --output=logs/counterfactual_%j.out
#SBATCH --error=logs/counterfactual_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ──────────────────────────────────────────────────────────────────────
# Score counterfactual pairs on Tricky Random (dataset 9)
#
# This script trains each model from scratch on the full training set,
# then scores the 47K matched-histogram counterfactual pairs.
#
# Usage:
#   sbatch scripts/slurm/score_counterfactual.sh DL    # LSTM, Transformer, RNNTransformer
#   sbatch scripts/slurm/score_counterfactual.sh BERT   # BERT from scratch
# ──────────────────────────────────────────────────────────────────────

MODE=${1:-DL}

echo "============================================"
echo "Counterfactual pair scoring — mode: $MODE"
echo "============================================"

if [ "$MODE" == "DL" ]; then
    python -m src.experiments.score_counterfactual_pairs \
        --model_type DL \
        --models LSTM,Transformer,RNNTransformer \
        --number_to_use 9 \
        --seed 9950

elif [ "$MODE" == "BERT" ]; then
    python -m src.experiments.score_counterfactual_pairs \
        --model_type LLM \
        --model_name google-bert/bert-base-uncased \
        --number_to_use 9 \
        --seed 9950

else
    echo "Unknown mode: $MODE. Use DL or BERT."
    exit 1
fi
