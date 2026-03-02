# MIMICIV

Research repository for evaluating ML/LLM architectures on clinical sequential data from MIMIC-IV.

## Installation

```bash
# Clone and install in editable mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"

# Or using conda
conda env create -f environment.yml
conda activate mimiciv
pip install -e .
```

## Project Structure

```
MIMICIV/
├── src/
│   ├── experiments/     # Main experiment scripts (LLM, DL, XGBoost)
│   ├── models/          # Model definitions and training utilities
│   ├── data/            # Data processing and prompt generation
│   ├── analysis/        # Metrics, visualization, exploration
│   ├── utils/           # Shared utilities
│   └── mimic/           # MIMIC-IV specific pipeline
├── notebooks/
│   ├── preprocessing/   # Data preparation notebooks
│   ├── models/          # Model exploration notebooks
│   ├── analysis/        # Analysis and visualization
│   └── exploratory/     # Scratch/test notebooks
├── scripts/slurm/       # SLURM job submission scripts
├── simulation/          # Synthetic data generation
├── data/                # Data directory (not tracked)
├── codes/               # Clinical code mappings (CCS)
└── docs/                # Documentation
```

## Running Experiments

All experiment scripts are in `src/experiments/`. Run from the repo root:

### LLM Experiments

```bash
# 8B model (full precision)
python src/experiments/LLM_fraction_experiment.py \
    --number_to_use 9 \
    --model_name meta-llama/Llama-3.1-8B

# 70B model (quantized + LoRA)
python src/experiments/LLM_fraction_experiment.py \
    --number_to_use 9 \
    --model_name meta-llama/Llama-3.1-70B \
    --use_quantization --peft

# Tiny model (local testing)
python src/experiments/LLM_fraction_experiment.py \
    --number_to_use 9 --tiny --tiny_type 1M
```

### Deep Learning Baselines

```bash
python src/experiments/DL_baselines.py --help
python src/experiments/DL_TR_baselines.py --help
```

### XGBoost Baselines

```bash
python src/experiments/XGBoost_fraction_experiment.py --help
```

## HPC (SLURM)

SLURM scripts are in `scripts/slurm/`. Example:

```bash
sbatch scripts/slurm/LLM_fraction_experiment.slurm
```

## Features

- Best-model checkpointing (saves at best epoch, reloads before test)
- 4-bit quantization support for large models
- Optimal F1 threshold selection on validation set
- Multiple prompt formats for clinical narratives
- Fraction experiments (1%, 10%, 30%, 50%, 75%, 100% of data)
