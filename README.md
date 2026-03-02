# MIMICIV - Sequential Clinical Data Analysis

Research repository for evaluating ML/LLM architectures on clinical sequential data, comparing how different model families learn from temporal medical event sequences.

## Research Overview

This project investigates:

1. **Data efficiency**: How much training data do different architectures need? (1%, 10%, 30%, 50%, 75%, 100%)
2. **Sequential understanding**: Can models learn meaningful patterns from ordered clinical events?
3. **Pre-training benefits**: Do pre-trained LLMs outperform architectures trained from scratch?
4. **Positional information**: How important is event ordering vs. bag-of-events representations?

### Models Evaluated

| Category | Models |
|----------|--------|
| **Pre-trained LLMs** | Llama-3.1-8B, Llama-3.1-70B, Qwen models |
| **LLMs from scratch** | Tiny Llama/Qwen (0.1M - 50M params) |
| **Deep Learning** | Transformer, BiLSTM, LSTM, GRU, CNN1D, BiLSTM-Transformer hybrid |
| **Traditional ML** | XGBoost with ordinal/categorical/TF-IDF/LLM embeddings |

### Clinical Outcomes

- **MIMIC-IV**: 90-day mortality prediction for CKD (Chronic Kidney Disease) cohort
- **Simulation**: Synthetic sequential data for controlled experiments

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd MIMICIV

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Alternative: using conda
conda env create -f environment.yml
conda activate mimiciv
pip install -e .
```

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (recommended)
- ~16GB GPU memory for 8B models
- ~40GB GPU memory for 70B models (with 4-bit quantization)

## Project Structure

```
MIMICIV/
├── src/
│   ├── experiments/      # Main experiment scripts
│   │   ├── LLM_fraction_experiment.py      # LLM fine-tuning
│   │   ├── DL_baselines.py                 # LSTM/GRU baselines
│   │   ├── DL_TR_baselines.py              # Transformer/CNN/hybrid
│   │   └── XGBoost_fraction_experiment.py  # Traditional ML
│   ├── models/           # Model definitions
│   ├── data/             # Data processing & prompts
│   ├── analysis/         # Metrics & visualization
│   ├── utils/            # Utilities
│   └── mimic/            # MIMIC-IV pipeline
├── notebooks/
│   ├── preprocessing/    # Data preparation
│   ├── models/           # Model exploration
│   ├── analysis/         # Results analysis
│   └── exploratory/      # Scratch notebooks
├── scripts/
│   ├── slurm/            # HPC job scripts
│   └── containers/       # Apptainer definitions
├── simulation/           # Synthetic data generation
├── data/                 # Data directory (not tracked)
├── codes/                # Clinical code mappings (CCS)
├── paper/                # LaTeX and figures
└── docs/                 # Documentation
```

## Data

### MIMIC-IV (Real Clinical Data)

Requires [PhysioNet](https://physionet.org/) credentialed access to MIMIC-IV v3.1.

**Cohort**: Patients with Chronic Kidney Disease (CKD)
**Target**: 90-day mortality (binary)
**Features**: Diagnosis codes mapped to CCS (Clinical Classification Software)
**Format**: Delimited sequences of CCS codes

```
# Example sequence
"CCS001\x1fCCS045\x1fCCS123\x1fCCS067"
```

**Preprocessing pipeline**: `src/mimic/prepare_training_data.py`

### Simulation Data (Synthetic)

For controlled experiments without PHI concerns.

```bash
cd simulation
python main.py
```

**Properties**:
- 300,000 sequences (200K negative, 100K positive)
- 5 events per sequence
- Vocabulary: 26 letters + 10 digits
- Format: `"A1\x1fB2\x1fC3\x1fD4\x1fE5"`

## Running Experiments

All commands should be run from the repository root.

### LLM Fine-tuning

```bash
# Llama 8B (full precision, ~16GB VRAM)
python src/experiments/LLM_fraction_experiment.py \
    --model_name meta-llama/Llama-3.1-8B \
    --number_to_use 9 \
    --batch_size 8 \
    --epochs 20 \
    --lr 2e-5

# Llama 70B (4-bit quantized + LoRA, ~40GB VRAM)
python src/experiments/LLM_fraction_experiment.py \
    --model_name meta-llama/Llama-3.1-70B \
    --number_to_use 9 \
    --use_quantization \
    --peft \
    --batch_size 4

# Tiny model for testing (CPU/small GPU)
python src/experiments/LLM_fraction_experiment.py \
    --tiny --tiny_type 1M \
    --number_to_use 9

# Cold start (random init, no pre-training)
python src/experiments/LLM_fraction_experiment.py \
    --model_name meta-llama/Llama-3.1-8B \
    --cold_start \
    --number_to_use 9
```

**Key arguments**:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model ID | meta-llama/Llama-3.1-8B |
| `--number_to_use` | Dataset identifier | 9 |
| `--batch_size` | Training batch size | 8 |
| `--epochs` | Max epochs | 20 |
| `--lr` | Learning rate | 2e-5 |
| `--max_length` | Max token length | 512 |
| `--peft` | Use LoRA fine-tuning | False |
| `--use_quantization` | 4-bit quantization | False |
| `--cold_start` | Random initialization | False |
| `--early` | Early stopping metric (auc/loss/f1) | loss |
| `--fractions` | Data fractions to test | 0.01,0.10,0.30,0.50,0.75,1.0 |
| `--seed` | Random seed | 9550 |

### Deep Learning Baselines

```bash
# LSTM/BiLSTM/GRU with hyperparameter search
python src/experiments/DL_baselines.py \
    --model_type bilstm \
    --csv_to_use 9 \
    --n_trials 20

# Transformer encoder
python src/experiments/DL_TR_baselines.py \
    --model_type transformer \
    --csv_to_use 9

# BiLSTM-Transformer hybrid
python src/experiments/DL_TR_baselines.py \
    --model_type rnn_transformer \
    --csv_to_use 9
```

**Available architectures**:
- `lstm`, `bilstm`, `gru` (DL_baselines.py)
- `transformer`, `cnn`, `mlp`, `rnn_transformer` (DL_TR_baselines.py)

### XGBoost Baselines

```bash
# Ordinal encoding (A=1, B=2, ..., Z=26)
python src/experiments/XGBoost_fraction_experiment.py \
    --csv_to_use 9 \
    --run_basic

# TF-IDF encoding (character n-grams)
python src/experiments/XGBoost_fraction_experiment.py \
    --csv_to_use 9 \
    --run_tfidf

# LLM embeddings (Llama mean pooling)
python src/experiments/XGBoost_fraction_experiment.py \
    --csv_to_use 9 \
    --run_llm \
    --model_name meta-llama/Llama-3.1-8B
```

## Prompt Formats

Multiple prompt templates for converting clinical sequences to text (`src/data/prompts.py`):

| Prompt | Description |
|--------|-------------|
| `no_narrative_prompt` | Unordered set of clinical facts |
| `naive_narrative_prompt` | Visit-by-visit chronological history |
| `compact_narrative_prompt` | Compressed format with chronic vs. new conditions |
| `compact_no_time_prompt` | Compact without temporal information |
| `full_narrative` | Complete temporal history with exact timing |
| `temporal_causal_prompt` | Optimized for next-token prediction |

## HPC (SLURM)

Job scripts are in `scripts/slurm/`:

```bash
# Submit a job
sbatch scripts/slurm/LLM_fraction_experiment.slurm

# Check queue
squeue -u $USER
```

Configure environment variables in SLURM scripts:
- `HF_HOME`: HuggingFace cache directory
- `SCRATCH`: Scratch space for outputs

## Configuration

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"  # For gated models (Llama)
export SCRATCH="/path/to/scratch"          # Output directory
export CUDA_VISIBLE_DEVICES="0"            # GPU selection
```

### Data Fractions

All experiments support testing across different training data sizes:

```python
fractions = [0.01, 0.10, 0.30, 0.50, 0.75, 1.0]
```

This enables analysis of data efficiency and learning curves.

## Output

Results are saved to the specified output directory:

```
results/
├── {model}_{fraction}_{seed}/
│   ├── metrics.json        # AUC, F1, accuracy
│   ├── predictions.csv     # Test set predictions
│   └── checkpoints/        # Model weights
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mimiciv-llm-research,
  author = {Michele Chini},
  title = {Do Large Language Models Detect and Exploit Decisive Sequential Information?},
  publisher = {GitHub},
  url = {https://github.com/...}
}
```

## License

[Add license information]

## Acknowledgments

- MIMIC-IV dataset from PhysioNet
- HuggingFace Transformers library
- Meta AI for Llama models
