# MIMICIV
Repo with all the passages for preprocessing pipeline of MIMICIV on DataCrunch

## LLM Fraction Experiment Scripts

There are three copies of the LLM fraction experiment script. **Use `src/LLM_fraction_experiment.py`** for all runs.

| File | Status | Notes |
|------|--------|-------|
| `src/LLM_fraction_experiment.py` | **Use this one** | Handles both small and large models. Saves best-epoch checkpoint and supports quantization-compatible device placement. |
| `src/LLM_fraction_experiment_quant.py` | Deprecated | Older version that skips best-model checkpointing (evaluates test on last epoch's weights). Was a workaround for 70B RAM concerns, but no longer needed. Safe to delete. |
| `LLM_fraction_experiment.py` (repo root) | Deprecated | Original version before quantization support. Uses `--number_to_use` as `int` instead of `str`, lacks quantization device guards, and missing `0.1M` tiny size. |

### Usage

8B model (full precision):
```bash
python src/LLM_fraction_experiment.py --number_to_use 9 --model_name meta-llama/Llama-3.1-8B
```

70B model (quantized + LoRA):
```bash
python src/LLM_fraction_experiment.py --number_to_use 9 --model_name meta-llama/Llama-3.1-70B --use_quantization --peft
```

Tiny model (local testing):
```bash
python src/LLM_fraction_experiment.py --number_to_use 9 --tiny --tiny_type 1M
```

### Key differences in `src/LLM_fraction_experiment.py`

- **Best-model checkpointing**: saves backbone and classification_head state dicts separately at the best epoch, reloads before test evaluation. With 4-bit quantization a 70B model is ~35GB, which fits in RAM.
- **Quantization guards**: skips `.to(device)` when `--use_quantization` is set, since `device_map='auto'` already places the model on GPU.
- **Optimal threshold**: finds the F1-maximizing threshold on the validation set before evaluating on test.
