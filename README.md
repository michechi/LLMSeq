# MIMICIV

Research repository for evaluating ML/LLM architectures on clinical sequential data from MIMIC-IV.

## LLM Fraction Experiment

Use `src/LLM_fraction_experiment.py` for all LLM runs.

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

### Features

- Best-model checkpointing (saves at best epoch, reloads before test)
- 4-bit quantization support for large models
- Optimal F1 threshold selection on validation set
