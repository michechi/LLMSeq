# OC full fine-tuning rebuttal arm (FOX)

**Question:** do the paper's OC decoder conclusions (Llama + LoRA fails the
order-critical noisy task) hold under **full fine-tuning**, i.e. is LoRA the
bottleneck rather than the architecture?

**Design:** Llama-3.2-1B and Llama-3.1-8B on OC NOISY (`tricky_rnd`, dataset
tag `9`, tracked in `data/simulation/tested/*_9.csv`), 100% of the training
data (400K rows), via the exact paper pipeline
(`repro/src/experiments/LLM_fraction_experiment.py`). The recipe is the
as-run tag-9 letter recipe verbatim (`scripts/slurm/wrapper_to_launch.sh`
defaults): batch 64, max_length 50, AdamW lr 2e-5, epochs ≤ 20, patience 3
on val loss, threshold tuned on val, seed 8888. **Only the unfreezing
changes:** `--peft` is dropped (and for 8B, `--use_quantization`, which was
an A100-40GB memory workaround — full FT cannot train 4-bit weights and the
H200's 141 GB doesn't need it). transformers 4.55.4 vs the paper's 4.47.1 is
the same recorded stack deviation as the KIP runs.

## Run

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/oc_fullft
mkdir -p logs
export HF_TOKEN=<token>       # gated meta-llama; or cached huggingface-cli login
sbatch oc_fullft_llama1b.slurm    # ~4 h cap
sbatch oc_fullft_llama8b.slurm    # ~24-30 h cap, usually early-stops long before
```

Seed override: `OC_SEED=9550 sbatch ...` (default 8888 = the as-run wrapper
default). For extra seeds just resubmit with different `OC_SEED` values —
each run writes its own timestamped results file.

## Outputs

- `results/oc_fullft/llm_fraction_experiment_meta-llama_Llama-3.2-1B_9_<ts>.csv`
  (and the 8B analogue). No `_lora` suffix in the filename = full FT. One row
  (fraction 1.0) with val/test AUC + F1, threshold, epochs, wall time.
- `results/oc_fullft/fox_oc_fullft_llama{1b,8b}_seed<seed>.trainlog.txt` —
  the full job log with per-epoch val metrics.
- Both are tracked; commit them back like the KIP results
  (`git add results/oc_fullft && git commit && git pull --rebase && git push`).

## Interpretation

- Paper baseline (LoRA): decoders sit at/near chance on OC noisy.
- Full FT also ~0.5 AUC → the conclusion holds; LoRA was not the limiter.
- Full FT ≫ 0.5 → real finding: report the delta and flag that the paper's
  decoder claim is adapter-specific. Check the per-epoch trainlog before
  believing either outcome (flat train loss = genuinely not learning).

No checkpoints are saved by this pipeline (results CSV + log only).
