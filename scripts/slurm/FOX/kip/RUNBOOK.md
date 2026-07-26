# KIP on FOX — runbook (NeurIPS rebuttal, 2026-07)

Self-contained instructions for running the remaining small-model KIP jobs on
FOX (Educloud, SLURM). No outside context needed: task background is in
`/HANDOFF.md` (repo root) and `repro/src/analysis/mechanism_id/kip_report.md`;
this file is the exact command sequence.

FOX runs everything from the NLPL **2024a** module stack (torch 2.6.0 +
cuda 12.6, python 3.12.3, transformers 4.55.4); there are no containers.
Do NOT use the older 2022b stack (transformers 4.47.1, torch 2.1.2 +
cuda 12.0): it predates the H200 nodes and crashes on them with
`CUDA error: no kernel image is available` (verified 2026-07-26).
transformers 4.55.4 vs the paper's 4.47.1 is a recorded, accepted deviation;
the LSTM/Transformer runs are pure torch and unaffected.

All jobs request `--partition=accel --gpus=nvidia_h200_nvl:1` (H200 NVL, per
2026-07-26 site instruction; `sinfo` shows 6 of them in accel). Fallbacks in
accel per `sinfo -o "%P %G"`: `a100`/`a100_80` (work with either stack),
`h100nv` (sm_90 — also needs the 2024a stack).

Everything below assumes the FOX checkout at `$HOME/MIMICIV` (override by
exporting `REPO_ROOT` before `sbatch`; every script honors it).

---

## 0. What will run (submit order)

| # | script | what | array | wall | expectation |
|---|--------|------|-------|------|-------------|
| 1 | `kip_d_lstm_m6.slurm` **PRIORITY** | LSTM m6 ordered, seeds 9550–9552, patience 5 | 0–2 | 2 h | open question — the rebuttal's key number |
| 2 | `kip_a_anchor.slurm` | LSTM m4 ordered, seed 9550, patience 5 | 0 | 1 h | **test_auc = 1.0000** (matches local A100; gate for all other runs) |
| 3 | `kip_b_lstm_m4_seeds.slurm` | LSTM m4 ordered, seeds 9570–9574, patience 5 | 0–4 | 1.5 h | likely 1.0000 per seed |
| 4 | `kip_c_bert_m4.slurm` | BERT m4 ordered, seeds 9551+9552, max_len 512 | 0–1 | 20 h | likely chance (local seed 9550: 0.5002) |
| 5 | `kip_e1_transformer_m6.slurm` | Transformer m6 ordered, seeds 9550–9552, patience 5 | 0–2 | 2 h | likely chance (m4 was chance 6/6) |
| 6 | `kip_e2_bert_m6.slurm` | BERT m6 ordered, seeds 9550–9552, max_len 64 | 0–2 | 8 h | unknown |
| later | `kip_f_m6_controls.slurm` | modes (b)+(c) on any m6 family that converged | 0–5 | 8 h | chance (control) |
| optional | `kip_g_llama_m4.slurm` | Llama-3.2-1B LoRA m4 ordered, seeds 9551+9552 | 0–1 | 8 h | likely chance (local seed 9550 flat at ~0.503) — **needs HF token**, see below |

Seed note for kip_c: the task list said "seeds 2–3", read as the 2nd and 3rd
canonical seeds (9551, 9552) — 9550 already ran locally. Edit `SEEDS` in the
script if literal 2 and 3 were intended.

All jobs append one row each to `results/kip_training.csv` (in-repo, tracked,
`site=fox` plus host/GPU/SLURM-id columns) and drop per-epoch curves and
training logs into `results/kip_curves/` (tracked, `fox_*` prefix).
Checkpoints go to `$REPO_ROOT/checkpoints/kip/` (gitignored; BERT arms total
roughly 4 GB — keep them until the kip_f controls have run, since mode (c)
re-loads the mode-(a) checkpoints).

## 1. Get the code

```bash
cd $HOME/MIMICIV
git fetch origin
git checkout Rebuttals_NeurIPS
git pull --ff-only origin Rebuttals_NeurIPS
```

## 2. Regenerate + verify the data (login node, ~10–20 min, CPU only)

The 14 canonical KIP files (`data/simulation/tested/{X,y}_{train,val,test}_kip_m{4,6}.csv`
+ 2 `*_rule.json` sidecars) are tracked in git, so the pull already placed
them. Regenerating them in place and checking nothing changed proves the
generator is byte-identical across sites; the shuffled controls are NOT
tracked and must be built here.

```bash
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3
module load nlpl-bitsandbytes/0.46.1-foss-2024a-Python-3.12.3

# quick import check before anything else (catches a missing package in the
# stack in seconds instead of a failed job):
python3 -c "import torch, transformers, peft, sklearn, pandas; \
print(torch.__version__, transformers.__version__, peft.__version__)"

cd $HOME/MIMICIV/repro
export DATA_DIR=$HOME/MIMICIV/data
python3 -u -m src.generators.kip --build --m 4 6
python3 -u -m src.generators.kip --verify --m 4 6      # must end "VERIFY OK"
python3 -u -m src.generators.kip_sanity                # see PASS/FAIL note below
python3 -u -m src.data.kip_shuffles                    # build + verify, seed 777
```

**Sanity-suite expectation:** `12/12 passed` if the module stack provides
xgboost. If it does not, the suite prints `10/12 passed  -- 2 FAILED` where
BOTH failures read `xgboost missing` (check "5 count purity (XGBoost)", m=4
and m=6) — that outcome is also acceptable, because count purity was already
validated locally at 12/12 (tracked:
`repro/src/analysis/mechanism_id/results/kip_sanity.txt`). **Any other FAIL
line means STOP — do not train.**

### sha256 comparison against the manifest (byte-identical proof)

```bash
cd $HOME/MIMICIV
sha256sum -c scripts/slurm/FOX/kip/kip_data.sha256   # 28 files, all "OK"
git status --short data/simulation/tested            # must print NOTHING
```

If any file mismatches: STOP, do not train. Likely cause: different
pandas/sklearn versions formatting CSVs differently (the generator itself is
pure-stdlib `random` + fixed `train_test_split(random_state=999)`). Fallback:
`rsync` the `data/simulation/tested/` and `data/simulation/kip_shuffled/`
trees from the local machine, or open an issue on the branch.

*(If the login node cannot run this, submit it as a batch job instead:)*

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/kip
mkdir -p logs
sbatch kip_0_setup_data.slurm
```

## 3. (Once) create the log dir

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/kip
mkdir -p logs
```

## 4. HF cache (no token, no prefetch required)

`bert-base-uncased` is ungated and FOX compute nodes have internet (the
existing FOX templates download Llama/Qwen inside jobs), so the BERT jobs
simply download it (~440 MB) into `$HF_HOME` on first use. **No HF token is
needed or wanted for these jobs.**

Optional, to avoid one download per BERT task: pre-fetch once on the login
node into a persistent dir and export `KIP_HF_CACHE` before submitting (the
jobs honor it). The `cache_dir=` argument below is required — the training
driver reads the FLAT `<cache>/models--bert-base-uncased` layout, not the
`<cache>/hub/...` layout a plain `HF_HOME` download would produce:

```bash
# (needs the step-2 module stack loaded for transformers)
export KIP_HF_CACHE=$HOME/hf_cache_kip
python3 - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="$KIP_HF_CACHE")
AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, cache_dir="$KIP_HF_CACHE")
print("bert-base-uncased cached OK")
PY
ls -d $KIP_HF_CACHE/models--bert-base-uncased   # must exist
# keep KIP_HF_CACHE exported in the same shell you run sbatch from
```

## 5. Submit

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/kip
sbatch kip_d_lstm_m6.slurm          # PRIORITY — the rebuttal's key number
sbatch kip_a_anchor.slurm           # cross-site anchor, must hit 1.0000
sbatch kip_b_lstm_m4_seeds.slurm
sbatch kip_c_bert_m4.slurm
sbatch kip_e1_transformer_m6.slurm
sbatch kip_e2_bert_m6.slurm
```

The arrays are independent; SLURM runs them as GPUs free up. **Before trusting
anything else, check the anchor**: its row in `results/kip_training.csv` must
show `test_auc = 1.0` (locally: epochs_done 4, val_auc 1.0). If it does not,
something differs between sites — stop and investigate before reading meaning
into any m6 number.

## 6. Monitor

FOX jobs write ONE log per task (stdout+stderr merged into the `.out` file),
so the per-epoch python logging and the final `DONE` summary are all there.

```bash
cd $HOME/MIMICIV
squeue --me                                          # queue state
ls -lt scripts/slurm/FOX/kip/logs/ | head            # per-task logs: <name>_<jobid>_<task>.out
tail -f scripts/slurm/FOX/kip/logs/kip_d_lstm_m6_<jobid>_0.out
grep -h "DONE" scripts/slurm/FOX/kip/logs/*.out      # driver's one-line summary per finished run
python3 -c "
import pandas as pd
df = pd.read_csv('results/kip_training.csv')
cols = ['timestamp','model','dataset','mode','seed','epochs_done','val_auc','test_auc','test_f1','site','slurm_job_id']
print(df[df.get('site').fillna('local') == 'fox'][cols].to_string(index=False))"
```

(The python one-liner needs the module stack from step 2 loaded for pandas.)

Each run appends its results row the moment it finishes (fcntl-locked append),
so a later crash never loses earlier rows. Failed task? Fix, then resubmit just
that index: `sbatch --array=<idx> <script>`.

## 7. Follow-up: modes (b)/(c) on converged m6 checkpoints

For each family whose mode-(a) m6 `test_auc` clearly beats chance (say >0.9):

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/kip
sbatch --export=ALL,KIP_MODEL=LSTM        kip_f_m6_controls.slurm
# and/or KIP_MODEL=Transformer / KIP_MODEL=BERT
```

Both controls should land at chance (~0.5 AUC); that is the point — the signal
must live in the ordering. Mode (c) reuses the mode-(a) checkpoint saved by
kip_d/e1/e2 in `$REPO_ROOT/checkpoints/kip/` (`FileNotFoundError` = mode (a)
didn't run here, or CKPT_ROOT changed between runs).

## 7b. Optional: Llama seeds (kip_g)

Completes the Llama m4 triplet (seed 9550 runs locally as Block D).
`meta-llama/Llama-3.2-1B` is **gated** — this is the one job that needs your
HF credentials. Either have a cached `huggingface-cli login` (the script
falls back to `~/.cache/huggingface/token`) or export the token in the shell
you submit from; it is never echoed or logged:

```bash
cd $HOME/MIMICIV/scripts/slurm/FOX/kip
export HF_TOKEN=<your token>        # skip if huggingface-cli login was done
sbatch kip_g_llama_m4.slurm
```

~2.5 GB model download per task unless `KIP_HF_CACHE` is set (step 4);
~7–10 min/epoch expected on H200, 20-epoch cap, patience 3 on val loss.

## 8. Commit results back to the branch

```bash
cd $HOME/MIMICIV
git add results/kip_training.csv results/kip_curves
git status --short          # expect only the CSV + kip_curves files
git commit -m "FOX KIP results: <which blocks finished>"
git pull --rebase origin Rebuttals_NeurIPS   # CSV has merge=union: local+fox rows both survive
git push origin Rebuttals_NeurIPS
```

Never commit checkpoints (`checkpoints/` is gitignored) or the SLURM `logs/`
dir (gitignored). If the rebase ever leaves a duplicate header line inside the
CSV (union merge artifact — possible when both sites edited the same region),
delete the stray header line; rows are self-contained (the summary script
tolerates it either way).

## Troubleshooting

- **`CUDA error: no kernel image is available for execution on the device`**:
  the job ran with the 2022b module stack (torch 2.1.2 + cuda 12.0, built
  before the H200s, no sm_90 kernels). Use the 2024a module block exactly as
  in the scripts; or switch the `#SBATCH` GPU line to `--gpus=a100_80:1`
  (A100s accept either stack; `h100nv` needs 2024a like the H200).
- **sbatch rejects `--gpus=nvidia_h200_nvl:1`**: list valid gres types with
  `sinfo -o "%P %G"` and adjust the type name; the older FOX templates used
  `--gpus=a100:1` and every KIP job fits an A100-40GB.
- **`ModuleNotFoundError: peft` (or transformers)**: the module stack didn't
  load — rerun the `module use -a ... && module load nlpl-...` block; the
  driver imports transformers/peft even for LSTM runs.
- **`FileNotFoundError ... ordered checkpoint first`**: mode (c) needs THIS
  site's mode-(a) checkpoint in `$REPO_ROOT/checkpoints/kip/` — run
  kip_d/e1/e2 first with the same CKPT_ROOT.
- **BERT job dies at model load with `OSError ... couldn't find them in the
  cached files`**: only possible with a stale `KIP_HF_CACHE` — either unset it
  (jobs then download fresh) or re-run the step-4 snippet exactly as written
  (it passes `cache_dir=` so the layout matches what the driver reads).
- **Row missing after a run**: the driver appends only on success — check the
  task's `.out` log tail.
- **Fresh Claude session on FOX needs context**: read `/HANDOFF.md`,
  `repro/src/analysis/mechanism_id/kip_report.md`, then this file.
