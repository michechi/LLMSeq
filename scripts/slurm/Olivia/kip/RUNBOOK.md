# KIP on Olivia — runbook (NeurIPS rebuttal, 2026-07)

Self-contained instructions for running the remaining small-model KIP jobs on
Olivia. No outside context needed: task background is in `/HANDOFF.md` (repo
root) and `repro/src/analysis/mechanism_id/kip_report.md`; this file is the
exact command sequence.

Everything below assumes the Olivia checkout at
`/cluster/home/michechi/MIMICIV` (override by exporting `REPO_ROOT` before
`sbatch`; every script honors it).

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

Seed note for kip_c: the task list said "seeds 2–3", read as the 2nd and 3rd
canonical seeds (9551, 9552) — 9550 already ran locally. Edit `SEEDS` in the
script if literal 2 and 3 were intended.

All jobs append one row each to `results/kip_training.csv` (in-repo, tracked,
`site=olivia` plus host/GPU/SLURM-id columns) and drop per-epoch curves and
training logs into `results/kip_curves/` (tracked). Checkpoints go to
`/cluster/work/projects/nn12048k/michechi/results/kip/checkpoints/`.

---

## 1. Get the code

```bash
cd /cluster/home/michechi/MIMICIV
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
CONTAINER=/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif
export APPTAINER_BIND="/cluster/home/michechi,/cluster/work/projects/nn12048k"
command -v apptainer >/dev/null || module load Apptainer

cd /cluster/home/michechi/MIMICIV/repro
apptainer exec --env DATA_DIR=/cluster/home/michechi/MIMICIV/data "$CONTAINER" \
    python3 -u -m src.generators.kip --build --m 4 6
apptainer exec --env DATA_DIR=/cluster/home/michechi/MIMICIV/data "$CONTAINER" \
    python3 -u -m src.generators.kip --verify --m 4 6          # must end "VERIFY OK"
apptainer exec --env DATA_DIR=/cluster/home/michechi/MIMICIV/data "$CONTAINER" \
    python3 -u -m src.generators.kip_sanity                    # see PASS/FAIL note below
apptainer exec --env DATA_DIR=/cluster/home/michechi/MIMICIV/data "$CONTAINER" \
    python3 -u -m src.data.kip_shuffles                        # build + verify, seed 777
```

**Sanity-suite expectation:** the container has no `xgboost`, so the full
suite prints `10/12 passed -- 2 FAILED` where BOTH failures read
`xgboost missing` (check "5 count purity (XGBoost)", m=4 and m=6) and the exit
code is 1. That exact outcome is expected — count purity was validated locally
(`repro/src/analysis/mechanism_id/results/kip_sanity.txt`, 12/12). **Any other
FAIL line means STOP — do not train.** For a clean 8/8 exit-0 run instead:
append `--skip_counts`.

### sha256 comparison against the manifest (byte-identical proof)

```bash
cd /cluster/home/michechi/MIMICIV
sha256sum -c scripts/slurm/Olivia/kip/kip_data.sha256   # 28 files, all "OK"
git status --short data/simulation/tested               # must print NOTHING
```

If any file mismatches: STOP, do not train. Likely cause: different
pandas/sklearn versions formatting CSVs differently (the generator itself is
pure-stdlib `random` + fixed `train_test_split(random_state=999)`). Fallback:
`rsync` the `data/simulation/tested/` and `data/simulation/kip_shuffled/`
trees from the local machine, or open an issue on the branch.

*(If apptainer is unavailable on the login node, run step 2 as a batch job
instead — it performs build/verify/sanity(--skip_counts)/shuffles/sha-check
in one go:)*

```bash
cd /cluster/home/michechi/MIMICIV/scripts/slurm/Olivia/kip
mkdir -p logs
sbatch kip_0_setup_data.slurm
```

## 3. (Once) create the log dir

```bash
cd /cluster/home/michechi/MIMICIV/scripts/slurm/Olivia/kip
mkdir -p logs
```

## 4. Pre-fetch bert-base-uncased into the HF cache (login node)

Compute nodes may lack internet; jobs run with `HF_HUB_OFFLINE=1`.
`bert-base-uncased` is ungated — **no HF token needed or wanted on Olivia.**

**Layout matters:** the training driver passes `cache_dir=$HF_HOME` to
`from_pretrained`, which reads the FLAT layout
`<cache>/models--bert-base-uncased` — NOT the `<cache>/hub/...` layout that a
plain `HF_HOME`-only download produces. The snippet below therefore passes
`cache_dir` explicitly so the files land exactly where the jobs will look.

```bash
CONTAINER=/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif
CACHE=/cluster/work/projects/nn12048k/michechi/cache
export APPTAINER_BIND="/cluster/home/michechi,/cluster/work/projects/nn12048k"
apptainer exec --env HF_HOME=$CACHE "$CONTAINER" python3 - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir="$CACHE")
AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, cache_dir="$CACHE")
print("bert-base-uncased cached OK")
PY
ls -d $CACHE/models--bert-base-uncased   # MUST exist; if missing, do not submit BERT jobs
```

If the login node needs a proxy for outbound HTTPS, first export the same
proxy the existing templates use:
`export https_proxy=http://10.63.2.48:3128/ http_proxy=http://10.63.2.48:3128/`.

## 5. Submit

```bash
cd /cluster/home/michechi/MIMICIV/scripts/slurm/Olivia/kip
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

Python logging (recipe, per-epoch metrics, the final `DONE` summary) goes to
**stderr**, i.e. the `.err` files; the `.out` files carry only the bash echo
header. The jobs archive each run's `.err` as
`results/kip_curves/olivia_<run>.trainlog.txt`.

```bash
cd /cluster/home/michechi/MIMICIV
CONTAINER=/cluster/work/projects/nn12048k/michechi/extended-pytorch/extended-pytorch.sif
export APPTAINER_BIND="/cluster/home/michechi,/cluster/work/projects/nn12048k"
squeue --me                                              # queue state
ls -lt scripts/slurm/Olivia/kip/logs/ | head             # per-task logs: <name>_<jobid>_<task>.out/.err
tail -f scripts/slurm/Olivia/kip/logs/kip_d_lstm_m6_<jobid>_0.err
grep -h "DONE" scripts/slurm/Olivia/kip/logs/*.err       # driver's one-line summary per finished run
apptainer exec --env LLMSEQ_ROOT=$PWD "$CONTAINER" python3 -c "
import pandas as pd
df = pd.read_csv('results/kip_training.csv')
cols = ['timestamp','model','dataset','mode','seed','epochs_done','val_auc','test_auc','test_f1','site','slurm_job_id']
print(df[df.get('site').fillna('local') == 'olivia'][cols].to_string(index=False))"
```

Each run appends its results row the moment it finishes (fcntl-locked append),
so a later crash never loses earlier rows. Failed task? Fix, then resubmit just
that index: `sbatch --array=<idx> <script>`.

## 7. Follow-up: modes (b)/(c) on converged m6 checkpoints

For each family whose mode-(a) m6 `test_auc` clearly beats chance (say >0.9):

```bash
cd /cluster/home/michechi/MIMICIV/scripts/slurm/Olivia/kip
sbatch --export=ALL,KIP_MODEL=LSTM        kip_f_m6_controls.slurm
# and/or KIP_MODEL=Transformer / KIP_MODEL=BERT
```

Both controls should land at chance (~0.5 AUC); that is the point — the signal
must live in the ordering. Mode (c) reuses the mode-(a) checkpoint saved by
kip_d/e1/e2 on this cluster (`FileNotFoundError` = mode (a) didn't run here).

## 8. Commit results back to the branch

```bash
cd /cluster/home/michechi/MIMICIV
git add results/kip_training.csv results/kip_curves
git status --short          # expect only the CSV + kip_curves files
git commit -m "Olivia KIP results: <which blocks finished>"
git pull --rebase origin Rebuttals_NeurIPS   # CSV has merge=union: local+olivia rows both survive
git push origin Rebuttals_NeurIPS
```

Never commit checkpoints (they live outside the repo on /cluster/work) or the
SLURM `logs/` dir (gitignored). If the rebase ever leaves a duplicate header
line inside the CSV (union merge artifact — possible when both sites edited
the same region), delete the stray header line; rows are self-contained.

## Troubleshooting

- **Import error in container** (`transformers`/`peft`): wrong container —
  jobs must use `extended-pytorch.sif`, not the bare base image.
- **`FileNotFoundError ... ordered checkpoint first`**: mode (c) needs THIS
  site's mode-(a) checkpoint — run kip_d/e1/e2 before kip_f.
- **BERT job dies at model load (`OSError ... couldn't find them in the
  cached files`) despite step 4**: the cache layout is wrong — the driver
  reads `<cache>/models--bert-base-uncased` (flat `cache_dir` layout), not
  `<cache>/hub/models--bert-base-uncased`. Re-run the step-4 snippet exactly
  as written (it passes `cache_dir=` for this reason) and confirm
  `ls -d /cluster/work/projects/nn12048k/michechi/cache/models--bert-base-uncased`.
- **Row missing after a run**: the driver appends only on success — check the
  task's `.err` log.
- **Fresh Claude session on Olivia needs context**: read `/HANDOFF.md`,
  `repro/src/analysis/mechanism_id/kip_report.md`, then this file.
