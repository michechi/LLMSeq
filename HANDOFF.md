# HANDOFF — KIP task state (NeurIPS rebuttal)

Last updated: 2026-07-26 (local blocks C/D still in flight — check
`results/kip_training.csv` for rows newer than this file).

## What KIP is

KIP (Key-Inversion Parity) is the rebuttal's new task variant: each sequence
(20 uppercase letters) contains the `m` hidden key letters exactly once;
reading them in sequence order through the sampled bijection `kappa` gives a
permutation `pi`, and `Y = 1 iff inv(pi) is even`. Tags `kip_m4`, `kip_m6`.
The hidden rule (S, kappa) is **sampled per dataset** and stored beside the
CSVs as `<TAG>_rule.json` — always load the sidecar, never hardcode key sets.
Full definition, sanity ladder, and CPU-probe results:
`repro/src/analysis/mechanism_id/kip_report.md`.

Headline CPU facts: all 12 sanity checks pass; fixed-lag linear probes sit at
chance and fixed-lag XGBoost tops out at 0.524 (flagged marginal in the
report); XGBoost on the pairwise precedence matrix hits AUC 1.000 at m=4 but
~0.502 at m=6; an MLP on precedence bits hits 1.000 at both m. So m=4 is
solvable from pairwise order statistics, m=6 requires more than pairwise —
that contrast is the point of the GPU runs.

## Data

- Canonical: `data/simulation/tested/{X,y}_{train,val,test}_kip_m{4,6}.csv`
  + `kip_m{4,6}_rule.json` (14 files, **tracked in git**). 400K/50K/50K rows.
- Shuffled controls (modes b/c): `data/simulation/kip_shuffled/` (not tracked;
  deterministic rebuild via `python -m src.data.kip_shuffles`, seed 777).
- Regeneration: `python -m src.generators.kip --build --m 4 6` (seed 959693,
  deterministic, byte-identical across sites); `--verify` re-derives labels;
  `python -m src.generators.kip_sanity` is the 12-check suite.
- Cross-site proof: `scripts/slurm/Olivia/kip/kip_data.sha256` (28 files).
- All repro-tree code needs `DATA_DIR=<repo>/data` exported and runs as
  `cd <repo>/repro && python -m src.<module>`.

## Training driver and conventions

One run = one invocation of `repro/src/experiments/kip_gpu_training.py`
(`--model {LSTM,Transformer,BERT,Llama1B} --tag kip_m{4,6}
--mode {ordered,shuffled_train,shuffled_eval} --seed N`). It reuses the paper
scripts' exact recipes (see its docstring), saves a checkpoint + per-epoch
`curve.json` (DL families), and appends one fcntl-locked row to
`results/kip_training.csv`.

- **Modes**: (a) `ordered` = train/test on real data; (b) `shuffled_train` =
  train on per-sequence-shuffled data, original labels (expect chance);
  (c) `shuffled_eval` = evaluate the mode-(a) checkpoint on shuffled test
  (expect chance). (c) needs the same-site mode-(a) checkpoint.
- **Patience**: DL patience 3 = "as-run" legacy (what every committed paper
  invocation executed); **patience 5 = the paper-appendix recipe and the
  going-forward convention** (`--dl_patience 5`). It mattered: at patience 3
  LSTM m4 hit 1.0 on only 2/3 seeds; at patience 5, 3/3. BERT keeps its own
  script defaults (patience 3, early=val-loss) — no BERT patience flag exists.
- **Seeds**: canonical triplet 9550/9551/9552; extension seeds 9570–9574.
  Report per-seed, never only the mean.
- **BERT max_length**: 512 = paper fidelity (used for kip_m4 ordered); 64 =
  user-approved deviation for kip_m6 + controls (longest real prompt ~43
  tokens; recorded in the recipe column either way).
- **Results CSV**: tracked in git, append-only, `merge=union` in
  .gitattributes. Columns include provenance (`site,host,gpu,slurm_job_id`);
  `site=local` is the A100 box, `site=olivia` is Sigma2 Olivia. Rows written
  by processes that predate the 2026-07-26 schema migration may lack the last
  4 fields — pandas reads them as NaN; backfill with
  `,local,hazy-book-sings-fin-03,NVIDIA A100-SXM4-80GB,` if needed.
  Patience-5 m4 rows from 2026-07-25 live in `results/kip_training_p5.csv`
  (kept separate so they never mix with the as-run patience-3 block).
- **Curves**: DL runs write `curve.json` next to the checkpoint (added
  2026-07-26 — local runs completed before then have per-epoch curves only in
  `logs/kip/*.log`); cluster jobs copy curves + the `.err` training logs into
  `results/kip_curves/` (tracked). BERT curves are the per-epoch lines in the
  training log — python logging goes to stderr.
- **`epochs_done` caveat**: for DL rows the column records the BEST epoch
  (early-stopping selection), for BERT/Llama rows the number of epochs
  actually executed. Don't compare it across families; DL's full history is
  in `curve.json`.

## Results so far (as of 2026-07-26 morning; per-seed test_auc)

| family | tag | mode (a) ordered | controls (b)/(c) |
|---|---|---|---|
| LSTM p5 | m4 | **1.0000 / 1.0000 / 1.0000** (9550-52, `kip_training_p5.csv`) | — |
| LSTM p3 | m4 | 1.0 / 1.0 / 0.498 (seed 9552 collapsed at patience 3) | all ~0.50 ✓ |
| Transformer (p3+p5) | m4 | chance, 6/6 runs (~0.50, epochs_done 1–3) | all ~0.50 ✓ |
| BERT (512, peft) | m4 | 0.5002 (seed 9550 only, 7.3 h) | ~0.50 ✓ (b @64, c @512) |
| all | m6 | local Block C in flight (tmux kip_blockC) | pending |
| Llama-3.2-1B LoRA | m4 | local Block D in flight (tmux kip_blockD) | pending |

Interpretation so far: only the LSTM learns KIP at m=4, consistent with the
CPU probes (pairwise precedence suffices at m=4). m=6 — where pairwise
stops sufficing — is the decisive experiment, hence its priority on Olivia.

## Olivia task list (cross-site runs)

Package: `scripts/slurm/Olivia/kip/` — **start at `RUNBOOK.md` there**; it is
self-contained (data regen → sha256 verify → BERT prefetch → sbatch order →
monitoring → committing rows back).

1. `kip_d_lstm_m6.slurm` — **PRIORITY**: LSTM m6 (a), 3 seeds, patience 5.
2. `kip_a_anchor.slurm` — anchor LSTM m4 (a) seed 9550 p5; must reproduce
   test_auc 1.0000 (cross-site validation gate).
3. `kip_b_lstm_m4_seeds.slurm` — LSTM m4 (a), new seeds 9570–9574, p5.
4. `kip_c_bert_m4.slurm` — BERT m4 (a), seeds 9551+9552 @512 ("seeds 2–3" of
   the canonical triplet; 9550 done locally).
5. `kip_e1_transformer_m6.slurm` + `kip_e2_bert_m6.slurm` — m6 (a), 3 seeds
   each (BERT @64).
6. `kip_f_m6_controls.slurm` — follow-up modes (b)/(c) per converged m6 family.

## Environments

- **Local A100 box** (`hazy-book-sings-fin-03`, site=local): venv at
  `/root/kip-venv` (system python is stdlib-only). Run as
  `cd /root/LLMSeq/repro && DATA_DIR=/root/LLMSeq/data /root/kip-venv/bin/python -m src.<module>`.
  Training only inside detached tmux; never kill a session to inspect it.
  Llama (Block D) needs HF_TOKEN in the tmux env — never echo/log it.
- **Olivia (Sigma2, site=olivia)**: SLURM account NN12048K, partition accel,
  container `extended-pytorch.sif` (transformers 4.47.1 + peft + sklearn;
  **no xgboost** — sanity check 5's XGB arm reports "xgboost missing" there,
  the only acceptable sanity failure). No HF token on Olivia; BERT weights
  pre-fetched to the permanent cache, jobs run `HF_HUB_OFFLINE=1`.
  Checkpoints: `/cluster/work/projects/nn12048k/michechi/results/kip/checkpoints/`.
