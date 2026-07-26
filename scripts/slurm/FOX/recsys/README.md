# RecSys audit grid on FOX — job package

Full experimental spec: **"RECSYS AUDIT" section of `/HANDOFF.md`** (self-
contained: dataset gates, protocol facts of ref [25], audit-variant
definitions, consumer contract). This directory only packs it into SLURM.

Prerequisites already satisfied (2026-07-26): data rsynced to
`~/MIMICIV/data/recsys/30music/{split,audit}`, `sha256sum -c` 16/16 OK on
FOX, `eval_loader --check-all` PASSED (10/10 files). Job 1 re-runs all three
gates anyway before training — byte checks AND the target-identity/contract
check must pass or the job aborts.

## Jobs (≤2, sequential drivers, resumable)

| job | contents | wall |
|---|---|---|
| `recsys_job1_grid.slurm` | SASRec+GRU4Rec+BERT4Rec × seeds {17,32,45} ordered training; every checkpoint evaluated on ordered + 9 shuffle files; MostPopular/ItemKNN/Markov-1/Markov-2 on the same 10 inputs; report table | 36 h |
| `recsys_job2_shuffled_train.slurm` | **OPTIONAL — only after the Job-1 table is reviewed**: GRU4Rec shuffled-train, seed 17 | 12 h |

```bash
cd $HOME/MIMICIV && git pull --ff-only origin Rebuttals_NeurIPS
cd scripts/slurm/FOX/recsys && mkdir -p logs
sbatch recsys_job1_grid.slurm
# STOP. Review results/recsys_audit/report_job1.md before considering job 2.
```

The models are a few M parameters: one GPU, and if the H200s are saturated
by the Q1 full-FT jobs, `sbatch --gpus=a100_80:1 recsys_job1_grid.slurm`
runs identically without touching them.

## Outputs (all tracked, merge=union on the CSVs)

- `results/recsys_audit/training.csv` — one row per trained cell (best epoch,
  best val NDCG@10, batch, wall, site, SLURM id).
- `results/recsys_audit/metrics.csv` — one row per (model|baseline,
  eval_input): HR@10 / NDCG@10 / Jaccard@10 with per-row user-bootstrap 95%
  CIs, Δabs/Δrel vs ordered with paired CIs, order_invariant flag.
- `results/recsys_audit/recs/*.csv.gz` — top-10 lists per run (offline
  recomputation of every number).
- `results/recsys_audit/report_job1.md` — the deliverable table: models ×
  {ordered, full_shuffle, keep_last, early_half} (mean over 3 shuffle seeds,
  bootstrap CIs, Δs), Jaccard@10, and the baseline ladder (reach +
  drop_explained per rung).
- Checkpoints + per-epoch curves: `checkpoints/recsys/` (gitignored).

Per-epoch val metrics stream into the job log and `*_curve.json` next to
each checkpoint.

## Commit back

```bash
cd $HOME/MIMICIV
git add results/recsys_audit
git commit -m "RecSys audit: Job 1 grid results (fox)"
git pull --rebase origin Rebuttals_NeurIPS && git push origin Rebuttals_NeurIPS
```

## Notes

- RecBole 1.2.0 (MIT) provides the model classes and objectives; training/
  eval loops are ours so data ingestion obeys the audit consumer contract
  (RecBole's own dataset pipeline re-sorts by timestamp — forbidden on
  30Music's tied timestamps). Installed in-job via `pip install --user
  recbole==1.2.0` if missing.
- Batch 512 with single-target prefix samples ≈ [25]'s batch 4 × 128
  per-position losses; recorded per run in training.csv.
- Resume: re-submitting Job 1 after a wall-time kill skips finished cells
  (checkpoint + recs-file markers) and continues.
