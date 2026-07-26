# RecSys audit grid — FOX runbook

Cold-start runbook for the 30Music order-sensitivity grid (reviewer Q3).
Protocol + full grid spec: **RECSYS AUDIT section of /HANDOFF.md** (read it
first). Everything below assumes checkout at `$HOME/MIMICIV` on branch
`Rebuttals_NeurIPS` and the data already rsynced + verified (done 2026-07-26:
sha256 16/16 OK, eval_loader --check-all 10/10 OK).

## 0. One-time setup (login node)

    cd ~/MIMICIV/scripts/slurm/FOX/recsys
    ./recsys_setup.sh          # pip --user --no-deps recbole==1.2.1 + leaf deps,
                               # then import smoke test ("SETUP OK")

If data needs re-verification (e.g. after re-rsync):

    cd ~/MIMICIV/data/recsys/30music/audit && sha256sum -c audit_files.sha256
    cd ../split && sha256sum -c split_files.sha256
    cd ~/MIMICIV/repro && python -m src.recsys.eval_loader \
      --audit-dir ~/MIMICIV/data/recsys/30music/audit \
      --split-dir ~/MIMICIV/data/recsys/30music/split --check-all

## 1. Submit (≤2 jobs; Job 2 is gated on the Job-1 table)

    cd ~/MIMICIV/scripts/slurm/FOX/recsys && mkdir -p logs
    sbatch recsys_job1_grid.slurm      # ONE array job = the whole main grid:
                                       # 9 tasks (3 models x seeds 17/32/45),
                                       # each trains + evals its checkpoint on
                                       # all 10 inputs; task 0 also runs the
                                       # four baselines. Resumable per task.

The models are tiny (few M params) — 9 single-GPU tasks that queue behind,
never disturb, the Q1 full-FT jobs. If the H200s are saturated:
`sbatch --gpus=a100_80:1 recsys_job1_grid.slurm` runs identically.

**Job 2 — DO NOT SUBMIT YET.** Only after the Job-1 table is reviewed:

    sbatch recsys_job2_shuffled_train.slurm   # GRU4Rec shuffled-train, seed 17

Time guesses (H200; batch 2048 causal / 256 BERT4Rec — per-model, see the
recsys_job1_grid.slurm header for the memory math): training ≤48 h/cell
(early stop patience 5 usually far sooner; A100 smoke: ~0.08 s/step at batch
512, ~10k steps/epoch at 2048), eval ~1 h/cell, baselines ~2 h on task 0
(Markov-2 table build alone is ~15 min).

## 1b. Report (login node, CPU, ~2 min) — then STOP

After the array finishes (`squeue --me` empty of recsys_job1):

    cd ~/MIMICIV/repro
    python -m src.recsys.report \
      --audit-dir ~/MIMICIV/data/recsys/30music/audit \
      --recs-dir ~/MIMICIV/results/recsys_audit/recs \
      --out ~/MIMICIV/results/recsys_audit/report_job1.md

That file IS the Job-1 deliverable (main table: models × ordered/
full_shuffle/keep_last/early_half with bootstrap CIs + Δs, Jaccard@10,
baseline ladder with reach + drop_explained per rung). **STOP here** —
commit it back (step 4) and review before any thought of Job 2.

## 2. Sanity gates (check BEFORE trusting the grid)

1. `results/recsys_audit/training.csv`: every cell converged (best_epoch > 0,
   best_val_ndcg10 for SASRec/GRU4Rec roughly near the anchor's val 0.115 —
   not decimal-equal, different objective granularity; see HANDOFF).
2. **Anchor bridge**: RecBole-SASRec seed 17 ordered test HR@10 should be
   within 2x of the anchor's 0.19692 (scripts/recsys/30music_anchor_result.json).
   If not, STOP and investigate before running anything else downstream.
3. Baseline sanity rows: MostPopular and ItemKNN carry
   `order_invariant=true`, Jaccard@10 = 1 and Δ = 0 across all variants BY
   CONSTRUCTION (they see only the window multiset). Markov-1/2 must show
   real drops on full_shuffle (they read the last one/two items).

## 3. What each run appends

- `results/recsys_audit/training.csv` — one row per training cell
  (fcntl-locked, provenance columns site/host/gpu/slurm_job_id).
- `results/recsys_audit/metrics.csv` — one row per (model|baseline,
  eval_input): HR@10 / NDCG@10 (+95% user-bootstrap CIs, B=1000 seed 4242),
  Jaccard@10 vs same-model ordered lists, Δabs/Δrel (+paired CIs).
- `results/recsys_audit/recs/*.csv.gz` — top-10 lists per run for offline
  recomputation.
- checkpoints in `$REPO_ROOT/checkpoints/recsys/` (gitignored, keep until
  the rebuttal is submitted).

## 4. Commit results back

    cd ~/MIMICIV
    git add results/recsys_audit && git commit -m "FOX recsys grid rows" && git push

`results/recsys_audit/*.csv` have merge=union in .gitattributes (same
convention as kip_training.csv), so parallel appends from both sites merge
cleanly.

## Notes / gotchas

- Eval inputs are read ONLY via `src.recsys.eval_loader` (the contract
  enforcer). If it raises AuditContractViolation, the data is corrupted or
  re-sorted — do NOT work around it; re-verify checksums and re-rsync.
- `recsys_setup.sh` uses `--no-deps` everywhere so pip cannot install a
  user-site numpy/torch that shadows the module stack. If you add packages,
  keep that discipline.
- BERT4Rec trains with RecBole's full standard objective (mask_ratio 0.2
  cloze + ft_ratio 0.5 mask-last batches — both branches of MaskItemSequence
  replicated; recbole==1.2.1 PINNED on FOX via recsys_setup.sh; the local
  A100 smoke ran 1.2.0 — identical model-constructor contract, verified).
  SASRec/GRU4Rec with CE over the full catalog at the last position of each
  augmented prefix.
- All ids are +1-shifted inside the models (RecBole pad token 0) and shifted
  back before anything is written; recs files contain RAW [25]-encoding ids.
