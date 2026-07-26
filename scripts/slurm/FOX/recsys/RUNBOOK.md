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

## 1. Submit order

    cd ~/MIMICIV/scripts/slurm/FOX/recsys && mkdir -p logs
    sbatch recsys_a_train.slurm                          # 9 cells: 3 models x seeds 17/32/45
    sbatch recsys_c_baselines.slurm                      # independent of (a)
    # after (a) completes (or per finished cells):
    sbatch --dependency=afterok:<jobid_a> recsys_b_eval.slurm
    sbatch recsys_d_gru_shuffletrain.slurm               # OPTIONAL mode-(b) block

Time guesses (H200; batch 2048 causal / 256 BERT4Rec — per-model, see
recsys_a_train.slurm header for the memory math): training ≤48 h/cell (early
stop patience 5 usually far sooner; A100 smoke: ~0.08 s/step at batch 512,
~10k steps/epoch at 2048), eval ~1 h/cell, baselines ~2 h total (Markov-2
table build alone is ~15 min).

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
  replicated, recbole==1.2.1 PINNED); SASRec/GRU4Rec with CE over the full
  catalog at the last position of each augmented prefix.
- All ids are +1-shifted inside the models (RecBole pad token 0) and shifted
  back before anything is written; recs files contain RAW [25]-encoding ids.
