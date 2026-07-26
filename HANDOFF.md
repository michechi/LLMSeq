# HANDOFF — NeurIPS rebuttal state (KIP + RecSys audit)

Two arms: **KIP** (synthetic Key-Inversion Parity, below) and the **RECSYS
AUDIT** (reviewer Q3, real order-sensitive task — see the section at the end;
it is self-contained for a cold H200 session).

# KIP task state

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
- Cross-site proof: `scripts/slurm/FOX/kip/kip_data.sha256` (28 files;
  identical copy in the Olivia package).
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
  `site=local` is the A100 box, `site=fox` is FOX (Educloud), `site=olivia`
  is Sigma2 Olivia (unavailable as of 2026-07-26). Rows written
  by processes that predate the 2026-07-26 schema migration may lack the last
  4 fields — pandas reads them as NaN; backfill with
  `,local,hazy-book-sings-fin-03,NVIDIA A100-SXM4-80GB,` if needed.
  Patience-5 m4 rows from 2026-07-25 live in `results/kip_training_p5.csv`
  (kept separate so they never mix with the as-run patience-3 block).
- **Curves**: DL runs write `curve.json` next to the checkpoint (added
  2026-07-26 — local runs completed before then have per-epoch curves only in
  `logs/kip/*.log`); cluster jobs copy curves + the full training log into
  `results/kip_curves/` (tracked). BERT curves are the per-epoch lines in the
  training log; python logging goes to stderr, so that's the single merged
  `.out` on FOX and the `.err` on Olivia (which splits streams).
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
stops sufficing — is the decisive experiment, hence its priority on FOX.

## FOX task list (cross-site runs)

Package: `scripts/slurm/FOX/kip/` — **start at `RUNBOOK.md` there**; it is
self-contained (data regen → sha256 verify → sbatch order → monitoring →
committing rows back). All jobs request `--gpus=nvidia_h200_nvl:1` (site
instruction 2026-07-26). An equivalent Olivia package exists at
`scripts/slurm/Olivia/kip/` but Olivia is unavailable as of 2026-07-26.

1. `kip_d_lstm_m6.slurm` — **PRIORITY**: LSTM m6 (a), 3 seeds, patience 5.
2. `kip_a_anchor.slurm` — anchor LSTM m4 (a) seed 9550 p5; must reproduce
   test_auc 1.0000 (cross-site validation gate).
3. `kip_b_lstm_m4_seeds.slurm` — LSTM m4 (a), new seeds 9570–9574, p5.
4. `kip_c_bert_m4.slurm` — BERT m4 (a), seeds 9551+9552 @512 ("seeds 2–3" of
   the canonical triplet; 9550 done locally).
5. `kip_e1_transformer_m6.slurm` + `kip_e2_bert_m6.slurm` — m6 (a), 3 seeds
   each (BERT @64).
6. `kip_f_m6_controls.slurm` — follow-up modes (b)/(c) per converged m6 family.
7. `kip_g_llama_m4.slurm` (optional) — Llama-1B LoRA m4 (a), seeds 9551+9552,
   completing the triplet with local Block D's seed 9550.
8. `kip_h_llama_m6.slurm` (optional) — Llama-1B LoRA m6 (a), seeds 9550–9552,
   completing per-model m6 coverage; controls via kip_f with
   KIP_MODEL=Llama1B. The Llama arms are the only KIP FOX jobs needing HF
   credentials (gated model); token read from env or cached login, never
   logged.

Separate rebuttal arm (NOT KIP): the RecSys audit grid — full spec in the
"RECSYS AUDIT" section at the end of this file.

Separate rebuttal arm (NOT KIP): `scripts/slurm/FOX/oc_fullft/` — does the
OC (tricky_rnd, tag 9) decoder conclusion survive FULL fine-tuning instead
of LoRA? Llama-3.2-1B + Llama-3.1-8B, paper recipe verbatim minus --peft
(8B also minus 4-bit), seed 8888, via LLM_fraction_experiment. Results into
`results/oc_fullft/` (tracked). See the README there.

## Environments

- **Local A100 box** (`hazy-book-sings-fin-03`, site=local): venv at
  `/root/kip-venv` (system python is stdlib-only). Run as
  `cd /root/LLMSeq/repro && DATA_DIR=/root/LLMSeq/data /root/kip-venv/bin/python -m src.<module>`.
  Training only inside detached tmux; never kill a session to inspect it.
  Llama (Block D) needs HF_TOKEN in the tmux env — never echo/log it.
- **FOX (Educloud, site=fox — the active cluster)**: SLURM account ec12,
  `--partition=accel --gpus=nvidia_h200_nvl:1`, no containers — the NLPL
  **2024a** module stack provides python 3.12 + torch 2.6.0/cu12.6 +
  transformers 4.55.4 + peft + sklearn
  (`module use -a /fp/projects01/ec30/software/easybuild/modules/all/`).
  The 2022b stack (transformers 4.47.1, torch 2.1.2/cu12.0) has no H200
  kernels — "no kernel image" crash (2026-07-26); 4.55.4 vs the paper's
  4.47.1 is a recorded, accepted deviation.
  Compute nodes have internet; bert-base-uncased downloads in-job, ungated,
  **no HF token needed or wanted**. Checkout at `$HOME/MIMICIV`; checkpoints
  in `$REPO_ROOT/checkpoints/kip/` (gitignored, ~4 GB for the BERT arms —
  keep until kip_f ran). If the sanity suite lacks xgboost there, the only
  acceptable failures are the two "xgboost missing" lines (12/12 was
  validated locally, tracked in
  `repro/src/analysis/mechanism_id/results/kip_sanity.txt`).
- **Olivia (Sigma2, site=olivia — UNAVAILABLE as of 2026-07-26)**: package
  retained at `scripts/slurm/Olivia/kip/` in case it comes back: account
  NN12048K, partition accel, container `extended-pytorch.sif`, no HF token,
  BERT pre-fetched + `HF_HUB_OFFLINE=1`, checkpoints under
  `/cluster/work/projects/nn12048k/michechi/results/kip/checkpoints/`.

---

# RECSYS AUDIT (reviewer Q3) — 30Music order-sensitivity grid

Second real order-sensitive task with shuffle controls, mirroring the protocol
of ref [25] (Klenitskiy et al., "Does It Look Sequential?", RecSys '24,
arXiv:2408.12008; code github.com/Antondfger/Does-It-Look-Sequential). Their
repo has **NO license file**: it is cloned ONLY to local scratch
(`/root/recsys_scratch/Does-It-Look-Sequential`, with our run drivers beside
it) — their code is never copied into this repo. The pipeline was
independently re-implemented in `repro/src/recsys/` and verified equivalent.

## Status (local A100 box, 2026-07-26) — all gates PASSED

- **Dataset: 30Music** (first choice, no fallback needed). Official direct
  SharePoint download linked from https://remaplab.deib.polimi.it/resources/
  (the old recsys.deib.polimi.it page 404s; the `download.aspx?share=<token>`
  URL form avoids the 403 on the raw share link). Raw =
  `relations/events.idomaar`, 31,351,945 play events, converted by
  `src.recsys.convert_30music` (0 parse skips; sha256 of source + output in
  `30M.csv.convert_meta.json`, folded into the audit manifest).
- **Preprocessing gate vs their published stats: PASS** — users **43,762
  (exact)**, interactions **22,604,876 (exact)**, mean length 516.54
  (ref ≈516.5); their published "822,507 items" is reproduced **exactly** as
  the TEST-split vocabulary (prep-level nunique is 839,099).
- **Split** (their code; validation sampling np.random.seed(17), recorded —
  their code leaves it unseeded): train 42,712 users / 20,119,073 events /
  834,223 items; validation 500 users / 225,297 events; test 25,518 users /
  18,453,009 events / 822,507 items. Median prep length 261. 97.81% of test
  targets are in the train-split vocab (97.84% incl. validation).
- **Independent re-implementation** (`src.recsys.preprocess --verify`):
  all four outputs (prep, train, validation, test) **row-for-row IDENTICAL**
  to their pipeline's on the same raw CSV.
- **Anchor** (their code verbatim via observation-only shims, SASRec seed 17,
  batch 4, patience 5; 65 min wall-clock, under the 2 h cap):

  | metric | [25] published (5-seed mean) | ours (seed 17) |
  |---|---|---|
  | HR@10 ordered | 0.198 | **0.19692** |
  | HR@10 shuffled | 0.020 | **0.02006** |
  | HR@10 rel. drop | −90% | **−89.81%** |
  | NDCG@10 ordered | 0.136 | **0.12995** |
  | NDCG@10 shuffled | 0.010 | **0.01015** |
  | NDCG@10 rel. drop | −92% | **−92.19%** |
  | Jaccard@10 | 0.12 | 0.0594 |

  Same qualitative picture (near-exact on HR/NDCG; Jaccard same order of
  magnitude, seed-dependent). Machine-readable copy:
  `scripts/recsys/30music_anchor_result.json`; rec lists + log archived in
  `results/recsys_audit/anchor/` (tracked). Recorded environment deviation:
  torch 2.13 / PL 2.6.5 / recommenders 1.2.1 vs their older pins.

## Protocol facts of [25] (recon from their code)

- Preprocessing: iterative loop until min user len ≥5 AND min item count ≥5:
  drop users <5 events → drop items <5 occurrences → collapse consecutive
  repeats (i-i-j → i-j) on (user, time)-sorted rows; then LabelEncoder
  (sorted-unique → consecutive ids) for items, then users.
- Split: global 90% quantile of ALL timestamps (pandas linear interpolation).
  Train users = 2nd event ≤ boundary (their post-boundary events dropped);
  test users = last event > boundary (FULL history kept — train/test user
  sets overlap by design); validation = 500 random train users, removed from
  train.
- Model input: last 128 items of history-minus-target (`max_length: 128`);
  target = last item per val/test user.
- Training: full-catalog cross-entropy (tied item-embedding head), Adam
  lr 1e-3, batch 4 (their 30Music setting), max 100 epochs, early stop on
  val NDCG@10, patience 5, best checkpoint restored.
- Eval: FULL catalog, `filter_seen: False` (seen items NOT filtered);
  HR@10 = `recommenders` recall@10 (single target ⇒ hit rate); NDCG@10 same
  package; Jaccard@10 = mean per-user Jaccard between ordered-input and
  shuffled-input top-10 lists.
- Their shuffle control: permute the ENTIRE history-minus-target
  (np.random.seed(random_state), one global seeding) BEFORE the 128
  truncation ⇒ the model sees different ITEMS, not just different order.
  Kept ONLY for the anchor; the audit grid uses the window variants below.
- Their published 30Music GRU4Rec drops (for grid reference): HR@10 −95%,
  NDCG@10 −96%, Jaccard@10 0.02.

## Audit data (main-result files; OUR protocol, deliberately ≠ anchor shuffle)

Built by `src.recsys.audit_shuffles` from the canonical split files. Per test
user: drop target → fix ordered visible window = last min(128, len) input
rows (exactly what a max_length=128 model consumes) → permute item values
WITHIN the window only (timestamps stay in their slots). Every condition
shows the model the same item multiset and the same target; only order
varies.

- Variants: `full_shuffle` (whole window), `keep_last` (last window item
  fixed), `early_half` (first ⌊W/2⌋ permuted, second half intact) × shuffle
  seeds {101, 102, 103} ⇒ 9 files, drop-in schema for `test_30Music.csv`.
- Mode-(b) extras: `{train,validation}_30Music_full_shuffle_s101.csv` —
  full-sequence per-user shuffles for the optional shuffled-train block.
- Determinism: `random.Random(f"{seed}:{variant}:{user_id}")` per user —
  machine/library independent. One shuffle per user per variant×seed, saved
  to disk; EVERY downstream consumer reads the SAME files. Never re-shuffle
  per consumer.
- **CONSUMER CONTRACT (load-bearing, ENFORCED IN CODE)**: files are
  stable-sorted by (user_id, timestamp); sequence order within equal
  timestamps = file row order. Group consecutive rows per user; do NOT
  re-sort by timestamp alone and do NOT feed these files through [25]'s
  unmodified `LMDataset` (it uses an unstable single-key sort; 30Music has
  tied timestamps — manifest records the exposure: 4,095 test users have
  ties, 95 could see a different window multiset and 58 a different
  last-window item ONLY under a contract-violating loader).
  **Enforcement**: the grid harness MUST obtain eval sequences via
  `src.recsys.eval_loader.load_eval_input(variant_csv, audit_dir)` — on every
  load it validates file structure (contiguous user blocks, non-decreasing
  timestamps) and re-verifies, per user, window-multiset equality vs
  `ordered_windows_30Music.csv`, target equality vs `targets_30Music.csv`,
  and the variant-specific invariant (keep_last / early_half / ordered),
  raising `AuditContractViolation` on any mismatch. Negative-tested against
  corrupted targets, cross-user item swaps (invisible to global-multiset
  checks), and timestamp-only re-sorts — all fail loudly.
- Verification baked into the build (on the SHIPPED bytes): targets
  byte-identical across ordered + all 9 variants, non-window rows untouched,
  per-user window multiset preserved → `target_identity_check: PASS`.
- Manifest (tracked): `scripts/recsys/30music_audit_manifest.json` (raw
  checksum + converter provenance, machine-readable preprocessing config,
  prep + split + ordered-window + per-variant×seed checksums, target-identity
  result, tie stats, git rev) and flat `sha256sum -c` files:
  `scripts/recsys/30music_audit_files.sha256`,
  `scripts/recsys/30music_split_files.sha256` (copies live beside the data).

## File layout

    data/recsys/30music/                 # gitignored; manifests tracked in scripts/recsys/
      30M.csv (+.convert_meta.json)      # converted raw (user,item,timestamp)
      prep_30Music.csv                   # preprocessing output (their code)
      split/{train,validation,test}_30Music.csv + split_files.sha256
      own/                               # our re-implementation outputs (equivalence check)
      audit/                             # ← THE GRID CONSUMES EXACTLY split/ + audit/
        targets_30Music.csv
        ordered_windows_30Music.csv
        test_30Music_{full_shuffle,keep_last,early_half}_s{101,102,103}.csv
        {train,validation}_30Music_full_shuffle_s101.csv
        audit_manifest.json + audit_files.sha256

## THE FULL GRID (H200 / FOX)

**Models** — train on ordered `train_30Music.csv`, early-stop on ordered
validation, training seeds **{17, 32, 45}**:

| model | implementation | config (matched in scale to [25]) |
|---|---|---|
| SASRec | RecBole ≥1.2 (MIT) | n_layers 2, n_heads 2, hidden 64, dropout 0.1, max_len 128, loss CE (full softmax), Adam lr 1e-3 |
| GRU4Rec | RecBole | embedding 64, hidden 64, 1 layer, dropout 0.1, max_len 128, loss CE, lr 1e-3 |
| BERT4Rec | RecBole | n_layers 2, n_heads 2, hidden 64, mask ratio 0.2 (RecBole default), max_len 128, loss CE, lr 1e-3 |

Shared training protocol: max 100 epochs, early stop patience 5 on val
NDCG@10, full-catalog eval, NO seen-item filtering (matches [25]
`filter_seen: False`). Training objective = RecBole-standard (prefix
augmentation; CE at last position for SASRec/GRU4Rec; BERT4Rec cloze with
mask_ratio 0.2 + ft_ratio 0.5 mask-last batches, replicated from
recbole==1.2.1 PINNED). Batch size is a recorded compute detail
(training.csv column): 2048 for SASRec/GRU4Rec, 256 for BERT4Rec (its cloze
CE logits are [B, 25, 839100] — B=2048 would need >170 GB and OOM an H200;
measured ~61 GiB peak at B=256). Document per model:
config dump + any equivalence checks (RecBole implementations are
scale-matched, not bit-identical, to [25]'s — the anchor bridges protocols;
flag if RecBole-SASRec ordered HR@10 deviates from the anchor's 0.197 by >2×).

**Baselines** (CPU-cheap, implement in our repo — no copied code):

- MostPopular: train-split popularity ranking, same list for all users
  (order-invariant ⇒ expected Δ ≈ 0; sanity row).
- ItemKNN (bag-of-items): cosine on the binary user×item train matrix;
  score(i) = Σ_{j∈window} sim(i,j) (order-invariant by construction).
- Markov-1: score(i) = train count(last_item → i).
- Markov-2: score(i) = train count((prev,last) → i), backoff to Markov-1 on
  unseen (prev,last) context, then to popularity (deterministic ladder).

**Eval inputs per model/baseline** (10 each): ordered (`test_30Music.csv`,
window = last 128 minus target) + 3 variants × 3 shuffle seeds from `audit/`.
Consume files via `src.recsys.eval_loader.load_eval_input` (contract
enforcement built in — see above); never read variant CSVs directly.

**Optional final block** (protocol uniformity with synthetic mode (b);
precedent in [25]'s extended journal version): GRU4Rec shuffled-train —
train on `train_30Music_full_shuffle_s101.csv`, early-stop on
`validation_30Music_full_shuffle_s101.csv`, eval on
`test_30Music_full_shuffle_s101.csv`, 1 seed (17).

**Metrics** per (model, eval input): HR@10, NDCG@10; Jaccard@10 vs the SAME
model's ordered top-10 lists (per-user mean); Δabs = shuffled − ordered;
Δrel = Δabs/ordered; user-bootstrap 95% CIs (resample the 25,518 test users
with replacement, B = 1000, numpy seed 4242) for each metric AND each Δ.
Report per shuffle seed and mean across the 3 shuffle seeds.

**Outputs**: one row per (model, eval_input, train_seed, shuffle_seed) into
`results/recsys_audit/metrics.csv` (tracked — results/recsys_audit/** is
un-gitignored; add `merge=union` to .gitattributes BEFORE parallel appends,
mirroring kip_training.csv) + top-10 lists per run under
`results/recsys_audit/recs/` (user_id → 10 items, parquet or csv.gz) so
Jaccard/CIs can be recomputed offline.

## H200 bootstrap (cold start)

1. Environment: NLPL 2024a stack (see KIP section: `module use -a
   /fp/projects01/ec30/software/easybuild/modules/all/`; python 3.12 + torch
   2.6.0/cu12.6) + `pip install --user recbole` (MIT; no HF token needed).
2. Pull branch `Rebuttals_NeurIPS`; rsync data (below); verify bytes AND
   contract (both must pass before any training):
       cd $REPO_ROOT/data/recsys/30music/audit && sha256sum -c audit_files.sha256
       cd ../split && sha256sum -c split_files.sha256
       cd $REPO_ROOT/repro && python -m src.recsys.eval_loader \
         --audit-dir $REPO_ROOT/data/recsys/30music/audit \
         --split-dir $REPO_ROOT/data/recsys/30music/split --check-all
3. Feed sequences per the CONSUMER CONTRACT — in practice: EVAL sequences
   come from `src.recsys.eval_loader.load_eval_input` ONLY (it re-verifies
   the invariants on every load and raises on violation). For TRAINING files
   (ordered or mode-(b)), group consecutive rows per user; if converting to
   RecBole atomic files, derive a per-user position column from file row
   order and use IT as the time field — do not let RecBole re-sort tied
   timestamps.
4. Grid: **packed into SLURM at `scripts/slurm/FOX/recsys/`** (see the README
   there) — `recsys_job1_grid.slurm` = 3 models × 3 seeds training + 10 evals
   each + all baselines + report table (sequential, resumable, one GPU;
   smoke-tested end-to-end on the local A100 2026-07-26);
   `recsys_job2_shuffled_train.slurm` = optional GRU4Rec mode-(b), submit
   ONLY after the Job-1 table is reviewed. Harness code:
   `repro/src/recsys/{recbole_grid,train_grid,eval_grid,baselines,report}.py`
   (RecBole 1.2.0 model classes + our contract-honoring loops).
5. Commit results rows back on the branch (`git add results/recsys_audit`).

## rsync (run FROM the local A100 box; NOT yet executed)

    rsync -avz --progress \
      /root/LLMSeq/data/recsys/30music/split \
      /root/LLMSeq/data/recsys/30music/audit \
      <fox-user>@fox.educloud.no:~/MIMICIV/data/recsys/30music/

(`30M.csv`/`prep_30Music.csv` are not needed on FOX — the grid consumes only
split/ + audit/. Full local rebuild if ever needed: `src.recsys.convert_30music`
→ their preprocessing (or `src.recsys.preprocess`, verified identical) →
`src.recsys.audit_shuffles`; every seed is recorded in the manifest.)
