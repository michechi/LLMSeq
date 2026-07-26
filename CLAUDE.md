# LLMSeq

Paper repo (NeurIPS submission + rebuttal experiments): can sequence models
learn order-dependent hidden rules? Synthetic letter-sequence tasks + MIMIC-IV
audit.

## Layout

- `repro/` — the reproducible package; all python runs as
  `cd repro && DATA_DIR=<repo>/data python -m src.<module>`.
- `data/simulation/tested/` — canonical datasets (KIP files tracked in git).
- `results/` — aggregated results; `results/kip_training.csv` is the
  cross-site KIP results table (append-only, merge=union).
- `scripts/slurm/{Olivia,FOX}/` — cluster job templates;
  `scripts/kip/` — local A100 block runners.

## Active work: KIP (NeurIPS rebuttal)

**Read `HANDOFF.md` (repo root) first** — task definition, conventions
(patience 5 = paper recipe; modes a/b/c; per-seed reporting; site column),
results so far, and the cross-site run plan.

- On Olivia (Sigma2): follow `scripts/slurm/Olivia/kip/RUNBOOK.md` exactly.
- On the local A100 box: python lives at `/root/kip-venv/bin/python` (system
  python has no pip); export `DATA_DIR=/root/LLMSeq/data`; training runs only
  in detached tmux sessions — never in the assistant's foreground shell, and
  never kill a tmux session to inspect it (use capture-pane or the logs in
  `logs/kip/`). Never echo or log HF_TOKEN.

Background reading: `repro/src/analysis/mechanism_id/kip_report.md` (KIP
definition + CPU probe ladder), `repro/REPRODUCING.md` (paper pipeline).
