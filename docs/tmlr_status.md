# TMLR work and current evidence

The current writing target is *Auditing Sequence Prediction with Known Label Rules*. The working source is in `paper/tmlr/`; its provenance tags and TODOs are deliberate. The cleanup is based on `NLDL` commit `0743bcf223022664a3885058c909814fe8669ce4` from September 9, 2026.

The latest snapshot contains OC matched-completion code and result CSVs added after the earlier manuscript audit. Their presence updates the inventory; it does not establish that the manuscript's corresponding verification tasks are complete. Reconcile tables, checkpoints, data hashes and exact protocols before updating those claims.

| Question | Available location | Next check |
|---|---|---|
| OC labels and corrected generation | `repro/src/oc_completion/oracle.py`, `gen_datasets.py`; generation manifest | Check labels and source provenance; distinguish historical tag `_6` from corrected `ocdet` |
| Pair counts versus learned rules | `repro/src/oc_completion/gen_pairs.py`, `eval_pairs.py`; `results/matched_completion/` | Verify pair construction, complete run matrix, checkpoints and predictions against the draft |
| Parity and curriculum | `repro/src/experiments/parity_decomposition_*`, `repro/src/ablations/` | Recover original seeds and compare training budgets |
| Full fine-tuning | Historical model scripts and supplied rebuttal report | Locate the exact reported configurations, trainable counts and predictions |
| Key-inversion parity | Report retained in the draft | Identify the matching generator and full archive; hidden-subset parity code is not automatically this experiment |
| Clinical diagnosis order | `repro/src/mimic/` | Preserve visible events before shuffling; verify recurrent padding and paired uncertainty within credentialed access |
| 30Music recommendation | Report retained in the draft | Recover candidate protocol, preprocessing, retraining conditions and paired predictions |

The planning budget is **300 H200-hours**, with clinical access available. Archive recovery takes priority over repeating experiments that already exist. This cleanup launched no training and made no scientific corrections to experiment implementations.

The AKI trajectory audit (`src/mimic/aki/`) and cancer work (`mimic_analysis/`) are preserved as separate projects. Their results do not replace the diagnosis-sequence study without a separately defined scientific change.

Older overview, task list and analysis notes are retained in [`archive/`](archive/). Treat them as historical context; their titles and conclusions are not the current paper's claims.
