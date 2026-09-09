# LLMSeq

Research code for evaluating what sequence-prediction results establish about label rules, input features and trained models. The current writing target is **TMLR**. Historical experiments and the separate AKI and cancer projects remain available.

Start with [the manuscript folders](paper/README.md) and [TMLR status and remaining work](docs/tmlr_status.md). The draft is a working document: existing result files are evidence to verify, not a claim that every experiment has been reproduced.

## Manuscripts

| Version | Main source | Notes |
|---|---|---|
| **TMLR** — current writing target | [`paper/tmlr/main.tex`](paper/tmlr/main.tex) | Working draft and supplied style |
| **NLDL** — preserved | [`paper/NLDL/main.tex`](paper/NLDL/main.tex) | [Open submission items](paper/NLDL/TODO_NLDL.md) |
| **NeurIPS** — earlier manuscript | [`paper/neurips/paper.tex`](paper/neurips/paper.tex) | [Figures, earlier drafts and missing dependencies](paper/neurips/README.md) |

Earlier general notes are grouped in [`docs/archive/`](docs/archive/); rebuttal and proof PDFs remain in [`docs/review_history/`](docs/review_history/).

## Code and results

| Work | Location |
|---|---|
| OC rules, regenerated datasets, matched-completion diagnostics | [`repro/src/oc_completion/`](repro/src/oc_completion/) |
| Earlier feature and parity analyses | [`repro/src/analysis/mechanism_id/`](repro/src/analysis/mechanism_id/) |
| Paper experiment implementations | [`repro/src/experiments/`](repro/src/experiments/) |
| Small retained result tables | [`results/`](results/), [`paper_tables/`](paper_tables/) |
| Clinical diagnosis-sequence study | [`repro/src/mimic/`](repro/src/mimic/) |
| Separate AKI trajectory project | [`docs/mimic_aki_audit.md`](docs/mimic_aki_audit.md), [`src/mimic/aki/`](src/mimic/aki/) |
| Separate cancer project | [`mimic_analysis/README_cancer.md`](mimic_analysis/README_cancer.md) |
| Historical scripts and notebooks | [`src/`](src/), [`simulation/`](simulation/), [`notebooks/`](notebooks/) |

The top-level `src/` and `repro/src/` implementations differ. They have not been merged or treated as interchangeable. Run TMLR reproduction modules with the launcher below; run the AKI project from the repository root as documented in its guide.

## Start without training

These commands inspect the repository and run the existing CPU tests. They do not launch a training sweep or access clinical data.

```bash
python scripts/check_repository.py
python -m pip install numpy pandas scikit-learn scipy pytest
bash scripts/run_repro.sh -m pytest tests/test_oc_completion.py -q
```

The stored-data test skips if historical synthetic files have not been restored. To recover the **exact** synthetic files from a local pre-cleanup checkout:

```bash
python scripts/restore_artifacts.py --group synthetic-data --source-root /path/to/original/LLMSeq --dry-run
python scripts/restore_artifacts.py --group synthetic-data --source-root /path/to/original/LLMSeq
```

The utility checks every file against the manifest and refuses to overwrite different files. It can also restore from local Git objects when pre-cleanup history is available; see [data handling](data/README.md). Restoration creates ignored local files. Clinical data and logs are excluded from this utility.

For model dependencies, install the reproduction package with `python -m pip install -e './repro[dev]'` in a suitable environment. Existing scripts include machine-specific launchers and experimental recipes; inspect settings before running them. The old `reproduce_main.sh --tiny` command includes model training and is not the initial CPU check.

## Storage and results

Git stores code, configuration, provenance manifests, reviewed small result tables, and manuscript source. Datasets, checkpoints, raw logs and bulk generated figures stay outside version control. Do not replace historical synthetic files by newly generated ones under the same name: known label discrepancies make exact provenance necessary.

JSON files are no longer ignored globally, so run configurations and small manifests can be committed. [The repository check](scripts/check_repository.py) rejects common accidental additions, including clinical data paths, large artifacts, raw logs, notebook outputs and recognizable credentials. This is a basic check, not proof that a file is suitable for public release.

[Cleanup notes](docs/repository_cleanup.md) explain what was removed from the tracked tree and how it is preserved. The local cleanup keeps datasets and logs on disk as ignored files, with backups of changed notebooks and retired documents. Earlier Git commits remain available. Old history still requires separate attention before this repository can be considered cleared of previously tracked clinical files and credentials.
