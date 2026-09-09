# Repository cleanup

Prepared from the latest `NLDL` snapshot, commit `0743bcf223022664a3885058c909814fe8669ce4`, on September 9, 2026. TMLR is the main documented project; AKI and cancer remain available at their existing paths.

The original tracked tree was 861 files, 317.75 MiB. The local folder's much larger roughly 18 GiB size is mainly ignored raw MIMIC data. Git's existing object history is about 290 MiB. These are different measures: reducing tracked files does not delete local raw data or shrink old Git history.

## What changed

- Removed bulk datasets, generated figure galleries, raw logs, compiled paper PDFs, OS files and backups from the proposed tracked tree.
- Retained all existing Python, shell, SLURM, package, protocol and source-hash files unchanged. Their import and HPC paths stay intact; the verified AKI source manifest still passes.
- Retained small result tables and paper tables byte for byte, and preserved existing paper figures and sources.
- Cleared saved outputs and execution counts from valid notebooks while preserving every code and Markdown cell's source. One already-invalid notebook (`notebooks/preprocessing/check_data_creation.ipynb`) was excluded rather than silently reconstructed; its original remains in the source checkout.
- Added a TMLR guide and the working manuscript source. Older overview, TODO and analysis notes moved to `docs/archive/legacy_*.md`; the root review/proof PDFs moved to `docs/review_history/`.
- Added checksum-based restoration, a launcher selecting the correct reproduction package, a basic repository check, and a CI workflow. JSON settings and provenance files are no longer globally ignored.

Removed categories, measured from the original tracked files:

| Category | Files | MiB |
|---|---:|---:|
| junk | 13 | 0.22 |
| generated-figures | 296 | 18.48 |
| restricted-clinical | 18 | 99.84 |
| synthetic-data | 50 | 174.66 |
| raw-logs | 20 | 0.67 |
| invalid-notebook | 1 | 0.12 |
| compiled-paper | 2 | 5.52 |

## Evidence and recovery

Synthetic datasets and retired generated reports/builds have exact checksums in `data/manifests/retired_artifacts.csv`. `scripts/restore_artifacts.py` can recover them from a local source checkout or existing local Git objects without overwriting different files. Data restored at its original paths remains ignored. Exact recovery was tested for all 50 synthetic files; the existing OC suite also passed using those restored files.

Raw logs and clinical files are deliberately excluded from the restoration utility. They remain in the original local checkout and old history. The cleanup does not assert that missing run archives, inference outputs or paper verification tasks have been recovered.

## Public history and credentials

The public `NLDL` tree contains processed MIMIC files, a clinical-text export, and eight clinical logs with a recognizable Hugging Face credential. No token value or patient record is reproduced here. The corresponding paths are absent from this proposed tree, and saved notebook outputs were removed. The credential needs revocation/rotation; its current validity was not tested.

This branch does **not** erase prior versions. Resolving public exposure and shrinking Git history require a separate operation covering affected branches, tags and accessible copies. Coordinate that work before replacing remote history; no force-push, branch deletion, repository-visibility change or token action has been performed. Passing the basic file check is not a full privacy/security review.

## Use the cleanup safely

The approved cleanup is applied locally without deleting datasets or raw logs. Changed notebooks and documents are backed up in a separate local directory, and the pre-cleanup commit has a local backup branch. A snapshot ZIP is also available for a separate checkout. Do not switch the original data-bearing working directory directly to a commit that deletes its tracked datasets: Git can remove those files during the switch. Create a separate worktree, or securely back up/move data outside the checkout first. Point experiment data paths at the intended external data directory.

A supplied Git bundle contains the cleanup commit relative to the existing base. It is for importing/reviewing the branch; it does not sanitize its ancestral history. A snapshot ZIP contains the proposed files without Git history. Neither artifact has been published to GitHub.


## Remaining storage

Inspected the tracked tree at `e5a64be` on September 9, 2026, before this folder organization: **480 files, 16.84 MiB**. No file exceeds 5 MiB, and no byte-identical duplicate files larger than 10 KB were found. Moving files changes where they appear, not their storage cost.

| Item | Size | Decision |
|---|---:|---|
| `paper_tables/counterfactual_pairs_tricky_rnd.csv` | 4.11 MiB | Keep: read by the counterfactual scorer and SLURM launchers |
| `codes/CCS_PCS_mapping.csv` and `codes/CCS_DX_mapping.csv` | 3.46 MiB together | Keep: clinical reference mappings; the diagnosis pipeline and an analysis notebook refer to these mappings |
| Seven older frequency/distribution PNGs in `paper/neurips/figures/` | 2.81 MiB | Candidates for a later archive outside Git: no tracked TeX references; preserved because reproducible replacements have not been verified |
| `results/matched_completion/pair_results_breakdown_smoke.csv` | 0.20 MiB | A small smoke-run result; keep distinguishable from the full result |
| Local `paper/NLDL/main.pdf` and TeX build files | About 1.3 MiB | Ignored generated files; can be regenerated, preserved locally |
| Local `.venv/` and caches | About 53 MiB | Ignored environment/cache; affects disk usage, not the tracked tree |
| Local `data/` | About 18 GiB | Ignored research data; not disposable repository clutter |
| Local `.git/` | About 291 MiB | Existing history; moving current files does not shrink it |

The seven unreferenced plots are `all_positions_frequency.png`, `all_positions_frequency_compact.png`, `bigram_distribution.png`, `letter_frequency_by_position.png`, `ngram_vignette.png`, `positions_heatmap.png`, and `trigram_distribution.png`. The four other retained PNGs are referenced by the older drafts and remain with them in `paper/neurips/figures/`.

This organization moves all loose manuscript sources and figures under `paper/neurips/`, groups older drafts in its `archive/`, and groups legacy general notes in `docs/archive/`. `paper/NLDL/` and `paper/tmlr/` remain. Links and figure lookup paths are updated; scientific content, code, configurations, results, mappings, provenance manifests and clinical data are preserved. Nothing was removed from the tracked file set or pushed remotely.
