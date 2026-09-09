# NeurIPS manuscript

Start with [`paper.tex`](paper.tex), moved from `paper/paper.tex`. This is the existing NeurIPS 2026 source; the folder organization does not revise the paper or convert it to a newer template. The current writing target remains [`../tmlr/`](../tmlr/).

## Contents

| Path | Contents |
|---|---|
| [`paper.tex`](paper.tex) | Main NeurIPS source |
| [`figures/`](figures/) | All 11 PNG assets formerly loose in `paper/` or `paper/n_grams_plots/` |
| [`archive/paper_neurips.tex`](archive/paper_neurips.tex) | Earlier NeurIPS draft |
| [`archive/paper_revisioned.tex`](archive/paper_revisioned.tex) | Historical **ICML 2026** revision; it also requests the `icml2025` bibliography style |
| [`archive/Experiments.tex`](archive/Experiments.tex), [`archive/Appendix_Training.tex`](archive/Appendix_Training.tex) | Standalone section fragments, not included by the main source |
| [`archive/draft.tex`](archive/draft.tex) | Separate clinical landmark/mortality section fragment |

Archived drafts are retained for comparison. Run LaTeX from their containing directory if restoring a historical build. Figure paths now point to `figures/` or `../figures/`; the older drafts' misspelled `tricky_deterministic_ngrams.png` reference now matches the existing singular `tricky_deterministic_ngram.png`. The manuscript wording, numbers, equations and draft line counts are preserved.

## Missing build dependencies

The main source already lacked these files before organization:

- `neurips_2026.sty`
- `example_paper.bib`
- `auc_combined_main_1row.png`, `auc_combined_full_1row.png`, `BERT.png`, `Llama.png`

Restore the original style and bibliography next to `paper.tex`, and the figures into `figures/`, before attempting a complete build. They have not been reconstructed or replaced by results from another experiment. The older drafts have additional missing figures and template dependencies.

The main source currently references none of the 11 retained PNGs. Four n-gram plots are referenced by the archived drafts. The other seven frequency/distribution plots total about 2.81 MiB and have no references in the tracked TeX sources; they remain as historical assets until their provenance and need are resolved. See [the storage audit](../../docs/repository_cleanup.md#remaining-storage).

The historical compiled PDF is listed at its original path, `paper/NeurIPS.pdf`, in the recovery manifest. That provenance path is intentionally unchanged.
