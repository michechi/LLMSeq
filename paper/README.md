# Manuscripts

| Folder | Start here | Purpose |
|---|---|---|
| [`tmlr/`](tmlr/) | [`main.tex`](tmlr/main.tex) | Current TMLR working draft; [build instructions](tmlr/README.md) |
| [`NLDL/`](NLDL/) | [`main.tex`](NLDL/main.tex) | Preserved NLDL version; [open items](NLDL/TODO_NLDL.md) |
| [`neurips/`](neurips/) | [`paper.tex`](neurips/paper.tex) | Earlier NeurIPS manuscript, with figures and older drafts grouped below it |

Each version retains its own text. Results and tables shared with the experiment scripts stay in [`../results/`](../results/) and [`../paper_tables/`](../paper_tables/).

The former loose `paper/*.tex` and PNG files are now under `neurips/`. Its [`archive/`](neurips/archive/) contains older complete drafts and unused section fragments; it includes an ICML-formatted revision, identified in the [NeurIPS folder guide](neurips/README.md). Review PDFs remain in [`../docs/review_history/`](../docs/review_history/).

Run LaTeX from the folder containing the chosen source. Build products remain ignored. The NeurIPS source has missing dependencies, and NLDL uses placeholders for missing figures; see their notes before treating either as ready for submission.
