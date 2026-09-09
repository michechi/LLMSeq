# TMLR working manuscript

Build `main.tex` with the supplied official anonymous TMLR style:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The generated PDF and TeX build products are ignored. This is the September 9, 2026 working draft. All reported results, provenance tags and verification TODOs are retained. No new scientific results were added during repository cleanup.

Reconcile the draft with the latest OC matched-completion result files in `results/matched_completion/` before marking corresponding TODOs complete. See [the work list](../../docs/tmlr_status.md). The original anonymous repository URL in the draft also needs author review before submission.

The official template is from [JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file), commit `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`; its files are unmodified. `STYLE-LICENSE.txt` accompanies those files.
