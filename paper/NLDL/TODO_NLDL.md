# NLDL submission — open items

Everything here is something I could not resolve without inventing content.
Nothing in `main.tex` or `references.bib` was fabricated to paper over these.

## 1. Missing figure assets (blocking for a real submission)

Four PNGs are referenced by `paper/paper.tex` but exist nowhere in the
repository, in git history, or in the compiled `NeurIPS.pdf`:

| File | Used in | Status |
|---|---|---|
| `auc_combined_main_1row.png` | **Figure 1, main text** | missing |
| `auc_combined_full_1row.png` | Appendix G (full results) | missing |
| `BERT.png` | Appendix J (attention) | missing |
| `Llama.png` | Appendix J (attention) | missing |

They are wired through `\figureorplaceholder`, so the document compiles and
shows a labelled box in each slot. **Drop the real PNG next to `main.tex`
(or in `paper/`) and it is picked up automatically — no edit needed.**

`auc_combined_main_1row.png` is normally produced by
`repro/src/analysis/plot_main_figure.py`, but `results/` currently holds only
6 JSONs from a `parity_decomp_dryrun`, not the main sweep, so I could not
regenerate it from your own data. Regenerating from partial results would
have produced a figure that misstates the results, so I did not.

**Page-count caveat:** the main text is 8 pages *with placeholders*. The real
Figure 1 is a 1×3 AUC panel and will occupy more vertical space than the
placeholder box. Re-check the count after dropping the figures in; see §5 for
where to claw back space if needed.

## 2. Bibliography provenance

`example_paper.bib` was **never committed to this repository** (confirmed
against all branches and the full history). `references.bib` was rebuilt from:

- **36 entries** transcribed from the rendered reference list of
  `paper/NeurIPS.pdf` — your own compiled bibliography.
- **9 entries** absent from that PDF, verified against authoritative records:
  `du2017deeplog`, `hahn-2020-theoretical`, `hcup_ccs`, `johnson2023mimiciv`,
  `johnson2024mimiciv31`, `kdigo2024ckd`,
  `liu2019robertarobustlyoptimizedbert`, `sumida2017disease_trajectories_esrd`,
  `teinemaa2019outcome`.

All 45 cite keys resolve; zero undefined citations.

### Items to confirm

- **`hcup_ccs` — ICD-9 vs ICD-10.** The verified entry is the **ICD-9-CM** CCS
  tool. Single-level CCS for ICD-9-CM has ~285 diagnosis categories; the paper
  says you mapped to **293** categories. If your MIMIC-IV cohort uses ICD-10
  codes, the correct resource is **CCSR for ICD-10-CM**
  (<https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp>). Please
  confirm which tool produced the 293 categories and swap the entry if needed.
- **`kdigo2024ckd` author.** Cited under the corporate author *KDIGO CKD Work
  Group* (the canonical form, and what PubMed uses). Crossref instead lists 26
  named authors led by Paul E. Stevens. Corporate form kept deliberately.
- **`XGBoost_LLMs_HF`.** Resolved to Alshraideh et al., *Cureus* 2025 (heart
  failure + generative LMs) — inferred from the `_HF` suffix and the PDF
  bibliography. Confirm this is the reference you intended.
- **DOIs not available:** `hcup_ccs` (web resource, none exists),
  `ramezanaliSeqBenchTunableBenchmark2025` (no DOI found — the PDF entry
  carries only a month/year), and the several arXiv-only entries, which carry
  eprint IDs rather than journal DOIs. Nothing was invented to fill these.

## 3. Cross-reference repaired

`\ref{appendix:bert}` was referenced in the curriculum appendix but **the label
exists nowhere in `paper.tex`** — it rendered as `??` in your NeurIPS build
too. I repointed it to `\ref{appendix:training}`, the appendix that actually
contains the BERT LoRA configuration. Marked with a `TODO(author)` comment at
the site. Confirm the target.

## 4. Submission metadata

- `\paperID{42}` is a placeholder — replace with your **OpenReview submission
  number**. NLDL desk-rejects full papers that omit it.
- `\vol{V}` stays as-is until acceptance.
- Class is `[fullpaper]`, which defaults to `review`: anonymized, line numbers
  on. Switch to `[fullpaper,final]` only for camera-ready.
- The anonymous code URL is `anonymous.4open.science/r/anonymous_neurips_seqllm/`.
  It does not leak author identity, but the slug does say *neurips*. Consider
  re-publishing under a venue-neutral slug.

## 5. Page budget

Main text is **8 pages** (references begin on p.8 and do not count toward the
limit; the appendix is unlimited). NLDL allows 5–8 pages, so you are exactly at
the ceiling with no slack.

If the real Figure 1 pushes you over, the cheapest cuts, in order:

1. **§4.2 Models and baselines** — the four `\paragraph` blocks are dense; the
   $k$-gram/lag-aware baseline paragraph can lose ~40 words without losing an
   experimental detail.
2. **§5.3 Mechanism-identification audit** — a single paragraph that summarizes
   Appendix I; it can shrink to 3 sentences since the appendix carries it.
3. **§1 Introduction paragraph 3** — the "recent evidence motivates this
   concern" list overlaps §2 Related work.
4. Make Table 2 (`tab:results_summary`) a `table*` — it is currently narrow
   enough for one column, but starring it can improve float packing.

Do **not** cut the theorem or definition statements; they are verbatim by
requirement.
