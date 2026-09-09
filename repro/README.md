# Sequence-prediction reproduction code

This directory contains implementations used for the TMLR revision and historical NeurIPS/ICML experiments. Start with the [repository guide](../README.md) and [current evidence status](../docs/tmlr_status.md).

Run modules from the repository root through the launcher:

```bash
bash scripts/run_repro.sh -m pytest tests/test_oc_completion.py -q
bash scripts/run_repro.sh -m src.oc_completion.gen_datasets --help
bash scripts/run_repro.sh -m src.oc_completion.gen_pairs --help
```

The launcher selects this directory's `src` package, sets data and output roots, and uses `${PYTHON:-python}`. For example, `PYTHON=/path/to/environment/bin/python bash scripts/run_repro.sh -m pytest tests/test_oc_completion.py -q`. It does not load `.env` automatically.

| Component | Code | Role |
|---|---|---|
| OC matched-completion work | `src/oc_completion/` | Rule, corrected dataset generation, pair construction, training and scoring |
| Feature analysis | `src/analysis/mechanism_id/` | Historical label checks, feature comparisons and diagnostics |
| Model experiments | `src/experiments/` | BERT, decoders, small neural models and feature baselines |
| Parity and ablations | `src/data/`, `src/ablations/` | Historical controls; verify task definitions and run settings |
| Clinical diagnosis sequences | `src/mimic/` | Credentialed cohort preparation and model comparisons |

Generated datasets are restored from a checksum manifest or generated under an explicitly new version. They are not shipped in this checkout. See [data handling](../data/README.md). The historical tag `_6` and corrected `ocdet` files must remain distinguishable.

[REPRODUCING.md](REPRODUCING.md) retains earlier commands for reference. It is not a verified end-to-end TMLR reproduction protocol. Historical seed counts, exact data usage, generator versions and model recipes still require checks. Files in `results_reference/` do not substitute for missing predictions.

The `repro/` subtree retains its existing [MIT license](LICENSE). No additional license has been assigned to the rest of the repository during cleanup.
