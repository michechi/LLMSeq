# Local data and provenance

Dataset files are intentionally absent from the tracked tree. The original checkout still contains its data. The cleanup has not deleted that local copy, regenerated labels, or changed any stored results.

`manifests/retired_artifacts.csv` records the exact size, SHA-256, original Git blob and source commit for synthetic datasets and retired generated figures/paper builds. Retained OC generation manifests remain at their original paths under `simulation/oc_completion/`.

To recover synthetic files from a local original checkout, run from the repository root:

```bash
python scripts/restore_artifacts.py --group synthetic-data --source-root /path/to/original/LLMSeq
```

Alternatively, when the original commit is still present in local Git history:

```bash
python scripts/restore_artifacts.py --group synthetic-data --from-git
```

Use `--dry-run` to validate without writing. Use `--group generated-figures` or `--group compiled-paper` for those archived outputs. Existing files with different hashes cause an error; there is no overwrite option. The utility never downloads anything and cannot restore clinical files or logs.

Do not describe regeneration as exact recovery without checking hashes. Historical tag `_6`, tag `_9`, parity files and corrected `ocdet`/`ocnoisy` files have different provenance and must remain distinguishable.

Raw and derived MIMIC-IV data must stay in the credentialed environment. Configure paths using the [clinical reproduction guide](../repro/data/mimic/README.md) or [AKI guide](../docs/mimic_aki_audit.md). No clinical table, sequence, label file, serialized visit record or patient-level prediction belongs in this public checkout. Removing current files does not remove their earlier Git history.
