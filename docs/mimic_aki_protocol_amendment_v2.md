# MIMIC-IV AKI trajectory audit: protocol amendment v2

Amendment approved: 2026-08-11 UTC

Parent protocol: `configs/mimic_aki_a100_audit_v1.yaml`

Parent YAML byte SHA-256:
`f57e6d7aa9b12eb67779be09145780d5b0a1f6777e3b0bca186bee570d5694cd`

Parent semantic configuration SHA-256:
`4ce68faea065397ddc83b901dac7fc5b65fdffc6bc85c94888470778051fef1a`

Amended protocol: `configs/mimic_aki_cpu_smd015_amendment_v2.yaml`

## Trigger and status

The v1 preparation completed before this amendment. No model training,
prediction, or performance evaluation had been run. The v1 validation split
had post-match `duration_hours` SMD `-0.1074843234`, narrowly exceeding the
v1 absolute limit of `0.10`. All pair-count, class-presence, patient-split,
event-integrity, sequence-length, and provenance checks otherwise passed.

The v1 result remains immutable and recorded as non-estimable under its
original rule. This v2 analysis is an amended confirmatory analysis, not the
original fully prespecified confirmatory analysis.

## Approved changes

1. Retain `matching.maximum_absolute_smd: 0.10` as the default for every
   configured balance feature.
2. Add a single feature-specific override:
   `matching.maximum_absolute_smd_overrides.duration_hours: 0.15`.
3. Require the matching audit to report all four configured features in every
   split, together with each effective limit and a pass/fail flag. Non-finite
   post-match SMDs fail estimability.
4. Record `models.sequence_training.device: cpu` and `pin_memory: false`
   because the execution VM is CPU-only. No model architecture, optimization,
   seed, epoch, batch-size, or preprocessing choice changes.

All other cohort, episode, labeling, matching, split, representation, model,
control, seed, and metric definitions remain identical to v1.

## Rationale and limitation

An absolute SMD below `0.10` is a common conservative convention, not a
universal boundary. Austin describes SMD balance diagnostics and notes the
absence of a universally agreed criterion; fixed cutoffs cannot by themselves
guarantee absence of bias. A recent large OHDSI/LEGEND observational study used
maximum post-adjustment SMD below `0.15` as an applied diagnostic. These sources
provide methodological context and precedent, not proof that an SMD of
`0.1075` is harmless in this predictive audit.

The duration-only `0.15` limit is therefore documented as an explicit,
outcome-blinded, post-preparation choice approved after observing the v1
balance diagnostic. The original `0.10` failure remains a reported sensitivity
result and limitation. The amendment must not be described as a universal
methodological consensus.

## Reproducibility requirements

- Freeze the amended YAML and its byte and semantic hashes before v2
  extraction.
- Use the identical amended file and semantic hash for extraction,
  preparation, and training.
- Write v2 into a new run root; do not overwrite or relabel v1 artifacts.
- Preserve this amendment, validation output, implementation hashes, console
  logs, manifests, balance tables, predictions, and aggregate metrics.

## References

- Austin PC. [Balance diagnostics for comparing the distribution of baseline
  covariates between treatment groups in propensity-score matched
  samples](https://pubmed.ncbi.nlm.nih.gov/19757444/). *Statistics in
  Medicine*. 2009;28:3083-3107. doi:10.1002/sim.3697.
- Austin PC. [An Introduction to Propensity Score Methods for Reducing the
  Effects of Confounding in Observational
  Studies](https://pubmed.ncbi.nlm.nih.gov/21818162/). *Multivariate
  Behavioral Research*. 2011;46:399-424. doi:10.1080/00273171.2011.568786.
- Hripcsak G, et al. [Assessing Covariate Balance With Small Sample
  Sizes](https://pubmed.ncbi.nlm.nih.gov/40772805/). *Statistics in Medicine*.
  2025;44:e70212. doi:10.1002/sim.70212.
- Kim DH, et al. [Real-world evidence for comparative safety of second-line
  antihyperglycemic agents in older adults with type 2
  diabetes](https://www.nature.com/articles/s41467-026-71307-0). *Nature
  Communications*. 2026;17:7195. doi:10.1038/s41467-026-71307-0.
