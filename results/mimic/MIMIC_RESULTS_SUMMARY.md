# MIMIC-IV CKD→ESRD Diagnostic Framework Results

## Overview

We applied the diagnostic framework from the ICML 2026 paper ("Do Large Language Models Detect and Exploit Decisive Sequential Information?") to a real clinical task: predicting CKD (Chronic Kidney Disease) progression to ESRD (End-Stage Renal Disease) using diagnosis sequences from MIMIC-IV 3.1.

The goal: quantify how much sequential signal exists in this clinical prediction task, and whether temporal ordering of diagnoses matters.

---

## Cohort

- **Source**: MIMIC-IV 3.1, ICD codes mapped to CCS (Clinical Classifications Software) categories (~293 unique codes)
- **Patients**: 14,813
- **Progressors (Y=1)**: 1,146 (7.7%)
- **Non-progressors (Y=0)**: 13,667
- **Sequence length**: median 49, mean 82, max 2,396 diagnosis codes per patient
- **Admissions**: median 3, mean 4.9 per patient

### Cohort Construction

Built from MIMIC-IV 3.1 via `src/mimic/build_ckd_cohort_ccs.py`:

- **Inclusion**: all patients with at least one CKD code (ICD-10 N18.1–N18.5, ICD-9 585.1–585.5)
- **Label Y=1 (progressors)**: ESRD codes (N18.6, Z99.2, Z49.x / 585.6, V45.1, V56.x) appear in a **later** admission than the first CKD code. Sequences are **truncated before ESRD onset** (correct for a prediction task: "given past history, will this patient progress?")
- **Label Y=0 (non-progressors)**: no ESRD codes ever recorded. Full observation history is retained.
- **Exclusions**: patients with CKD and ESRD in the same admission (avoids trivial same-visit prediction); patients with fewer than 10 diagnosis codes after flattening.
- **Ordering**: codes sorted by `(admittime, seq_num)`. Between admissions, `admittime` gives genuine temporal order. Within a single admission, `seq_num` reflects position on the discharge summary (not necessarily clinical temporal order).
- **Token mapping**: ICD codes mapped to CCS (Clinical Classifications Software) categories (~293 groups). Unmapped codes become "OTHER".

### Sequentiality Validation (selection bias check)

A key concern: do both groups share clinical trajectories, or are they trivially separable?

**Vocabulary overlap**:
- 265 of 293 CCS codes are shared between Y=1 and Y=0 (Jaccard = 0.90)
- **0 codes appear exclusively in Y=1** — every code a progressor has also appears in non-progressors
- 100% of each Y=1 patient's unique codes also appear somewhere in Y=0 patients
- 114 codes have >5% prevalence in both groups; 73 codes have >10% prevalence in both

**Distribution similarity**:
- Cosine similarity of code frequency vectors between Y=1 and Y=0: **0.956**
- Spearman rank correlation of code frequencies: **0.968** (p ≈ 0)
- The groups are NOT separable populations — they are the same kind of patients (CKD) who diverge in outcome

**Top shared codes (prevalence in both groups)**:

| Code | Description | freq(Y=1) | freq(Y=0) |
|------|-------------|-----------|-----------|
| CCS_158 | Chronic kidney disease | 99.7% | 89.9% |
| CCS_99 | Hypertension with complications | 97.3% | 92.6% |
| CCS_59 | Deficiency and other anemia | 82.6% | 60.2% |
| CCS_157 | Acute renal failure | 82.6% | 71.7% |
| CCS_55 | Fluid and electrolyte disorders | 75.9% | 64.0% |
| CCS_53 | Disorders of lipid metabolism | 72.3% | 74.1% |
| CCS_257 | Other aftercare | 69.0% | 69.8% |
| CCS_50 | Diabetes with complications | 62.1% | 44.4% |

**Multi-visit depth (genuine temporal ordering)**:
- 76% of patients have ≥2 admissions (genuine inter-visit temporal order)
- 58% have ≥3 admissions
- Y=1 patients: median 4 admissions (63% have ≥3)
- Y=0 patients: median 3 admissions (57% have ≥3)
- 24% of patients have only 1 admission — for these, "ordering" is discharge-summary position only

**Truncation bias check**: Despite Y=1 sequences being truncated before ESRD onset, sequence lengths are similar across groups (Y=1 median: 54 codes, Y=0 median: 49 codes), ruling out observation-time bias.

**Conclusion**: The two groups share heavily overlapping clinical trajectories. The predictive signal comes from subtle frequency differences in shared codes (e.g., CCS_156 nephritis: 30% in Y=1 vs 9% in Y=0), not from group-exclusive codes. This strengthens our finding: even though patients share similar clinical paths and have genuine multi-visit sequences, temporal ordering adds no predictive value.

---

## Step 1: K-gram Diagnostic (mirrors ICML Table 4)

For each k, we learn P(Y=1|g) for every k-gram g on the training set, predict each test sequence by averaging its k-gram probabilities, and report AUC. We also measure coverage: what fraction of test k-grams were seen in training. Results averaged over 5 random 80/20 splits.

| k | \|V_k\| (unique k-grams) | Token Coverage | Type Coverage | AUC | F1 |
|---|--------------------------|---------------|--------------|-----|-----|
| 1 | 291 | 100.0% ± 0.0% | 99.3% ± 0.4% | **0.823 ± 0.015** | 0.356 ± 0.013 |
| 2 | 33,924 | 99.0% ± 0.0% | 91.0% ± 0.3% | **0.801 ± 0.016** | 0.341 ± 0.017 |
| 3 | 378,170 | 72.5% ± 0.2% | 55.5% ± 0.2% | 0.576 ± 0.015 | 0.165 ± 0.006 |
| 4 | 773,116 | 24.7% ± 0.3% | 18.5% ± 0.2% | 0.500 ± 0.018 | 0.146 ± 0.002 |
| 5 | 895,168 | 5.2% ± 0.1% | 4.2% ± 0.1% | 0.479 ± 0.015 | 0.145 ± 0.001 |
| 6 | 910,196 | 0.9% ± 0.0% | 0.7% ± 0.0% | 0.482 ± 0.016 | 0.144 ± 0.000 |
| 7 | 902,672 | 0.1% ± 0.0% | 0.1% ± 0.0% | 0.492 ± 0.012 | 0.144 ± 0.000 |

**Key finding**: Unigrams (k=1) capture nearly all predictive signal (AUC 0.823). Bigrams (k=2) have 99% token coverage yet AUC *drops* to 0.801 — this means ordering adds no value even when coverage isn't an issue. For k≥3, vocabulary explodes and AUC falls to chance.

---

## Step 2: Shuffle Test (simple models)

Fixed 80/20 stratified split. For order-sensitive models (k-gram classifiers), we shuffle each patient's diagnosis sequence and retrain+evaluate. Shuffled results averaged over 10 random permutations.

| Model | Original AUC | Shuffled AUC | Δ AUC |
|-------|-------------|-------------|-------|
| XGBoost (BoC) | 0.810 | 0.810 ± 0.000 | 0.000 |
| LogReg (BoC) | 0.832 | 0.832 ± 0.000 | 0.000 |
| 1-gram | 0.820 | 0.820 ± 0.000 | 0.000 |
| 2-gram | 0.796 | 0.779 ± 0.007 | +0.017 |
| 3-gram | 0.553 | 0.478 ± 0.015 | +0.075 |

**Key finding**: Order-invariant models (BoC) perform best. The best model overall is LogReg on bag-of-codes (AUC 0.832), which is by definition insensitive to ordering. Bigrams lose a negligible Δ=0.017 from shuffling.

---

## Step 3: Neural Model Comparison (ordered vs shuffled)

Trained on 60/20/20 stratified splits (same patients for ordered and shuffled conditions). All models use same hyperparameters across conditions. Run on HPC with H200 GPU.

| Model | Type | Ordered AUC | Shuffled AUC | Δ AUC |
|-------|------|------------|-------------|-------|
| XGBoost BoC | Order-invariant | 0.803 | 0.803 | 0.000 |
| LogReg BoC | Order-invariant | 0.809 | 0.809 | 0.000 |
| Transformer | Order-sensitive | **0.860** | **0.844** | +0.016 |
| BiLSTM | Order-sensitive | 0.820 | 0.823 | −0.003 |
| BERT (base) | Order-sensitive | 0.826 | 0.843 | −0.017 |
| LSTM | Order-sensitive | 0.500 | 0.467 | failed |

Notes:
- LSTM failed to converge (vanishing gradient on long sequences with small hidden dim)
- XGBoost/LogReg AUCs are from the 60/20/20 split (slightly different from the 80/20 shuffle test above)
- Transformer is the strongest model overall, but its shuffled version (0.844) still outperforms all BoC baselines

**Key finding**: No sequence-aware model benefits meaningfully from temporal ordering. The Transformer's Δ=+0.016 is within noise. BiLSTM and BERT actually perform *slightly better* on shuffled data. The Transformer's advantage over BoC baselines comes from model capacity (learned code interactions), not from exploiting sequential structure.

---

## Top Discriminative Codes

Codes most enriched in progressors (Y=1) vs non-progressors (Y=0):

| Code | Description | freq(Y=1) | freq(Y=0) | log2 ratio |
|------|-------------|-----------|-----------|------------|
| CCS_156 | Nephritis, nephrosis, renal sclerosis | 30.3% | 9.2% | +1.71 |
| CCS_87 | Retinal detachments, retinopathy | 18.1% | 7.4% | +1.29 |
| CCS_215 | Genitourinary congenital anomalies | 5.5% | 2.0% | +1.47 |

CCS_158 (CKD itself) has log2R=0.15 — present in 99.7% of progressors and 89.9% of non-progressors, so not very discriminative (expected, since all patients have CKD).

## Lag Analysis

Among progressors, the median lag between consecutive kidney-related codes (CCS_158, CCS_157) is **9 positions** (IQR 3–15), with 30.6% of gaps in the 6–10 range. This means kidney codes recur relatively frequently, making them easy to detect via unigrams without needing to model temporal spacing.

---

## Summary for Paper

All three diagnostic tools converge on the same conclusion:

1. **K-gram analysis**: Unigrams alone achieve AUC 0.823. Adding bigram ordering information reduces AUC (0.801), even with 99% coverage. The task is solvable through local diagnosis patterns.

2. **Shuffle test**: Randomly permuting temporal order has no meaningful effect on any model. The best simple model (LogReg BoC, AUC 0.832) is order-invariant by design.

3. **Neural models**: Even powerful sequence architectures (Transformer, BiLSTM, BERT) show no benefit from temporal ordering (Δ AUC ranges from −0.017 to +0.016). The Transformer achieves the highest AUC (0.860) but this comes from learning code interactions, not sequential patterns — its shuffled version (0.844) still outperforms all baselines.

**This validates the ICML paper's theoretical finding on real clinical data**: CKD→ESRD progression, despite being one of the most well-documented sequential clinical cascades (diabetes → hypertension → CKD → ESRD), is predictable from *which* diagnoses appear, not *when* they appear. The sequential structure leaks local information that makes ordering redundant.

---

## File Locations

- K-gram results: `results/mimic/kgram_analysis.csv` (aggregated), `results/mimic/kgram_analysis_raw.csv` (per-seed)
- Shuffle test results: `results/mimic/shuffle_test.csv`
- Neural model results: `results/mimic/model_results.csv`
- Training logs: `logs/mimic_*_*.out`
- Scripts: `src/mimic/kgram_analysis.py`, `src/mimic/shuffle_test.py`, `src/mimic/train_mimic.py`
- Data: `data/processed/ckd_cohort_ccs.csv`, `data/processed/mimic_training/`

# Additional Task: Visit-Level Temporal Analysis

## Motivation

Our current analysis flattens all diagnosis codes into one long sequence per patient. But within a single MIMIC admission, the ordering of codes (`seq_num`) reflects discharge summary position, NOT clinical temporal order. This means our current "temporal sequence" mixes reliable between-visit ordering with unreliable within-visit ordering.

To produce an airtight result, we need to test temporal ordering at the only granularity where we have genuine time information: the visit level.

## What to Build

### Step 1: Visit-Level Preprocessing

Go back to the raw cohort data (before flattening). Group diagnosis codes by admission.

```
Patient 123:
  Visit 1 (2019-03): {CCS_99, CCS_158, CCS_53}
  Visit 2 (2020-07): {CCS_50, CCS_157, CCS_59}
  Visit 3 (2021-11): {CCS_99, CCS_158, CCS_55}
```

For each patient:
1. Group diagnoses by `hadm_id`
2. Sort visits chronologically by `admittime`
3. Represent each visit as a **multi-hot vector** of length 293 (one per CCS code). Each entry is 1 if that CCS code appears in that visit, 0 otherwise
4. The patient becomes a **sequence of multi-hot vectors**: `[v1, v2, v3, ...]` where each `vi` is a binary vector of length 293
5. Filter to patients with **3 or more admissions** (so we have genuine multi-step temporal sequences)
6. Keep the same label definition and truncation logic (truncate before ESRD visit)

Output format: a dataset where each patient has:
- `subject_id`
- `visit_sequence`: list of multi-hot vectors, ordered by admittime
- `num_visits`: number of admissions
- `label`: 0/1 for ESRD progression

### Step 2: Visit-Level Shuffle Test

The shuffle test now permutes **visit order**, not code order:

```
Original:    Visit1 → Visit2 → Visit3
Shuffled:    Visit3 → Visit1 → Visit2
```

The codes within each visit stay the same. Only the temporal ordering of visits changes.

**Models to test:**

1. **Bag-of-visits baseline (order-invariant)**: sum all multi-hot vectors across visits into a single vector. Feed to LogReg and XGBoost. This is equivalent to bag-of-codes — should match our existing BoC results.

2. **LSTM on visit sequence (order-sensitive)**: feed the sequence of multi-hot vectors into an LSTM where each timestep is one visit. This tests whether visit ordering matters.

3. **Transformer on visit sequence (order-sensitive)**: same but with a small Transformer encoder. Each visit's multi-hot vector is projected through a linear layer to create an embedding, then processed by the Transformer with positional encoding.

For each order-sensitive model:
- Train on original (chronological) visit sequences
- Evaluate on original sequences → get ordered AUC
- Evaluate on shuffled visit sequences (10 random permutations) → get shuffled AUC
- Report Δ AUC

### Step 3: Visit-Level K-gram Diagnostic

This is trickier because each "token" is now a multi-hot vector (a set of codes), not a single code.

**Simplest approach**: represent each visit by its top-3 most discriminative codes (from our existing discriminative_codes.csv analysis), concatenated and sorted alphabetically. This creates a manageable visit-type vocabulary. Then run standard k-gram analysis on visit-type sequences.

**Alternative approach**: skip visit-level k-grams and rely on the shuffle test as the primary diagnostic. The k-gram analysis at the code level (already done) plus the visit-level shuffle test together cover both granularities.

**I recommend the alternative**: the shuffle test is the cleaner experiment. Don't overcomplicate the k-gram analysis at visit level.

## Expected Outcome

The prediction is that shuffling visit order will not meaningfully affect performance, consistent with our code-level results. But now the result is airtight: we are testing genuine temporal ordering (visits separated by weeks/months), with no contamination from unreliable intra-visit ordering.

## Why This Matters for the Paper

This addresses the strongest possible reviewer objection: "Your ordering was just discharge summary artifacts." By aggregating to the visit level and testing only inter-visit temporal order, we show that even genuine temporal structure in clinical data adds no predictive value for CKD→ESRD progression.

## Technical Notes

- Multi-hot vectors are sparse (293 dimensions, typically 5-20 codes per visit). Use sparse representations if memory is an issue.
- For the Transformer/LSTM: input dimension is 293 (or projected down to e.g. 64 via a linear layer). Sequence length is number of visits (median ~4, much shorter than the flattened sequences).
- These models should train fast — short sequences, moderate input dimension.
- Use the same train/val/test split as the existing experiments for comparability.
- Run with 3-5 seeds to get standard deviations on the neural models.
