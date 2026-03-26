# ICML26 Rebuttals

## R1

We thank Reviewer 1 for their detailed and constructive feedback. We address each point below.

---

### Point 1: Definition of compliance and generalization of Theorem 3.3

**Reviewer concern:** The proof of Theorem 3.3 in Appendix C.1 uses the narrow (naive) case where $S = \mathcal{A}$, $\lambda = 1$, and the full sequence is non-decreasing (stars-and-bars argument), but the general compliance definition (Definition 3.1) allows arbitrary $S \subseteq \mathcal{A}$, $\lambda \geq 1$, and only requires the existence of an increasing subsequence of key symbols at lag-spaced positions. The reviewer correctly notes that the appendix does not prove the general case.

**Response:** The reviewer is correct that our original proof only covers the naive setting. We have now produced a **generalized proof** using a coupling argument that covers the full generality of Definition 3.1 (arbitrary $S$, $\kappa$, $\lambda$) under strict inequality. The key idea is as follows:

**Theorem (General Case).** Under Definition 3.1 with $|S| = m \geq 2$, any bijection $\kappa$, any $\lambda \geq 1$, $n \geq \lambda + 1$, and compliance probability $\rho \in (0,1)$: $I(Y^*; X_1) > 0$.

**Proof sketch.** Let $s_{\min}, s_{\max} \in S$ with $\kappa(s_{\min}) = 1$ and $\kappa(s_{\max}) = m$.

**Case 1: $S \subsetneq \mathcal{A}$ (non-key symbols exist).** Fix $a \in \mathcal{A} \setminus S$ and couple two sequences $\mathbf{X}$ and $\mathbf{X}'$, identical at positions $2, \ldots, n$, with $X_1 = s_{\min}$ and $X'_1 = a$. Since $a \notin S$, every compliant chain in $\mathbf{X}'$ avoids position 1 and remains valid in $\mathbf{X}$, so $\{\mathbf{X}' \text{ compliant}\} \subseteq \{\mathbf{X} \text{ compliant}\}$. Conversely, setting $X_{1+\lambda} = s_{\max}$ with all other positions $\geq 2$ filled with non-key symbols (positive probability since $S \subsetneq \mathcal{A}$) makes $\mathbf{X}$ compliant via chain $(1, 1{+}\lambda)$ with $\kappa(s_{\min}) = 1 < m = \kappa(s_{\max})$ (strict, since $m \geq 2$), while $\mathbf{X}'$ is not compliant (position 1 excluded, only one key symbol at $1{+}\lambda$ cannot form a chain of length $\geq 2$). Hence $P(Y^* = 1 \mid X_1 = s_{\min}) > P(Y^* = 1 \mid X_1 = a)$.

**Case 2: $S = \mathcal{A}$, $\rho < 1$.** Couple $\mathbf{X}$ ($X_1 = s_{\min}$) with $\mathbf{X}'$ ($X_1 = s_{\max}$), identical at positions $\geq 2$. Any chain in $\mathbf{X}'$ starting at position 1 requires $\kappa(s_{\max}) < \kappa(X'_{1+\lambda})$, which is impossible since $m$ is the maximum rank. So all chains in $\mathbf{X}'$ lie in positions $\geq 2$ and transfer to $\mathbf{X}$. Setting all positions $\geq 2$ to $s_{\max}$ (probability $(1/\ell)^{n-1} > 0$) makes $\mathbf{X}$ compliant via $(1, 1{+}\lambda)$ with $1 < m$, while in $\mathbf{X}'$ every element has rank $m$ and no strictly increasing pair exists. Hence $P(Y^* = 1 \mid X_1 = s_{\min}) > P(Y^* = 1 \mid X_1 = s_{\max})$.

In both cases $P(Y^* = 1 \mid X_1)$ is non-constant, so $I(Y^*; X_1) > 0$. $\square$

**Changes to the paper:**
- We add the generalized proof to the appendix, keeping the original stars-and-bars argument as a quantitative illustration for the naive setting.
- We tighten the theorem statement to explicitly parameterize by $S$ and $\lambda$, matching Definition 3.1.
- In the proof outline (Section 3.2), we clarify that the stars-and-bars calculation is an illustrative example for the naive setting, and reference the general proof in the appendix.
- We also correct the inconsistency between strict inequality in Definition 3.1 ($\kappa(X_{t_1}) < \cdots < \kappa(X_{t_k})$) and non-strict inequality in the naive setting description ($\mathcal{K}(X_1) \leq \cdots \leq \mathcal{K}(X_n)$), harmonizing both to use strict inequality throughout.

---

### Point 2: Uniform prior assumption and over-claimed generality

**Reviewer concern:** The compliance framework assumes $X_t \stackrel{\mathrm{iid}}{\sim} \mathrm{Uniform}(\mathcal{A})$, which is unrealistic for medical or text data. Claims in the intro and Section 3.2 that results extend to all "classification tasks based on event-level ordering" are not qualified with this assumption.

**Response:** The reviewer raises a valid point about qualification of claims. However, we note that the impossibility result is **substantially more general than the uniform assumption suggests**:

**The uniform distribution is not needed for the impossibility theorem.** Our generalized coupling proof (see Point 1 above) never uses the uniform distribution. The argument requires only:
1. **Independence (i.i.d.):** so that conditioning on $X_1$ does not change the distribution of $(X_2, \ldots, X_n)$, making the coupling valid.
2. **Full support:** $P(a) > 0$ for all $a \in \mathcal{A}$, so that the specific realizations in the strict-inequality step have positive probability.

The coupling between $\mathbf{X}$ and $\mathbf{X}'$ works identically under $X_t \stackrel{\mathrm{iid}}{\sim} P$ for *any* distribution $P$ with full support. We therefore **strengthen the theorem** to:

> *Under any i.i.d. distribution $P$ on $\mathcal{A}$ with $P(a) > 0$ for all $a$, the impossibility result holds: $I(Y^*; X_1) > 0$.*

This directly addresses the reviewer's concern: the result is not limited to uniform priors but holds for arbitrary i.i.d. distributions.

**What IS specific to the uniform assumption:**
- **Compliance probability formulas** (e.g., $\rho_{\mathrm{naive}} = \binom{n+\ell-1}{n}/\ell^n$): these are exact only under uniform.
- **Optimal performance bounds** (AUC = $1 - \pi$): these depend on the specific $\rho$ value.
- **Benchmark data generation:** uniform is a design choice to eliminate distributional confounds and isolate sequential structure.

**Regarding non-i.i.d. sequences (the practical concern):** The i.i.d. assumption is genuinely needed for the coupling. However, ordering constraints create local information through a structural mechanism: knowing a symbol's rank at any position restricts which compliant sequences are reachable. This mechanism is orthogonal to the marginal distribution. Non-uniform priors (where some events are rarer) typically *amplify* local information rather than eliminating it, since observing a rare high-rank event at an early position is even more informative about non-compliance.

**Changes to the paper:**
- We generalize the theorem statement to any i.i.d. distribution with full support (dropping "uniformly").
- We qualify claims in the introduction (line 146) and Section 3.2 (line 271) to read "under independent sampling" rather than implying universal applicability.
- We add a remark noting that the uniform assumption is specific to the quantitative bounds and benchmark design, not to the qualitative impossibility result.
- We discuss the non-i.i.d. case as a limitation and future direction.

---

### Point 3: Induction-heads counterexample

**Reviewer concern:** The induction-heads task is an "event-level ordering task" that transformers can solve, appearing to contradict the claim that such tasks cannot satisfy pure sequential dependence. The reviewer proposes a variant where $Y = (b + c) \bmod \ell$ (sum of the two symbols after a repeated token), which may satisfy pure sequential dependence while being predictable.

**Response:** We appreciate this thoughtful example. However, we believe it reflects a terminological ambiguity rather than a mathematical contradiction.

**Induction heads are not compliance tasks.** Our impossibility theorem (Theorem 3.3) applies to a precisely defined class: **compliance tasks** (Definition 3.1), where the label is $Y^* = \mathbf{1}\{\text{key symbols appear in prescribed order } \kappa \text{ at lag } \lambda\}$. This has a specific structure:

| Property | Compliance (our framework) | Induction heads |
|---|---|---|
| What determines the label | Rank ordering of key symbols under $\kappa$ | Identity of tokens after a repeated token |
| Key mechanism | "Are events in the right order?" | "What came after this token before?" |
| Fixed structure to learn | Key set $S$, ordering $\kappa$, lag $\lambda$ | None (any token can trigger) |
| Task type | Order verification | Pattern copying |

The induction-head task depends on **token identity and repetition** (finding a repeated token and copying what follows), not on **rank ordering under a hidden key**. These are fundamentally different types of sequential dependence.

**The reviewer's variant is valid but not a compliance task.** The proposed task $Y = (b + c) \bmod \ell$ is a clever construction that likely satisfies pure sequential dependence under uniform i.i.d. sampling (by alphabet symmetry, $P(Y = y \mid X_t = a) = P(Y = y)$ for all $t, a$). However, this does not contradict our theorem, which states:

> "**Compliance** tasks cannot satisfy pure sequential dependence."

This is compatible with other task types (pattern copying, modular arithmetic on copied tokens) satisfying pure sequential dependence. Indeed, our own **parity task** (Section 3.3) is introduced precisely because compliance tasks *cannot* have this property — we need a non-compliance task to test global integration.

**The source of confusion is our terminology.** The paper uses "event-level ordering tasks" and "classification tasks based on event-level ordering" in several places where we mean specifically "compliance tasks as defined in Definition 3.1." This vague phrasing allows a broader interpretation that includes induction heads.

**Changes to the paper:**
- We replace "event-level ordering tasks" with "ordering-based compliance tasks (Definition 3.1)" in lines 146, 164, and 271, removing the terminological ambiguity.
- We add a brief discussion of induction heads as a related but distinct type of sequential task (pattern copying vs. order verification), noting that our impossibility result applies specifically to compliance-based classification.
- We clarify that the theorem does not claim *no* task can have pure sequential dependence — only that compliance tasks cannot.

---

### Point 4: Parity on binary vocabulary

**Reviewer concern:** A simple transformer achieves perfect AUC on parity with vocabulary size 2 ($\ell = 2$, $n = 10$), contradicting the claim that parity is not learnable.

**Response:** The reviewer's experiment is correct: binary parity IS learnable by a small transformer. We acknowledge that our claims about parity are overstated. However, the two experimental settings are in fundamentally different complexity regimes:

| Parameter | Reviewer's experiment | Our experiment |
|---|---|---|
| Alphabet $\ell$ | **2** | **26** |
| Key set $\|K\|$ | 1 (trivial — only symbol "1") | 6 (hidden, random) |
| Sequence length $n$ | 10 | 20 |
| Input space size | $2^{10} = 1{,}024$ | $26^{20} \approx 10^{28}$ |
| Batch size | 1,024 (= **entire input space**) | Fixed 400K ($\approx 10^{-22}$ of space) |
| Key discovery needed? | No | Yes (find 6 among 26) |
| Distractor rate | 0% | ~77% |

The critical difference: the reviewer's setup covers the **entire input space in a single batch** (1,024 possible binary sequences of length 10 = 1,024 batch size). The model effectively memorizes a lookup table. Our models see $4 \times 10^5$ samples from a $10^{28}$-element space.

Moreover, our task requires solving **three interleaved problems simultaneously**: (1) identify which 6 of 26 symbols are key (feature selection), (2) count occurrences of each key symbol (counting), (3) compute parity of even-count key symbols (modular arithmetic). The reviewer's task reduces to XOR of 10 bits, a well-studied function known to be learnable.

**An informative asymmetry:** By Proposition 3.5, the reviewer's task has $p = |K|/|\mathcal{A}| = 1/2$, giving $I(Y^*; X_t) = 0$ exactly (zero local information). Our task has $p = 6/26 \neq 1/2$, giving $I(Y^*; X_t) > 0$ (nonzero local information). So our models fail despite having *more* local information than the reviewer's model, confirming that the failure is about **optimization complexity** (large alphabet, hidden keys, sparse coverage), not about information-theoretic limitations.

**Changes to the paper:**
- We revise line 458 (*"This is not a matter of scale or training data"*) which is directly contradicted by the reviewer's experiment. The revised text will read:

  > *"In our experimental regime ($\ell = 26$, $m = 6$, $n = 20$), every model performs at chance. While simpler instances such as binary parity ($\ell = 2$, $n = 10$) are learnable by small transformers [as demonstrated by the reviewer], task difficulty scales rapidly with alphabet size and key set complexity. The failure reflects the practical challenge of simultaneously discovering hidden key symbols, counting their occurrences, and computing parity — a combinatorial burden that overwhelms current architectures at realistic parameter scales."*

- We revise line 488 to remove the claim that parity failure is not about scale/data.
- We add a **sensitivity analysis** sweeping $\ell$ from 2 to 26 (holding other parameters proportional) to show the performance degradation curve. This will precisely characterize where the transition from learnable to unlearnable occurs, turning the reviewer's concern into a new empirical contribution.

---

### Point 5: No signal beyond k-grams

**Reviewer concern:** The claim that there is "no signal beyond k-grams" is incorrect. Induction heads (and in-context learning more generally) capture dependencies beyond k-grams, even out of distribution.

**Response:** The reviewer is correct that transformers possess mechanisms (induction heads, in-context learning) that capture dependencies beyond any fixed k-gram. We agree that our framing is ambiguous.

The paper's finding #3 (line 156: *"No signal beyond k-grams, no learning"*) conflates two distinct things:

1. **A property of the task:** k-gram baselines carry no discriminative signal on parity.
2. **A claim about model capabilities:** models are limited to k-gram-level patterns.

Statement (1) is a valid empirical observation. Statement (2) is too strong and is contradicted by the existence of induction-head mechanisms. Our actual finding is more nuanced: **on the specific parity task, the type of global computation required (counting modulo 2 over a hidden key set in a large alphabet) defeats all tested models.** This is not because models are "capped at k-gram level" — it is because the specific computation (feature selection + counting + modular arithmetic) is too hard in our parameter regime.

The reviewer's binary-parity counterexample (Point 4) further demonstrates this: a transformer succeeds on parity with zero local information ($p = 1/2$), proving that the failure is about task complexity, not a fundamental k-gram ceiling.

**Changes to the paper:**
- We rewrite finding #3 to be explicitly empirical and task-scoped:

  > *"Parity resists all architectures: When the task requires global integration with no local shortcuts (parity with $\ell = 26$, $m = 6$), none of the tested models exceed chance performance, from small LSTMs to 70B-parameter LLMs."*

- We remove the implication that models are fundamentally limited to k-gram patterns.
- We add a remark acknowledging that transformers have beyond-k-gram mechanisms (induction heads, in-context learning), and clarify that the parity failure reflects the specific difficulty of counting-based modular arithmetic over hidden keys in a large alphabet, not a general capability ceiling.
- We revise line 488 accordingly, replacing *"the task is simply invisible to any method that relies on local patterns"* with a more precise characterization of why the parity task is hard in our regime.

---
---

## R2

We thank Reviewer 2 for their constructive feedback on framing, practical relevance, and presentation. We address each point below.

---

### Point 1: Ambiguity of purpose and stance

**Reviewer concern:** The paper's stance on whether models should rely on semantics versus sequence ordering is ambiguous. It is unclear to what extent the demonstrated drawback affects real-world performance — one could argue that in many cases the event itself is more informative than its timing.

**Response:** The reviewer raises an important point, and we agree that the paper's stance should be clarified. The paper's contribution is **diagnostic, not prescriptive**: we do not argue that models *should* use ordering information, but rather provide a framework to determine *whether* they do and *how much* of a task is solvable through local patterns versus global sequential structure.

The reviewer's intuition — that "the event itself is often more informative than its timing" — is in fact supported by our framework. To demonstrate this concretely, we applied our diagnostic methodology to a **real clinical task: predicting progression from Chronic Kidney Disease (CKD) to End-Stage Renal Disease (ESRD) on MIMIC-IV** (14,813 patients, 7.7% progressors). Our findings:

- **Unigram features capture nearly all signal** (AUC = 0.823); bigrams add nothing.
- **Shuffling the temporal order** of diagnoses has negligible effect on model performance (Δ AUC ranges from -0.017 to +0.016 across all models).
- An **order-invariant logistic regression** on bag-of-codes achieves AUC = 0.832, matching or exceeding all sequential models.

This validates the reviewer's concern: for CKD→ESRD, *which* diagnoses appear is predictive, not *when* they appear. Our framework correctly diagnoses this.

The practical implication is clear: before deploying a sequential model where temporal structure is "presumed critical," practitioners should check whether ordering actually carries additional signal. Our framework — combining k-gram analysis, theoretical bounds, and shuffle tests — provides the methodology to answer this question systematically. For some tasks the answer will be "ordering doesn't matter" (as in CKD→ESRD); for others, ordering will carry genuine additional signal. The value of our contribution lies in providing the tools to distinguish these cases, not in prescribing which type of model to use.

**Changes to the paper:**
- We clarify in the introduction and discussion that the paper's contribution is a diagnostic framework, not a claim that ordering is always important.
- We add the MIMIC-IV case study as a validation of the framework on real clinical data, demonstrating how the diagnostic tools work in practice.
- We reframe the discussion to emphasize that the framework helps practitioners determine whether sequential structure matters for their specific task.

---

### Point 2: Lack of quantitative gap analysis

**Reviewer concern:** It would be stronger to quantify the performance gap between more semantic-based and more order-based models.

**Response:** We agree that quantifying this gap strengthens the paper. We provide three complementary quantitative analyses:

**1. K-gram gap analysis (already in paper, now emphasized).** The k-gram baseline tables (Table 4, Appendix) already quantify how much signal is captured at each locality level. For the tricky stochastic task ($m=6$, $\lambda=7$, $\pi=0.3$): the 2-gram baseline achieves AUC $\approx$ 0.58, while the theoretical ceiling is AUC* = 0.67. The "beyond-local-patterns" gap is approximately 0.09 AUC — small but present. LSTM reaches 0.661, capturing most of this gap; Llama-1B reaches only 0.587, barely exceeding the 2-gram baseline.

**2. Sensitivity grid (new, 29 configurations).** We conducted a comprehensive sensitivity analysis across a $3 \times 3 \times 3$ grid of parameters ($n \in \{15, 20, 25\}$, $m \in \{3, 6, 10\}$, $\lambda \in \{3, 7, 10\}$) with class balance fixed at $\rho = 0.293$ via resampling from a large pool. In the stochastic setting:

| Model | Range of AUC across grid | AUC* range | Fraction of configs at AUC* |
|-------|--------------------------|------------|---------------------------|
| BERT | 0.570 – 0.705 | 0.570 – 0.699 | ~90% (1 anomaly at $m=6,\lambda=1$) |
| LSTM | 0.563 – 0.700 | 0.570 – 0.699 | ~85% |
| Transformer (small) | 0.533 – 0.692 | 0.570 – 0.699 | ~60% (struggles at $m=10$, small $\lambda$) |
| Llama-1B | 0.543 – 0.600 | 0.570 – 0.699 | ~15% (only easy configs) |

This quantifies the gap between architectures across the full parameter space, normalized by per-configuration theoretical bounds.

**3. MIMIC-IV shuffle test (new).** On CKD→ESRD prediction, the gap between ordered and shuffled sequences is Δ AUC $\in [-0.017, +0.016]$ — statistically indistinguishable from zero. This quantifies the "ordering gap" on real data: for this task, it is effectively zero.

**Changes to the paper:**
- We add a summary table quantifying the "beyond-k-gram gap" for each architecture across task variants.
- We include the sensitivity grid results in the appendix with discussion of how the gap varies with task parameters.
- We include the MIMIC-IV shuffle test as a real-world quantification of the ordering gap.

---

### Point 3: Comparison between encoder and decoder models

**Reviewer concern:** The comparison may be skewed because binary classification naturally favors discriminative (encoder) models over generative (decoder) models.

**Response:** The reviewer raises a valid architectural point. Binary classification is indeed a more natural fit for encoder architectures. We acknowledge this and have strengthened our analysis with **mechanistic evidence** showing that the gap has a specific, identifiable cause beyond the classification/generation mismatch.

**Attention analysis (new).** We analyzed attention patterns for BERT and Llama-1B on the tricky stochastic task:

| Feature | BERT | Llama-1B |
|---------|------|----------|
| Lag-specialized attention heads | Yes — layers 3-5, attention ratio 3-4x between lag-spaced key positions | No — all attention ratios $\approx$ 1x |
| Key letter identification | Yes (early layers) | Yes (attends more to key letters) |
| Lag structure exploitation | Yes (peaks in mid layers) | No (flat across all 16 layers) |
| Distinguishes true compliance from noise | Yes (significantly different attention for true vs. noise-flipped) | Partial |

This reveals a specific mechanistic explanation: **bidirectional attention allows BERT to directly attend between lag-separated positions**, forming "lag-detector" heads. Llama's causal mask prevents this — it can identify key letters but cannot learn the lag structure, because it can only attend backward and cannot integrate information from future positions that would complete the compliance chain.

**Crucially, the limitation is surmountable with scale.** Qwen-14B (decoder-only) matches the theoretical AUC* ceiling of 0.670 (confirmed across 3 seeds: AUC = 0.670 $\pm$ 0.001). This shows that decoder-only models *can* solve the task, but require $\sim$140x more parameters than encoder architectures to do so. The gap is about **efficiency**, not fundamental capability.

We frame this explicitly in the revised paper (line 493 already states this): the efficiency gap reflects a mismatch between next-token prediction objectives and classification tasks, not a fundamental architectural limitation. The causal attention mask forces decoders to encode ordering information through their parameters rather than through bidirectional integration.

**Regarding the reviewer's question on optimization vs. architecture:** The evidence suggests both contribute. The causal mask is an architectural constraint (prevents direct lag-detection), but sufficient capacity allows the model to learn equivalent representations through its parameters (as demonstrated by Qwen-14B). This is consistent with the well-known result that autoregressive models are universal approximators but may require more capacity than bidirectional models for tasks requiring global context.

**Changes to the paper:**
- We add the attention analysis to the paper (or appendix), providing mechanistic evidence for the encoder-decoder gap.
- We explicitly acknowledge the reviewer's point about task-architecture mismatch in the discussion.
- We emphasize that the comparison reveals an *efficiency* gap, not a *capability* gap, supported by Qwen-14B's success at scale (3 seeds confirming robustness).

---

### Point 4: Figure readability

**Reviewer concern:** Figures are difficult to interpret due to thin, overlapping bars.

**Response:** We agree and will improve figure readability. Specifically:

- **Split figures into subpanels** by architecture category (encoders, decoders, baselines), reducing visual clutter.
- **Use line plots with confidence bands** for the scaling analysis (AUC vs. training data size), instead of grouped bars.
- **Reduce to the most informative models per figure** — the current figures plot all models simultaneously, making individual trends hard to follow. We will show 3-4 representative models per panel.
- **Increase line weight and marker size** for improved readability at print resolution.

We note that Reviewer 3 raised a similar concern (Point 3: "too many models plotted in single figures"), confirming that this is a presentation issue worth addressing.

**Changes to the paper:**
- Redesign Figures 2-4 with subpanels by architecture category.
- Add a summary comparison figure showing only the top-performing model from each category.
- Move detailed per-model figures to the appendix for completeness.

---
---

## R3

We thank Reviewer 3 for their constructive suggestions on scope, evaluation methodology, and presentation. We address each point below.

---

### Point 1: Underutilization of LLMs (no zero-shot/few-shot evaluation)

**Reviewer concern:** The evaluation does not leverage instruction following or in-context learning; models are used only as encoder-style classifiers via fine-tuning, missing zero-shot or few-shot settings with natural language task descriptions.

**Response:** This is a thoughtful concern, and we want to clarify why we made this design choice deliberately.

Our framework intentionally strips away semantic content to isolate sequential pattern exploitation. In a zero-shot or few-shot setting, the LLM would attempt to leverage linguistic knowledge — which is precisely the confound we aim to eliminate. Consider what a zero-shot prompt would require:

> *"Classify whether this letter sequence is compliant: BXDCAQZHELMPTKDWSANJ"*

The model has no way to determine compliance without knowing the hidden key set $\mathcal{S}$, the ordering $\kappa$, and the lag $\lambda$. Revealing these parameters in the prompt would give away the answer, turning the task into instruction execution rather than pattern learning. Few-shot examples (input-output pairs without explanation) would require the model to *induce* the hidden rule — which is essentially what fine-tuning does, but with far fewer examples and no gradient signal.

More fundamentally, the paper tests whether models can **learn** sequential structure from data. Zero-shot evaluation tests whether models already **know** something about the task from pretraining. Since our tasks are synthetic and bear no relation to natural language, zero-shot success would be uninformative about sequential reasoning capabilities.

That said, to directly address the reviewer's question: we can confirm that in preliminary tests, providing LLMs with few-shot examples (5-10 labeled sequences) and a natural language task description yields chance-level performance, as expected — the hidden combinatorial structure cannot be deduced from a handful of examples. This confirms that fine-tuning is the appropriate evaluation paradigm for our framework.

We acknowledge that testing LLMs in their "native" instruction-following mode on real sequential tasks (where semantic understanding can assist) is an important complementary evaluation. Our MIMIC-IV case study (see Point 4) provides a step in this direction, applying the diagnostic framework to real clinical data.

**Changes to the paper:**
- We add a paragraph in Section 4 (Experimental Setup) explicitly justifying the fine-tuning approach: the tasks are designed so that instruction following provides no advantage, because the task structure is hidden and must be learned from data.
- We note zero-shot/few-shot evaluation on real-world sequential tasks as a direction for future work.

---

### Point 2: Weak training signal for parity

**Reviewer concern:** The parity task provides extremely sparse supervision for the required global computation, so the observed failure does not conclusively support a fundamental architectural limitation.

**Response:** The reviewer is correct, and we agree that the parity failure should not be framed as evidence of a fundamental architectural limitation. This concern aligns with Reviewer 1's Point 4 (binary parity counterexample), and we address both together.

We acknowledge two complementary explanations for the parity failure:

1. **Computational hardness:** Parity is outside AC⁰ — it is provably hard for bounded-depth circuits, and gradient-based learning on this function class faces well-documented challenges (Hahn 2020, Bhattamishra et al. 2020). The sparse binary supervision (a single bit label for a global property) provides minimal gradient signal for the required computation.

2. **Task complexity in our regime:** Our parity task ($\ell = 26$, $m = 6$, $n = 20$) requires simultaneously solving three interleaved problems: (i) identifying which 6 of 26 symbols are key (feature selection), (ii) counting occurrences of each key symbol (counting), and (iii) computing parity of even-count key symbols (modular arithmetic). As Reviewer 1 demonstrated, binary parity ($\ell = 2$, $n = 10$) is learnable by a small transformer — confirming that the failure is about task complexity in our regime, not a fundamental impossibility.

The evidence for **local pattern reliance** extends beyond parity alone:
- Model performance across ALL tasks (naive, tricky deterministic, tricky stochastic, parity) correlates with k-gram baseline signal — a consistent pattern, not a single observation.
- Our **attention analysis** (new) shows that BERT develops specialized "lag-detector" heads (layers 3-5, attention ratio 3-4x at lag-spaced key positions) for the compliance task where local patterns exist, but no equivalent structure emerges for parity. This provides mechanistic evidence that models exploit the specific local structure when available.
- The tricky tasks provide a key contrast: models succeed on tasks requiring global structure that *happen to leak local signal*, and fail on parity where no such signal exists.

Regarding alternative training strategies (curriculum learning, auxiliary objectives): we agree these are promising directions. Starting with short sequences and gradually increasing length, or providing auxiliary supervision on key-set membership, could potentially support learning global counting behavior. We note this as future work.

**Changes to the paper:**
- We revise the parity discussion to frame the failure as reflecting **task complexity in our parameter regime** rather than a fundamental architectural limitation.
- We add a discussion of computational hardness (AC⁰ bounds) as a complementary explanation.
- We note curriculum learning and auxiliary objectives as future directions.

---

### Point 3: Poor visual presentation

**Reviewer concern:** Too many models are plotted in single figures, reducing readability and making comparisons across architectures and scales difficult.

**Response:** We agree — both Reviewer 2 (Point 4) and Reviewer 3 flag this, confirming it needs attention. We will:

- **Split figures into subpanels** by architecture category (encoders, decoders/LLMs, baselines), so each panel shows 3-4 models with clear trends.
- **Use line plots with confidence bands** for scaling analyses (AUC vs. training data size), replacing grouped bars.
- **Add a summary comparison figure** showing only the top-performing model from each architecture category, providing a clean cross-architecture comparison.
- **Move detailed per-model figures to the appendix** for completeness.
- **Increase line weight and marker size** for readability at print resolution.

**Changes to the paper:**
- Redesign Figures 2-4 with architecture-grouped subpanels.
- Add summary figure with one representative model per category.
- Move full-detail figures to appendix.

---

### Point 4: Highly abstract synthetic setting

**Reviewer concern:** The gap between letter-based tasks and real temporal reasoning may limit the practical implications of the conclusions.

**Response:** We agree that bridging from synthetic to real data is important for practical impact. The synthetic design is intentional — it allows controlled manipulation of parameters (key set, ordering, lag, noise) that are hopelessly confounded in real sequential data. But we recognize the need to demonstrate that the framework transfers.

To this end, we have conducted a **clinical validation on MIMIC-IV** (14,813 patients, CKD→ESRD progression, 7.7% positive rate):

**Applying the diagnostic framework:**

| Diagnostic tool | Finding |
|-----------------|---------|
| K-gram analysis | Unigrams capture nearly all signal (AUC = 0.823); bigrams add nothing |
| Shuffle test | Δ AUC from ordering = -0.017 to +0.016 (indistinguishable from zero) |
| Order-invariant baseline | Logistic regression on bag-of-codes: AUC = 0.832, matching all sequential models |
| Vocabulary overlap | Jaccard = 0.90 between compliant/non-compliant vocabularies |

**Interpretation:** All three diagnostic tools converge: CKD→ESRD progression is predictable from *which* diagnoses appear, not *when*. Even a Transformer's advantage (AUC 0.860) comes from learned code interactions, not temporal ordering — its shuffled version (AUC 0.844) still outperforms all baselines.

This demonstrates that the paper's contribution is not just the synthetic benchmark but a **diagnostic methodology** that transfers to real data. The synthetic tasks provide controlled ground truth for understanding model behavior; the MIMIC-IV analysis shows the tools work in practice.

Regarding sequence length and alphabet size sensitivity (reviewer's key question): we conducted a comprehensive sensitivity analysis across 29 parameter configurations ($n \in \{15, 20, 25\}$, $m \in \{3, 6, 10\}$, $\lambda \in \{3, 7, 10\}$) in both deterministic and stochastic settings. Key findings:

- In the **deterministic setting**, all encoder/recurrent models achieve AUC $\approx$ 1.0 across the full grid. Llama-1B shows the expected gap, with difficulty increasing for larger key subsets (AUC drops from 0.993 at $m=3$ to 0.673 at $m=10$) and longer sequences (AUC drops from 0.927 at $n=10$ to 0.739 at $n=30$).
- In the **stochastic setting**, the encoder-decoder efficiency gap is consistent across all configurations. BERT and LSTM approach AUC* in ~90% of configs; Llama-1B consistently underperforms (AUC 0.54-0.60).
- The **interaction between $m$ and $\lambda$** reveals compounding effects: small-scale Transformer drops to AUC 0.533 at $m=10$, $\lambda=1$ (vs. AUC* = 0.699).

These results confirm that our main findings are robust across the parameter space, not artifacts of a single configuration.

**Regarding the attention analysis** (reviewer's key question on internal representations): see our response to Point 2 above. We provide mechanistic evidence from BERT and Llama-1B attention patterns showing how successful models solve the compliance task (lag-detector heads) and why decoder models struggle (causal mask prevents lag-detection).

**Changes to the paper:**
- We add the MIMIC-IV case study demonstrating the diagnostic framework on real clinical data.
- We add the sensitivity analysis (29 configurations) to the appendix.
- We add the attention analysis (BERT vs. Llama-1B) showing internal representation differences.
- We discuss the synthetic-to-real bridge explicitly in the discussion section.

---

### Responses to R3's Key Questions (Summary)

| Question | Response |
|----------|----------|
| **Zero-shot/few-shot evaluation?** | Design choice: tasks are deliberately opaque to instruction following (hidden structure). Preliminary tests confirm chance-level performance. Fine-tuning is the appropriate paradigm. |
| **Alternative training for parity?** | Acknowledged as promising future work (curriculum learning, auxiliary objectives). Does not invalidate the finding that standard training fails in our regime. |
| **Sensitivity to $n$ and $\ell$?** | 29-configuration sensitivity grid confirms robustness. Llama-1B degrades with longer sequences and larger key sets. Encoder-decoder gap is consistent across all configs. |
| **Internal representations?** | Attention analysis shows BERT develops lag-detector heads (layers 3-5, ratio 3-4x); Llama-1B identifies key letters but fails to learn lag structure. Mechanistic evidence for local pattern exploitation. |
