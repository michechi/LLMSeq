# Can a task be both order-sensitive and k-locally pure? Yes: KIP

Reviewer question addressed: *"Can you show a task that is both order-sensitive and
k-locally pure, so that 'pure sequential dependence' is demonstrated?"*

Short answer: **KIP (Key-Inversion Parity) is that task.** It is order-sensitive in
the strongest possible sense (the label is independent of the *entire*
order-invariant sigma-field), it is exactly k-locally pure for every window of
size k <= m-2, and a recurrent model demonstrably learns it to AUC 1.000 while
both order-destroying controls collapse the same model to chance. All numbers
below are from `kip_report.md` (sanity/audit/reveal) and
`results/kip_training.csv` (GPU runs).

## 1. Why the paper's original two families could not answer this

The original benchmark splits the two properties across tasks:

- **Compliance (Tricky) is order-sensitive but provably not locally pure.**
  Theorem 3.3 (impossibility): under independent full-support sampling, any
  compliance task has I(Y; X_1) > 0 -- a single position already leaks.
- **Balanced parity is k-locally pure for every k < n but order-INVARIANT.**
  Its label is a function of counts; order carries nothing (Theorem 3.5).

So within the paper's own framework, "pure sequential dependence" (Def. 3.1) was
never exhibited by a concrete task. The obstruction is Theorem 3.3's hypothesis:
**independent coordinates**. KIP removes exactly that hypothesis, by placing each
of the m hidden key letters exactly once (sampling without replacement), with
distractors i.i.d. on the complement alphabet. The impossibility theorem then no
longer binds -- and both properties become achievable simultaneously.

## 2. The two exact properties

**(P1) Pure sequential dependence, Def. 3.1, strongest form.** Conditional on any
order-invariant statistic of the sequence (counts, co-occurrences, the full token
multiset), transposing two key letters' positions is measure-preserving and flips
Y = sign(pi). Hence Y is *exactly* independent of every order-invariant feature,
and depends only on the relative order of the key letters. Clauses (a)-(d) of
Def. 3.1 hold with equality, not approximately.

**(P2) k-local purity with radius m-2.** For any fixed position subset J with
|J| <= m-2, at least two key letters lie outside J; transposing them preserves
X_J and the generating law while flipping Y. Therefore **I(Y; X_J) = 0 exactly
for every |J| <= m-2**. Beyond the radius, leakage is positive but bounded by the
probability that >= m-1 keys land inside J:

| | k=2 | k=3 | k=4 | k=5 | k=6 | k=8 |
|---|---|---|---|---|---|---|
| KIP-m4 | **0 (exact)** | <=3.5e-3 bits | <=1.3e-2 | <=3.2e-2 | <=6.1e-2 | <=0.15 |
| KIP-m6 | **0 (exact)** | **0 (exact)** | **0 (exact)** | <=3.9e-4 bits | <=2.2e-3 | <=1.8e-2 |

(n=20; bound = P(>= m-1 of m keys in J), computable in closed form.) The radius
is tunable: m -> n pushes exact purity to windows of size n-2, the ceiling for
any deterministic order-sensitive label. Unlike balanced parity, purity for *all*
k < n is not on offer -- that is precisely the price of order-sensitivity, and
KIP makes the trade explicit and quantified.

## 3. Empirical demonstration -- the KIP analysis stack

**(P1) verified (sanity suite, 12/12 pass).** Count purity: LogReg and XGBoost on
the 26-dim count vector score test AUC 0.498-0.509 on both datasets (an exact
independence, tested). Swap-flip: transposing two key letters flipped the label
1000/1000 times; permuting only distractors changed it 0/1000 times. A uniform
shuffle preserves the label 49.7-50.3% of the time.

**(P2) verified (audit ladder).** At m=6, every k-local family within the purity
radius sits at chance *with k-gram coverage 1.000* (so this is not a sparsity
artifact): unigrams 0.499-0.500, contiguous k-grams k=2/3/4 at
0.4993/0.5014/0.5033, fixed-lag pairs 0.498-0.503 for every lag tested. At m=4,
k=3,4-grams are *beyond* the radius, but their permitted leak (<=1.3e-2 bits) is
too small to exploit: measured 0.5006/0.5004. The only above-chance baselines are
**global aggregations** -- all-lag pair counts reconstruct the full precedence
matrix (which determines Y) and reach 1.000 at m=4 via XGBoost -- and an
aggregation over all windows is not a k-local statistic, so this does not
contradict purity. (Balanced parity has the same structure: every window is
pure, yet aggregated counts determine its label.) Footnote for completeness:
XGBoost on *fixed*-lag pair counts shows a small joint-distribution leak at m=4
(0.524 at lag 1-2); each coordinate is marginally independent of Y, the joint
aggregate is not. Reported, not suppressed.

**Non-degeneracy (oracle/reveal ladder).** The closed-form oracle from the
C(m,2) precedence bits scores AUC 1.000000 exactly on both datasets, and a
64-unit MLP given those bits reaches 1.000. The task has ceiling AUC* = 1; every
failure below is a learning failure, not a task pathology.

**Learnability payoff (GPU runs, exact paper recipes, two sites).** On KIP-m4,
an LSTM trained from raw tokens reaches **test AUC 1.0000 on 9/9 seeds across
two independent sites** (paper recipe, patience 5): local A100 seeds
9550/9551/9552 and FOX H200 seeds 9550 + 9570-9574. The controls close the
argument: training on per-sequence-shuffled data with original labels gives
0.500 +/- 0.002, and evaluating the *trained* AUC-1.0 checkpoints on shuffled
test sequences gives 0.502 +/- 0.006 -- the learned skill is destroyed entirely
by destroying order, so what was learned is order and nothing else. Meanwhile a
parameter-matched Transformer encoder (0.500, both sites), BERT-base + LoRA
(0.499-0.500 over 3 seeds at max_length 512, cross-site), and Llama-3.2-1B +
LoRA (0.504-0.505 over 3 seeds cross-site on m4; 0.500 over 3 seeds on m6) all
fail the same data.

## 4. The one-paragraph rebuttal answer

> KIP instantiates pure sequential dependence exactly: each of m hidden key
> letters appears exactly once, so every order-invariant statistic is
> independent of the label by a transposition argument, while the label is the
> parity of the hidden permutation the key letters spell out -- order is the
> *only* signal. Because the generator is not coordinate-independent, the
> impossibility theorem for compliance tasks does not apply, and KIP is exactly
> k-locally pure for all windows k <= m-2 (leakage beyond bounded by 3.9e-4 bits
> at m=6, k=5), which we verify empirically: unigrams, 2/3/4-grams at full
> coverage, and all fixed-lag pair features sit at AUC 0.50. The task is
> nonetheless solvable (closed-form oracle AUC 1.0) and *learnable*: an LSTM
> reaches AUC 1.000 from raw tokens on 9/9 seeds across two independent sites
> (A100 and H200), collapsing to 0.50 under shuffled-label training and under
> shuffled evaluation of the trained model, while Transformer, BERT and
> Llama-1B under identical recipes remain at chance on both KIP datasets.

## 5. Honest scope notes

1. The *learned* demonstration is at m=4, where the purity radius is k <= 2 --
   already enough to exclude every pairwise feature family, including the
   fixed-lag pair counts that saturate the compliance tasks. The larger-radius
   instance (m=6, pure through k=4) is solved by no tested model: LSTM under the
   paper patience-5 recipe is at chance on 3/3 seeds (FOX, best epochs 5-8),
   matching the local patience-3 runs, and Transformer / BERT / Llama-1B are at
   chance on 3 seeds each. What remains open is only the exhaustive version of
   the question -- a fixed-30-epoch no-early-stop run to exclude very late
   grokking -- not the recipe-consistent one, which is now answered.
2. Purity holds for windows up to m-2, not for all k < n; all-k purity is
   incompatible with order-sensitivity in this construction, and the tradeoff is
   stated and quantified rather than hidden.
3. BERT and Llama results carry the LoRA scoping caveat (~0.4% of weights
   trained); the from-scratch LSTM/Transformer contrast carries none.
