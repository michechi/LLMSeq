# ICML26 Review Concerns

## R1

### Weaknesses / Concerns

1. **Definition of compliance**  
   Could the authors please clarify the difference in the more general defintion of compliance (Definition 3.1) and the seemingly more narrow definition of compliance (I'll call it narrow-compliance here) used in Theorem 3.3 for the stars-and-bars proof which appears to assume the full sequence is non-decreasing. The authors currently use the more narrow definition of compliance (full sequence, not subsequence), but claim to prove that any event-level ordering task cannot satisfy pure sequential dependence (as there is mutual information between the label and a particular event index). This claim is not justified with the current proof. I think the authors need to show that to show this for compliance defintion 3.1, one must perform a surjection from the narrow-compliance formulation (using a non-decreasing sequence) to the general compliance definition 3.1 case. Perhaps the authors should be a bit more clear that the proof outline being an intuitive example for the narrow-compliance definition, and the appendix needs to actually show the general case, currently the appendix does not and there is not clarification about switching tot his narrow-compliance defintion toy example.

2. **Uniform prior assumption and over‑claimed generality**  
   (2) The authors use "compliance task" synonymously with "event-level ordering tasks" in line 210 (left column) when claiming Theorem 3.3's implication is that "classification tasks based on event-level ordering cannot satisfy pure sequential depednence". The compliance task and the broader class of event-level ordering tasks, however, are two very different problem classes.

    The current definition of "compliance tasks" actually assumes a uniform prior (each event in the sequence is sampled iid from a uniform distribution over the alphabet). This is a unrealistic assumption for any medical or text dataset. Making general claims about all sequences based on a single proof on data with a uniform prior distribution is not rigorous or correct. The authors must strongly reduce the claims they make. Specifically the claims made in the intro (left column 060-065) and section 3.2 (left column 210- right column 170) must be explicitly qualified with the fact that the authors assume an unrealistic uniform prior. The authors present a nice toy example, but the implications are claimed to extend to "classification tasks based on event-level ordering", and that is simply an incorrect statement.

3. **Induction‑heads counterexample**  
   (3) Additionally, the broader set of tasks that depend on event order, which the authors call "event-level ordering tasks" should include the well known induction-heads task which transformers are well known for being able to solve. This task is the crux of what we call in-context learning [1]. It looks something like this: we have some sequence of letters and we essentially want to copy forward based on the context. So if we see something like this: ..., a, b, ... a, we use the context that b comes after a in the context of this sequence, so we predict b. In the event that the last token does not appear previously, the label Y is just uniformly sampled. In this case P(Y=a | X_t=a) is a bit larger than P(Y=a) since seeing it in the history implies it's in the future so the re is some leakage. I think the authors can get around this by making Y a combo of two tokens.

    So we can make an inductions head variant like this: ... a, b, c, ... a, and the label . I.e. we take the next two symbols after a, sum them up modulo them (I'm just using  to convert b and c to integers and $k^{-1} to map them back to letters) so we get a new letter and that new letter Y follows a uniform distribution, and still follows a uniform distirbution if one conditions on b, c, or a. This I think meets the pure sequence dependence definition, and Y is completely predictable from the sequence, and explicitly depends on event-level ordering as it is copying forward ordered events. I believe the contradicts a major contribution claim of the paper that "clasification tasks based on event-level ordering cannot satisfy pure sequential dependence".

4. **Parity on binary vocabulary**  
   - The claim that parity is not learnable is contradicted by a simple experiment with vocabulary size 2, where a transformer achieves perfect AUC.  
   - The authors need to defend why their experiments showed failure (e.g., due to vocabulary size or fixed parameters) and possibly revise their claim.

5. **No‑signal‑beyond‑k‑grams claim**  
   - The claim that there is no signal beyond k‑grams for the tested models is incorrect. Induction‑heads (and more generally in‑context learning) can capture dependencies longer than k‑grams, even out of distribution.

### Key Questions for Authors (Summary)

- How do you generalize Theorem 3.3 from non‑decreasing sequences to the general definition of compliant sequences?  
- Can you significantly modify your claims to be modest and realistic, acknowledging the uniform prior assumption and that the proof covers only a subset of tasks?  
- Would you review the literature (e.g., induction heads) that directly contradicts the claim that event‑level ordering tasks cannot satisfy pure sequential dependence?  
- Why was parity not learnable in your experiments while a simple transformer learns it on a binary vocabulary?  
- Can you retract the claim about no signal beyond k‑grams?

---

## R2

### Weaknesses / Concerns

1. **Ambiguity of purpose and stance**  
   - The paper’s stance on whether models should rely on semantics versus sequence ordering remains ambiguous.  
   - It is unclear to what extent the demonstrated drawback affects real‑world performance. One could argue that in many cases the event itself is more informative than its timing.

2. **Lack of quantitative gap analysis**  
   - It would be stronger to quantify the performance gap between more semantic‑based and more order‑based models.

3. **Comparison between encoder and decoder models**  
   - The comparison may be skewed because binary classification naturally favors discriminative (encoder) models over generative (decoder) models.

4. **Figure readability**  
   - Figures are difficult to interpret due to thin, overlapping bars.

### Key Questions for Authors

- Can you provide more quantitative intuitions (data, performances, or references) on the gap between semantic‑based and order‑based models?  
- Given that causal masking forces autoregressive models to encode information “autoregressively in parameters,” does this make them fundamentally unsuitable for global ordering understanding, and to what extent is this failure due to optimization rather than architecture?

---

## R3

### Weaknesses / Concerns

1. **Underutilization of LLMs**  
   - Evaluation does not leverage instruction following or in‑context learning; models are used only as encoder‑style classifiers via fine‑tuning, missing zero‑shot or few‑shot settings with natural language task descriptions.

2. **Weak training signal for parity**  
   - The parity task provides extremely sparse supervision for the required global computation, so the observed failure does not conclusively support a fundamental architectural limitation.

3. **Poor visual presentation**  
   - Too many models are plotted in single figures, reducing readability and making comparisons across architectures and scales difficult.

4. **Highly abstract synthetic setting**  
   - The gap between letter‑based tasks and real temporal reasoning may limit the practical implications of the conclusions.

### Key Questions for Authors

- Have you evaluated pretrained LLMs in zero‑shot or few‑shot settings with explicit natural language descriptions, instead of only fine‑tuning them as classifiers?  
- Did you explore alternative training objectives or curriculum strategies for the parity task that might better support learning global counting behavior?  
- How sensitive are the results to sequence length and alphabet size? Would longer sequences or different parameter regimes change the relative performance between encoders and decoders?  
- Can you provide additional analysis on internal representations (e.g., attention patterns, hidden state probing) to support claims about reliance on local versus global information?