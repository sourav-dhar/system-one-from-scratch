# Narrow Decision Models (NDM)

> **Family:** encoder (prefix-tree-masked bidirectional encoder with separate state, question and option readouts and a set head) · **Team:** Softplan · **Released:** arXiv [2610.25553](https://arxiv.org/html/2610.25553) (submission date not recorded in the notes we read) · **License:** not disclosed / not read · **Code/weights:** none recorded in our notes · **Sources read:** the arXiv HTML paper, including §4, §10 (Table 2 and §10.4) and App. B. **Gaps:** no code was available to check against the text; no comparison with the hosted System One models (Jev and others) is recorded in our notes.

## Summary

NDM encodes one state together with all of its questions and options in a single pass of a small multilingual encoder, mmBERT-base. A **prefix-tree attention mask** makes each block see only itself and its ancestors. As a result:

- the state is encoded once, and its K/V can be cached exactly;
- no question or option sees a sibling, so answers are invariant to option order and to which other questions are present.

Decisions are read from three kinds of vector:

- a **question vector** that fuses the separately pooled state and question;
- per-option vectors scored by a small, position-free **set head**, which adds option interaction;
- a per-option **adequacy logit**, computed before the set head, which gives a "none fits" signal.

Noul uses no option embeddings at all.

Results:

- Invariance holds to within $1.5\times10^{-5}$ in fp32.
- The set head improves NLL from 1.181 to 0.799 over independent options.
- Adequacy recovers 0.66 of absent answers, against 0.25 for an explicit "none" option.
- On real legal documents, NDM agrees 91.2% with a production LLM, with ECE 0.011.

## Architecture

**Backbone.** mmBERT-base, 22 layers. The 197M-parameter embedding table is frozen, the 110M-parameter body is trained, and 6.38M new parameters are added.

**Prefix-tree mask.**

- The input is a tree: state → questions → options.
- A token attends only to tokens in its own block and in its ancestor blocks.
- Positions are logical, so sibling options share position ids.
- **The state is encoded once, and no option or question sees another.**

Formally, token $j$ may attend to token $k$ iff $\mathrm{block}(k)\in\{\mathrm{block}(j)\}\cup\mathrm{anc}(\mathrm{block}(j))$ *(our rendering of the description)*.

**Readouts.** With $h_S$ the pooled state, $h_{Q_i}$ the pooled question $i$ and $h_{o}$ the pooled options:

$$v_{Q_i}=\mathrm{proj}\big[h_S;\,h_{Q_i}\big]+e_{\mathrm{type}},\qquad p_{\mathrm{noul}}=\sigma\big(\mathrm{logit}([h_Q;\,h_S])\big).$$

- **Question vector:** the state and question are pooled *separately* and then fused, plus a type embedding.
- **Set head:** 2 pre-norm transformer layers, width 384, 6 heads, no positional encoding, zero-initialised. It runs over the option vectors, so options can interact while the head stays permutation-equivariant.
- **Adequacy logit:** one per option, computed *before* the set head. It serves as the abstention ("none fits") signal.
- **Noul:** computed from $[h_Q;h_S]$ with **no option embeddings**.

**What is cached.**

- The **state K/V** is cached exactly.
- State encoding costs 6.5 ms at 256 tokens and 59 ms at 4,096 tokens. Each additional question costs 0.8–4.9 ms.
- Because options attend to their ancestors (state and question), the option vectors depend on the request. They are computed once per request and shared by the heads, but they cannot be reused across requests the way CLM's or Jeva's option embeddings can *(inference from the mask description)*. Our other notes describe NDM's set head as operating on "cached" or "independent" option vectors; we read this as per-request caching *(inference)*.

**Passes per request** *(inference)*: one encoder pass over the state, which can be served from the K/V cache, plus incremental passes for each question subtree (0.8–4.9 ms each), then the small heads.

| Property | Value |
|---|---|
| Backbone | mmBERT-base, 22 layers; frozen 197M embedding table; 110M body trained; 6.38M new parameters |
| Mask | prefix tree (own block + ancestors); logical positions shared by siblings |
| State reuse | state encoded once; exact K/V cache (6.5 ms @256 tok, 59 ms @4,096 tok) |
| Question vector | $\mathrm{proj}[h_S;h_{Q_i}]$ + type embedding |
| Option interaction | set head: 2 pre-norm layers, width 384, 6 heads, no positions, zero-init |
| Null | per-option adequacy logit (before the set head) |
| Noul | $\sigma(\mathrm{logit}([h_Q;h_S]))$, no option embeddings |
| Invariance | option order and sibling questions, within $1.5\times10^{-5}$ (fp32) |
| Calibration | per-primitive temperature in [0.25, 10]; domain temperature shrunk toward the synthetic one |

## Training

**Losses (proper scoring rules per type).**

- **Choice:** $-\log\sum_{i\in\mathcal A}p_i+\mathrm{Brier}$, where $\mathcal A$ is the set of accepted answers. Probability mass on *any* accepted answer is rewarded.
- **Score:** log loss + ranked probability score (RPS), an ordinal proper score.
- **Noul:** log loss + Brier.

**Optimisation (App. B).** Learning rate 2e-5 for the body and 2e-4 for the new parts, with layer-wise learning-rate decay of 0.95.

The data sources are not detailed in our notes, apart from a synthetic set used as the temperature prior and the legal-document deployment.

## Calibration

- **Temperatures:** a bounded per-primitive temperature $T\in[0.25,10]$, plus a domain temperature shrunk toward the synthetic-data temperature.
- **Effect:** Choice ECE falls from 0.188 to 0.033.
- **Temperatures do not transfer across domains:** the recorded evidence is MASSIVE NLL 0.123 → 0.232. Our notes do not record the exact transfer protocol; it is presumably a temperature fit elsewhere applied to MASSIVE *(inference)*.
- **Real legal documents:** ECE 0.011.

## Results

**Invariance (Table 2).**

- All invariance tests hold within $1.5\times10^{-5}$ in fp32. Negative controls differ by 2–14.
- Option shuffle: zero argmax changes on 905 questions, against 41% for Qwen3-0.6B likelihood scoring.

**Heads on the option vectors (§10.4):**

| Scoring | NLL | Accuracy |
|---|---|---|
| Independent options | 1.181 | 0.655 |
| + sibling summary statistics | 1.266 | 0.655 |
| **Set head** | **0.799** | **0.705** |

Hand-crafted sibling statistics make NLL worse. A learned set head improves both NLL (−0.382) and accuracy (+5.0 points) *(inference: arithmetic)*.

**Abstention.**

- Recall of absent answers: 0.25 with an explicit "none" option vs **0.66 with the adequacy logit**.
- AUROC: our notes record "0.79 vs 0.76" in the same order, i.e. explicit option 0.79 and adequacy 0.76. If that order is correct, adequacy wins on recall at the operating point but not on AUROC. The order was not re-verified.

**Block vs pair encoding.** Block encoding gives NLL 0.819, against 1.024 for pair encoding: −0.204, CI −0.312 to −0.107. We read "block" as all options in one prefix-tree pass and "pair" as separately encoded (state, option) pairs *(inference)*. The listed values differ by 0.205 (rounding).

**Real legal documents.** 91.2% agreement with the production LLM; macro-F1 0.80 vs 0.61 for TF-IDF + logistic regression; ECE 0.011.

No independent evaluation of NDM appears in the sources we read.

## Serving and efficiency

- The state is the expensive part: 6.5 ms at 256 tokens and 59 ms at 4,096 tokens. With an exact state K/V cache, each extra question costs 0.8–4.9 ms.
- The heads are small: 6.38M new parameters in total.
- The prefix-tree mask makes many-question requests cheap without sacrificing isolation between questions. This contrasts with Bongard's question-aware encoding, where adding questions changes answers [@bongard] *(inference)*.

## Flaws and failure modes

### (a) NDM's own flaws, with evidence

- **Temperatures do not transfer across domains** (MASSIVE NLL 0.123 → 0.232). Per-domain calibration data is required.
- **Adequacy vs explicit "none" is mixed.** Adequacy has better recall (0.66 vs 0.25) but, as recorded, a lower AUROC (0.76 vs 0.79).
- **Options are request-dependent**, so option representations cannot be cached across requests the way bi-encoder option embeddings can *(inference)*.
- **Small backbone, narrow scope.** The name and the legal-document deployment suggest NDM is meant as a domain model *(inference)*. Our notes record no comparison against Jev or other general System One models.
- **Code not inspected**; all mechanisms are from the paper text.

### (b) Flaws of other systems it claims to fix

- **Option-order sensitivity of likelihood scoring.** 0 argmax changes on 905 questions vs 41% for Qwen3-0.6B.
- **Independent option scoring (IIA).** The set head gives NLL 1.181 → 0.799.
- **The flat explicit "none" option.** Adequacy recall is 0.66 vs 0.25.
- **Pair encoding.** Block encoding gives NLL 0.819 vs 1.024 and encodes the state only once *(the second clause is inference)*.
- **Generic baselines on the target domain.** Macro-F1 is 0.80 vs 0.61 for TF-IDF + logistic regression.

## Implications for our design

*(All items are inference unless cited.)*

**Adopt**

- **Separate the state vector from the question vector, then fuse them** with $\mathrm{proj}[h_S;h_Q]$. This is the direct fix for CLM's question-suffix collapse, where the state cosine is 0.958 with the question appended vs 0.452 without (CLM #15) [@clm-repo]. It also lets one state embedding serve every question and is cacheable.
- **A Noul head on $[h_Q;h_S]$ with no option embeddings.** This avoids the yes/no label-word blind spot of bi-encoders (cosine gaps of about $10^{-3}$) [@sbert2s1] and CLM's Noul template.
- **A set head over option vectors.** Applied to CLM-style *request-independent* cached option embeddings, it would add option interaction without giving up cross-request caching. NDM's own option vectors are request-dependent, so this transfer is untested.
- **An adequacy logit for "none fits"**, evaluated against Jeva's coverage gate [@jeva] on both recall and AUROC.
- **Proper-score losses per type:** accepted-set log loss + Brier for Choice; log loss + RPS for Score.
- **Bounded per-primitive temperatures**, fit per domain.

**Avoid** fitting temperatures on one domain and shipping them to another.

## References

- [@ndm] Narrow Decision Models, arXiv 2610.25553: https://arxiv.org/html/2610.25553
- [@clm-repo] CLM issue #15: https://github.com/Contrastive-LM/CLM/issues/15
- [@sbert2s1] sbert2s1, arXiv 2610.02486: https://arxiv.org/html/2610.02486
- [@jeva] Jeva, arXiv 2610.18773: https://arxiv.org/html/2610.18773
- [@bongard] Bongard, arXiv 2609.39111: https://arxiv.org/html/2609.39111
