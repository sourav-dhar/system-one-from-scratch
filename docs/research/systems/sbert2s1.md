# sbert2s1 ("From Retrieval to Typed Decisions")

> **Family:** bi-encoder (sentence-embedding conversions to typed decisions, compared against cross heads) · **Team:** not recorded in our notes (repository owner `pritamdeka`) · **Released:** arXiv [2610.02486](https://arxiv.org/html/2610.02486) (submission date not recorded in the notes we read) · **License:** not disclosed / not read · **Code/weights:** [pritamdeka/sbert2s1](https://github.com/pritamdeka/sbert2s1) [@sbert2s1-repo] · **Sources read:** the arXiv HTML paper (RQ1, RQ2; Tables 3, 4, 6, 10 and 12) and the code file [`research/s1/model.py`](https://github.com/pritamdeka/sbert2s1/blob/main/research/s1/model.py). **Gaps:** the backbone encoders' identities and the training-set sizes are not recorded in our notes.

## Summary

sbert2s1 asks how far a sentence-embedding bi-encoder can be pushed toward typed System One decisions, and what is lost on the way. It defines four conversions:

- **Z:** a frozen encoder;
- **B:** the same bi-encoder form, trained end to end;
- **C:** a cross head in the style of Laya;
- **PFR:** C plus a cosine prior.

All four are calibrated with per-(type, K-bucket) temperatures. Its two findings matter most for a CLM-style design:

- The trained bi-encoder (B) **matches the cross head on seen tasks but loses 9.4 points held-out** (23.4 vs 32.8).
- **Label-word options (yes/no, true/false) carry almost no cosine signal**: the gaps are about $10^{-3}$, and no encoder is reliably above chance.

Other findings:

- B is perfectly order-invariant (0.0% flips) where the cross head flips 15.7%.
- Fine-tuning destroys B's retrieval ability unless the prior is frozen.
- Laya's RLCD estimator is 3.6–15× too large in scale.

## Architecture

**Conversions** ([model.py](https://github.com/pritamdeka/sbert2s1/blob/main/research/s1/model.py)):

| Variant | Form | Notes |
|---|---|---|
| Z (zero-shot) | $z_k=\alpha_\tau\cos\!\big(e(s),\,e(x\oplus o_k)\big)$, frozen encoder $e$ | **The question $x$ goes into the hypothesis, together with option $o_k$; the state $s$ is embedded alone** |
| B (bi-encoder) | the same form, trained end to end | state embeddings can be precomputed |
| C (cross) | Laya-style cross head with one `[MASK]` per option | joint encoding |
| PFR | C $+\ \alpha\cdot\cos$ prior, with a zero-initialised scorer | starts as the cosine prior |

The scale $\alpha$ starts at 20 and has its own learning rate and no weight decay. Probabilities are a softmax over $z_k$ per question.

**Passes per request** *(inference)*:

- **Z and B:** one encoder pass for the state, which can be cached across questions, plus one pass per (question ⊕ option) hypothesis. Hypotheses are request-independent given the question, so they can be cached across states.
- **C and PFR:** one joint pass per question.

**Contrast with CLM.** CLM appends the question to the *state* and leaves options bare (`f"{state}\n\n{instructions}"`; [schema.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py)). sbert2s1 appends the question to the *option*. This keeps the state embedding question-independent and gives short options context *(inference)*.

| Property | Value |
|---|---|
| Families compared | Z (frozen bi-encoder), B (trained bi-encoder), C (Laya-style cross head), PFR (C + cosine prior) |
| Bi-encoder layout | state alone; question ⊕ option as the hypothesis |
| Score | $\alpha_\tau\cos$, $\alpha$ initialised at 20 (own learning rate, no weight decay) |
| Cached (Z/B) | state embedding; question ⊕ option embeddings *(inference)* |
| Calibration | per-(type, K-bucket) temperatures, K ∈ {2, 3–5, 6–10, ≥11}, fit by L-BFGS on $\log T$ |
| Order invariance | B 0.0% flips by construction; C 15.7% |

## Training

- **B:** the bi-encoder form, trained end to end on typed decisions. Fine-tuning damages its retrieval ability unless the prior is frozen (see Results).
- **Objectives compared:** Laya's RLCD estimator and a leave-one-out variant, PG-LOO, which corrects the estimator's scale (Tables 4 and 6). We assume these train the cross-head variants (C and PFR) *(inference)*.
- **PFR:** its scorer is zero-initialised, so training starts from the cosine prior.
- The training-set composition and sizes are not recorded in our notes.

## Calibration

- Temperatures are fit per (type, K-bucket), with K buckets {2, 3–5, 6–10, ≥11}, by L-BFGS on $\log T$.
- **After temperature scaling, every training objective sits at 3.6–4.4 ECE** (Table 4). The choice of objective matters little for calibration once a temperature is fit.

## Results

**Bi-encoder vs cross head (RQ2; Table 10).** On seen tasks, B is within −0.6 points of C (not significant). Held out, B loses 9.4 points: 23.4 vs 32.8.

**Label-word options (RQ1; Table 3).** For yes/no and true/false options, cosine gaps are about $10^{-3}$, and **no encoder is reliably above chance**.

**Permutation flips (Table 12):**

| Model | Flips |
|---|---|
| B | 0.0% |
| PFR | 0.3% |
| C | 15.7% |
| Laya, fine-tuned | 9.8% |

**Retrieval after fine-tuning.** B's SciFact nDCG@10 falls from 67.6 to 34.3, unless the prior is frozen.

**Estimator scale (Tables 4 and 6).**

- Laya's RLCD estimator is 3.6–15× too large in scale.
- The leave-one-out version (PG-LOO) fixes this and gains +2.5.
- After temperature scaling, every objective reaches 3.6–4.4 ECE.

**Routing.** Sending the least-confident 20% of decisions to Gemma-4-31B raises macro accuracy from 59.5 to 68.4.

No independent evaluation of sbert2s1 appears in the sources we read.

## Serving and efficiency

Our notes record no latency figures. The Z and B variants inherit bi-encoder economics *(inference)*:

- one state embedding reused across questions;
- cached hypothesis embeddings;
- permutation invariance.

The 20% escalation result (59.5 → 68.4) shows that the study's calibrated models can serve as a cheap first stage in a cascade. Our notes do not record which variant was routed.

## Flaws and failure modes

### (a) Flaws the study documents in its own bi-encoder variants

- **Held-out generalisation.** B loses 9.4 points to C held-out (23.4 vs 32.8), although it ties on seen tasks. This is the clearest evidence in our sources that pure bi-encoders lose held-out generalisation.
- **Label-word blind spot.** Yes/no and true/false options give cosine gaps of about $10^{-3}$, with no encoder reliably above chance.
- **Catastrophic forgetting of retrieval.** SciFact nDCG@10 falls from 67.6 to 34.3 when the prior is not frozen.
- **Low absolute held-out accuracy** for both B and C (23.4 and 32.8), as recorded.

### (b) Flaws of other systems it identifies or fixes

- **Laya's RLCD estimator** is 3.6–15× too large in scale. PG-LOO fixes it (+2.5).
- **Order sensitivity of cross heads**: C 15.7% and fine-tuned Laya 9.8%, against B's 0.0%.
- **Uncalibrated objectives.** Per-(type, K) temperatures bring every objective to 3.6–4.4 ECE.

## Implications for our design

*(All items are inference unless cited.)*

**For CLM:**

- CLM's Noul texts ("Yes. This is true: …" / "No. This is false: …") and short Score levels are exactly the label-word case where cosine gaps are about $10^{-3}$. This matches CLM issue #3 (state sensitivity 0.000) [@clm-repo]. Noul should not be a cosine between two near-identical option texts. Narrow Decision Models' Noul head on $[h_Q;h_S]$ is the alternative [@ndm].
- **Put the question on the option side** ($e(x\oplus o_k)$) or embed it separately. Do not append it to the state.
- **Expect a held-out gap for any pure bi-encoder.** Budget for a joint or late-interaction component (Jeva-style [@jeva]) and measure held-out tasks explicitly.
- **Freeze or regularise toward the retrieval prior** if retrieval quality must survive decision fine-tuning.
- **Adopt per-(type, K-bucket) temperatures fit by L-BFGS on $\log T$**, and the confidence-based escalation cascade.

## References

- [@sbert2s1] "From Retrieval to Typed Decisions", arXiv 2610.02486: https://arxiv.org/html/2610.02486
- [@sbert2s1-repo] https://github.com/pritamdeka/sbert2s1 — [`research/s1/model.py`](https://github.com/pritamdeka/sbert2s1/blob/main/research/s1/model.py)
- [@laya-card] Laya model card: https://huggingface.co/convaiinnovations/laya
- [@clm-repo] CLM, incl. issue #3 and `schema.py`: https://github.com/Contrastive-LM/CLM
- [@ndm] Narrow Decision Models, arXiv 2610.25553: https://arxiv.org/html/2610.25553
- [@jeva] Jeva, arXiv 2610.18773: https://arxiv.org/html/2610.18773
