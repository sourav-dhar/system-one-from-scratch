# this-that-model-1.0

> **Family:** decoder readout (hybrid delta-rule / full-attention LM; answer-slot hidden states scored against the unembedding rows of the declared labels) · **Team:** FLock.io · **Released:** arXiv [2609.23886](https://arxiv.org/html/2609.23886) (submission date not recorded in the notes we read) · **License:** MIT · **Code/weights:** [FLock-io/this-that-model](https://github.com/FLock-io/this-that-model); weight location not recorded in the notes we read · **Sources read:** the arXiv HTML paper (Eqs. 2, 5 and 8–10; Tables 1 and 3), plus this model's rows in the Rafe & Das stress test [@rafedas]. **Gaps:** the code was not read; the training data, the label tokenisation for multi-token labels and the size of the training set are not recorded in our notes.

## Summary

this-that-model-1.0 is a roughly 2B hybrid language model (delta-rule plus full-attention layers, built on decider-2b). It answers typed questions by reading the hidden state at each question's answer slot and scoring it **only against the unembedding rows of that question's declared labels**.

- **Serving:** N questions share one forward pass and are conditionally independent. A schema-first layout lets one cached prefix serve many states.
- **Training:** log loss plus Brier, and an unbiased U-statistic Brier reward estimated from 32 draws.
- **Safety mechanism:** a weight-interpolation "rollback" between base and fine-tuned weights.

Results:

- On a 68-question comparison it reports 0.941 accuracy and Brier 0.042, against Jev's 0.765 / 0.133, at 30.9 ms. The paper itself notes that the gap rests on about 12 questions.
- In an independent stress test it is among the most robust models to yes/no identifier swaps: 4.2–6.5 flips per 100, against Jev's 50.5.

## Architecture

**Backbone.** A 1.88–2B hybrid of delta-rule (linear-recurrent) and full-attention layers. The base model is decider-2b.

**Head (Eq. 2).** For question $j$ with declared label set $\mathcal L_j$ and answer-slot hidden state $h_j$:

$$p(\ell\mid j)=\frac{\exp(w_\ell^\top h_j)}{\sum_{\ell'\in\mathcal L_j}\exp(w_{\ell'}^\top h_j)},\qquad \ell\in\mathcal L_j,$$

where $w_\ell$ is the unembedding row of label $\ell$. This is our rendering of "scored against the unembedding rows of the declared labels only". How multi-token labels are handled is not recorded.

**Many questions per pass.** N questions share one pass and are conditionally independent: each question has its own answer slot.

**Schema-first layout (Eq. 5).** The question schema comes first, as a prefix that is reused across $m$ states. Its computation is therefore amortised over many states, the reverse of the usual "state first, questions branch" layout *(inference from the description)*.

**Rollback.** Weights can be interpolated as $\theta(\lambda)=\theta_0+\lambda\Delta$, between the base $\theta_0$ and the fine-tuning delta $\Delta$.

**Passes per request** *(inference)*: one forward pass per state over the cached schema prefix plus the state, which answers all N questions. Labels are unembedding rows, so there is no option-encoding cost, but options are limited to the declared label vocabulary.

| Property | Value |
|---|---|
| Backbone | 1.88–2B hybrid delta-rule + full attention (base: decider-2b) |
| Head | answer-slot hidden state × unembedding rows of declared labels only (Eq. 2) |
| Questions per pass | N, conditionally independent |
| Cached | schema-first prefix reused across $m$ states (Eq. 5) |
| Loss | log loss + Brier; unbiased U-statistic Brier reward from 32 draws (Eqs. 8–10) |
| Rollback | $\theta(\lambda)=\theta_0+\lambda\Delta$ |
| Latency | 30.9 ms (Table 1 setting) |

## Training

- **Supervised loss:** log loss + Brier.
- **Reward:** an unbiased U-statistic estimate of the Brier score from 32 sampled draws (Eqs. 8–10). We read "unbiased" as removing the bias of scoring a probability estimated from the same samples *(inference)*.
- **Rollback:** $\theta(\lambda)=\theta_0+\lambda\Delta$. We read it as allowing a partial retreat toward the base model when fine-tuning regresses *(inference)*.
- Data sources and sizes are not recorded in our notes.

## Calibration

- Calibration is trained in through the Brier terms; no post-hoc temperature is recorded in our notes.
- Table 1 Brier: 0.042, against Jev's 0.133.

## Results

**Authors' results:**

| Evaluation | this-that | Jev | Notes |
|---|---|---|---|
| Table 1 (68 questions) | accuracy 0.941, Brier 0.042, 30.9 ms | 0.765 / 0.133 | the paper notes the gap rests on about 12 questions |
| Table 3 (2,250 questions) | 0.844 | 0.803 | in-distribution for this model |
| Multi-step arithmetic | 0.560 | n.r. | weakness |
| Branch-shaped families | 0.818 | n.r. | weakness |

**Independent results (Rafe & Das [@rafedas]):**

- *Yes/no identifier swap:* this-that is in the most robust group, with Kev-9B, Nimble and decider, at 4.2–6.5 flips per 100. Jev flips 50.5 and Laya 91.3.
- *Descriptions only:* it gains 6.8 points on CLINC when given descriptions only, as recorded in our notes.
- *Cascade:* this-that → Jev on CLINC reaches 0.927 at 0.43 of Jev's cost (Table 10). Weak first stages, such as Laya and Kev-0.8B, escalate 99–100% of decisions.

## Serving and efficiency

- 30.9 ms per request in the Table 1 setting.
- All questions on a state share one pass.
- The schema prefix is computed once and reused across states.
- In the Rafe & Das cascade, this-that works as a first stage that cuts cost to 0.43 of Jev's while keeping CLINC at 0.927.

## Flaws and failure modes

### (a) this-that's own flaws, with evidence

- **Small headline evaluation.** Table 1 has 68 questions, and the paper itself says the gap over Jev rests on about 12 of them.
- **In-distribution comparison.** The larger Table 3 (2,250 questions) is in-distribution for this model, which favours it against Jev.
- **Weak multi-step reasoning:** 0.560 on multi-step arithmetic and 0.818 on branch-shaped families.
- **Label-vocabulary head.** Options are scored through unembedding rows of declared labels, so rich option descriptions do not enter the score through the head. Whether this explains the descriptions-only result on CLINC is not established *(inference)*.

### (b) Flaws of other systems it claims to fix

- **Overconfident or miscalibrated decision models.** Brier 0.042 vs Jev's 0.133 on Table 1.
- **Sensitivity to yes/no identifiers.** 4.2–6.5 flips per 100 vs Jev's 50.5 (independent).
- **Cost of re-reading the schema for every state.** Schema-first prefix reuse (Eq. 5).
- **Irreversible fine-tuning regressions.** The $\lambda$-rollback.

## Implications for our design

*(All items are inference unless cited.)*

**Adopt**

- **The $\lambda$-rollback** ($\theta_0+\lambda\Delta$) for any adapter or LoRA we add. For a CLM-style head it is cheap to keep, but every rollback step changes the option cache and so requires re-embedding.
- **Schema-first prefix caching.** For a bi-encoder, the analogue is placing shared, stable context first, so that prefix caching hits across steps.
- **An unbiased U-statistic Brier reward** if we use sampled-reward training.
- **A strong small first stage in a cascade** (0.927 at 0.43 of Jev's cost), gated on calibrated confidence.

**Evaluate** our model on the Rafe & Das yes/no swap. CLM's Noul template, which uses near-identical yes/no option texts, is untested there.

## References

- [@thisthat] this-that-model-1.0, arXiv 2609.23886: https://arxiv.org/html/2609.23886
- Code: https://github.com/FLock-io/this-that-model
- [@rafedas] Rafe & Das, "Benchmarking System One vs trained classifiers", arXiv 2610.00346: https://arxiv.org/html/2610.00346
