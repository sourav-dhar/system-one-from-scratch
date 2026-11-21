# Bongard

> **Family:** encoder–decoder (state encoded once by a bidirectional encoder; one decoder branch per question; bilinear judgment head) · **Team:** AgentBull Pte Ltd; authors listed as Ding, Jin, Ji · **Released:** arXiv [2609.39111](https://arxiv.org/html/2609.39111) (submission date not recorded in the notes we read) · **License:** weights under Gemma terms; code license not recorded · **Code/weights:** [AgentBull/bongard](https://github.com/AgentBull/bongard) [@bongard-repo]; weights [AgentBull/bongard-mini](https://huggingface.co/AgentBull/bongard-mini) · **Sources read:** arXiv HTML paper; `JudgmentHead` in the repo's `model.py`, found by grep. **Gaps:** the per-stage corpus sizes and the number of sandbox rounds are not stated anywhere we read, and no training data is released.

## Summary

Bongard reads the state once with the bidirectional encoder of an unchanged T5Gemma-2 4B-4B. Each question then gets its own decoder branch over the shared encoder output. A small bilinear head scores each candidate's end-marker state against the decision marker.

Training has three stages:

1. supervised judgments with proper-scoring losses;
2. a "JEPA" representation-alignment stage that introduces **pack centring**, because raw readout cosines were saturated at 0.93–0.996;
3. sandbox action-outcome training against outcome distributions from rollouts or exact oracles.

Results:

- DecisionBench 1.0: 78.05% (4th of 61) with ECE 0.063, against Jev's 0.128.
- Same-item comparison with Jev on 24 benchmarks: Bongard wins 6, ties 1 and loses 17.
- Latency: 36 ms per short request, and 221 ms for 32 questions in one request.

## Architecture

**Backbone.** T5Gemma-2 4B-4B, unchanged: 34 encoder + 34 decoder layers, $d=2560$, a bidirectional encoder and a frozen SigLIP vision tower.

**Passes per request.**

- The state, rendered as compact JSON, is encoded **once**.
- From stage 3 on, the distinct question instructions ("II") are encoded together with the state.
- Each question gets its own decoder branch. Branches run in groups of up to 8, so a request needs about $\lceil N/8\rceil$ decoder calls for $N$ questions.
- Inside the decoder, attention over the question-prefix partition and the shared-state partition is merged by log-sum-exp, in the style of Hydragen [@hydragen].

**Judgment head** (`JudgmentHead` in [model.py](https://github.com/AgentBull/bongard)). With $h_i$ the decoder hidden state at candidate $i$'s end marker and $g$ the state at the decision marker,

$$z_i=w^\top h_i+\frac{(W_c h_i)^\top (W_g g)}{\sqrt{256}} .$$

**Typed outputs.**

- Noul $=\sigma(z_{\text{true}}-z_{\text{false}})$.
- Score returns a distribution over levels and its expectation.

**Head size.** The introduction says 1.3M parameters; Table 2 says 4.26M. With $w\in\mathbb R^{2560}$ and $W_c,W_g\in\mathbb R^{256\times2560}$, the bilinear head alone has $2560+2\cdot655{,}360=1{,}313{,}280$ parameters. That matches the 1.3M figure, so the 4.26M figure presumably counts additional components *(inference: arithmetic)*.

**Cache.** An optional encoder-output cache is available. On a 1B-1B model with a 1,924-token state, it cut latency from 1,165 to 115 ms with bit-identical logits. A hit requires identical II, because the question instructions are part of the encoder input from stage 3 on.

**Order.** Bongard is **not order-invariant**: each $h_i$ sees the candidates before it. An optional mode, off by default, averages over cyclic rotations.

| Property | Value |
|---|---|
| Backbone | T5Gemma-2 4B-4B (34+34 layers, $d=2560$), unchanged; frozen SigLIP vision tower |
| State encoding | bidirectional encoder, once per request (with the question instructions from stage 3) |
| Question handling | one decoder branch per question, groups of ≤8; log-sum-exp attention merge |
| Head | bilinear judgment head over candidate-end and decision markers ($\sqrt{256}$ scaling) |
| Head size | 1.3M (intro) vs 4.26M (Table 2) |
| Cached | optional encoder output; exact (bit-identical) but requires identical II |
| Option interaction | yes (decoder sees earlier candidates); not permutation-invariant |
| Calibration | per-primitive (and per-K where data allow) temperatures, 1.2–2.15, bound to the checkpoint hash |

## Training

**Stage 1 (supervised).**

- Proper-scoring losses.
- Data: program-exact synthesis, retrieval judgments filtered by a teacher reranker, and structured lookups.
- Corpus size not stated.

**Stage 2 ("JEPA").** The stage uses JEPA pairs derived from stage 1, with objective

$$L=L_{\mathrm{SFT}}+L_{\mathrm{view}}+\lambda\,(L_{\mathrm{content}}+L_{\mathrm{answer}}),\qquad \lambda=0.004 .$$

- **Pack centring** is computed per view type within each physical batch $S$: $c(x)_i=\dfrac{x_i-\mathrm{sg}[\mu_S]}{\lVert x_i-\mathrm{sg}[\mu_S]\rVert}$, with a stop-gradient on the batch mean.
- Content loss: symmetric InfoNCE at $\tau=0.1$, with split groups excluded as negatives.
- Answer loss: $\mathrm{softplus}\big(-(u\cdot p-u\cdot n)/\tau\big)$.
- Motivation: before centring, raw readout cosines were **0.93–0.996**, i.e. strongly anisotropic.

**Stage 3 (sandbox).**

- Every legal action is labelled with an outcome distribution, taken either from rollout frequencies under the stated policy or from an exact oracle (DP, endgame solvers).
- Environments: games, grid worlds, business simulators and computer use.
- Loss: direct cross-entropy against the outcome distribution. The paper justifies this by noting that the expected REINFORCE gradient is half the negative Brier gradient.
- The final model averages checkpoints from rounds 8–15.

**Compute (App. E).**

- Stages 1 and 2 ran on one B200 for about 51 h and 23.4 h.
- Optimiser: 8-bit AdamW with stochastic rounding.
- Stage 3 uses an NVFP4 FFN.

## Calibration

- One temperature per primitive, and per K where data allow. Final values are 1.2–2.15, bound to the checkpoint hash.
- External ECE falls from 0.141 to 0.030 across stage 3.
- DecisionBench ECE is 0.063, against Jev's 0.128.
- Table 10's "fair-chance" KL from uniform falls from 0.173 to 0.0009.

## Results

**Stage 2 (Table 6):**

| Metric | Before → after |
|---|---|
| Re-expression panel | 75.7 → 85.9 |
| Logical complements | 44.8 → 71.1 |
| NLL | 0.613 → 0.437 |

A λ = 0 vs λ = 0.004 control (400 updates) raises within-source QA retrieval from 2.3% to 56.6%.

**Stage 3:** the frozen panel goes from 50.6 to 64.8 (Brier 0.482 → 0.289), and external ECE from 0.141 to 0.030.

**Table 10 (stage 2 → final):**

| Metric | Stage 2 → final |
|---|---|
| Hard-style decisions | 34.9 → 50.5 |
| Frontier-style decisions | 45.2 → 73.9 |
| DecisionBench balanced | 64.3 → 68.7 |
| Fair-chance KL from uniform | 0.173 → 0.0009 |

**Encoder-mask ablation (Table 3a; 1B-1B, frozen):**

| Mask | Accuracy | NLL |
|---|---|---|
| Bidirectional | 69.8 | 0.677 |
| Forward-only | 64.6 | 2.944 |
| Mixed | 57.3 | n.r. |

**DecisionBench 1.0** [@decisionbench]: 78.05%, 4th of 61; ECE 0.063 vs Jev 0.128.

**Same-item comparison with Jev (Table 12, 24 benchmarks):** Bongard wins 6, ties 1, and Jev wins 17.

| Benchmark | Bongard | Jev |
|---|---|---|
| Banking77 | 93.1 | 79.8 |
| JudgeBench | 55.4 | 79.7 |
| Prompt injection | 56.9 | 75.9 |

No independent evaluation of Bongard appears in the sources we read.

## Serving and efficiency

- Median latency is 36 ms on short items (RTX PRO 6000, per the landscape notes).
- 32 questions in one request take 221 ms, against 1.16 s as separate requests. This is the payoff of encoding the state once.
- The paper estimates about \$4.4 per million decisions.
- The encoder cache cuts a repeated 1,924-token request from 1,165 to 115 ms (1B-1B) with bit-identical logits. Hits require identical question instructions.

## Flaws and failure modes

### (a) Bongard's own flaws, with evidence

- **Stage 2 is evaluated on representations only.** Its gains (re-expression, complements, retrieval) come without a decision-level control.
- **It tends to accept wrong worked answers**, and trails Jev on JudgeBench (55.4 vs 79.7) and prompt injection (56.9 vs 75.9).
- **Question-set dependence.** With question-aware encoding (from stage 3), adding other questions to a request changes answers. The same coupling breaks cache hits unless the question instructions are identical.
- **Not order-invariant.** Each candidate sees the earlier ones; rotation averaging is optional and off by default.
- **Loses the same-item comparison with Jev on 17 of 24 benchmarks.**
- **Inconsistent head size** (1.3M vs 4.26M), and no released training data.

### (b) Flaws of other systems it claims to fix

- **Anisotropic contrastive readouts** (raw cosines 0.93–0.996). Pack centring plus λ = 0.004 raises retrieval from 2.3% to 56.6%. CLM's state embeddings show the same symptom (cosine 0.958 with the question appended, 0.947 after the head; CLM #15) *(inference: same pathology)*.
- **Repeated encoding of a shared state across questions.** Encoding the state once gives 221 ms vs 1.16 s for 32 questions.
- **Overconfidence.** Per-primitive temperatures, plus stage-3 proper-scoring training, take external ECE from 0.141 to 0.030.
- **High-variance RL for calibrated decisions.** Direct cross-entropy against outcome distributions replaces REINFORCE, justified by the gradient relation above.

## Implications for our design

*(All items are inference unless cited.)*

**Adopt**

- Pack (batch) centring for contrastive readouts, or at least a check of the cosine distribution of our 512-d outputs. CLM's state cosines (0.947–0.958) indicate the same saturation.
- Outcome-distribution labels from sandboxes, the best-evidenced data move for agentic decisions: panel 50.6 → 64.8, Brier 0.482 → 0.289.
- A bilinear head over marker states as a cheap typed readout.
- Encoder-output caching with exactness tests (bit-identical logits).

**Test, with mixed evidence.** Bongard's 1B probe favours a bidirectional encoder: 69.8 vs 64.6, with NLL 0.677 vs 2.944. In decoder backbones the evidence points the other way: Small Heads finds bidirectional attention costs 3.5–6.0 points in-distribution and 5.4–11.8 out-of-distribution [@smallheads], and Jeff-1.1 finds −0.2 ± 1.1 [@jeff].

**Avoid** making the state encoding depend on the question set, as Bongard's question-aware encoding does. It breaks caching and isolation between questions.

## References

- [@bongard] Bongard, arXiv 2609.39111: https://arxiv.org/html/2609.39111
- [@bongard-repo] https://github.com/AgentBull/bongard · weights: https://huggingface.co/AgentBull/bongard-mini
- [@decisionbench] DecisionBench: https://huggingface.co/datasets/Hanno-Labs/decision-bench
- [@hydragen] Hydragen (shared-prefix attention), arXiv 2402.05099
- [@smallheads] Small Heads, Strong Decisions, arXiv 2611.15592: https://arxiv.org/html/2611.15592
- [@jeff] Jeff-1.1, arXiv 2610.17205: https://arxiv.org/html/2610.17205
- [@clm-repo] CLM issue #15: https://github.com/Contrastive-LM/CLM/issues/15
