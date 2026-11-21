# Jeva ("System One Models from First Principles")

> **Family:** late interaction (two-tower encoder with a request-side decision block over cached option memory) · **Team:** Qi Li and Xinchao Wang (NUS) · **Released:** arXiv [2610.18773](https://arxiv.org/html/2610.18773) (submission date not recorded in the notes we read) · **License:** Jeva-4B weights Apache-2.0 · **Code/weights:** [LiQiiiii/Jeva](https://github.com/LiQiiiii/Jeva) [@jeva-repo] (234 stars); [LIQIIIII/Jeva-4B](https://huggingface.co/LIQIIIII/Jeva-4B), which ships `head.pt`, `backbone_adapter/` and `backbone/` · **Sources read:** the arXiv HTML paper except its final 683 characters (boilerplate), Eqs. 1–10, Props. 1–3, App. A.1, B.1, B.7, B.8 and C.1; code files [`matrix_decision.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/matrix_decision.py), [`cache.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/cache.py), [`model.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/model.py) and [`backbone.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/backbone.py). **Gaps:** the paper contains no ablation tables; the teacher and the training datasets are not named; no frozen-backbone results are reported. Earlier landscape and data-inventory notes recorded Jeva's weights as "not confirmed / not found" and its training as "not described", because they were based on the abstract only. The later full read supersedes them for weights and training; the training data remains unreleased.

## Summary

Jeva keeps the defining property of a bi-encoder: every candidate is encoded **without the request**, so option representations can be cached across requests. It adds a single late-interaction "decision block" in which the request vector attends over the cached option keys and values. Options never attend to each other, and only the request state evolves. Probabilities come from dot products between the refined request vector and the fixed, normalised option vectors. A separate **coverage gate** on the maximum score yields a "none fits" probability without adding a candidate.

The paper proves three properties of this design:

- one block can express set-dependent preferences that no independent scorer can (Prop. 1);
- the coverage null avoids the $\log K$ penalty of a flat-softmax null (Prop. 2);
- outputs are exactly permutation-equivariant and unchanged when cached options are substituted (Prop. 3).

Jeva-4B (Qwen3.5-4B with rank-64 LoRA, d = 768, one block) reports Typed Decisions 78.30 against CLM-8B's 43.05, 0.00% permutation flips, null-detection AUROC 98.66, and a 27.98× p50 speed-up with full option reuse. The paper contains no component ablations, so the separate contributions of LoRA, the block and the curriculum cannot be identified.

## Architecture

**Encoding** (paper §3; Eqs. 1–6 cover encoding, block and readout; checked against [`model.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/model.py)).

$$z_0=P_x\,E_\theta\big(T_x(t,\iota,x)\big),\qquad u_i=\mathrm{nrm}\big(P_o\,E_\theta(T_o(t,o_i))\big),$$

- $E_\theta$ is the Qwen3.5-4B backbone with last-non-pad pooling (width 2,560).
- $T_x$ is the request template. It holds the decision type $t$, the question $\iota$ and the state $x$.
- $T_o$ is the option template. **It contains only the decision type and the candidate text**, so $u_i$ does not depend on the request.
- $P_x$ and $P_o$ are bias-free `Linear(hidden→d)` maps with $d=768$, and $\mathrm{nrm}$ is L2 normalisation.

**Decision block** ([`matrix_decision.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/matrix_decision.py)). Per head $h$, the state queries the option keys with cosine attention (query and keys L2-normalised) divided by a learned temperature $\tau_\ell$ (`tau_set`, initialised at 0.15). The paper presents this as entropy-regularised reading (Eq. 10). In our notation:

$$a^h_i=\operatorname{softmax}_i\!\left(\frac{\mathrm{nrm}(q^h)^\top\mathrm{nrm}(k^h_i)}{\tau_\ell}\right),\qquad m=W^O\,\mathrm{Concat}_h\sum_i a^h_i\,v^h_i .$$

The state then receives two scaled residual updates:

$$\tilde z=z+\alpha\,m,\quad \alpha=\tau\,\sigma(\gamma)/\sqrt{L}\ \ (\alpha/\tau\ \text{initialised at }0.6);\qquad z'=\tilde z+L^{-1/2}\,\mathrm{SwiGLU}\big(\mathrm{RMSNorm}(\tilde z)\big).$$

- The query, key, value and output projections are initialised at the identity, and the FFN output at zero, so the block starts as a near no-op.
- **Only the state evolves**; the option vectors stay fixed. **Options never attend to each other**, so every interaction between options passes through the state.
- The per-head $k^h_i,v^h_i$ are computed from the option side and cached per layer (see below). We assume they are projections of $u_i$, since $u_i$ is the only option-side encoder output *(inference)*.

**Readout and coverage** (§3; App. A.1, B.7).

$$s_i=z_L^\top u_i,\qquad q=\mathrm{softmax}(s/\tau_{\mathrm{dec}}),\qquad \rho=\sigma\!\left(\frac{\max_i s_i-b_{\mathrm{cov}}}{\tau_{\mathrm{cov}}}\right),\qquad p_\perp=1-\rho,\qquad p_i=\rho\,q_i .$$

- $u_i$ is normalised but $z_L$ is not, so $\lVert z_L\rVert$ carries the logit scale.
- All temperatures are parameterised as $10^{-4}+\mathrm{softplus}(\eta)$; $\tau_{\mathrm{dec}}$ and $\tau_{\mathrm{cov}}$ are initialised at 0.10.

**Typed outputs.**

- **Noul** $=q_{i^+}$, the probability of the positive Boolean candidate. The method-level notes describe these as two cached Boolean candidates, "false" and "true".
- **Score** $=\sum_i q_i v_i$, where the $v_i$ are numeric rubric values that are **not** fed to the encoder.
- **Null** comes from the coverage gate, not from an extra candidate.

**Sizes.**

- The paper uses $L=1$, $d=768$, $H=8$ heads and FFN width 3,072. The code's default depth is 8.
- The paper reports 13.37M trainable parameters, which the method notes attribute to "the projections and core".
- The repo formula for the stack, $4d^2+3d\,d_{\mathrm{ffn}}$, gives 9,437,184 at $d=768$. The two projections add $2\times2560\times768=3{,}932{,}160$, for a total of 13,369,344. So the reported 13.37M matches the projections plus the block alone and excludes the LoRA adapter *(inference: arithmetic)*.

**What is cached** ([`cache.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/cache.py)).

- Per option: the normalised $u_i$ plus every layer's per-head $K$ and $V$ (`FactorizedOptionMemory`).
- Never cached: the request state and its refinement.
- Two modes:
  - **Set** reuses an exact ordered option set.
  - **Hybrid** reuses individual options inside new sets.
- The cache is LRU with entry and byte caps.
- Any LoRA update changes $E_\theta$, so all option caches must be invalidated per checkpoint.

**Passes per request** *(inference)*:

- One backbone pass for the request text (type + question + state).
- One backbone pass per option *not* in the cache.
- One decision block whose cost, with cached K/V, is about $2d^2+2Kd+3d\,d_{\mathrm{ffn}}$ MACs, which is small next to a backbone pass.
- As in CLM, the question is part of the request template, so two questions about one state need two request passes.

### Propositions (as stated in the paper)

**Prop. 1 (set-dependent preference).** One decision block can prefer A in one candidate set and B in another set that contains both. No independent scorer with set-independent tie-breaking can do this (existence proof, App. B.1).

The method-level notes give the underlying limitation. With independent scores $g_i$, $\log(q_i/q_j)=(g_i-g_j)/\tau$ does not depend on the other options, so adding or removing options cannot reverse a preference. This is the IIA (Luce) property of bi-encoders such as CLM.

**Prop. 2 (coverage vs flat null).** Adding options that score below the current maximum leaves $\rho$ unchanged. A flat-softmax null with logit $b_0$ competing against $K$ options of score $s$ instead gives

$$p_\perp^{\mathrm{flat}}=\frac{e^{b_0/\tau}}{e^{b_0/\tau}+K e^{s/\tau}}=\sigma\!\left(\frac{b_0-s}{\tau}-\log K\right),$$

a penalty of $\log K$ that grows with the candidate count. The middle expression is our derivation of the paper's closed form.

**Prop. 3 (symmetry and cache exactness).** Under exact arithmetic, permuting the options permutes the probabilities and leaves $z_L$, the null probability and the Noul/Score outputs unchanged. Substituting cached option entries leaves all outputs unchanged.

### Summary table

| Property | Value |
|---|---|
| Backbone | Qwen3.5-4B, last-non-pad pooling (2,560-d), rank-64 LoRA (α 128, dropout 0.05) |
| Projections | bias-free $P_x,P_o$: 2,560 → 768; $u_i$ L2-normalised |
| Decision block | $L=1$ (code default 8), 8 heads, cosine attention with learned $\tau_\ell$, SwiGLU FFN 3,072, RMSNorm, identity/zero init |
| Option interaction | only through the state; options never attend to each other |
| Scoring | $s_i=z_L^\top u_i$; softmax at $\tau_{\mathrm{dec}}$; $\lVert z_L\rVert$ acts as the scale |
| Null | coverage gate $\rho=\sigma((\max s-b_{\mathrm{cov}})/\tau_{\mathrm{cov}})$ |
| Trainable (reported) | 13.37M (matches projections + block; LoRA placement ambiguous) |
| Cached | $u_i$ + per-layer per-head K/V per option; Set and Hybrid modes; LRU with entry and byte caps |
| Permutation behaviour | exactly equivariant (Prop. 3); 0.00% measured flips |
| Typed outputs | Choice, Noul ($q_{i^+}$), Score ($\sum q_iv_i$), null |

## Training

**Losses** (Eq. 9 and auxiliaries).

- *Open set:* $\ell=\mathrm{BCEWithLogits}(g,c_y)-\sum_i y_i\log q_i$, where $g=\mathrm{logit}(\rho)$ and $c_y=\sum_i y_i$ indicates whether any candidate is correct.
- *Closed set:* candidate cross-entropy.
- *Noul auxiliaries:* squared error on the positive candidate's probability, plus a hinge margin for hard labels.
- *Score auxiliaries:* error on the weighted expectation and on the cumulative distribution.
- **No loss weights are given.**

**Backbone and optimiser.**

- Qwen3.5-4B with rank-64 LoRA (α 128, dropout 0.05), described in the text as "on the projections and decision head". The method notes describe the LoRA as being on the backbone, with the encoder not frozen. The paper's wording is ambiguous about where the LoRA sits.
- AdamW with separate parameter groups and gradient clip 1.0.
- A `tune_mode="frozen"` option exists ([`backbone.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/backbone.py); App. B.8), but **no frozen-mode results are reported**.

**Curriculum** (App. C.1; the sizes are approximate, include replay, and total about 1.1M):

| Stage | Content | Size |
|---|---|---|
| 1 | Basic typed decisions, intents, action selection | ~80k |
| 2 | Preferences, semantics, general reasoning | ~500k |
| 3 | Counterfactual and verification | ~250k |
| 4 | Evidence-grounded and open-set | ~60k |
| 5 | Reading and logic | ~20k |
| 6 | Hard examples and replay: ~15k teacher-blended 7:3 (original : teacher), kept only if the teacher's top answer matches under two option shuffles; 30k replay mixing original targets and the previous model's probabilities | ~60k |
| 7 | Targeted refinement | ~10k |
| 8 | Long texts and large pools | ~80k |
| 9 | Retention | ~2k |
| 10 | Final scoring with mixed replay | ~70k |

**The datasets and the teacher are not named.**

## Calibration

- Calibration is built into the parameterisation: learned $\tau_{\mathrm{dec}}$, $\tau_{\mathrm{cov}}$ and $\tau_\ell$, plus the unnormalised $\lVert z_L\rVert$, which acts as a per-input scale *(inference)*.
- The Noul and Score heads are trained with proper-score-like auxiliaries (squared error; CDF error).
- **No post-hoc temperature scaling is described.**
- The only calibration metric in our notes is Noul Brier: Jeva 0.0184 vs Jev 0.0342 (Table 2).
- No ECE is recorded in our notes.

## Results

**Table 1 (accuracy, %).** CLM was given an appended candidate, "None of the above. None of the listed options applies to this request."

| Model | Typed | BANKING77 K=8 | BANKING77 K=16 | MASSIVE (1st)¹ | MASSIVE K=16 | Mind2Web task | Mind2Web (2nd)² | Mind2Web (3rd)² |
|---|---|---|---|---|---|---|---|---|
| Jeva-4B | 78.30 | 92.05 | 91.62 | 91.09 | 90.22 | 79.99 | 62.25 | 66.23 |
| CLM-8B | 43.05 | 26.95 | 18.15 | 33.36 | 23.23 | 21.73 | 21.16 | 19.79 |
| InternDecision-4B | 78.70 | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. |
| Kev-4B | 66.00 | n.r. | 82.47 | n.r. | n.r. | n.r. | n.r. | n.r. |

¹ Presumably K=8, by analogy with BANKING77 *(inference)*. ² The column labels were not recorded in our notes. "n.r." means not recorded in our notes. InternDecision-4B scores 0.40 above Jeva on Typed.

**Table 2 (robustness and typed outputs).**

| Metric | Jeva | Comparators |
|---|---|---|
| Choice flips under option permutation | **0.00%** (mean TV $3.17\times10^{-8}$; 0 flips on 600 items) | Jev-1.13 17.00%; InternDecision-4B 22.67%; JevK5 26.17% |
| Noul Brier | 0.0184 | Jev 0.0342 |
| Score MAE | 0.2315 | n.r. |

**Null detection** (BANKING77, K=16; 3,080 requests, 588 with the answer absent):

| Model | AUROC |
|---|---|
| Jeva | 98.66 |
| Kev | 93.61 |
| InternDecision-4B | 84.27 |
| JevK5 | 79.44 |

At a 5.02% false-positive rate, Jeva catches 93.88% of the absent answers.

**ToolRet (K up to 2,048):**

| Model | Aggregate | K=32 | K=2,048 |
|---|---|---|---|
| Jeva | 78.40% | 86.90 | 69.88 |
| JevK5 | 64.00 | n.r. | 67.47 (about 28 s) |

- Batching cuts Jeva's p50 at K=2,048 from 94.459 s to 12.040 s (7.85×).
- InternDecision fits only 65 of 84 requests at K=32.

**Nested candidate sets** (K from 32 to 2,048, 83 queries): Jeva is correct at every size on 58/83 queries, CLM on 7/83.

**Cache reuse** (one H200; the Typed Decisions timings include cold starts):

| Setting | Result |
|---|---|
| MLDR, full reuse, Hybrid mode | p50 2,102.0 → 75.1 ms (27.98×; median paired speed-up 27.86×) |
| MLDR, partial reuse | 1.95× |
| MLDR, Set mode | stays near 2.1 s |
| Typed Decisions, shared banks | 93.1 → 46.6 / 46.7 ms; 99.2% of candidate encodings skipped |
| Maximum probability difference across cache conditions | 0.00736 |

**Independent evaluations.** No independent evaluation of Jeva appears in the sources we read.

## Serving and efficiency

- The serving cost is dominated by one request-side backbone pass plus any uncached option passes. The decision block adds about $2d^2+2Kd+3d\,d_{\mathrm{ffn}}$ MACs per request *(inference)*.
- At $d=768$ the block has about 9.4M parameters (repo formula). A $d=512$ variant would have about 4.2M *(inference)*.
- Hybrid caching is what delivers the 27.98× MLDR speed-up. Set mode only helps when the exact ordered option set repeats.
- For very large K, batching the option encodings matters more than the block: 94.459 s → 12.040 s at K=2,048.
- Cache exactness holds empirically up to a maximum probability difference of 0.00736, which is consistent with Prop. 3 under finite precision *(inference)*.

## Flaws and failure modes

### (a) Jeva's own flaws, with evidence

- **No component ablations.** No experiment removes the decision block or the coverage head, or varies $L$ and $d$. The gains of LoRA, the block and the curriculum cannot be separated.
- **No frozen-backbone results**, although `tune_mode="frozen"` exists. In a related setting, Jeff-1.1 found a head-only model on a frozen backbone scored below zero-shot readout (78.9 vs 79.6 Capability) [@jeff], so the frozen-mode question is material.
- **Unnamed teacher and data.** The curriculum stages are described only by content.
- **Prop. 1 is an existence result.** It shows that set-dependent preferences are expressible, not that training learns them. The nested-set result (58/83) is the empirical counterpart.
- **Prop. 2 and the attention path** *(inference)*. Added options also enter the block's attention and can therefore move $z_L$. Our notes do not record how the proof handles this.
- **Internal inconsistencies.**
  - The paper gives both 46 and 50 MLDR requests.
  - The code's default depth is 8; the paper uses 1.
  - The LoRA placement is worded ambiguously.
- **Request side not cached.** As in CLM, the question is inside the request template, so per-question request passes are needed *(inference)*.
- **Large-K accuracy still degrades**, from 86.90 at K=32 to 69.88 at K=2,048 on ToolRet.

### (b) Flaws of other systems it claims to fix

| Flaw | Systems | Jeva's evidence |
|---|---|---|
| IIA: an independent scorer cannot express set-dependent preference | CLM and pure bi-encoders | Prop. 1; nested sets 58/83 vs CLM 7/83; Typed 78.30 vs 43.05 |
| Flat-softmax null biased by $\log K$ | explicit "none" candidates | Prop. 2; AUROC 98.66 vs 93.61 (Kev) and 84.27 (InternDecision-4B) |
| Option-order flips | joint or listwise models | 0.00% vs Jev-1.13 17.00%, InternDecision-4B 22.67%, JevK5 26.17% |
| Re-encoding every candidate per request; context limits at large K | joint models | Hybrid reuse 27.98×; InternDecision fits only 65/84 requests at K=32 |

## Implications for our design

*(All items are inference unless cited.)*

**Adopt.** This is the closest published upgrade path for a CLM-style model, because it keeps the same pooling, the linear projections and the cached options. Concretely:

1. Keep two towers and put the decision type into the option template.
2. Replace "cosine × capped scale" with an unnormalised $z_L$ dotted against normalised $u_i$, so that $\lVert z_L\rVert$ acts as a per-input temperature.
3. Add one cross-attention block (state queries, cached option K/V) and a coverage head trained with Eq. 9.
4. Extend the cache to store per-layer K/V alongside $u_i$, and invalidate it on any backbone or adapter change.

The method notes estimate about 1 week for the block and coverage head on frozen features, plus 2–3 weeks for LoRA and a curriculum.

**Run the ablations Jeva did not.** Specifically: block vs no block, coverage vs explicit "none" option, $L\in\{0,1,2\}$, $d\in\{512,768\}$, and frozen vs LoRA backbone.

**Avoid** relying on Jeva's numbers to predict frozen-backbone performance: none are reported, and the Jeff-1.1 result above points the other way.

**Fix the request side as well.** Jeva leaves the question in the request template. Embedding state and question separately, as Narrow Decision Models do [@ndm], would make the state embedding reusable across questions.

## References

- [@jeva] Li & Wang, "Jeva: System One Models from First Principles", arXiv 2610.18773: https://arxiv.org/html/2610.18773
- [@jeva-repo] https://github.com/LiQiiiii/Jeva — [`matrix_decision.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/matrix_decision.py), [`cache.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/cache.py), [`model.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/model.py), [`backbone.py`](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/backbone.py)
- Jeva-4B weights: https://huggingface.co/LIQIIIII/Jeva-4B
- [@clm-repo] https://github.com/Contrastive-LM/CLM (comparison baseline)
- [@jeff] Jeff-1.1 / "Releasing the Inner Jev", arXiv 2610.17205: https://arxiv.org/html/2610.17205
- [@ndm] Narrow Decision Models, arXiv 2610.25553: https://arxiv.org/html/2610.25553
- [@colbert] ColBERT (late-interaction background), arXiv 2004.12832
