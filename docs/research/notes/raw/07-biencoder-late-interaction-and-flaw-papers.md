# Bi-encoder, late-interaction and structured System One models, and the evaluations that expose their flaws: primary-source survey (read 2026-11-21)

CLM-v0.1-8B is the weakest System One model in every independent evaluation that includes it. The evidence points to the head and how texts are laid out, not to the frozen backbone or the harness. The released head ignores the state when options are short labels. Appending the question to the state collapses distinct states to nearly the same embedding. A logistic regression on the same frozen Qwen3-8B embeddings beats a fine-tuned CLM head. Jeva and the Narrow Decision Models paper show how to add option interaction and a "none fits" outcome while keeping option embeddings cacheable.

**How I read things.**
- CLM: I read the full code (every file under `src/clm/`, the `train/` files, `evaluation/bon_eval.py`, preprocessing, the T-Rex example and its result JSONs), the open issues, the HF card, and both result images (read directly).
- Code for other systems: Jeva's decision-stack and cache files, SanSi's model and loss, the sbert2s1 model file, Bongard's head (grep), and S1-Arena's CLM adapter.
- Papers: all 19 arXiv HTML papers, paginated with offsets. Section 7 lists everything I could not read.
- Conventions: "(inference)" marks my own reasoning. Bracketed links point to the source; Ttable and section numbers are the paper's own.

---

## 0. Headline findings for a from-scratch builder

1. **Every independent CLM evaluation agrees it is weak.**
   - S1-Arena: 41.4% accuracy, Brier 0.749. That is **−17.8 pp below its own backbone's answer-token readout**, and the only pair in the study where Brier also gets worse ([2610.21244](https://arxiv.org/html/2610.21244), Table 1, Fig. 5).
   - Jeva Table 1: Typed Decisions test 43.05 ([2610.18773](https://arxiv.org/html/2610.18773)).
   - Reading Right: decision accuracy 0.123 ([2610.18579](https://arxiv.org/html/2610.18579), Table 2).
   - JevAdvBench: clean accuracy 51.3 ([2609.31142](https://arxiv.org/html/2609.31142), Table 3).
   - S1-Arena queried CLM through the official `clm.Engine.answer`, keeping the last 2,048 tokens (the end that holds the question). So the low score is **not** a harness or truncation artifact ([S1-Arena `s1arena/models/clm.py`](https://github.com/TianyuCodings/S1-Arena)).
2. **The failures trace to the head and the text layout (inference, strongly evidenced).**
   - Short option labels make the released head ignore the state: state sensitivity 0.000, and "Very angry" wins at 1.00 for calm states, even with the levels reversed ([CLM #3](https://github.com/Contrastive-LM/CLM/issues/3)).
   - Appending the question to the state drives the mean cosine between different states to 0.958, against 0.452 for the state alone ([#15](https://github.com/Contrastive-LM/CLM/issues/15)).
   - A logistic regression on the *same frozen Qwen3-8B embeddings* beats the fine-tuned CLM head on Banking77 (0.900 vs 0.795) ([#13](https://github.com/Contrastive-LM/CLM/issues/13)).
3. **What works without giving up cached options.**
   - Jeva's one-block late-interaction "decision stack" plus a coverage gate, over cached per-option K/V ([2610.18773](https://arxiv.org/html/2610.18773); [code](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/matrix_decision.py)).
   - NDM's set head on independent option vectors ([2610.25553](https://arxiv.org/html/2610.25553) §10).
   - Both keep request-independent option computation that can be cached.
4. **Head size matters little; backbone, data and readout layout matter.** Small Heads finds a 0.53M cosine head equals a 27.3M head when options are read **in the joint prompt**. Going from 0.6B to 4B adds about 11 out-of-distribution points ([2611.15592](https://arxiv.org/html/2611.15592) Tables 3 and 7).

---

## 1. CLM (Contrastive-LM/CLM, CLM-v0.1-8B)

### Identity
- Apache-2.0 code and weights.
- GitHub repo created 2026-09-23, last push 2026-10-05, 3,212 stars (GitHub API). The engine reports `RELEASE="2026-09-19"` ([engine.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py)).
- Authors: Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré, Mirhoseini ([README](https://github.com/Contrastive-LM/CLM) citation).
- There is no paper; the Notion blog is the only write-up.
- The README says scaling experiments and data pipelines "live in the research repo's `main` branch," which is not public ([README](https://github.com/Contrastive-LM/CLM)).

### Architecture (from code)

**Encoder**
- Run as `vllm serve Qwen/Qwen3-8B --runner pooling` with LAST-token pooling and `--max-model-len 2048`. The script also sets `--enforce-eager --enable-prefix-caching --max-num-seqs 32 --gpu-memory-utilization 0.35` ([serve_qwen3_8b.sh](https://github.com/Contrastive-LM/CLM/blob/main/serve_qwen3_8b.sh)).
- The client L2-normalises the 4096-d vectors ([embedder.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)).

**Heads** ([heads.py `make_head`](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py))
- Shape: `Linear(4096→w) → act → [(depth−2) × Linear(w,w) → LayerNorm → act] → Linear(w→512)`. LayerNorm comes *before* the activation; residual is optional.
- Fine-tuning defaults: width 1536, depth 3, LayerNorm on, GELU, projection 512 ([finetune.py](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)).
- Parameter count: each head is 9,443,840 parameters, 18.9M for the pair (inference: my arithmetic).
- That matches the README's "75 MB reference head" at fp32, so the README's "20M-parameter trainable projection head" means the **pair**. This settles the 9.44M-vs-20M question from the prior notes (inference).

**Score**
- $\ell_i=\min(e^{\text{logit\_scale}},100)\cdot\cos(f_s(s),f_a(a_i))$, then a per-question softmax. An optional request `temperature` in (0, 100] divides the logits ([engine.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py), [heads.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py)).
- The released checkpoint has `logit_scale` = 4.6132 ([#15](https://github.com/Contrastive-LM/CLM/issues/15)). e^4.6132 ≈ 100.8, so the scale is **clamped at the cap of 100** (inference). That fits the overconfidence we and S1-Arena measured.

**Text layout** ([schema.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py))
- State text = prose rendering of the state + "\n\n" + the question's instructions (context first, question last).
- Choice option = its description, or the key if the description is empty, with no prefix.
- Score option = the level text.
- Noul options = `"true: Yes. This is true: {instr}"` and `"false: No. This is false: {instr}"`.
- `confidence` = top probability minus the mean of the others. This differs from TypeSafe's normalised margin.

**Other endpoints**
- `clm-raw` ablation: cosine in the raw 4096-d space × 100.
- `/v1/rank` endpoint.

**Caches**
- Host side: an LRU of 200k normalised embeddings ([embedder.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)).
- Device side: a `VectorArena` preallocated at start-up, 2% of GPU memory by default. 87.5% goes to 512-d projections and 12.5% to 4096-d raw vectors. Keys are `head@generation\x00text`, eviction is LRU, and the arena never grows ([cache.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/cache.py)).
- Example log line: "505.0 MB reserved (215,764×512d + 3,852×4096d)" ([README](https://github.com/Contrastive-LM/CLM)).

**Serving latency** (RTX 4090, server-side p50, fixed action set; [README](https://github.com/Contrastive-LM/CLM) "vector cache")

| Setting | 3 actions | 50 actions |
|---|---|---|
| New state each call | 28.6 → 28.0 ms | 28.8 → 28.1 ms |
| Revisited states | 1.7 → 0.6 ms | 2.0 → 0.7 ms |

- The README summarises this as about 2.8× faster for loops that revisit states.
- The HF card claims "13× faster than Jev" at about 1k candidates ([HF](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)).

### Training
- **Objective:** bidirectional InfoNCE, $L=-\frac{1}{2B}\sum_i[\log\frac{e^{s_i^\top a_i/\tau}}{\sum_j e^{s_i^\top a_j/\tau}}+\log\frac{e^{a_i^\top s_i/\tau}}{\sum_j e^{a_i^\top s_j/\tau}}]$. Mid-training adds hard negatives to the s→a denominator ([README](https://github.com/Contrastive-LM/CLM) "Training Algorithm").
- **Stages** ([README](https://github.com/Contrastive-LM/CLM) "Data Recipe"):
  - Pre-training: about 60M Nemotron DQA question-answer pairs.
  - Mid-training: about 30M Gemini 2.5 Flash-Lite hard negatives.
  - Post-training: about 1M trajectories from ADP, Endless-Terminals and LiteCoder-Terminal-SFT, mixed 60% agentic with 40% DQA replay.
- **Ablations** (README; about 100K held-out questions with 10 hard negatives each):

| Setting | Top-1 |
|---|---|
| Pre-training only | 52.1% |
| + mid-training | 69.2% |
| Hard negatives from the start | peaks at 62.4%, then overfits |
| Post-training with replay | 69 → 68.5% |
| Post-training on agentic data only | 56.2% |

- Our own measurement (51.7 / 69.2 / 68.6) reproduces these.
- **Scaling:** $L(X)\approx(X_c/X)^{\alpha_X}$; "encoder size yields the strongest gains"; $N^*\propto D^{1.02}$, about 310 tokens per parameter. **The exponents are not given anywhere I could read.**
- **Fine-tuning recipe** ([finetune.py](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)):
  - `clm` task:
    - Loss: bidirectional InfoNCE that masks other rows sharing the same `(task_id, step_idx)` with −inf. This is the "same-document masking."
    - Initialisation: `logit_scale` = log(1/0.07); scale clamped at 100.
    - Learning rate: 2e-3·√(1024/w)·√(batch/1024), AdamW, OneCycle (10% warm-up, cosine), gradient clip 1.0.
    - Batch 2048, 20 epochs, patience 5, checkpoints selected by within-task top-1.
    - Optional task-blocked sampler (4 tasks per batch).
    - Embeddings: up to 8,192 tokens. States go through the chat template keeping the tail; actions keep the head.
  - `choice` task:
    - Soft annotator targets.
    - `--loss infonce`: bidirectional in-batch over the batch's distinct option texts. `--loss softce`: per-question softmax.
    - Learning rate 5e-4, batch 256, maximum length 2,048 (keep the tail).
- **Selection evaluation** ([bon_eval.py](https://github.com/Contrastive-LM/CLM/blob/main/evaluation/bon_eval.py)): the trajectory score is the mean cosine over the last `window=12` steps. Best-of-N uses the exact expectation over uniform N-subsets, with ties broken uniformly.
- **No calibration step exists in the code.** Temperature is only a manual request parameter.

### Calibration
- None ships beyond the learned scale, which sits at the cap. Our temperature fit (ECE 0.065 → 0.013) is the fix the authors did not ship.

### Results reported by the authors (read from the repo images)

**Zero-shot (`zero-shot.png`)**

| Task | CLM success | CLM latency | Jev success | Jev latency |
|---|---|---|---|---|
| T-Rex | 5/5 | 16.5 ms | 5/5 | 149.8 ms |
| BFCL v4 | 95.2% | 76.8 ms | 99.2% | 125.5 ms |
| WikiRacing | 26/30 | 79.8 ms | 30/30 | 225 ms |
| Super Mario | 5/5 | 33.5 ms | 5/5 | 132.6 ms |

**Fine-tuned verifier (`agentic.png`)**

| Benchmark | Setup | Pass@1 baseline | CLM | Jev | CLM latency | Jev latency |
|---|---|---|---|---|---|---|
| DeepSWE | Bo4, 38 tasks | 73.7% | 81.6% (31/38) | 71.1% | 79 ms | 449 ms (5.7×) |
| Terminal-Bench 2.1 | Bo5, 30 tasks | 84.0% | 87.6% | 83.1% | 32 ms | 131 ms (4.1×) |

- So CLM beats Jev on DeepSWE by **4 tasks out of 38** (31 vs 27), and the pass@1 baseline by 3 (inference).

**T-Rex result JSONs** ([clm_realtime.json](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/clm_realtime.json), [jev_realtime.json](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/jev_realtime.json))

| Metric | CLM | Jev |
|---|---|---|
| Agreement with the planner's best move | **0.658** | **0.987** |
| Shield interventions (summary field) | **4,883** | **28** |
| Model p50 | 2.6 ms | 131.9 ms |

- The options in this task literally read "Safe… Best" and "Unsafe… Collision" ([t_rex README](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md)).
- The summary's intervention count equals the per-seed shield, arrival and emergency saves added together (inference: they sum exactly).
- So the "5/5 survived" parity comes mostly from the safety harness (inference).

### Independent results (CLM rows)

| Evaluation | CLM-v0.1-8B | Reference |
|---|---|---|
| S1-Arena, non-agentic ([Table 1](https://arxiv.org/html/2610.21244)) | 41.4% / Brier 0.749 / 41 ms / 19 GPU-s per 1k | Qwen3-8B answer-token 59.3 / 0.715; Jev 72.6 / 0.349 |
| S1-Arena per domain | Cls 35.6, Know 48.3, **Social 22.1**, **Pro 28.2**, Ret 45.9, Struct 40.7, Code 40.1, Fore 35.5, Dialog 40.8, Batch 42.2, Judge 46.0, Safety 54.7, Route 49.6 | Table 1 |
| S1-Arena, agentic ([Table 2](https://arxiv.org/html/2610.21244)) | step accuracy 40.6, loop success 1.7% | Qwen3-8B 59.9 / 2.5; Vega 67.8 / 26.5 |
| Jeva Table 1 ([2610.18773](https://arxiv.org/html/2610.18773)) | Typed 43.05; BANKING77 K=8 26.95, K=16 18.15; MASSIVE 33.36 / 23.23; Mind2Web 21.73 / 21.16 / 19.79 | Jeva-4B 78.30 / 92.05 / 91.62 / 91.09 / 90.22 / 79.99 / 62.25 / 66.23 |
| Jeva nested set (K from 32 to 2,048, 83 queries) | correct at every size on **7/83** | Jeva 58/83 |
| Reading Right ([Table 2](https://arxiv.org/html/2610.18579)) | DD 0.123, UU 0.092, U1D1 1.1%, U0D0 79.6%, p50 / p95 39.2 / 475.9 ms; ECE worse than Vega's 26.6% | JEV DD 0.763 |
| JevAdvBench ([Tables 2–3](https://arxiv.org/html/2609.31142)) | clean 51.3, ASR 34.5 ±3.8, **S1 option order 0.0**, S3 near-miss 60.6, H1 60.2; 0.8 gate coverage 77.6 / risk 8.1 / wrong 6.3 | Jev ASR 46.3 |
| Intent routing ([#13](https://github.com/Contrastive-LM/CLM/issues/13)) | Banking77 seen: zero-shot 0.051 → fine-tuned 0.775 (InfoNCE) / 0.795 (soft-CE). Unseen: 0.440 / 0.397. Typed-decisions: 0.355 → 0.686 / 0.661 | **LogReg on the same frozen embeddings: 0.900 (Banking77) and 0.737 (typed-decisions)** |

### Flaws documented in the CLM repo
1. **Short option labels make it ignore the state.**
   - Score levels written as short labels give state sensitivity **0.000**, the same option whatever the order, and the same failure for `choice`. Adding one trailing period changes sensitivity from 0.256 to 1.059 ([#3](https://github.com/Contrastive-LM/CLM/issues/3), comment by xvshiting).
   - Reproduced on four backends: vLLM, MLX, MPS and llama.cpp.
2. **The question suffix collapses distinct states.**
   - Mean cosine between different BoolQ states: 0.958 with the question appended, 0.452 for the state alone, 0.947 after the state head.
   - Result: BoolQ `noul` ≈ 0.133 on every item (498/500 "no"); Emotion answers "sadness" on 460/500 ([#15](https://github.com/Contrastive-LM/CLM/issues/15)).
3. **Truncation keeps the wrong end.**
   - Since vLLM 0.13 the pooling runner truncates from the right, so long states lose the question.
   - Cosine to the correct tail embedding is −0.0006. 139/293 DecisionBench-medium states exceed 2,048 tokens ([#6](https://github.com/Contrastive-LM/CLM/issues/6), [#29](https://github.com/Contrastive-LM/CLM/issues/29)).
4. **Retrieval at scale fails.**
   - `/v1/rank` over 29,852 tldr commands: 0/20 top-1, against BM25's 8/20.
   - Hubness: one candidate appears in the top-10 for 9/20 unrelated queries.
   - Re-ranking BM25's top-50 drops to 2/20 ([#15](https://github.com/Contrastive-LM/CLM/issues/15) comment).
5. **README numbers do not reproduce.** Urgency is 0.41 in the README but 0.84 when run; cold-cache tokens 106 vs 98 ([#15](https://github.com/Contrastive-LM/CLM/issues/15)).
6. **Non-English inputs are near random.** Chinese accuracy is reported below 30% (anecdotal; [#22](https://github.com/Contrastive-LM/CLM/issues/22)).
7. **Concurrency bugs in the arena.**
   - Slot leak on concurrent misses ([#23](https://github.com/Contrastive-LM/CLM/issues/23), [#27](https://github.com/Contrastive-LM/CLM/issues/27)).
   - The thrash fallback runs while holding the lock ([#28](https://github.com/Contrastive-LM/CLM/issues/28)).
   - `val_top1` ignores the same-step mask ([#35](https://github.com/Contrastive-LM/CLM/issues/35)).

### Adopt / avoid
- **Adopt:**
  - The disaggregated state/action cache, including the arena design with generation-stamped keys.
  - Same-document masking.
  - The two-stage hard-negative curriculum.
  - Replay during post-training.
  - Task-disjoint folds.
  - The exact best-of-N expectation in evaluation.
- **Avoid:**
  - Short-label or boilerplate option texts.
  - Appending the question to the state before last-token pooling.
  - Shipping a logit scale stuck at the cap with no calibration.
  - Leaving truncation to the server default.
  - Option-independent scoring with no null outcome.

---

## 2. Jeva (arXiv 2610.18773; [HF](https://huggingface.co/LIQIIIII/Jeva-4B), [GitHub](https://github.com/LiQiiiii/Jeva))

### Identity
- Qi Li and Xinchao Wang (NUS).
- Jeva-4B weights are Apache-2.0: `head.pt`, `backbone_adapter/`, `backbone/`.
- The repo has 234 stars ([GitHub API](https://github.com/LiQiiiii/Jeva)).

### Architecture (paper §3 and Eqs. 1–6, checked against the code)
- **Encoding:**
  - $z_0=P_xE_\theta(T_x(t,\iota,x))$ and $u_i=\mathrm{nrm}(P_oE_\theta(T_o(t,o_i)))$, with last-non-pad pooling.
  - **The option template contains only the decision type and the candidate**, so options do not depend on the request.
  - $P_x$ and $P_o$ are bias-free `Linear(hidden→d)` ([model.py](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/model.py)).
- **Decision block** ([matrix_decision.py](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/matrix_decision.py)):
  - Attention: the state queries option keys with cosine attention (q and k L2-normalised) divided by $\tau_\ell$ (`tau_set`, initialised at 0.15). This is entropy-regularised reading, Eq. 10.
  - Values: $m=W^O\mathrm{Concat}_h\sum_ia^h_iv^h_i$.
  - Updates: $\tilde z=z+\alpha m$ with $\alpha=\tau\,\sigma(\gamma)/\sqrt L$ ($\alpha/\tau$ initialised at 0.6), then $z'=\tilde z+L^{-1/2}\mathrm{SwiGLU}(\mathrm{RMSNorm}(\tilde z))$.
  - Initialisation: q, k, v and o projections start at the identity; the FFN output starts at zero.
  - **Only the state evolves**; option vectors stay fixed.
- **Readout:**
  - $s_i=z_L^\top u_i$, with $u$ normalised but $z$ not, so $\|z\|$ carries the scale.
  - $q=\mathrm{softmax}(s/\tau_{dec})$.
  - Coverage $\rho=\sigma((\max_is_i-b_{cov})/\tau_{cov})$; $p_\perp=1-\rho$; $p_i=\rho q_i$.
  - Temperatures are $10^{-4}+\mathrm{softplus}(\eta)$; $\tau_{dec}$ and $\tau_{cov}$ initialised at 0.10 (App. A.1, B.7).
- **Sizes:** L=1, d=768, H=8, FFN 3,072. Code default depth is 8; the paper uses 1.
- **Cached:** per option, the normalised $u_i$ plus every layer's per-head K and V (`FactorizedOptionMemory`). Never cached: the request state and the refinement.
- **Cache modes:**
  - Set: reuses an exact ordered option set.
  - Hybrid: reuses individual options inside new sets.
  - The cache is LRU with entry and byte caps ([cache.py](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/cache.py)).
- **Typed outputs:**
  - Noul = $q_{i^+}$, the positive Boolean candidate.
  - Score = $\sum_iq_iv_i$, where $v_i$ are numeric rubric values that are not fed to the encoder.
  - Null comes from the coverage gate, not from an extra candidate.

### Propositions (as stated in the paper)
- **Prop. 1 (set-dependent preference):** one block can prefer A in one candidate set and B in another set that contains both. No independent scorer with set-independent tie-breaking can do this (existence proof, App. B.1).
- **Prop. 2:** adding options below the current maximum leaves ρ unchanged. A flat-softmax null instead gives $p_\perp^{flat}=\sigma((b_0-s)/\tau-\log K)$, a log K penalty.
- **Prop. 3:** under exact arithmetic, permutations permute the probabilities and leave $z_L$, the null probability and Noul/Score outputs unchanged. Cached substitution leaves all outputs unchanged.

### Training
- **Losses:**
  - Open-set: $\ell=\mathrm{BCEWithLogits}(g,c_y)-\sum_iy_i\log q_i$, with $g=\mathrm{logit}(\rho)$ and $c_y=\sum_i y_i$ (Eq. 9).
  - Closed-set: candidate cross-entropy.
  - Noul auxiliaries: squared error on the positive candidate's probability, plus a hinge margin for hard labels.
  - Score auxiliaries: error on the weighted expectation and on the cumulative distribution. No loss weights are given.
- **Backbone and optimiser:**
  - Qwen3.5-4B with rank-64 LoRA (α 128, dropout 0.05) "on the projections and decision head." Total trainable is 13.37M; where the LoRA sits is ambiguous in the text.
  - AdamW with separate parameter groups, gradient clip 1.0.
  - A `tune_mode="frozen"` option exists in [backbone.py](https://github.com/LiQiiiii/Jeva/blob/main/src/jeva/_runtime/backbone.py) and App. B.8, but **no frozen-mode results are reported.**
- **Curriculum (App. C.1; approximate sizes include replay; about 1.1M total):**

| Stage | Content | Size |
|---|---|---|
| 1 | Basic typed decisions, intents, action selection | ~80k |
| 2 | Preferences, semantics, general reasoning | ~500k |
| 3 | Counterfactual and verification | ~250k |
| 4 | Evidence-grounded and open-set | ~60k |
| 5 | Reading and logic | ~20k |
| 6 | Hard examples and replay: ~15k teacher-blended 7:3, kept only if the teacher's top answer matches under two shuffles; 30k replay mixing original targets and the previous model's probabilities | ~60k |
| 7 | Targeted refinement | ~10k |
| 8 | Long texts and large pools | ~80k |
| 9 | Retention | ~2k |
| 10 | Final scoring with mixed replay | ~70k |

- **Datasets and teacher are not named.**

### Results
- Tables 1 and 2 are given in §1 above, with Jeva and all baselines. Table 2 highlights:
  - Choice flips under permutation: Jeva **0.00%** (mean TV 3.17e-8), Jev-1.13 17.00%, InternDecision-4B 22.67%, JevK5 26.17%.
  - Noul Brier: Jeva 0.0184, Jev 0.0342.
  - Score MAE: Jeva 0.2315.
- **Null detection** (BANKING77 K=16; 3,080 requests, 588 with the answer absent):
  - AUROC: Jeva 98.66, KEV 93.61, InternDecision-4B 84.27, JevK5 79.44.
  - Jeva catches 93.88% of absent answers at 5.02% false-positive rate.
- **ToolRet (K up to 2,048):**
  - Jeva 78.40% aggregate, 86.90 at K=32, 69.88 at K=2,048. JevK5 64.00 aggregate, 67.47 at K=2,048 (about 28 s).
  - Batching cuts p50 at K=2,048 from 94.459 s to 12.040 s (7.85×).
  - InternDecision fits only 65/84 requests at K=32.
- **Cache reuse:**
  - MLDR, full reuse: Hybrid p50 2,102.0 → 75.1 ms (27.98×; median paired speedup 27.86×). Partial reuse: 1.95×. Set mode stays near 2.1 s.
  - Typed Decisions shared banks: 93.1 → 46.6 / 46.7 ms, skipping 99.2% of candidate encodings.
  - Maximum probability difference across cache conditions: 0.00736.
  - Hardware: one H200; Typed Decisions timings include cold starts.
- **How baselines were run:** CLM got an appended candidate "None of the above. None of the listed options applies to this request."

### Flaws
- No ablation removes the decision stack or the coverage head, or varies L and d.
- No frozen-backbone results.
- Teacher and data sources are unnamed.
- Prop. 1 is an existence result only.
- Inconsistent counts: 46 vs 50 MLDR requests.
- **Adopt** (inference): this is the closest published upgrade path for CLM. Same pooling, linear projections and cached options, plus one interaction block and a coverage gate.

---

## 3. Bongard (arXiv 2609.39111; [code](https://github.com/AgentBull/bongard))

### Architecture
- **Backbone:** T5Gemma-2 4B-4B, unchanged: 34+34 layers, d=2560, bidirectional encoder, frozen SigLIP vision tower. AgentBull Pte Ltd; weights at `AgentBull/bongard-mini` under Gemma terms.
- **Passes:** the state (compact JSON) is encoded once. From stage 3 on, the distinct question instructions ("II") are encoded with it. Each question gets its own decoder branch, in groups of up to 8 (about ⌈N/8⌉ decoder calls).
- **Attention merge:** question-prefix and shared-state partitions are combined by log-sum-exp, Hydragen-style.
- **Judgment head:** $z_i=w^\top h_i+(W_ch_i)^\top(W_gg)/\sqrt{256}$, where $h_i$ is the candidate-end marker and $g$ the decision marker. Confirmed in `JudgmentHead` in [model.py](https://github.com/AgentBull/bongard).
- **Typed outputs:** Noul = $\sigma(z_{true}-z_{false})$; Score returns a level distribution and its expectation.
- **Head size inconsistency:** 1.3M in the introduction vs 4.26M in Table 2.
- **Cache:** an optional encoder-output cache. On a 1B-1B model with a 1,924-token state it cut 1,165 → 115 ms with bit-identical logits, but hits need identical II.
- **Not order-invariant:** each $h_i$ sees earlier candidates. An optional, default-off mode averages over cyclic rotations.

### Training
- **Stage 1:** supervised, proper-scoring losses. Corpus size not stated.
- **Stage 2 (JEPA):**
  - $L=L_{SFT}+L_{view}+\lambda(L_{content}+L_{answer})$ with λ = 0.004.
  - **Pack centring:** $c(x)_i=(x_i-sg[\mu_S])/\|x_i-sg[\mu_S]\|$, computed per view type within each physical batch.
  - Content loss: symmetric InfoNCE with τ = 0.1, split groups excluded as negatives.
  - Answer loss: $\mathrm{softplus}(-(u\cdot p-u\cdot n)/\tau)$.
  - Diagnosis: raw readout cosines were **0.93–0.996 (anisotropy)** before centring.
- **Stage 3 (sandbox):**
  - Label every legal action with an outcome distribution, from rollout frequencies under the stated policy or from an exact oracle.
  - Direct cross-entropy, justified because the REINFORCE expected gradient is half the negative Brier gradient.
  - Final model averages checkpoints from rounds 8–15.
- **Compute:** stages 1–2 on one B200 for about 51 h and 23.4 h; 8-bit AdamW with stochastic rounding; NVFP4 FFN in stage 3 (App. E).
- **Calibration:** one temperature per primitive, and per K where data allow; final values 1.2–2.15, bound to the checkpoint hash.

### Results
- **Stage 2 (Table 6):** re-expression panel 75.7 → 85.9; logical complements 44.8 → 71.1; NLL 0.613 → 0.437.
- **λ=0 vs 0.004 control (400 updates):** within-source QA retrieval 2.3% → 56.6%.
- **Stage 3:** frozen panel 50.6 → 64.8 (Brier 0.482 → 0.289); external ECE 0.141 → 0.030.
- **Table 10, stage 2 → final:** hard-style 34.9 → 50.5; frontier-style 45.2 → 73.9; DecisionBench balanced 64.3 → 68.7.
- **Encoder mask ablation (Table 3a, 1B-1B frozen):** bidirectional 69.8 (NLL 0.677), forward-only 64.6 (NLL 2.944), mixed 57.3.
- **DecisionBench 1.0:** 78.05%, 4th of 61; ECE 0.063 vs Jev 0.128.
- **Same-item comparison with Jev (Table 12, 24 benchmarks):** Bongard wins 6, ties 1, Jev wins 17. Examples: Banking77 93.1 vs 79.8; JudgeBench 55.4 vs 79.7; prompt injection 56.9 vs 75.9.
- **Latency:** 36 ms median on short items. 32 questions in one request take 221 ms vs 1.16 s as separate requests. About $4.4 per million decisions.

### Limitations and assessment
- Stage 2 gains are shown on representations, with no decision-level control.
- It tends to accept wrong worked answers.
- With question-aware encoding, adding other questions changes answers.
- **Adopt:** pack centring for anisotropic contrastive readouts (inference: this is the same pathology as CLM's state-cosine 0.958); outcome-distribution labels from sandboxes; the bilinear head.
- **Avoid:** making the state encoding depend on the question set (it breaks caching and isolation).

---

## 4. SanSi (arXiv 2610.07730; [code](https://github.com/minnesotanlp/Sansi))

### Architecture (Gan, Lee, Kang; UMN)
- **Backbone:** Ouro-1.4B (or 2.6B), 24 shared layers looped: $h_t=N(F_\theta(h_{t-1}))$, run for T=8 loops without re-injecting the input.
- **Prompt:** `<state>\n\nQuestion:…\nOptions: (A)…\nAnswer:`.
- **Readout after every loop:** $\ell_t=(Wz_t+U_tV_tz_t)/\tau_t$, where W is the frozen LM-head rows for the letters and $U_tV_t$ a rank-16 correction (zero-initialised U, so it starts as the identity), with $\tau_t=\exp(a_t)$ ([model.py `CoupledReadout`](https://github.com/minnesotanlp/Sansi/blob/main/sansi/model.py)).
- **Trainable:** LoRA r64 (α 128) at 60.6M plus 0.27M of readouts.
- **Loss:** $\frac1T\sum_t[\mathrm{CE}+\mathrm{Brier}]$. Targets are one-hot, uniform for unanswerable items, or the crowd distribution ([loss.py](https://github.com/minnesotanlp/Sansi/blob/main/sansi/loss.py)).
- **Hyperparameters (App. B):** AdamW β=(0.9, 0.95); learning rate 1e-4 (LoRA) and 1e-3 (readout); 200 warm-up steps then cosine to 10%; 1,000 steps × 16 items; options shuffled on every draw.
- **Data:** 12,800 training items from 20 sources; 10,027 test items from 59 sources.

### Results (Table 2)

| Model | Accuracy | ECE |
|---|---|---|
| SanSi (8 loops) | 72.0 | 0.093 |
| SmolLM2-1.7B | 58.4 | — |
| Qwen3.5-4B | 73.8 | — |
| Kev-4B (authors' data) | 74.3 | — |
| SanSi-2.6B | 75.8 | — |

- Jev API reaches 78.9 overall; on sources none of their models trained on, Jev 83.5 vs SanSi-2.6B 71.9. Released Kev-4B scores 68.2 overall and 72.0 on that subset (App. Table 14).
- **Per loop:** 58.4 → +8.5 (loop 2) → +3.5 (loop 3) → +1.2 (loop 4); loops 5–8 add +0.4 together.
- **Ablations (Table 3):**
  - Training only the last loop drops loop 1 to 35.6.
  - Without the Brier term, ECE at loop 8 is 0.104 vs 0.093.
  - **Averaging the probabilities of all 8 loops cuts ECE to 0.044 at no cost.**
- **Stopping:** Ouro's exit gate reaches the same 72.0 with 44% fewer loops.
- **Depth tasks:** liar chains hold to depth 3 at 1 loop and 11 at 4 loops.
- **As a GRPO reward model:** correct answers rise from 37.0% to 52.1%.
- **Costs and limits:**
  - 8 loops cost 7.7× one pass.
  - Gains require looped pre-training: adding loops to SmolLM2 drops it to 33.5.
  - Many questions from a cached prefix would need K/V stored for every loop.
- **Adopt:** a CE+Brier per-readout loss; averaging probabilities across depths. **Not compatible** with a frozen single-pass encoder (inference).

---

## 5. Small Heads, Strong Decisions (arXiv 2611.15592; Rastogi; promised code repo returns 404)

### Design
- **Backbone:** Qwen3-0.6B with LoRA r32 (α 64) on all projections, and a **joint prompt** (state, all questions and options in one sequence). Five readouts compared:

| Readout | Parameters | Mechanism |
|---|---|---|
| Cosine | 0.53M | last-token pooling of question and each option, projected to 256-d, RMS-normalised, scale min(e^θ, 100) initialised at 10 |
| Joint schema (Clef) | 27.3M | cross-attention to a token memory plus 4 decoder layers |
| Bilinear-MLP | 1.05M | — |
| Learned letter | 0.26M | — |
| LM letter | 0 | frozen LM-head rows |

- **Data:** 15 sources, 220,264 rows per epoch. Soft cross-entropy averaged over questions.
- **Calibration:** two-level temperature (global, clamped to [0.05, 20], times a per-type factor clamped to [0.5, 5]).
- **Evaluation:** single seed, paired bootstrap.

### Findings
1. **Readout makes little difference (Table 3).** In-distribution: cosine 82.0, joint 81.9, bilinear 82.0. Out-of-distribution: cosine 78.0, joint 76.1, LM letter 79.4 (that lead comes entirely from Belebele, and depends on the letter codes, Table 5).
2. **Cost (Table 4):** cosine 6.4 ms per row vs 13.6 for joint; 6.9 vs 8.7 GPU-hours to train. Head and prompt regime are confounded.
3. **Scale dominates (Table 7).** Cosine head ID/OOD: 0.6B 82.0/78.0, 4B 85.9/88.9, 8B 85.8/89.8, 14B 86.6/90.8.
4. **Base, instruct and thinking checkpoints tie** within 0.25 points in-distribution (Tables 8–9).
5. **Retrieval-tuned checkpoints add nothing (Table 10/17).** Qwen3-Embedding-8B 89.2 and Reranker-8B 88.7 vs the general 8B at 89.8 out-of-distribution.
6. **Pooling and attention (Table 6/15):** last/last pooling is best (78.0 OOD). **Bidirectional attention costs 3.5–6.0 points ID and 5.4–11.8 OOD.**
7. **Training stability:** LoRA at 2e-4 diverged on Qwen3-8B (cosine head); 1e-4 trains normally.
8. **Calibrated ECE:** about 0.033–0.034 ID and 0.057–0.077 OOD.

### Implications for us (inference)
- The cosine head works **when options are encoded in context with the state.** CLM's encodes them context-free.
- So "small head" is fine. "Context-free option" is the variable to test.
- Qwen3-Embedding as a backbone is not expected to help.

---

## 6. AnyS1 (arXiv 2611.23751; USTC/NTU; project page only, no data link)

### Design
- **Backbone:** Qwen3.5-4B-Base, LoRA r16, one sequence ⟨state, question, ⟨opt⟩o_i⟨/opt⟩…, ⟨dec⟩⟩.
- **Head:** a pointer projection ($d_p$=256) plus **one position-free 4-head bidirectional self-attention layer over the ⟨dec⟩ and ⟨/opt⟩ vectors**, zero-initialised residuals.
- **Data:** Any-DP, 25,986 consensus states mined from 1,072 SKILL.md skills (Table 1; teacher agreement 89.2%).
- **Loss:** label-smoothed cross-entropy, ε = 0.1. 1 epoch, learning rate 1e-4.
- **Gate:** delegate to System-1 if $\max p\ge\tau$. In cold mode, τ = 0.918 is fit on 153 examples.

### Results
- **Table 4:** option-interaction head 83.8 Dev vs standard pointer 82.7; Jev 78.4; Kev-4B 57.8.
- **Table 3c (calibration objective):**

| Objective | ECE | Gated agreement |
|---|---|---|
| Label smoothing | 0.094 | 97.7% |
| Hard-label CE | 0.146 | 92.4% |
| CE + Brier | 0.148 | 93.7% |

- **Table 3b:** LLM-written synthetic states 65.0 vs real container states 83.8 (confounded by corpus size, 84,481 vs 25,986).
- **Table 2:** 75–91% episode cost savings. The gate-free All-S1 setting collapses ALFWorld to 40.3 (vs 95.5 with the gate).
- **Latency:** 21.8–31.7 ms per decision on an H100.
- **Adopt:** label smoothing for calibration under shift; real tool-output states; a confidence gate tuned for agreement with System-2.

---

## 7. Narrow Decision Models (arXiv 2610.25553; Softplan)

### Architecture
- **Backbone:** mmBERT-base (22 layers; frozen 197M embedding table; 110M body trained; 6.38M new parameters).
- **Prefix-tree mask:** a token sees only its own block and its ancestors. Positions are logical, so sibling options share positions. **The state is encoded once, and no option or question sees another.**
- **Readouts:**
  - Question vector = proj[$h_S$; $h_{Q_i}$] + a type embedding.
  - **Set head:** 2 pre-norm layers, width 384, 6 heads, no positions, zero-initialised.
  - Adequacy logit per option, computed *before* the set head.
  - Noul = σ(logit([$h_Q$; $h_S$])), with no option embeddings.
- **State K/V cache:** exact. State encoding takes 6.5 ms at 256 tokens and 59 ms at 4,096; each extra question costs 0.8–4.9 ms.
- **Losses:** Choice: −log Σ over the accepted set + Brier. Score: log loss + RPS. Noul: log loss + Brier.
- **Optimisation:** learning rate 2e-5 (body) and 2e-4 (new parts); layer-wise decay 0.95 (App. B).
- **Calibration:** bounded per-primitive temperature T ∈ [0.25, 10]; a domain temperature shrunk toward the synthetic one.

### Results (§10)
- **Invariance tests (Table 2):** all within 1.5e-5 in fp32; negative controls differ by 2–14.
- **Option shuffle:** zero argmax changes on 905 questions, vs 41% for Qwen3-0.6B likelihood scoring.
- **Heads on cached vectors:**

| Scoring | NLL | Accuracy |
|---|---|---|
| Independent options | 1.181 | 0.655 |
| + sibling summary statistics | 1.266 | 0.655 |
| **Set head** | **0.799** | **0.705** |

- **Abstention:** recall from an explicit "none" option 0.25 vs **adequacy 0.66**; AUROC 0.79 vs 0.76.
- **Block vs pair encoding:** NLL 0.819 vs 1.024 (−0.204, CI −0.312 to −0.107).
- **Temperature:** Choice ECE 0.188 → 0.033. Temperatures do not transfer across domains (MASSIVE NLL 0.123 → 0.232).
- **Real legal documents:** 91.2% agreement with the production LLM, macro-F1 0.80 vs 0.61 for TF-IDF+LR, ECE 0.011.
- **Adopt** (inference):
  - Separate the state vector from the question vector, which is the direct fix for CLM's suffix collapse.
  - A set head over cached option vectors.
  - An adequacy head for "none."
  - Noul without option embeddings.

---

## 8. From Retrieval to Typed Decisions / sbert2s1 (arXiv 2610.02486; [code](https://github.com/pritamdeka/sbert2s1))

### Conversions ([research/s1/model.py](https://github.com/pritamdeka/sbert2s1/blob/main/research/s1/model.py))
- **Z:** $z_k=\alpha_\tau\cos(e(s),e(x\oplus o_k))$ with a frozen encoder. **The question goes into the hypothesis; the state goes alone.**
- **B:** the same form, trained end to end. State embeddings can be precomputed.
- **C:** Laya cross head with [MASK]-per-option.
- **PFR:** C plus α·cos, with a zero-initialised scorer.
- α starts at 20 with its own learning rate and no weight decay.
- **Calibration:** per-(type, K-bucket) temperatures, K in {2, 3–5, 6–10, ≥11}, fit by L-BFGS on log T.

### Findings
- **B matches C on seen tasks (−0.6, not significant) but loses 9.4 held-out points** (§RQ2; Table 10: B 23.4 vs C 32.8 held-out).
- **Label-word options** (yes/no, true/false): cosine gaps are about 1e-3, and **no encoder is reliably above chance** (RQ1, Table 3).
- **Permutation flips (Table 12):** B 0.0%, PFR 0.3%, C 15.7%, Laya fine-tuned 9.8%.
- **Fine-tuning destroys retrieval:** SciFact nDCG@10 67.6 → 34.3 for B, unless the prior is frozen.
- **Laya's RLCD estimator is miscalibrated in scale:** it is 3.6–15× too large (Table 6); the leave-one-out version (PG-LOO) fixes it. After temperature scaling, every objective sits at 3.6–4.4 ECE (Table 4).
- **Routing:** sending the least-confident 20% to Gemma-4-31B raises macro accuracy 59.5 → 68.4.

### Implications for us (inference)
- This is the clearest evidence that pure bi-encoders lose held-out generalisation.
- Yes/no and short-label options are an intrinsic blind spot. CLM's Noul texts and short Score levels hit exactly this.

---

## 9. Beveridge (arXiv 2611.18655; [code](https://github.com/sathiskumarjothi0-oss/bev-decisions-pypi))

### Design
- **Large-C (398.3M):** ModernBERT-large with questions and options *before* the document (up to 4,096 tokens).
- **Engineered features:** 22 typed value features per number or date; a parameter-free rule engine that emits exact facts (cmp, sum, count, window).
- **Decision network (3.5M):** evidence attention, a shared recurrent layer applied 3 times with halting, option logits from MLP([z; o; z⊙o]), plus an exact yes/no path.
- **Calibration:** per-type temperature from a grid; Large-C uses 1.5 / 1.2 / 1.0.

### Results
- **Table 2:** Beveridge-Bench 93.0 vs Jev 87.9. But SST-5 33.2, Case holdings 42.0, Banking77 63.6 vs Jev 79.2.
- **Ablations (Table 4):**
  - Without exact facts: −17.4.
  - Without the operator path: −11.5.
  - **One recursive step instead of three: +0.0**, so the recursion is unnecessary.
- **Shuffling options changes 3.4% of answers** (§10).
- **Automation at 95% accuracy:** 76% of decisions vs Jev 59% (Table 6).
- **Latency:** 4.9 ms on an H200 (Table 7).
- **Assessment:** gains come from domain-engineered numeric features on the authors' own suite. Limited relevance to a general bi-encoder (inference).

---

## 10. this-that-model-1.0 (arXiv 2609.23886; FLock.io; MIT; [code](https://github.com/FLock-io/this-that-model))

### Design
- **Backbone:** 1.88–2B hybrid delta-rule and full-attention model (base: decider-2b).
- **Head:** the hidden state at each answer slot is scored against the unembedding rows of the declared labels only (Eq. 2). N questions share one pass and are conditionally independent.
- **Schema-first layout:** the prefix is reused across m states (Eq. 5).
- **Loss:** log loss plus Brier, and an unbiased U-statistic Brier reward from 32 draws (Eqs. 8–10).
- **Rollback:** weight interpolation θ(λ)=θ₀+λΔ.

### Results
- **Table 1 (68 questions):** accuracy 0.941, Brier 0.042 vs Jev 0.765 / 0.133; 30.9 ms. The paper notes the gap rests on about 12 questions.
- **Table 3 (2,250 questions, in-distribution for this model):** 0.844 vs Jev 0.803.
- **Weaknesses:** multi-step arithmetic 0.560; branch-shaped families 0.818.
- **Rafe & Das:** gains 6.8 points on CLINC if given descriptions only.
- **Adopt:** the λ-rollback idea and schema-first prefix caching.

---

## 11. Evaluation and flaw papers

### 11.1 System1-Arena (arXiv 2610.21244; UT Austin; [code](https://github.com/TianyuCodings/S1-Arena))

**Protocol**
- Non-agentic: 78,362 questions, 87 tasks, 13 domains.
- Agentic: 3,481 recorded steps (step accuracy) and 138 environments (127 count toward loop success).
- Typed questions: boolean, choice (2–255 options), score (2–10 levels).
- Metrics: accuracy and Brier (0–2). ECE is defined (top-label, 10 equal-mass bins) **but no per-model ECE appears** in what I could read.
- Failures score zero.
- **Latency:** p50 per single request. Local models on an idle H100 (A100 for agentic runs); hosted models include network time. Realtime mode charges a fixed per-model latency, rounded to 10 ms (App. B, D).
- **Model inputs:** CLM keeps the last 2,048 tokens (via the official library). Laya cuts to 512 tokens. Jev is called through the Vercel gateway; Microsoft-Decision-1 through OpenRouter.

**Results:** Table 1 rows for every System-1 model and the per-domain table are summarised in §1 and in my earlier notes. Key rows:

| Model | Accuracy | Brier | p50 |
|---|---|---|---|
| Vega-27B | 73.0 | 0.343 | 228 ms |
| Jev 1.13 | 72.6 | 0.349 | 321 ms |
| Microsoft-Decision-1 | 70.9 | 0.369 | 304 ms |
| OpenAI Decisions | 68.6 | 0.424 | 140 ms |
| Kai-0.6B | 50.3 | 0.537 | 72 ms |
| Laya | 44.5 | 0.660 | 136 ms |
| CLM | 41.4 | 0.749 | 41 ms |

- Best closed LLM: Opus 5.5 at 81.1 / 0.232.
- Agentic loop success: Gemma-4-31B 32.9, Vega 26.5, Jev 17.4, CLM 1.7.

**Flaws found**
- **System-1 vs backbone (Fig. 5):** Vega +2.9, MS +5.6, Kai +11.0, **CLM −17.8**, OpenAI Decisions −2.6.
- **No-answer rates:** 13.2–13.7% of single steps for Jev, Vega and MS-Decision-1 (coverage depresses their accuracy).
- **Open LLM overconfidence:** Gemma-4-31B puts ≥0.999 on its chosen option in 87% of questions.

**Recommended fix:** none explicit. The paper's framing is "which decisions can be left to System 1."

### 11.2 TypedBench (arXiv 2610.11392; DFKI, Imperial, Mila)

**Protocol**
- Accuracy is the median over friendly framings, with the range reported.
- Stability via JSD and flip rates.
- ECE is reported as **ECE / null**, where the null is a Bernoulli resampling floor.
- A fitted temperature T > 1 means overconfident.
- Ordinal metrics: QWK, RPS, monotonicity.
- **Value of calibration** = cost(argmax) − cost(Bayes).
- **CLM not evaluated.**

**Flaws**

| System | Finding |
|---|---|
| Jev | Priority accuracy 0.874 [0.708, 0.946]; T = 0.66 and 0.21 (underconfident); **Bayes decision costs more than argmax (−1,220 per 10⁴)**; out-of-scope AUROC 0.77 |
| Laya | Priority 0.343 (prior 0.338), T = 5.04; collapses before 100 options; ignores negation (0.10–0.28); upper-casing −0.305 |
| Kev-9B | 0.565, value −3,180; Kev 0.8B/4B/9B p50 62/208/325 ms |
| Gemma 4 31B (constrained) | ECE/null 25.4 |
| Benchmark itself | DistilBERT scores 1.000 (states are templated); 273/500 phishing states duplicated before deduplication |

**Fixes:** report accuracy as a range over wordings; calibrate per model, primitive and question; judge probabilities by the decision cost they induce.

### 11.3 Benchmarking System One vs trained classifiers (Rafe & Das, arXiv 2610.00346)

**Protocol**
- TypeSafe wire format for every model, neutral option identifiers o1…oK.
- Yes/no swap under four identifier pairs.
- Option counts 5 to 150.
- Learn-then-Test coverage.
- Fine-tuned DeBERTa-v3-large baseline. **CLM not evaluated.**

**Flaws**
- **Fine-tuned DeBERTa wins nearly everything:** D1 0.793, CLINC 0.953, emotion 0.938, vs Jev 0.732 / 0.913 / 0.504.
- **Yes/no swap flips per 100:** Jev 50.5 (AUC 0.89 → 0.31); Laya 91.3; Kev-0.8B 14.3; Kev-9B, Nimble, decider and this-that 4.2–6.5; Gemma 17.2.
- **Out-of-scope:** Jev accepts 0.310 of out-of-scope inputs at a held-out threshold, vs Kev-9B 0.015.
- **Option count:** Laya drops from 0.852 at 20 options to 0.020 at 50.
- **Calibration:** Kev's stored temperature raises its ECE from 0.051 to 0.377.
- **Cascades (Table 10):** this-that → Jev on CLINC reaches 0.927 at 0.43 of Jev's cost. Weak first stages (Laya, Kev-0.8B) escalate 99–100% of decisions.
- The design-implications section is cut off in the HTML.

### 11.4 "Consider, Then Choose" / Runoff (arXiv 2610.18233; NTU)

**Method**
- Pointwise score: $f=\log P(\mathrm{yes}|x,q,o)-\log P(\mathrm{no}|x,q,o)$, softmaxed to π.
- Consideration set: the smallest top set with π-mass ≥ α = 0.99 and size ≥ k_min = 3.
- Choose: cyclic-shift listwise averaging over that set only. Final $P=\pi(C)\,r$.
- Cost: K + m² option slots instead of K².

**Findings**
- Listwise flip rates grow with K: 4.7% → 17.0% (Qwen3-8B, Banking20).
- More than 90% of flips involve the top-3 pointwise candidates.
- **Table 2:** Runoff flips 0.0 on open models, vs 11.6 for AnyJev averaging (Qwen3-8B); Jev 5.3 → 0.9.
- **Table 4:** 3–4× faster than AnyJev at K=20, but slower at K ≤ 6.
- **Table 3:** Laya native flip 30.6.
- **Ablation (Table 5):** pointwise only 63.3, Runoff 69.0.
- **Relevance (inference):** bi-encoders are already order-invariant. This is a fix for listwise models, and it shows that pointwise-only scoring costs about 6 points.

### 11.5 Reading Right, Acting Wrong (arXiv 2610.18579; CMU)

**Protocol**
- 19,663 paired states.
- Each decision (D) is paired with fact probes (U).
- Reported outcomes: U1D0 ("read the facts right, acted wrong"), plus probe-then-decide and code-composition variants.

**Findings**
- The six most accurate systems put **73–85% of their errors after correct reading**.
- Errors cluster: all 14 competent systems fail together on 141 states.
- **Fix:** ask the facts as narrow questions and compose the action in code. Jev 0.791 → 0.996; Clef → 1.000; Qwen3.5-9B → 0.965.
- On natural states, correct facts add only 2.9–7.1 points. On ALFWorld, adding the **next sub-goal** gives +16.3 for Jev.
- **CLM:** DD 0.123, UU 0.092, so CLM fails at *reading* (U0D0 79.6%), not at composition.
- The appendix gives no CLM input details beyond "authors' serving code."

### 11.6 Beyond Calibration: probability axioms (arXiv 2609.33209; Northeastern, Shenyang)

**Protocol**
- 160 ChaosNLI and PubMedQA items with 3 labels each.
- Checks: complement $s+n=1$, partition, disjunction, monotonicity, cross-format, bundling.
- A repeat-noise floor (NF).

**Findings**

| Check | Jev | Qwen3.8-27B first-token | Qwen verbalized |
|---|---|---|---|
| Negation-complement MAD | **0.064** | 0.293 | 0.122 |
| Partition sum | 1.14 (exceeds 1 on 159/160 items) | 0.71 | — |

- Jev's noise floor is 0.013.

**Fixes**
- Use `choice` for partitions.
- Never treat per-option yes/no probabilities as a distribution.
- Don't transfer thresholds across question formats.
- Run label-free coherence checks.

**CLM relevance (inference):** a bi-encoder's per-option softmax is coherent across a partition by construction, but `noul` vs `choice` coherence is untested.

### 11.7 JevAdvBench (arXiv 2609.31142)

**Protocol**
- 862 questions; 12 black-box attack types; 5,586 variants kept after review.
- An attack succeeds if ≥2/3 attacked calls are wrong.
- Jev's noise floor is 0.17%.

**Findings**
- **Jev:** ASR 46.3 (task override H1 92.0, flawed reasoning C3 77.2); 23.5% risk at the 0.8 gate.
- **Open models:** 34.0–51.7.
- **CLM v0.1:** ASR 34.5, **option-order attack 0.0**, near-miss options **60.6**, gate "wrong" rate 6.3, but clean accuracy only 51.3. CLM falls in a different order than Jev.
- **Methodological caveats:** 68% of the labels are Jev's own stable answers; adaptive search was run only on Jev.

**Fix:** treat the state as untrusted. Raising the gate to 0.99 still leaves 7.9% wrong. No defence is evaluated.

### 11.8 ProbabilityBench (arXiv 2610.17016; Tokyo / RIKEN; [code](https://github.com/ishida-lab/ProbabilityBench))

- **Protocol:** exact instance-level $p^*$ in three simulated worlds (Bayes error 0.019–0.167); expected KL to $p^*$, clipped to [0.001, 0.999]; multicalibration.
- **Findings:**
  - Decision models hold 10 of 14 cost–quality frontier points.
  - Jev's KL improves with demonstrations (0.248 → 0.203); Luna's worsens (0.199 → 0.411).
  - Clipping hits 27% of DeepSeek V4 Flash's outputs.
- **CLM:** not evaluated. Kev appears only in the price table.
- **Useful as:** an exact-probability calibration test bed.

### 11.9 Calibration and Risk-Budgeted Decisions (arXiv 2610.19811)

- **Protocol:** forced binary action with a conservative fallback; control the false omission rate (FOR) at α ∈ {0.05, …, 0.5}. Three threshold rules:
  - Naive: τ = α.
  - Empirical: best calibration-set threshold.
  - Learn-then-Test (LTT): binomial fixed-sequence testing at δ = 0.10.
- **Findings (Table 3, 47 model–dataset pairs):**

| Rule | Exceeds the budget | Notes |
|---|---|---|
| Naive | 42.0% | max excess 73.26 pp |
| Empirical | 51.7% | — |
| LTT | 7.8% | returns no policy in 129/282 cases |

- Span-01 and Solar-Decide never certify at α = 0.15.
- Kev-ft (9,988 records, 2 epochs) certifies 6/6 datasets.
- **PCA-Ridge** (logistic regression on Qwen3-8B embeddings) is competitive: S-Labs TNR 99.8 at 10.0% risk (Table 4).
- **Fixes:** use LTT; fine-tune a local model when a general model fails to certify.
- **Relevance (inference):** frozen Qwen3-8B embeddings plus a linear probe make a strong, cheap baseline. That matches CLM issue #13.

### 11.10 Typed Decision Models: Early Evidence Audit (arXiv 2609.32160; Northeastern)

- **Protocol:** 28 papers posted 19–24 Sep; a 14-item checklist.
- **Findings:**
  - Only **5/26** papers include a label log-probability baseline.
  - **1/27** test option-name invariance.
  - **3/20** report latency both with and without caching.
  - 12/27 lack confidence intervals.
  - Jev trails strong models by 14.6 on JudgeBench and 19.8 on hard RM-Bench.
  - Repeat runs change up to 1.33% of decisions.
- **Checklist IDs:** B1–B3 (baselines), T1–T4 (tests), R1–R3 (reporting), C1–C4 (calibration).
- **Fixes:**
  - Same-backbone label log-probability baselines.
  - A cheap generative cascade baseline.
  - Option renaming tests.
  - Latency with and without cache.
  - Frozen cascade thresholds.

---

## 12. Cross-cutting table: flaw → fix → evidence → cost → works with cached option embeddings?

| # | Flaw (systems affected) | System(s) that fixed it | Evidence | Cost of the fix | Works with cached option embeddings? |
|---|---|---|---|---|---|
| 1 | Options scored independently, so one option's preference can't depend on the others (CLM, any pure bi-encoder) | Jeva decision stack; NDM set head; AnyS1 option-interaction layer | Jeva Prop. 1, nested set 58/83 vs CLM 7/83, Typed 78.30 vs 43.05 (Table 1); NDM NLL 1.181 → 0.799, accuracy 0.655 → 0.705 (§10.4); AnyS1 +1.1 Dev (Table 4) | Jeva-size stack: 4d²+3d·d_ffn ≈ 9.4M parameters at d=768 (repo formula); about 4.2M at d=512 (inference). Per request about 2d²+2Kd+3d·d_ffn MACs with cached K/V, tiny next to an 8B encoder pass (inference) | **Yes** for Jeva (per-layer option K/V cached) and for a set head over cached vectors. AnyS1 needs one joint sequence (**No**) |
| 2 | Question suffix collapses last-token state embeddings (CLM, cosine 0.958) | NDM: separate $h_S$ and $h_Q$, then question vector = proj[$h_S$; $h_Q$]; sbert2s1: question goes into the hypothesis, state embedded alone | CLM #15 (0.452 for the state alone); NDM §4; sbert2s1 model.py | One extra short embedding per question, cacheable; the state embedding becomes reusable across questions (inference) | **Yes** |
| 3 | Yes/no and short-label options carry no signal (CLM Noul and Score; bi-encoders generally) | NDM Noul from [$h_Q$; $h_S$] with no option embeddings; Bongard σ($z_{true}$−$z_{false}$) from a contextual readout; descriptive option texts | sbert2s1: label cosine gaps about 1e-3 (RQ1); CLM #3 sensitivity 0.000 vs 1.059 with sentence options | A small Noul head on the state+question vector; rewrite options as statements | **Yes** |
| 4 | Overconfidence; logit scale stuck at the cap (CLM Brier 0.749; Gemma ECE/null 25.4) | Per-type (and per-K) temperature: Small Heads, Laya, NDM, Bongard, Beveridge, sbert2s1 | Ours: ECE 0.065 → 0.013; NDM 0.188 → 0.033; sbert2s1: every objective 3.6–4.4 after scaling | Negligible; needs held-out data per domain, because temperatures don't transfer (NDM) | **Yes** |
| 5 | No "none fits" outcome; a flat softmax null is biased by log K (CLM, Jev) | Jeva coverage gate; NDM adequacy head | Jeva AUROC 98.66 vs KEV 93.61; NDM recall 0.66 vs 0.25 for an explicit option | One scalar head on the max score; a BCE term | **Yes** |
| 6 | Option-order flips (Jev 17%, Laya 30.6–66.8%, C heads 15.7%) | Bi-encoders, Jeva, NDM (by construction); Runoff (wrapper for listwise models) | JevAdvBench S1: CLM 0.0; sbert2s1 B 0.0%; Jeva 0.00%; Runoff 0.0 vs 11.6 | Free for bi-encoders; Runoff costs K+m² slots | **Yes** (already true for CLM) |
| 7 | Yes/no swap flips (Jev 50.5/100, Laya 91.3) | Fine-tuned decision checkpoints (Kev-9B, Nimble, decider, this-that: 4.2–6.5) | Rafe & Das stress test | Train with neutral or randomised identifiers (inference) | Yes (untested for CLM: inference) |
| 8 | Long states lose the question to truncation (CLM served with vLLM ≥0.13) | Keep the tail (S1-Arena adapter; PRs #6/#29) | Cosine −0.0006 → 1.0 (PR #6) | Two lines | Yes |
| 9 | Reads the facts right but acts wrong (Jev 73.1% of errors, Gemma 80.9%) | Fact probes composed in code; next-sub-goal block | Jev 0.791 → 0.996 on rule states; ALFWorld +16.3 | A second batch of cheap questions | Yes (orchestration layer) |
| 10 | Anisotropic readouts, raw cosine 0.93–0.996 (Bongard stage 1; CLM states 0.947) | Bongard pack centring | Retrieval 2.3% → 56.6% with λ=0.004 (§4.2 control) | Training-time only | Yes |
| 11 | Hard-label CE overconfident under shift | Label smoothing ε=0.1 (AnyS1); CE+Brier (SanSi, NDM); soft targets | AnyS1 ECE 0.094 vs 0.146; SanSi ECE 0.093 vs 0.104 without Brier | Free | Yes |
| 12 | RLCD estimator scale blows up (Laya-style training) | PG-LOO or direct proper-score loss (NDM, Bongard) | sbert2s1: 3.6–15× inflation; PG-LOO +2.5 (Table 4/6) | Free | Yes |
| 13 | Thresholds blow the risk budget (42–52% of settings) | Learn-then-Test | Exceedances 7.8% (Table 3) | Needs ≥22–45 calibration negatives; often refuses to certify | Yes |
| 14 | Weak zero-shot / held-out generalisation of a pure bi-encoder (CLM; B −9.4 held-out) | Joint encoding (Small Heads, Kev); LoRA (Jeva); bigger backbone | Small Heads 0.6B → 4B: +11 OOD; sbert2s1 B vs C | Loses cross-request option caching if options become joint | **No** for joint encoding; Jeva-style LoRA keeps it (**Yes**) |
| 15 | Multi-step reasoning in one pass (this-that 0.560 arithmetic) | SanSi loops; Beveridge exact facts | SanSi +13.5; Beveridge −17.4 without facts | 7.7× compute (SanSi); hand-built features (Beveridge) | No (SanSi); Yes (feature path) |
| 16 | Adversarial state injection (Jev ASR 46.3; all open models 34–52) | None demonstrated | JevAdvBench | — | — |
| 17 | Untrained head underuses the frozen features (CLM fine-tuned 0.795 < LogReg 0.900 on the same embeddings) | Linear probe or PCA-Ridge baselines | CLM #13; Calibration & Risk Table 4 | Trivial | Yes (state-side only) |

---

## 13. Adopt / avoid summary for a from-scratch builder (inference unless cited)

**Keep from CLM**
- Frozen-encoder embedding caches with generation-stamped keys.
- Same-document masked InfoNCE.
- The pre → hard-negative → replay curriculum (its ablations: 52.1 → 69.2; 62.4 when hard negatives start from scratch; 56.2 without replay).
- The bi-encoder's permutation invariance (JevAdvBench S1 = 0.0).

**Add**
1. State and question embedded *separately*, then fused, as in NDM (fixes #15).
2. A Noul head that does not depend on option text.
3. Descriptive option rendering, or question ⊕ option on the option side, as in sbert2s1.
4. A Jeva-style one-block decision stack with a coverage gate over cached option K/V.
5. Per-(type, K) temperature plus CE+Brier or label smoothing.
6. Pack or batch centring for anisotropy.
7. A same-backbone baseline set: answer-token readout, logistic regression on frozen embeddings, fine-tuned DeBERTa.
8. Learn-then-Test thresholds.
9. Tail truncation.

**Evaluate on**
- S1-Arena (the adapter already exists).
- Jeva's Typed / BANKING77 / MASSIVE / ToolRet / nested-set protocol.
- Rafe & Das yes/no swaps.
- TypedBench framing ranges.
- Option-renaming tests (checklist item T3).

**Avoid**
- Short-label or boilerplate options.
- A question suffix before last-token pooling.
- A shipped scale at the cap with no calibration.
- Making the state encoding depend on the question set (Bongard's II): it breaks caching.
- Treating per-option Noul probabilities as a distribution.

---

## 14. What I could not read or verify

- **CLM Notion blog:** JS-rendered. Via r.jina.ai only the headline text came through. **No scaling exponents, hyperparameters or per-task tables found**, and the research repo `main` branch isn't public.
- **CLM ECE in independent evaluations:** S1-Arena ECE values are absent from the visible text. Reading Right says only "worse than Vega's 26.6%." Neither gives an exact number.
- **Jeva:** the final 683 characters (boilerplate). There are no ablation tables in the paper at all. Teacher and data sources are unnamed. No frozen-mode results.
- **Bongard:** per-stage corpus sizes and sandbox round counts are not stated anywhere I read.
- **SanSi:** appendices E–K were truncated (stopping-rule Table 22, calibration Table 28).
- **Small Heads:** the code repo `Ritvik19/DecideAnything` returns 404. The exact attention mask between options in the joint prompt isn't specified in what I read.
- **S1-Arena:** the last part of App. D / Table 5 was truncated, and the per-environment System-1 results aren't attributed to models.
- **TypedBench:** App. F (partial) through J were not read.
- **Rafe & Das:** §6–8 (design implications) truncated.
- **Calibration & Risk:** App. C.2 tail truncated.
- **ProbabilityBench:** App. C and Tables 5–7 not read.
- **Reading Right:** no CLM input-format details in the appendix.

Sources: [CLM repo](https://github.com/Contrastive-LM/CLM) · [CLM HF](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) · [CLM Notion](https://contrastive-lm.notion.site/) · [Miraheze CLM](https://ai.miraheze.org/wiki/CLM) · [VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests) · [Kwok on X](https://x.com/jackyk02/status/2102905335925424285?lang=en) · [Jeva](https://arxiv.org/html/2610.18773) / [Jeva code](https://github.com/LiQiiiii/Jeva) / [Jeva-4B](https://huggingface.co/LIQIIIII/Jeva-4B) · [Bongard](https://arxiv.org/html/2609.39111) / [code](https://github.com/AgentBull/bongard) · [SanSi](https://arxiv.org/html/2610.07730) / [code](https://github.com/minnesotanlp/Sansi) · [Small Heads](https://arxiv.org/html/2611.15592) · [AnyS1](https://arxiv.org/html/2611.23751) · [NDM](https://arxiv.org/html/2610.25553) · [sbert2s1](https://arxiv.org/html/2610.02486) / [code](https://github.com/pritamdeka/sbert2s1) · [Beveridge](https://arxiv.org/html/2611.18655) · [this-that](https://arxiv.org/html/2609.23886) · [S1-Arena](https://arxiv.org/html/2610.21244) / [code](https://github.com/TianyuCodings/S1-Arena) · [TypedBench](https://arxiv.org/html/2610.11392) · [Rafe & Das](https://arxiv.org/html/2610.00346) · [Runoff](https://arxiv.org/html/2610.18233) · [Reading Right](https://arxiv.org/html/2610.18579) · [Beyond Calibration](https://arxiv.org/html/2609.33209) · [JevAdvBench](https://arxiv.org/html/2609.31142) · [ProbabilityBench](https://arxiv.org/html/2610.17016) · [Risk-Budgeted](https://arxiv.org/html/2610.19811) · [TDM Audit](https://arxiv.org/html/2609.32160)
