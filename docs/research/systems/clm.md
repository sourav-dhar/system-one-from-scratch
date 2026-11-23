# CLM (Contrastive Language Models, CLM-v0.1-8B)

> **Family:** bi-encoder (two-tower contrastive scorer over a frozen decoder backbone) · **Team:** Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré, Mirhoseini (README citation block); described as Stanford + NVIDIA in the landscape sources · **Released:** engine constant `RELEASE="2026-09-19"`; GitHub repository created 2026-09-23, last push 2026-10-05, 3,212 stars when read · **License:** Apache-2.0 (code and weights) · **Code/weights:** [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) [@clm-repo]; [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) [@clm-card]; published pre/mid/post-trained heads [@clm-heads] · **Sources read:** every file under `src/clm/`, the `train/` files, `evaluation/bon_eval.py`, preprocessing, the T-Rex example and its result JSONs, the open GitHub issues, the HF card, both README result images, and S1-Arena's CLM adapter. **Gaps:** there is no paper; the [Notion blog](https://contrastive-lm.notion.site/) [@clm-blog] is JavaScript-rendered and only its headline was retrievable; the research repository's `main` branch (scaling experiments, data pipelines) is not public; no scaling-law exponents, hyperparameter tables or per-task tables were found.

## Summary

CLM-v0.1-8B is a two-tower contrastive decision model.

- **Encoding:** a frozen Qwen3-8B, served by vLLM's pooling runner with last-token pooling, embeds the state text and each action text independently.
- **Heads:** two small MLPs project the 4096-d embeddings to 512-d.
- **Decision:** a per-question softmax over cosines times a learned scale capped at 100.
- **Caching:** action embeddings never depend on the request, so they are cached on the host and in a preallocated GPU arena. A revisited state is scored in under a millisecond on an RTX 4090.

**Training.** The heads are trained with bidirectional InfoNCE in three stages: about 60M Nemotron DQA pairs, then about 30M Gemini hard negatives, then about 1M agentic trajectories with 40% replay. The authors report held-out hard-negative top-1 of 52.1% → 69.2% → 68.5%, which this project reproduces (0.5173 / 0.6916 / 0.6863, n = 191,161).

**Independent evaluations** put CLM at or near the bottom of every typed-decision comparison:

- S1-Arena: 41.4% accuracy and Brier 0.749, 17.8 pp below answer-token readout of its own backbone [@s1arena].
- Jeva: Typed Decisions 43.05 [@jeva].
- Reading Right: decision accuracy 0.123 [@readingright].
- JevAdvBench: clean accuracy 51.3 [@jevadvbench].

**Mechanisms.** The repository's issues document why:

- short option labels make the head ignore the state (#3);
- appending the question to the state collapses distinct states (#15);
- right truncation drops the question from long states (#6, #29);
- a logistic regression on the same frozen embeddings beats the fine-tuned head (#13).

**Calibration.** The scale sits at its cap and no calibration ships. One fitted temperature cuts ECE about 5–10× in our measurements, with top-1 unchanged.

## Architecture

**Encoder.** [`serve_qwen3_8b.sh`](https://github.com/Contrastive-LM/CLM/blob/main/serve_qwen3_8b.sh) runs `vllm serve Qwen/Qwen3-8B --runner pooling` with LAST-token pooling and `--max-model-len 2048`, plus `--enforce-eager --enable-prefix-caching --max-num-seqs 32 --gpu-memory-utilization 0.35`. The client L2-normalises the 4096-d vectors ([embedder.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)). Only the heads are trained.

**Heads.** `make_head` in [heads.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py) builds

$$h_0=\phi(W_{\text{in}}\mathbf e),\qquad h_k=\phi\big(\mathrm{LN}(W_k h_{k-1})\big)\ \ (k=1,\dots,\text{depth}-2),\qquad f(\mathbf e)=W_{\text{out}}\,h_{\text{depth}-2},$$

i.e. `Linear(4096→w) → act → [(depth−2) × (Linear(w,w) → LayerNorm → act)] → Linear(w→512)`. LayerNorm comes *before* the activation; a residual connection is optional.

- **Fine-tuning defaults** ([finetune.py](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)): width 1536, depth 3, LayerNorm on, GELU, projection 512.
- **`heads.py` defaults** (as recorded by the method-level notes): depth 2, GELU, no LayerNorm, projection 512. These are function defaults rather than CLI defaults, so the two need not conflict *(inference)*.
- **Parameter count.** Under the fine-tuning defaults, one head has 9,443,840 parameters (6,292,992 + 2,360,832 + 3,072 + 786,944), so the pair has 18.9M.
- **Reconciling the README.** At fp32 the pair is about 75.5 MB, which matches the README's "75 MB reference head". The README's "20M-parameter trainable projection head" therefore means the **pair**. The landscape notes' unreconciled "about 20M per head" is resolved by this *(inference: arithmetic)*.

**Score** ([engine.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/engine.py), [heads.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py)):

$$\ell_i=\min\!\big(e^{\lambda},100\big)\cdot\cos\!\big(f_s(\mathbf e_s),\,f_a(\mathbf e_{a_i})\big),\qquad p_i=\frac{\exp(\ell_i/T)}{\sum_{j=1}^{K}\exp(\ell_j/T)},$$

- $\lambda$ is the learned `logit_scale`, and $T\in(0,100]$ is an optional request `temperature`.
- The released checkpoint has $\lambda=4.6132$ ([#15](https://github.com/Contrastive-LM/CLM/issues/15)). Since $e^{4.6132}\approx100.8$, the scale is clamped at the cap of 100 *(inference)*.
- **Other scoring paths:** a `clm-raw` ablation scores by raw 4096-d cosine × 100, and a `/v1/rank` endpoint ranks candidates.
- **Confidence:** `confidence` is the top probability minus the mean of the others. TypeSafe's normalised margin, $(p_{\max}-1/K)/(1-1/K)$ [@jev-hume], is a different quantity.

**Text layout** ([schema.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py)):

| Type | State-tower text | Action-tower text(s) | Output |
|---|---|---|---|
| all | `f"{state}\n\n{instructions}"`: prose rendering of the state, then the question's instructions (context first, question last) | — | — |
| `choice` | as above | option description, or the bare key if empty; no prefix, no question or type context | softmax over options |
| `score` | as above | the level text | $\sum_i i\,p_i$; no ordinal structure in the loss |
| `noul` | as above | `"true: Yes. This is true: {instr}"` and `"false: No. This is false: {instr}"` (key plus default description) | $p(\texttt{true})$ |

**Caches.**

- **Host:** an LRU of 200k normalised embeddings ([embedder.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)).
- **Device:** a `VectorArena` preallocated at start-up from 2% of GPU memory. 87.5% of it holds 512-d projections and 12.5% holds 4096-d raw vectors. Keys are `head@generation\x00text`, eviction is LRU, and the arena never grows ([cache.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/cache.py)).
- **Example log line:** "505.0 MB reserved (215,764×512d + 3,852×4096d)". At 4 bytes per element this is 441.9 + 63.1 = 505.0 MB, with an 87.5% share for the projections, so the arena stores fp32 *(inference: arithmetic)*.

**Passes per request** *(inference)*:

- **Cost:** one backbone pass over the state text, unless cached, plus one pass per uncached action text; then one state-head MLP, $K$ cached action projections, $K$ cosines and a softmax.
- **Per question:** the instructions are appended to the state, so each question on the same state needs its own state pass.
- **Sharing:** `choice` actions are shared across questions and requests. `noul` actions embed the instructions, so they are reused only when the same question repeats.

| Property | Value |
|---|---|
| Backbone | Qwen3-8B, frozen; vLLM `--runner pooling` |
| Pooling / embedding | last token; 4096-d; L2-normalised client-side |
| Heads | state and action MLPs 4096→1536→1536→512 (fine-tune defaults); 9,443,840 params each, 18.9M pair *(inference)* |
| Scoring | $\min(e^\lambda,100)\cdot\cos$, per-question softmax, optional request temperature |
| Released scale | $\lambda=4.6132$, so at the cap of 100 *(inference)* |
| Served context | 2,048 tokens; fine-tuning embeds up to 8,192 |
| State–option interaction / null | none / none |
| Permutation invariance | by construction |
| Cached | host LRU (200k); GPU arena (2% of memory; 512-d and 4096-d slots; generation-stamped keys) |
| Typed outputs | `choice`, `score`, `noul`; `/v1/rank`; `clm-raw` |

## Training

**Objective** (README "Training Algorithm"; [@cpc]):

$$L=-\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{s_i^\top a_i/\tau}}{\sum_j e^{s_i^\top a_j/\tau}}+\log\frac{e^{a_i^\top s_i/\tau}}{\sum_j e^{a_i^\top s_j/\tau}}\right].$$

Mid-training adds hard negatives to the state→action denominator only.

**Stages and data** (README "Data Recipe"; dataset cards):

| Stage | Data | Details |
|---|---|---|
| Pre | ≈60M Nemotron DQA pairs (question = state, answer = action) | [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) `High-Quality-DQA`: 7,806,965 documents, 62,451,070 pairs; context p1 / median / p99 = 254 / 764 / 1,935 tokens. Vectors [@clm-vectors]: 62,451,070 rows, ≈1.02 TB fp16, no README or license |
| Mid | ≈30M Gemini 2.5 Flash-Lite hard negatives | Up to 10 per pair, ≈95.5% coverage; ≈1 in 20 near-binary questions has a "negative" that paraphrases the answer [@nemotron-text]. Vector shard 0 only [@clm-negs]: 3M pairs, 26,696,585 negatives, ≈229 GB |
| Post | ≈1M trajectories: ADP [@adp], Endless-Terminals, LiteCoder-Terminal-SFT | 60% agentic / 40% DQA replay |
| Task heads | DeepSWE PRM: 405,919 steps, 4,701 trajectories, 113 tasks, MIT, embeddings only; typed decisions: [@typed-decisions] | DeepSWE train and eval (44,409 steps, 449 rollouts) share the 113 tasks, split only by task folds |

**Licensing.** The upstream Nemotron data is gated under the NVIDIA Data Agreement: "solely for the purpose of internal training", with no redistribution. The ungated mirrors are tagged "nvidia-open-data-license", which looks inconsistent with that agreement *(inference)*.

**Ablations** (README; about 100K held-out questions × 10 hard negatives):

| Setting | Top-1 |
|---|---|
| Pre-training only | 52.1% |
| + mid-training | 69.2% |
| Hard negatives from the start | peaks at 62.4%, then overfits |
| Post-training with replay | 69 → 68.5% |
| Post-training on agentic data only | 56.2% |

**Scaling.** The README states $L(X)\approx(X_c/X)^{\alpha_X}$, that "encoder size yields the strongest gains", and $N^*\propto D^{1.02}$ (about 310 tokens per parameter). **No exponents or constants are given anywhere we read.**

**Fine-tuning recipe** ([finetune.py](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)).

- **`clm` task:**
  - Loss: bidirectional InfoNCE, with other rows sharing `(task_id, step_idx)` masked to $-\infty$ ("same-document masking").
  - Scale: `logit_scale` initialised at $\log(1/0.07)$, clamped at 100.
  - Learning rate: $2\times10^{-3}\sqrt{1024/w}\sqrt{\text{batch}/1024}$, i.e. ≈$2.31\times10^{-3}$ at $w=1536$, batch 2048 *(inference: arithmetic)*.
  - Optimiser and schedule: AdamW; OneCycle (10% warm-up, cosine); gradient clip 1.0; batch 2048; 20 epochs; patience 5.
  - Checkpoint selection by within-task top-1; optional task-blocked sampler (4 tasks per batch).
  - Inputs: up to 8,192 tokens; states via the chat template keep the **tail**, actions keep the **head**.
- **`choice` task:**
  - Soft annotator targets.
  - `--loss infonce`: bidirectional in-batch over distinct option texts. `--loss softce`: per-question softmax.
  - Learning rate 5e-4, batch 256, maximum length 2,048 (keep the tail).

**Selection evaluation** ([bon_eval.py](https://github.com/Contrastive-LM/CLM/blob/main/evaluation/bon_eval.py)). A trajectory's score is the mean cosine over its last `window=12` steps. Best-of-N is the exact expectation over uniform N-subsets, with ties broken uniformly.

## Calibration

**What ships.** The code contains no calibration step. Temperature is only a manual request parameter, and the learned scale sits at the cap.

**Independent evidence.**

- S1-Arena: Brier 0.749 vs 0.715 for its own backbone's answer-token readout. CLM is the only System-1/backbone pair whose Brier worsens (+0.034). S1-Arena defines ECE (top-label, 10 equal-mass bins), but no per-model ECE appears in the text we read.
- Reading Right: CLM's ECE is only described as worse than Vega's 26.6%.

**This project's measurements** (authors' published heads [@clm-heads] re-scored with our evaluation code; `results/benchmark/heads.json`):

| Head | ECE at scale 100, full hard test | Fitted scale (chunk 0028) | ECE on chunk 0029 (n = 95,929), before → after |
|---|---|---|---|
| pre | 0.1587 | 51.46 | 0.1598 → 0.0154 |
| mid | 0.0637 | 69.55 | 0.0648 → 0.0131 |
| post | 0.0653 | 69.36 | 0.0668 → 0.0141 |

- Top-1 is unchanged by the fit.
- The fitted scales correspond to temperatures of about 1.94 / 1.44 / 1.44 relative to the shipped 100, so all three heads are overconfident at the cap *(inference)*.
- The ECE reductions are about 10× / 5× / 4.7× *(inference)*.
- The missing step is ordinary temperature scaling [@guo2017].

## Results

**Authors' zero-shot results** (`zero-shot.png`):

| Task | CLM success | CLM latency | Jev success | Jev latency |
|---|---|---|---|---|
| T-Rex | 5/5 | 16.5 ms | 5/5 | 149.8 ms |
| BFCL v4 | 95.2% | 76.8 ms | 99.2% | 125.5 ms |
| WikiRacing | 26/30 | 79.8 ms | 30/30 | 225 ms |
| Super Mario | 5/5 | 33.5 ms | 5/5 | 132.6 ms |

**Authors' fine-tuned verifier results** (`agentic.png`):

| Benchmark | Setup | Pass@1 baseline | CLM | Jev | CLM latency | Jev latency |
|---|---|---|---|---|---|---|
| DeepSWE | Bo4, 38 tasks | 73.7% | 81.6% (31/38) | 71.1% | 79 ms | 449 ms (5.7×) |
| Terminal-Bench 2.1 | Bo5, 30 tasks | 84.0% | 87.6% | 83.1% | 32 ms | 131 ms (4.1×) |

- On DeepSWE, CLM leads Jev by 4 of 38 tasks (31 vs 27) and the baseline by 3 (28) *(inference)*.
- These numbers come from fine-tuned task heads; the README lists this as a limitation.

**T-Rex result JSONs** ([clm_realtime.json](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/clm_realtime.json), [jev_realtime.json](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/results/jev_realtime.json)):

| Metric | CLM | Jev |
|---|---|---|
| Agreement with the planner's best move | 0.658 | 0.987 |
| Shield interventions (summary field) | 4,883 | 28 |
| Model p50 | 2.6 ms | 131.9 ms |

- The options literally read "Safe… Best" and "Unsafe… Collision" ([t_rex README](https://github.com/Contrastive-LM/CLM/blob/main/examples/t_rex/README.md)).
- The intervention count equals the sum of per-seed shield, arrival and emergency saves. The 5/5 parity with Jev therefore comes mostly from the safety harness *(inference)*.
- The JSON's 2.6 ms and the image's 16.5 ms presumably time different spans; the sources do not say *(inference)*.

**Independent evaluations** (every CLM row in our sources):

| Evaluation | CLM-v0.1-8B | Reference rows |
|---|---|---|
| S1-Arena non-agentic, Table 1 [@s1arena] | 41.4%, Brier 0.749, p50 41 ms, 19 GPU-s per 1k requests | Qwen3-8B answer-token 59.3 / 0.715; Jev 72.6 / 0.349; Vega-27B 73.0 / 0.343; Laya 44.5 / 0.660 |
| S1-Arena per domain | Cls 35.6, Know 48.3, Social 22.1, Pro 28.2, Ret 45.9, Struct 40.7, Code 40.1, Fore 35.5, Dialog 40.8, Batch 42.2, Judge 46.0, Safety 54.7, Route 49.6 | Qwen3-8B letter readout: Social 46.8, Pro 59.2, Cls 56.6, Fore 39.4, Struct 63.5, Batch 68.7, Safety 69.5, Route 54.0 |
| S1-Arena agentic, Table 2 | step accuracy 40.6%, loop success 1.7% | Qwen3-8B 59.9 / 2.5; Vega 67.8 / 26.5; Jev 67.8 / 17.4 |
| Jeva Table 1 [@jeva] (CLM given an appended "None of the above" candidate) | Typed 43.05; BANKING77 K=8 26.95, K=16 18.15; MASSIVE 33.36 / 23.23; Mind2Web 21.73 / 21.16 / 19.79 | Jeva-4B 78.30 / 92.05 / 91.62 / 91.09 / 90.22 / 79.99 / 62.25 / 66.23 |
| Jeva nested sets (K 32 → 2,048; 83 queries) | correct at every size on 7/83 | Jeva 58/83 |
| Reading Right, Table 2 [@readingright] | DD 0.123, UU 0.092, U1D1 1.1%, U0D0 79.6%; p50 / p95 39.2 / 475.9 ms | Jev DD 0.763 |
| JevAdvBench, Tables 2–3 [@jevadvbench] | clean 51.3; ASR 34.5 ± 3.8; S1 option order 0.0; S3 near-miss 60.6; H1 60.2; 0.8 gate: coverage 77.6, risk 8.1, wrong 6.3 | Jev ASR 46.3 |
| Intent routing, [#13](https://github.com/Contrastive-LM/CLM/issues/13) | Banking77 seen: zero-shot 0.051 → fine-tuned 0.775 (InfoNCE) / 0.795 (soft-CE); unseen 0.440 / 0.397; typed-decisions 0.355 → 0.686 / 0.661 | Logistic regression on the same frozen embeddings: 0.900 / 0.737 |
| JevBench v1.5.4 ([BenchLM](https://benchlm.ai/md/decision-models.md)) | composite 1.51, rank 77 | — |

Notes on these rows:

- **Harness.** S1-Arena queried CLM through the official `clm.Engine.answer`, keeping the last 2,048 tokens ("the end that its training keeps") [@s1arena-repo]. Its low score is therefore not a harness artefact. Whether keeping only the tail costs CLM accuracy is untested, and the answer-token readout it is compared with saw full inputs *(inference)*.
- **Size of the gap.** The paper's text says −17.8 pp; the Table 1 values differ by 17.9 *(inference: arithmetic)*.
- **Ranking.** The landscape notes describe CLM as the lowest "of 10 System-1 models", and it is last among the key rows in our notes. We have not re-counted the Table 1 rows.
- **Agentic.** All five System-1 models lose step accuracy against their base (−1.2 to −19.3); CLM loses the most (59.9 → 40.6).
- **Reading Right:** CLM fails at *reading* (U0D0 79.6%), not at composition.
- **JevAdvBench caveats:** 68% of its labels are Jev's own stable answers, and adaptive search ran only on Jev. CLM's lower ASR goes with a lower clean accuracy *(inference)*.
- **Jev Decision Index:** lists CLM, but no score is recorded in our notes.
- **Risk-budget study:** a PCA-Ridge probe on Qwen3-8B embeddings is competitive (S-Labs TNR 99.8 at 10.0% risk) [@riskbudget], which agrees with #13.

**This project's reproduction** (published heads, our evaluation code, full held-out sets; `results/benchmark/heads.json`):

| Head | Tricky Q&A top-1 (n = 191,161) | General top-1 | README |
|---|---|---|---|
| pre | 0.5173 | 0.8347 | 52.1% |
| mid | 0.6916 | 0.8364 | 69.2% |
| post | 0.6863 | 0.8363 | 68.5% |

**Truncation.** Cutting the mid head's output to its first 64 / 128 / 256 dimensions gives tricky top-1 of 0.5102 / 0.5669 / 0.6328, against 0.6916 at 512.

**Reading** *(inference)*:

- The reproduction is within 0.4 points of the README at every stage.
- Mid-training adds 17.4 points on the hard-negative task but 0.17 on general retrieval.
- The 64-d prefix falls below the full-width pre-trained head, so no Matryoshka objective was used [@mrl].

## Serving and efficiency

**Latency** (README "vector cache"; RTX 4090, server-side p50, fixed action set). The arrow pairs are presumably without → with the arena *(inference)*.

| Setting | 3 actions | 50 actions |
|---|---|---|
| New state each call | 28.6 → 28.0 ms | 28.8 → 28.1 ms |
| Revisited states | 1.7 → 0.6 ms | 2.0 → 0.7 ms |

- The README summarises this as about 2.8× faster for loops that revisit states.
- Latency is nearly flat in K, because a cached action costs one 512-d cosine. A new state costs one 8B pass of at most 2,048 tokens *(inference)*.

**Speed claims and independent latencies.**

- The HF card claims "13× faster than Jev" at about 1k candidates.
- Secondary coverage reports "up to 9x faster than Jev in tests", and 4–9× on large reusable action sets ([VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests)).
- Independent measurements agree that CLM is fast: S1-Arena p50 41 ms, the fastest System-1 model there, at 19 GPU-s per 1k requests; Reading Right p50 / p95 39.2 / 475.9 ms.
- The 41 ms did not turn into loop success: S1-Arena realtime mode gives 1.7%.

**Interface.** Per the landscape study, CLM exposes the TypeSafe-style `POST /v1/systemone` shape ([repo](https://github.com/Contrastive-LM/CLM)).

**Unreleased variant.** The card announced a multimodal CLM-35B for "early October"; one news piece calls it CLM-35B-A3B MoE ([pasqualepillitteri](https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en)). The HF organisation shows only CLM-v0.1-8B and the DeepSWE heads. jackyk02 has embedded much of the post-training pool with Qwen3.6-35B-A3B-FP8, which suggests the next backbone *(inference)*.

## Flaws and failure modes

### (a) Own flaws, with evidence

1. **Short option labels make it ignore the state** ([#3](https://github.com/Contrastive-LM/CLM/issues/3)).
   - With short-label score levels, state sensitivity is 0.000: "Very angry" wins at 1.00 for calm states, even with the levels reversed. `choice` fails the same way.
   - One trailing period changes sensitivity from 0.256 to 1.059 (comment by xvshiting).
   - Reproduced on vLLM, MLX, MPS and llama.cpp.
   - sbert2s1 finds the same blind spot in bi-encoders generally: label-word cosine gaps of about $10^{-3}$, with no encoder reliably above chance [@sbert2s1].
2. **The question suffix collapses distinct states** ([#15](https://github.com/Contrastive-LM/CLM/issues/15)).
   - Mean cosine between different BoolQ states is 0.958 with the question appended, 0.452 for the state alone, and 0.947 after the state head.
   - BoolQ `noul` ≈ 0.133 on every item (498/500 "no"), and Emotion answers "sadness" on 460/500.
   - This is the same anisotropy Bongard measures at 0.93–0.996 [@bongard] *(inference)*.
3. **Truncation keeps the wrong end** ([#6](https://github.com/Contrastive-LM/CLM/issues/6), [#29](https://github.com/Contrastive-LM/CLM/issues/29)).
   - Since vLLM 0.13 the pooling runner truncates from the right, so long states lose the question.
   - The cosine to the correct tail embedding is −0.0006, and 139/293 DecisionBench-medium states exceed 2,048 tokens.
   - Keeping the tail, as #6/#29 propose, restores a cosine of 1.0.
4. **The head underuses the frozen features** ([#13](https://github.com/Contrastive-LM/CLM/issues/13)). The fine-tuned heads reach 0.775 / 0.795 on Banking77, against 0.900 for a logistic regression on the same embeddings (typed-decisions: 0.686 / 0.661 vs 0.737).
5. **Retrieval at scale fails** ([#15](https://github.com/Contrastive-LM/CLM/issues/15), comment).
   - `/v1/rank` over 29,852 tldr commands scores 0/20 top-1, against BM25's 8/20.
   - Hubness: one candidate appears in the top 10 for 9/20 unrelated queries.
   - Re-ranking BM25's top 50 with CLM gives 2/20.
6. **README examples do not reproduce** ([#15](https://github.com/Contrastive-LM/CLM/issues/15)): urgency 0.41 in the README vs 0.84 when run; cold-cache tokens 106 vs 98. The training ablation, by contrast, does reproduce (this project).
7. **Non-English inputs are near random** ([#22](https://github.com/Contrastive-LM/CLM/issues/22)): Chinese accuracy below 30% (anecdotal).
8. **Arena concurrency bugs.** There is a slot leak on concurrent misses ([#23](https://github.com/Contrastive-LM/CLM/issues/23), [#27](https://github.com/Contrastive-LM/CLM/issues/27)), and the thrash fallback runs while holding the lock ([#28](https://github.com/Contrastive-LM/CLM/issues/28)).
9. **Evaluation bug.** `val_top1` ignores the same-step mask ([#35](https://github.com/Contrastive-LM/CLM/issues/35)), so checkpoint selection is scored differently from the training loss *(inference)*.
10. **Overconfidence.** The scale is at the cap, S1-Arena Brier is 0.749, and ECE before fitting is 0.1587 / 0.0637 / 0.0653 (this project).
11. **No option interaction and no null outcome.** Independent scores obey IIA (Jeva Prop. 1 [@jeva]), which is consistent with nested-set correctness of 7/83 and a 60.6% near-miss attack success rate.
12. **Weak on rubric-style decisions** *(inference)*.
    - CLM loses least to its backbone where tasks resemble retrieval (Route −4.4, Fore −3.9) and most where an instruction must be applied to the state (Pro −31.0, Batch −26.5, Social −24.7).
    - Its training data teaches "which answer matches this question", not "apply instruction X to state Y".
    - The Noul template makes yes and no near-paraphrases.
13. **Data and evaluation hygiene.**
    - About 5% of near-binary Gemini negatives are false negatives.
    - DeepSWE train and eval share tasks, split only by task folds.
    - The README's own limitations: truncation beyond 2,048 tokens; heads tied to their encoder; probabilities relative to the candidate set.

### (b) Flaws of other systems it claims to fix

- **Per-request re-reading of every candidate in joint models (Jev).**
  - Claimed speed-ups: 13× at about 1k candidates, 4.1–5.7× on the verifiers, and 2.6 vs 131.9 ms model p50 on T-Rex. The independent latencies support the speed claim.
  - Accuracy parity holds only with fine-tuned verifier heads (DeepSWE +4 tasks).
  - Zero-shot, Jev matches or beats CLM: BFCL 99.2 vs 95.2, WikiRacing 30/30 vs 26/30, T-Rex agreement 0.987 vs 0.658.
- **Option-order sensitivity** (a design property, not an explicit claim). CLM is permutation-invariant by construction: JevAdvBench S1 = 0.0, against Jev's 17.00% permutation flips (Jeva Table 2).

## Implications for our design

*(Inference unless cited.)*

**Adopt**

- The disaggregated state/action cache: a fixed-size arena with generation-stamped keys, keyed by backbone version, head version and text hash.
- Same-document masked InfoNCE.
- The pre → hard-negative → replay curriculum, whose value the ablations show (52.1 → 69.2; 62.4 when hard negatives start at once; 56.2 without replay).
- Task-disjoint folds and the exact best-of-N expectation.
- Permutation invariance, reported with option-renaming tests.

**Avoid**

- **Short-label or boilerplate options (#3).** Use descriptive statements, or question ⊕ option on the option side, as sbert2s1 does [@sbert2s1].
- **The question suffix before last-token pooling (#15).** Embed state and question separately and fuse them with $\mathrm{proj}[h_S;h_Q]$, as NDM does [@ndm]. The state embedding then becomes reusable across questions.
- **A scale stuck at the cap with no calibration.** Fit per-type and per-K temperatures on held-out *datasets*. Our single temperature already reaches 0.013–0.015.
- **Server-default truncation.** Keep the tail explicitly.
- **Independent scoring without a null.** Add a Jeva block plus coverage gate over cached K/V [@jeva], or an NDM set head plus adequacy logit [@ndm].
- **Omitting a Matryoshka loss**, if small caches matter: our 64-d prefix loses 18.1 points [@mrl].

**Baselines to beat:**

- same-backbone answer-token readout (59.3 on S1-Arena);
- a logistic regression on the same embeddings (#13);
- PCA-Ridge [@riskbudget].

## References

- [@clm-repo] CLM code, README and issues (#3, #6, #13, #15, #22, #23, #27, #28, #29, #35; source files linked inline): https://github.com/Contrastive-LM/CLM
- [@clm-card] https://huggingface.co/Contrastive-LM/CLM-v0.1-8B · [@clm-blog] https://contrastive-lm.notion.site (not readable) · [@clm-heads] https://huggingface.co/jackyk02/qwen3_8b_head
- Data: [@clm-vectors] https://huggingface.co/datasets/Contrastive-LM/CLM-v0.1-Pretrain-Nemotron · [@clm-negs] https://huggingface.co/datasets/jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings · [@nemotron-text] https://huggingface.co/datasets/jackyk02/Nemotron-CC-2.1-HQA-Doc-Negatives · [@adp] https://huggingface.co/datasets/neulab/adp-v2 · [@typed-decisions] https://huggingface.co/datasets/LocalLLaMA/typed-decisions · DeepSWE: https://huggingface.co/datasets/Contrastive-LM/deepswe-clm-train-embeddings-8k
- Evaluations: [@s1arena] https://arxiv.org/html/2610.21244 · [@s1arena-repo] https://github.com/TianyuCodings/S1-Arena · [@jeva] https://arxiv.org/html/2610.18773 · [@readingright] https://arxiv.org/html/2610.18579 · [@jevadvbench] https://arxiv.org/html/2609.31142 · [@riskbudget] https://arxiv.org/html/2610.19811
- Related systems: [@sbert2s1] https://arxiv.org/html/2610.02486 · [@ndm] https://arxiv.org/html/2610.25553 · [@bongard] https://arxiv.org/html/2609.39111 · [@jev-hume] https://archerhume.com/posts/jevs-architecture-unmasked/
- Foundations: [@cpc] arXiv 1807.03748 · [@guo2017] arXiv 1706.04599 · [@mrl] arXiv 2205.13147
- Secondary: [VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests) · [BenchLM](https://benchlm.ai/md/decision-models.md) · [pasqualepillitteri](https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en) · [Miraheze](https://ai.miraheze.org/wiki/CLM) · [Kwok on X](https://x.com/jackyk02/status/2102905335925424285?lang=en)
- This project: `results/benchmark/heads.json`.
