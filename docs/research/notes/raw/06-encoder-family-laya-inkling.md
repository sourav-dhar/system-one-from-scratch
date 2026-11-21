I've finished the research for this family. Most primary sources were readable, but the Laya/Vega preprint itself was not. Where I could not read something, I say so and do not fill the gap. Anything marked **(inference)** is my own reasoning.

# Encoder / non-autoregressive decision models, plus Inkling: primary-source dossier (as of 2026-11-21)

## 0. What I could and could not read

| Source | Status |
|---|---|
| Laya code (`laya/common.py`, `train.py`, `calibrate.py`, `agent.py`, `serve.py`), `BENCHMARKS.md`, `research/README.md`, issues #131/#156/#185/#394/#741/#963, HF card | Read through `gh api` (raw). |
| **"From Laya to Vega" preprint** (DOI 10.20944/preprints202611.0781.v1) | **Not read.** Both preprints.org URLs (the manuscript page and a `download_pub` link found by search) returned HTTP 403. The Wayback Machine is blocked for my fetch tool. Semantic Scholar returned 404 for the DOI lookup and 429 for title search. No PDF exists in either repo (searched the trees). A curl retry with a browser User-Agent was denied by the permission classifier, and I did not try other workarounds. I could read only the Crossref metadata ([api.crossref.org/works/10.20944/preprints202611.0781.v1](https://api.crossref.org/works/10.20944/preprints202611.0781.v1)): authors Nandakishor M and Anjali M, posted 9 Nov 2026, CC BY 4.0, plus the abstract. **No equation or table below comes from the paper.** They come from the code, the cards and the docs. |
| vegaml code (`vegaml/_common.py`, `ttt.py`, `learn/learner.py`, `notebooks/vega_finetune.py`), docs, HF `nandakishorm/vega-U` `vega_config.json` | Read. The engine's **pre-training** code is not in the repo. The README says it ships "only in the training tree". |
| Inkling family: TML posts, HF cards and `config.json`, Tinker SDK docs, Tinker pricing, OpenRouter endpoint API, vLLM recipes, transformers docs | Read. |
| Von, Phocinae, Clef/Clef-flash (card, `joint_schema_model.py`, blog), GLiNER2.5-Decide (card, configs, arXiv 2507.18546), Intern-Decision (repo, arXiv 2610.22131 HTML up to Appendix C) | Read. |

---

## 1. Laya, the classic encoder line (Convai Innovations)

### Identity
- **Repo:** [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya). Created 2026-09-18, Apache-2.0, 35,758 stars (GitHub API).
- **Checkpoints** (all three in the HF repo [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)):
  - `laya`: ModernBERT-large, 421M parameters, 512 context.
  - `laya-multilingual`: mmBERT-base, 322M, 1024 context (up to 8k).
  - `laya-typed-decisions`: ModernBERT-large, 421M, 1024 context.
- **Code completeness:**
  - Shipped: inference, the fine-tune loop, calibration, the server, and ONNX / C++ / .NET / Rust / TS SDKs.
  - **Not shipped:** the base-checkpoint training corpus and the base RLCD run.
- **Lineage:** the author's dev.to post says a 2025 predecessor used frozen embeddings plus a PPO value network (arXiv 2503.23303, 2510.01237) ([dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)).

### Architecture (from `laya/common.py`)
- **Input layout** (`build_sequence` / `build_head`): `[CLS] "{type} question: {ins}" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`.
  - Each option is capped at 48 tokens.
  - If the option block exceeds `head_max_len` (192 for English, 256 for multilingual), every option is cut to `max(4, (head_max_len−16)//K)` tokens.
  - The state gets whatever room is left (`max_len − len(head) − 1`). It is right-truncated, or left-truncated for conversation lists.
  - Literal `[MASK]` strings in user text are replaced with a space.
  - Noul options are hard-coded as `false: …` / `true: …`.
- **Model** (`class DecisionModel`):
  - Backbone: ModernBERT-large (28 layers, d=1024, 16 heads per [dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)), fully fine-tuned.
  - h ← h + type_emb(qtype) (3 types).
  - Head: `nn.TransformerEncoder` with 2 layers (`d_model=d`, `nhead=d//64`, `ff=4d`, dropout 0.1, `norm_first=True`).
  - Gather h at each `[MASK]`, then `scorer = LN → Linear(d,d) → GELU → Linear(d,1)`.
  - Logits are masked to −1e4 and softmaxed per question.
  - Act/escalate head: `Linear(d+4,256) → GELU → Linear(256,n_act)` on `[h_CLS, top1, top1−top2, H(p)/log K, K/255]`.
  - A `temperature` buffer holds 3 values (one per type).
- **Forward passes:**
  - Each question is its **own row**. The state is tokenized once and reused across rows (`agent.py::_encode_state`), and the batch runs as one tensor.
  - Cost grows linearly with the number of questions: on a T4, 1/5/10/50 questions take 39.5/84.5/158.6/771 ms (English) ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) "Speed").
  - **Nothing is cacheable across questions** (inference): the encoder is bidirectional and the state sits after the options.
- **Order-blind "parallel layout" exists but ships disabled.** `parallel_layout` / `parallel_option_masks` give every option the same start position and block cross-option attention. The docstring says "sequential [is what] every published checkpoint was trained on", and the parallel mode needs transformers≥5 (`uses_parallel_layout`).

### Training objective (RLCD), written from the code
From `laya/common.py::proper_reward` and `laya/train.py::rlcd_loss`. For a question with K options, logits z, target distribution t (one-hot or soft), and fixed σ:

1. **Noise model:** ε ~ N(0, σ²I_K), with the per-row mean removed (`eps − eps.sum/k`). z̃ = stopgrad(z) + ε, q = softmax(z̃). Draw S samples (default 4).
2. **Reward:** r(q,t) = Σ_k t_k·max(log q_k, −9.21) + w_sph·(t·q)/‖q‖ − w_rps·𝟙[score]·Σ_k (CDF_q,k − CDF_t,k)²/(K−1).
   - Defaults: w_sph = 0.75 and w_rps = 1.0 in `train.py`.
   - `proper_reward` itself defaults to w_sph = 0.5.
3. **Baseline and advantage:** A = r − mean_samples(r), then divided by **one std over the whole batch** (`adv.std()`).
4. **Policy gradient:** log π(z̃|z) = −‖z̃−z‖²/(2σ²).
5. **Total loss:** L = −mean(A·log π) + **soft-CE(z,t)**. `loss="soft-ce"` drops the policy-gradient term.
6. **σ schedule:** linear per epoch from 0.4 to 0.1 (`sigma_at`).
7. **TD(λ):** `td_lambda_targets` (λ=1, a Monte-Carlo terminal target G backed up over a trajectory's prefixes for noul) is defined and unit-tested. However, **`train.py` never calls it** (repo code search hits only `common.py`, `__init__.py`, docs and tests).

**Discrepancy to report:** the author's post describes the base training as "pure policy gradient (zero supervised cross-entropy loss)", with G=8 samples, σ 1.0→0.3, w_sph 0.5, and an act cost matrix of +1 / −3 / escalate −0.5 ([dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)). The released loop uses PG + soft-CE, 4 samples, σ 0.4→0.1 and w_sph 0.75.

**Hyperparameters** (`TrainConfig`, "the defaults train the way the published notebook does"):

| Setting | Value |
|---|---|
| Epochs | 4 |
| Batch | micro 8 × accumulation 8 = 64 effective |
| Learning rate | encoder 2.5e-5, head 1e-4, min 1e-6 |
| Optimizer / schedule | AdamW, weight decay 0.01, cosine |
| Gradient clip | 1.0 |
| Precision | fp16 autocast on CUDA |
| Calibration holdout | min(400, 10%), seed 20260922 |
| Option shuffling | `shuffle_options=()`, i.e. **off by default** |

**Data:**
- Base: "100% human-labeled public datasets" in six families (support/intent, NLI/fact-checking, safety, guardrails, rubric ratings, multi-turn sales/support). Augmentations: option shuffle, paraphrase, JSON vs raw state, distractor questions. Sizes are not given ([dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)).
- Fine-tune: `LocalLLaMA/typed-decisions`, 6,000 decisions in the train split ([issue #741](https://github.com/NandhaKishorM/laya/issues/741)). Its `gold` field holds "a teacher's probabilities per question" ([docs/finetune.md](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)).

**The RL term adds nothing measurable.** [Issue #741](https://github.com/NandhaKishorM/laya/issues/741) ran a three-arm ablation (A100, typed-decisions, 1,200 test decisions):

| Arm | Accuracy | ECE | Brier |
|---|---|---|---|
| soft-CE | **80.50%** | 11.82% | **0.0501** |
| RLCD (PG + soft-CE) | 79.08% | **11.18%** | 0.0543 |
| direct proper-score gradient | 79.92% | 12.04% | 0.0516 |

The issue's explanation: soft-CE already *is* the expected log score, and the PG term is a high-variance estimate of a gradient that can be computed exactly. `train.py`'s docstring concedes this ("#741 measured no gain"). The Crossref abstract agrees: "a supervised baseline matches the RL objective on single-step questions".

### Calibration
- **Fitter** (`laya/calibrate.py::_fit_unclamped_temperature`): one scalar T per group, LBFGS on log T, minimizing the soft NLL of softmax(z/T).
- **Buckets:** per type, plus `temp_bucket` = type × K ∈ {2, 3–5, 6–10, 11+}.
- **Floors:** `MIN_BUCKET_N = 2000` and `MIN_TYPE_N = 10`. With `calib_max = 400`, fine-tunes therefore only ever get type-level temperatures (inference).
- **Clamp:** results are clamped to [0.5, 5]. This was added after the shipped `choice:11+` temperature of 0.1006 (a ~10× sharpener) published coin flips as 1.0 confidence ([issue #394](https://github.com/NandhaKishorM/laya/issues/394); `common.py::clamp_temperature`).
- **Abstention:** `fit_abstention_thresholds` uses a selective-risk cut (target error 0.10, bucket floor 100).
- **Numbers:** ECE as shipped versus after the refit is 0.466→0.081 (`laya`) and 0.314→0.106 (`laya-multilingual`, which shipped with no temperatures) ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) "Calibration").
- **Two confidence fields.** `confidence` for choice/score is 1−H/log K, which is not calibrated. `answer_confidence` is max p (`agent.py::_decode_answers`).

### Results (all in [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) unless noted)
- **typed-decisions** (400 cases / 2,000 decisions):

| Model | Accuracy | Brier | ECE |
|---|---|---|---|
| `laya-typed-decisions` (fine-tuned) | 0.766 | 0.061 | 0.213 |
| `laya` base | 0.362 | 0.315 | — |
| `laya-multilingual` base | 0.352 | — | — |
| Jev 1.13 (published) | 0.727 | 0.148 | 0.144 |

  Reference points: majority class 0.461, random 0.318, teacher ceiling 0.735. **The 0.766 has "no committed result file behind it yet."**
- **Public sets (N=400):**

| Dataset | laya | laya-typed-decisions | Jev |
|---|---|---|---|
| AG News | 0.950 | 0.953 | 0.910 |
| DAIR Emotion | 0.595 | 0.600 | 0.480 |
| banking77 | 0.425 | 0.492 | 0.870 |

- **Themes:**
  - Spam 0.993 and phishing 0.980 (both in the training mix).
  - Jailbreak 0.708 and toxicity 0.530 (both held out).
  - Routing 0.639 (held out); `laya-multilingual` scores 0.123 here.
- **Languages:**
  - MASSIVE, 51 languages, 20 options: macro accuracy 0.227 for English `laya` and 0.401 for multilingual.
  - Khmer on English `laya`: 0.000 accuracy at 0.952 confidence.
- **Option-order flip rate:** massive_intent.en 0.150 / 0.230 (English / multilingual), versus Jev 0.13.
- **Third-party:**
  - S1-Arena: 44.5% accuracy, Brier 0.660, p50 136 ms, rank 27/29. "The Laya library cuts an input to 512 tokens" ([arXiv 2610.21244](https://arxiv.org/html/2610.21244) Table 1, App. B).
  - Intern-Decision's seven suites: average 57.77, WildJailBreak 14.84, JevBench-Hard Brier 0.804 / ECE 0.246 ([Intern-Decision README](https://github.com/InternLM/Intern-Decision)).
  - Decision Index 0.2.1 (Cloudflare's run): BANKING77 14.3, CLINC150 3.2, MMLU 30.7, median 5.8 ms, p95 222.5 ms ([clef-flash card](https://huggingface.co/Cloudflare/clef-flash)).
  - JevBench v1.4: I 36.1, C 63.7, composite 30.3, sealed ECE 0.172 ([Von card](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md)).
  - fast-decisions: "Laya Router" 46.6% ([GLiNER2.5-Decide card](https://huggingface.co/fastino/GLiNER2.5-Decide)).

### Serving
- `laya-serve` exposes `POST /v1/systemone`, `/v1/systemone/batch` and `/v1/route`. Auth is optional through `LAYA_API_KEY`, and `LAYA_MAX_BATCH_TOKENS` defaults to 131072 ([serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)).
- TileLang fast path (RTX 4070 Ti S): 1 question 17.7→4.6 ms on `laya`, 14.1→2.8 ms on multilingual.
- CPU (4-core EPYC): 580 ms per question (English), 193 ms (multilingual).
- Independent capacity study on H100 NVL with TensorRT FP16: 105 decisions/s at p99 ≤ 50 ms and 175 decisions/s at p99 ≤ 130 ms. All 74 backend rows matched the FP32 parity answers ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)).

### Flaws, with evidence
1. **Near chance zero-shot** on typed-decisions, below the majority baseline ([HF card](https://huggingface.co/convaiinnovations/laya) "Honest Limits").
2. **Option-count collapse** at high K:
   - banking77 scores exactly 0.425 on both checkpoints, because 77 options share the 192/256-token head budget (about 3–4 tokens each).
   - At K=20, confidence separation between right and wrong answers is 0.009 ([#394](https://github.com/NandhaKishorM/laya/issues/394)).
3. **Yes/no label flips.** With `true`/`false` or `yes`/`no` label words, noul returns the negative label in both polarities at confidence 1.0 ([#156](https://github.com/NandhaKishorM/laya/issues/156)).
4. **Slot bias.** `laya-multilingual` never picks the first-listed score option (0/290 in English), and noul AUROC is 0.355 ([#131](https://github.com/NandhaKishorM/laya/issues/131)).
5. **Act head is dead.** It saturates because head activations reach about 300× the encoder's scale: act logits are [4499, −3677] and `act_probability` is always 1.0 ([#185](https://github.com/NandhaKishorM/laya/issues/185)).
6. **Collapse to the prior** on small fine-tunes (72 updates). The within-row logit range falls to 0.01, and 6 of 15 seeds never leave the ln K plateau ([#963](https://github.com/NandhaKishorM/laya/issues/963)).
7. **Over-confident as shipped**, and confidently wrong on unseen scripts.
8. **Long-document position sensitivity.** With 7,000 filler tokens, accuracy is 0.85 at position 0 and 0.40 at position 1.0 ([research/README.md](https://github.com/NandhaKishorM/laya/blob/main/research/README.md)).

**What it claims to fix in Jev:** latency (6–8× faster), weights that can be adapted, and post-refit ECE (0.081 vs 0.246). Caveat: Laya's own typed-decisions table shows Jev with *better* ECE (0.144 vs 0.213).

---

## 2. vegaml / Vega / vega-U (frozen reader + "particle in Gaussian wells")

### Identity
- **Repo:** [github.com/NandhaKishorM/vegaml](https://github.com/NandhaKishorM/vegaml). Created 2026-10-09, Apache-2.0, 315 stars.
- **Default model:** [nandakishorm/vega-U](https://huggingface.co/nandakishorm/vega-U).
- **Laya 0.5 routes `laya.Agent()` to vegaml by default.** The classic encoder now needs `vegaml=False` ([HF card](https://huggingface.co/convaiinnovations/laya)).

### Architecture (`vegaml/_common.py` plus the vega-U `vega_config.json`)

**Prompt** (`build_prompt`; state first by default):
```
Observation: {state} Measurement ({type}): {instructions} Possible outcomes: * {opt1} * {opt2} Outcome:
```
- Options are capped at 32 tokens.
- When the state is shorter than `query_first_max_state = 4096` tokens, the question is *also* placed before it ([README "How it works"](https://github.com/NandhaKishorM/vegaml)).

**Reader** (`Perception`):
- A frozen causal LM, run only up to its deepest tapped layer. The LM head never runs.
- For the 0.8B reader (`n_layers` 24): `feature_layers [13,19]` and `option_layers [19]`.
- Pooled spans: the state mean, question mean and final token at each feature layer, plus the mean of each option span.
- Long states: 16 chunk means (`memory_chunks`) feed a `MemoryReader`.

**vega-U port:** a per-reader standardize step, then a linear map per tapped layer into a shared 1024-d space ([docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md)).

**Engine** (`class VegaEngine`; latent_dim D=64, world_dim 256, proj_hidden 2048):
- Softmax scalar mixes over the tapped layers.
- World latent s = MLP(state), plus `foresight` s_next = WorldDynamics(s).
- probe = MLP(question) + type_emb; impulse = MLP(last token).
- z₀ and p₀ come from MLP([s, s_next, probe, impulse]).
- Each option k gets a well from MLP([o_k, probe]):
  - centre c_k (score levels are placed on a learned 1-D rail),
  - log depth log a_k ∈ [−4, 4],
  - width σ_k = 0.3 + softplus(·).
- **Potential:** U(z) = ½κ‖z‖² − Σ_k a_k·exp(−‖z−c_k‖²/2σ_k²), with κ = exp(log_kappa), initialized to 0.05.
- **Dynamics:** n_steps = 12, dt_max = 0.5, with a selective-SSM "thermostat":
  - u = ssm_in([z, p, ∇U, ‖∇U‖]); dt = dt_max·σ(u_dt); γ = softplus(u_fr).
  - h ← e^(−A·dt)·h + dt·u_b; force = 0.1·tanh(ssm_out(h)).
  - Soft-capped gradient: g_eff = ∇U/(1 + ‖∇U‖/20).
  - p ← p·e^(−γ·dt) − dt·g_eff + dt·force; z ← z + dt·p (symplectic Euler).
  - A particle is "settled" when its kinetic energy < 0.01 and ‖∇U‖ < 0.1.
- **Readout:** E_k = ‖z_T − c_k‖²/2σ_k² − log a_k, and P = softmax(−E/τ).
- **Size:** 14.3M parameters, no attention (README). The engine is about 55 MB.

**Forward passes:**
- **One reader pass per question.** "Three questions about one state mean three prompts and three passes", except for states ≥4096 tokens, where the state prefix is encoded once and each question continues from a copy of it (README "73,728-token context").
- Cacheable: the reader KV of long-state prefixes, and the per-question engine output (inference).

### Training
- **Engine pre-training is described only in prose; the code is not released.**
  - The engine is trained "on the settling behaviour, not on next-token likelihood".
  - It uses "a counterfactual objective [that] pairs items that share an answer space" and a one-step world-dynamics block "trained to imagine the next state".
  - It has 2 rank-32 low-rank adapters on 7 engine projections, with per-question sigmoid gates (README "What is new here").
- **vega-U** was trained jointly on Qwen3.5-0.8B and Gemma-3-1B over **708,657 uid-matched items**, records up to 73,728 tokens, with a cross-reader invariance loss ([vega-U card](https://huggingface.co/nandakishorm/vega-U)).
- **Attaching a new reader** (`vegaml.attach`) is label-free:
  - The anchor reader writes the targets.
  - A ridge port is fit per layer, then refined for a few hundred steps by distillation with an anchor MSE term ([docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md)).
- **Released adapter recipe** ([notebooks/vega_finetune.py](https://github.com/NandhaKishorM/vegaml/blob/main/notebooks/vega_finetune.py)):
  - CE on cached features, plus a gate BCE (positives = the task, negatives from `ag_news`).
  - AdamW with OneCycle, lr 3e-4, rank 32, 3 epochs, about 0.84M parameters.
  - **It was run only on a 96-row smoke test** (0.313→0.542 on 48 items, "not a result").

### Calibration
- **Per-decision "thermal" temperature:** log τ = b + w·[log1p(kinetic), log1p(nearest-well distance), steps_used/n_steps, log K], one θ vector per type (`vega_config.json` `calibration.theta`).
- **The gain over a scalar temperature is tiny:** held-out calibration NLL is 0.48719 (scalar) versus 0.48476 (thermal) (`heldout_calib_nll`).
- **Split-conformal thresholds** at α 0.10 / 0.05:
  - choice 0.744 / 0.803,
  - score 0.837 / 0.924,
  - noul 0.571 / 0.686.
- **Abstain thresholds:** risk 0.1 → 0.5965. **Unbound threshold:** 1.377.
- **User refit** ([docs/calibration.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/calibration.md)): a per-question temperature, then a conformal threshold. At least 10 examples are required, and 30–100 are recommended.

### Results (vega-U card and README, all self-reported)

| Reader | How attached | Held-out acc | Zero-shot acc | ECE | AURC | Acc @ 50% coverage |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | cold | 0.811 | 0.629 | 0.021 | 0.053 | 0.976 |
| Qwen3.5-4B | cold | 0.791 | 0.568 | 0.017 | 0.063 | 0.965 |
| Qwen3.5-0.8B | joint | 0.765 | 0.534 | 0.010 | 0.073 | 0.956 |
| gemma-3-1b-it | joint | 0.738 | 0.504 | 0.007 | 0.093 | 0.927 |

- The 27B was scored on n=21,008; the others on n=50,676.
- The ordinal `score` type sits at about 0.61 for every reader.
- **Learner** (14 templates): 84.7% on day one → 96.0% after 120 confirmed answers, versus Jev 89.5% → 92.6%.
- **4B reader:**
  - banking77: 87.5% / ECE 0.028, versus Jev at 79.4% as shipped.
  - typed-decisions: 73.7% / ECE 0.031, versus Jev at 74.7% / 0.047.
- **Losses it reports:** as a ToxicChat prompt-injection gate, Jev catches 79.1% at 0.3% refused versus 73.0% at 0.7% for vega. "On familiar labels… a plain classifier on the same features matches or beats the engine."

### Flaws, and what it claims to fix
- `unbound` "fired on 71% of normal banking77 rows".
- The "held-out" numbers are in-distribution splits of the 708k-item training mixture.
- The gain from thermal calibration is about 0.002 NLL.
- One reader pass per question for short states.
- No independent evaluation was found.
- **Fixes claimed relative to Laya:** zero-shot accuracy (0.534–0.629 versus near chance), the 512-token context (73,728), uncalibrated gating (conformal sets plus abstain), and the need to fine-tune (Learner).

---

## 3. Inkling (Thinking Machines Lab), and how to run it as an ablation arm

### Identity and architecture
All weights are Apache-2.0 per the HF tags and cards. Configs are from each repo's `config.json`.

| Model | Release | Total / active params | Layers × hidden | Experts | Context | Source |
|---|---|---|---|---|---|---|
| Inkling | 2026-07-15 | 975B / 41B | 66 × 6144 | 256 routed + 2 shared, top-6 | 1M | [post](https://thinkingmachines.ai/news/introducing-inkling/), [config](https://huggingface.co/thinkingmachines/Inkling) |
| Inkling-Small | 2026-07-30 | 276B / 12B | 42 × 4096 | 256 + 2, top-6 | 1M (OpenRouter serves 524k) | [card](https://huggingface.co/thinkingmachines/Inkling-Small) |
| Inkling 2 Small | 2026-11-05; weights posted 11-12 | 594B / 17B | 48 × 4608 | 384, top-8, no shared experts | 1M | [card](https://huggingface.co/thinkingmachines/Inkling-2-Small), [post](https://thinkingmachines.ai/news/inkling-2-small/) |
| Inkling 2 Mini | 2026-11-20 | 26.4B / 3.4B (5.4B effective) | 24 × 2304; layers 6–18 run 3× | 128, top-8 | 1M | [card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) |

Shared design details:
- Attention mixes sliding-window (512 in v1, 1024 in v2) and global layers at a 5:1 ratio.
- Relative position embeddings instead of RoPE, and short convolutions.
- 8 multi-token-prediction (MTP) heads.

### Training
- **Inkling:** 45T pretraining tokens; Muon optimizer; over 30M RL rollouts.
- **Graders:** a rubric grader plus a claims grader that verifies each claim with agentic web search.
- **Calibration**, quoted from [the Inkling post](https://thinkingmachines.ai/news/introducing-inkling/):
  - "We trained for calibration with RL against proper scoring rules on a large corpus of resolved real-world questions."
  - Plus "short-form factual QA with abstention-aware rewards: answering only pays off when the model is likely to be right".
  - Prompts that encourage or forbid hedging.
  - **The scoring rule, reward weights and corpus size are not disclosed.**
- **Inkling 2 Small:**
  - NVFP4-native pretraining: H1 8.4T, H2 3.8T at 512K, H3 about 320B, H4 about 34B tokens.
  - RL: 1,075 steps, 17M+ rollouts, 2,048 GPUs, StableReinforce.
  - "Forecasting tasks in RL used proper scoring rules", and an alignment phase used on-policy self-distillation ([post](https://thinkingmachines.ai/news/inkling-2-small/)).

### Published calibration and MCQ numbers (no ECE anywhere)

| Metric | Inkling | Inkling 2 Small | Inkling 2 Mini |
|---|---|---|---|
| ForecastBench Brier Index (no search / with search) | 61.1±0.79 / 63.7±0.82 | — | — |
| Prophet Arena Brier | 0.1617 | — | — |
| AA-Omniscience score | 2.1 (post) / 1.0% (HF card) — inconsistent | 16.1 | −8.6 |
| AA-Omniscience non-hallucination rate | — | 73.3% | 63.4% |
| GPQA-D | 87.2 | — | — |
| Global-MMLU-Lite | 88.7 | — | — |

- The forecasting numbers were measured on "a different checkpoint of Inkling than the one released" ([post](https://thinkingmachines.ai/news/introducing-inkling/)).
- Inkling 2 Small: "No Brier score, calibration error, or abstention rate is reported" ([post](https://thinkingmachines.ai/news/inkling-2-small/)).

### Is there a decision / System-One variant or a decision API?
**None that I found** in these sources:
- The HF org lists only Inkling(-NVFP4), Inkling-Small(-NVFP4), Inkling-2-Small(-MXFP8), Inkling-2-Mini, `TML-d24-varloop-research` and tokenizers ([HF API](https://huggingface.co/api/models?author=thinkingmachines)).
- The 2026 TML news index has no decision, classifier or calibration-model post ([news](https://thinkingmachines.ai/news/)).
- The Tinker SDK offers sampling, logprobs and training only ([samplingclient.md](https://github.com/thinking-machines-lab/tinker/blob/main/docs/api/samplingclient.md)).
- I found no third-party result that uses Inkling as a vegaml reader or System-One backbone (searched GitHub and the web). S1-Arena does not evaluate it.

### (1) Hosted access and log-probabilities

| Route | Inkling | Inkling-Small | Inkling 2 Small | Inkling 2 Mini |
|---|---|---|---|---|
| **Tinker** (prefill / sample, $ per M tokens) | 1.87 / 4.68 | 0.58 / 1.44 | 0.30 / 1.00 | 0.15 / 0.50 |
| Tinker logprobs | yes, all models | yes | yes | yes |
| OpenRouter providers with `logprobs`/`top_logprobs` | **none** (Together only, no logprobs) | **none** (DeepInfra fp4 only) | PrimeIntellect $0.15, Modal / Nebius / Together / Cloudflare $0.30 (input) | Io Net $0.08, PrimeIntellect $0.10 |
| Serverless input / output ($/M) | 1.00 / 4.05 | 0.30 / 1.20 | 0.30 / 1.00 | — |

- Tinker prices are from [tinker/pricing](https://thinkingmachines.ai/tinker/pricing/) as of 2026-10-19. "Forward-only requests from a training client are charged at the training rate."
- **Tinker logprob tools:** `sample(..., max_tokens=1, target_prompt_logprobs=…)` scores chosen candidate token ids at chosen prompt positions in a single prefill. There is also `topk_prompt_logprobs`, plus `prompt_logprobs_last_n`, which reuses the prefix cache.
- OpenRouter providers and parameters are from [openrouter.ai/api/v1/models/…/endpoints](https://openrouter.ai/api/v1/models).

**Practical consequence (inference):** a letter-readout baseline for Inkling or Inkling-Small runs only on Tinker or self-hosted. For Inkling 2 Small or 2 Mini, any of the listed providers works.

**The chat template injects a thinking-effort system message.** "none" maps to 0.0 ([chat_template.jinja](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/main/chat_template.jinja)). The 2 Mini card warns that minimum effort "does not fully disable reasoning". So read the answer token after forcing `<|message_model|><|content_text|>` with a raw-completions prompt (inference).

### (2) Self-hosting footprint
- **Inkling:**
  - BF16 is about 1.9 TB repo storage (952B parameters per the safetensors total).
  - NVFP4 is about 594 GB, with `vram_minimum_gb` 651, verified on GB200/B200/H200; Hopper falls back to BF16 dequantization ([vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling.yaml)).
- **Inkling-Small:** NVFP4 about 171 GB, VRAM minimum 180 GB, TP1 on B300/GB300 and TP2 on B200 ([recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-Small.yaml)).
- **Inkling 2 Small:**
  - NVFP4 (native): at least 500 GB, i.e. 2×B300 or 4×H200.
  - MXFP8: at least 800 GB ([card](https://huggingface.co/thinkingmachines/Inkling-2-Small)).
  - vLLM ≥0.35.0, VRAM minimum 473 ([recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-2-Small.yaml)).
- **Inkling 2 Mini:** about 20 GB mixed NVFP4/MXFP8. Runs on one DGX Spark or a Mac with ≥32 GB; H100/H200 verified. vLLM needs the `inkling-2-mini` docker image (no wheel yet).
- **Engines:** SGLang supports Inkling v1 and v2; llama.cpp has an `inkling2` converter (code search).

### (3) Extracting hidden states (frozen reader)
- **Tinker has no hidden-state API.** Searching the SDK for `hidden_states` returns no hits.
- **Self-host through transformers.** It has native `inkling`, `inkling2` and `inkling2_loop` model directories ([transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)), and the repos contain no remote code.
- **Caveat:** for Inkling 2 Small, "Currently only the `Inkling-2-Small-MXFP8` checkpoint is supported" in transformers ([inkling2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/inkling2.md)).
- **Inference:** `output_hidden_states=True` should work as for any HF model. vegaml's `find_layers` forward hooks would fire 3× on the looped core layers of 2 Mini, so you would need to choose which pass to tap. No vegaml port for any Inkling exists.

### How Inkling can serve the project, with costs (all inference)
1. **Zero-shot letter-readout arm.**
   - Assume about 1k prompt tokens per item. Then 78k S1-Arena items ≈ 78M prefill tokens: about **$12 on 2 Mini, $23 on 2 Small, $45 on Inkling-Small, $146 on Inkling** (Tinker prefill rates).
2. **Soft-target teacher with reasoning.**
   - Assume 261k items × (1k prompt + 2k thinking tokens) on 2 Small: about $78 prefill + $522 sampling.
   - Alternatively, average the K-candidate readout logprobs over 2–4 option permutations to get distributions directly, at roughly prefill cost only.
3. **Frozen-reader arm.** Inkling 2 Mini under a vegaml-style head on one 32–80 GB GPU. The bigger models are impractical as per-request readers.
4. **Recipe sources.**
   - **LoRA Without Regret** ([TML blog](https://thinkingmachines.ai/blog/lora/), 2025-09-29): use LoRA LR ≈ 10× the full-fine-tune LR; apply it to all layers, especially MLP/MoE; α=32 with A at 1/√d_in and B=0; LoRA dislikes large batches; rank 1 suffices for RL.
   - **On-Policy Distillation** ([TML blog](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025-10-27): the per-token advantage is −reverse-KL(student‖teacher) at discount 0; about 9–30× cheaper than SFT and about 50–100× cheaper than RL in their AIME setup; recipe at `tinker_cookbook/recipes/distillation`.
   - Tinker training costs $0.75/M tokens (2 Mini) and $1.50/M (2 Small) at 64K.

---

## 4. Von (wfzyx), archived
- **Identity:**
  - [github.com/wfzyx/von](https://github.com/wfzyx/von): Apache-2.0, 922 stars, **archived**; last commit 2026-10-19 points users to `LiquidAI/d1-omni-600M`.
  - Weights: [wfzyx/von](https://huggingface.co/wfzyx/von), 395M (ModernBERT-large), 8192 context.
  - Code is complete, including the training scripts.
- **Architecture** ([MODEL_CARD.md](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md), [models/option_marker.py](https://github.com/wfzyx/von/blob/main/src/von/models/option_marker.py)):
  - Input: `"{question} {state} [SEP] [MASK] opt1 [MASK] opt2 …"`, with forged `[MASK]`/`[SEP]` literals in user text neutralized by a zero-width joiner.
  - `OptionMarkerScorer`: LN → Linear(d, d/2) → GELU → LN → Linear(·,1).
  - **Order-invariant masks** (`build_independent_option_masks` and `build_option_invariant_position_ids`): option tokens see only the prefix and themselves, and position ids restart at the prefix end. On JevBench-hard, 1.1 flipped 49.5% of answers across 4 orderings; 1.2 flips 0.
  - Several questions are answered in one call (README) (inference: batched rows).
- **Training:**
  - About 290k-item balanced corpus (enterprise 25%, security 20%, safety 15%, linguistic 15%, triage 10%, ANLI/WANLI 15%), then synthetic two-hop and numeric continue-training.
  - Loss: CE + 0.5·Brier (`train_option_marker.py`: lr 3e-5, encoder ×0.5, scorer ×2.5, 3 epochs, batch 8 × accumulation 2), initialized from a `von-modernbert-rlcd` checkpoint (`train_rlcd.py`, also CE + Brier).
- **Calibration:**
  - Input-conditioned T = b + w_H·H_norm + w_len·log10(state_tokens)/4 + w_K·K/8, clamped to [0.3, 12].
  - Noul "band" rule: p′ = 0.8 + 0.1(p−0.5).
  - The map is fit in-sample on JevBench public and "does not carry to other distributions".
- **Results:**
  - JevBench v1.4: I 34.5, C 75.7, composite 27.5, sealed ECE 0.107.
  - Decision Index v0.2.1: 13.74; p50 32.8 ms on A10G.
  - jabr v2: 72.0%.
  - p50 0.096 s on 4 vCPU (OpenVINO), 0.023 s on A10G.
- **Post-mortem** ([docs/von-2.md](https://github.com/wfzyx/von/blob/main/docs/von-2.md)):
  - Seven encoder-side interventions all came back null under a paired McNemar test.
  - DI Full 8.2 (#101 of 114).
  - Von 2 switched to Qwen3.5-0.8B + LoRA r64 + Clef's JointSchemaHead and gained +6.5 pp pooled.
  - A persistent **noul yes-bias** (it picked yes on 224 of the 400 held-out noul items; the expected count was 195) was traced to data: there are no same-state negated-criterion twins.
- **Fixes it claims:** option-order flips (Laya and its own 1.1); a stat-gated reporting discipline.

## 5. Phocinae-Largha-150M-v1
- **Identity:** [github.com/Phocinae/Phocinae-Largha-150M-v1](https://github.com/Phocinae/Phocinae-Largha-150M-v1). Apache-2.0, 2026-10-08, 144.3M parameters, fp16 weights 288.6 MB.
- **Architecture** ([technical report](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/docs/technical-report.md), [config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/config.json)):
  - mmBERT-small: 22 layers × 384 hidden, 6 heads, 256k vocab, local window 128.
  - It uses **Laya's config schema**: `head_layers 2`, `max_len 512`, `head_max_len 192`, `act_costs {escalate: 0.5}` ([rl_agent_config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/rl_agent_config.json)). Inference: it is a Laya-architecture fine-tune.
- **Training:**
  - Data: typed-decisions train split (1,200 cases), plus flip-augmented option reorderings, plus about 2,400 machine-translated and about 1,400 native Chinese rows.
  - Loss CE + Brier; Adafactor; lr 2e-5 (encoder) / 1e-4 (head); **300 updates, 0.044 hours**.
- **Calibration:**
  - Shipped temperatures 0.866 / 0.808 / 0.662 (sharpening).
  - Shipped ECE 0.2519 → 0.0168 with the opt-in `calib/` column, at the cost of **Brier 0.0207 → 0.1218**.
- **Results:**
  - typed-decisions 0.791 (en) / 0.848 (zh). In-distribution, and "not a like-for-like head-to-head" with Laya / Jev.
  - JevBench public 0.5455.
  - Flip rate (any of 3 reorders) 0.0283.
  - 21.0 ms on an RTX 5090; 1.64 s per case on one CPU thread.
  - 16k / 32k context probes: 0.453 / 0.387.
- **Fix claimed:** option-order flips via flip augmentation (versus Laya at 3.7% in-domain and 19.4% out of domain).

## 6. Cloudflare Clef / Clef-flash (a decoder, despite the group name)
- **Identity:**
  - [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (Qwen3.8-27B) and [clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B), Apache-2.0, 2026-09-30. `clef-omni` was added 2026-10-09.
  - Blog: 2026-10-01 ([blog](https://blog.cloudflare.com/clef-decision-models)).
- **Architecture** (`joint_schema_model.py`):
  - One causal prefill over `system + "STATE:" + state + schema + "<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"`. **All questions of a record share one pass.**
  - Head config (`joint_head_config.json`): width 1024, 2 routing layers, 4 layers total, 16 heads.
  - EvidenceRoutingLayers: option queries = proj(option-span mean) + proj(mean of the LM **output embeddings** of the option tokens, the "lexical" vector) + proj(question mean). These cross-attend to the projected sequence memory.
  - Fields cross-attend to each other and to memory.
  - **Logit:** prior + σ(g)·(s·cos(field, option) + MLP[f; o; f⊙o; |f−o|]), where prior = exp(ρ)·cos(lexical_k, question + last-token).
- **Training (blog):**
  - The routing head is trained jointly with rank-256 LoRA. The blog says "frozen" backbone; the card says `base_model_relation: finetune` (inference: LoRA merged into the weights).
  - Loss: label-smoothed CE + Brier.
  - "RLCD" with ordinal partial credit, exact-record reward and a reference penalty.
  - Synthetic data with permuted fields and schemas; sizes not given; no ECE reported.
- **Results (Decision Index 0.2.1, internal run), Clef / flash / Jev:**

| Benchmark | Clef | Clef-flash | Jev |
|---|---|---|---|
| BANKING77 | 94.2 | 90.9 | 79.7 |
| MMLU-Pro | 65.9 | 65.3 | 82.7 |
| BBH | 73.7 | 68.9 | 92.9 |
| ForecastBench Brier | 13.9 | 10.6 | 17.4 |
| Median latency (ms) | 209.3 | 38.8 | 524.1 |

  Von's notes put flash at DI 57.1 versus Kev-9B at 38.5 ("recipe and head beat parameter count").
- **Serving:** SGLang `/v1/systemone` on an H200; Workers AI `@cf/cloudflare/clef`.

## 7. Fastino GLiNER2.5-Decide
- **Identity:**
  - [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide): 340M DeBERTa-v3-large (24 layers × 1024, 512 positions), Apache-2.0, 2026-09-23.
  - Variants: `-1B` (Ettin encoder-from-decoder 1B) and `-multi` (287M).
- **Mechanism** ([arXiv 2507.18546](https://arxiv.org/html/2507.18546)):
  - Input `[P] task ([L] ℓ1 [L] ℓ2 …) [SEP] text`.
  - logit_i = MLP(h_[L]i); softmax for single-label, sigmoid for multi-label.
  - **Several tasks are scored in one encoder pass.**
- **Training:**
  - GLiNER2 base: 254,334 GPT-4o-annotated / synthetic examples; 5 epochs; lr 1e-5 / 2e-5.
  - **Decide-specific training data and calibration are not disclosed.**
  - Ordinal scales are trained "as plain classes" (the card admits this).
- **Results:**
  - fast-decisions (17 domains × 300 held out): 60.2% versus JevK5 57.6% and Laya Router 46.6%.
  - GLiNER2 zero-shot: Banking77 0.70 (GPT-4o 0.78), average 0.72 (GPT-4o 0.84).
  - CPU: 130 ms at 5 labels, 208 ms at 50.

## 8. Intern-Decision (InternLM), a causal model, not an encoder
- **Identity:**
  - [github.com/InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision): Apache-2.0 code; weights 0.8B / 2B / 4B on Qwen3.5.
  - Training data and weights are not in the repo.
  - Paper: [arXiv 2610.22131](https://arxiv.org/html/2610.22131).
- **Method:**
  - The assistant turn holds a JSON skeleton, e.g. `{"team": "<decision>", …}`.
  - Eq. 1: z_{p_j−1} = LMHead(h_{p_j−1}).
  - Eq. 2: full-vocabulary CE at the shifted positions.
  - Eq. 3: P_j(c|x;T) = softmax over the legal candidates of z/T.
  - Options are letters (62 symbols); up to 16 fields per request; one causal pass for all fields.
  - "Earlier placeholders are visible to later fields… but their gold or predicted answers are not fed back."
  - **No option-order randomization**: "Field and option order remain unchanged."
- **Training (Table 9 / 10):**
  - 17,880 samples / 23,500 decisions, 5,500 with images.
  - Synthetic text from a deepseek-v4.1-flash teacher.
  - **Public data includes AG News, ToolACE, Typed Decisions and WildJailbreak**, which are four of its seven evaluation suites (inference: in-distribution).
  - 2 epochs / 156 updates; lr 2e-6; batch of 16 packed 8k sequences; 4 GPUs; vision tower frozen ([configs/training/qwen35.py](https://github.com/InternLM/Intern-Decision/blob/main/configs/training/qwen35.py)).
- **Calibration:** one scalar T per checkpoint, fit by NLL bisection over [0.01, 100] (`src/eval/fit.py`): 0.8B 2.748, 2B 2.101, 4B 1.992.
- **Results:**
  - 4B averages 90.02 versus Jev 88.74.
  - JevBench-Hard Brier 0.347 / ECE 0.065.
  - Known-distribution pilot: 0.550 / 0.089 versus Jev 0.595 / 0.130.
  - 44.16 ms on an RTX 4090 (289 tokens, 3 fields).
  - Von's notes report Decision Index Full 11.8 for the sub-1B model.
- **Limitations it admits:** no matched-data ablations, no comparison with a classification head, and latency without a stated protocol.

---

## 9. Cross-system synthesis for a from-scratch builder

**Passes per request.**
- One pass for all questions: Clef, Intern-Decision, GLiNER2.
- One encoder row per question: Laya (inference: Von and Phocinae likewise).
- One reader pass per question unless the state is ≥4k tokens: vega.

The causal "state first, schema after" layout (Clef, Intern-Decision) is the only one that also allows prefix-KV reuse (inference).

**Flaws and the systems that fix them:**

| Flaw | Fix | Evidence |
|---|---|---|
| Option-order flips (Laya 0.15 at K=20; Von 1.1 49.5%) | Order-invariant masks + reset position ids | Von 1.2: 0 flips |
| Option-count collapse (Laya K=77) | Avoid a fixed per-option token budget | Clef 94.2 on BANKING77 |
| Noul label-word bias (Laya #156; Von yes-bias) | Same-state negated twins, neutral option keys | Von polarity data / r4 hinge loss (no result reported) |
| Over-confident shipped temperatures | Clamp T, refit on held-out data, report ECE before and after | Laya #394 |
| A "dead" auxiliary head | Normalize its inputs | Laya #185 |

**Adopt:**
- Proper-score losses as **plain soft-CE + Brier**. Skip the noisy-logit PG: #741 shows it is equal or worse, and the Crossref abstract agrees.
- Shuffle options during training.
- Fit temperatures per type (and per K only when enough data exists) on a held-out split.
- Split-conformal sets and selective-risk abstention (vegaml).
- Paired McNemar gates for every claim (Von).
- Clef-style lexical prior from the output embeddings.

**Avoid:**
- Benchmarking on the training distribution (Intern-Decision, Phocinae, vega-U's "held-out").
- Confidence = normalized entropy.
- Fixed per-option token budgets.
- Encoder-only small backbones for reading-heavy tasks: Von's post-mortem and the Decision Index results put the top tier at ≥9B decoders.
