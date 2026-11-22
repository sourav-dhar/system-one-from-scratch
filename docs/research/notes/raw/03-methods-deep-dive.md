# Method-level deep dive: System One decision models and what each means for our CLM-style bi-encoder

**What I read.** I read the primary sources: the arXiv HTML pages, the repo code and READMEs, and the HF cards. The prior study is [landscape.md](00-landscape-full.md).

**What I could not fetch:**
- The CLM Notion blog returned only the word "Notion".
- The Laya preprint on preprints.org returned HTTP 403.
- The S1-Arena appendix on how CLM was queried did not appear in any of the text I fetched.

Where a paper is silent on something, I say so.

---

## 1. System1-Arena (arXiv 2610.21244)

**What it measures.**
- 87 non-agentic tasks across 13 domains, plus agentic step accuracy and closed-loop success over 127–138 environments ([HTML](https://arxiv.org/html/2610.21244)).
- CLM inputs: "for CLM we keep the last 2,048 tokens of each text", described as "the end that its training keeps". Laya is cut to 512 tokens. No other model's input is shortened ([HTML](https://arxiv.org/html/2610.21244)).
- CLM's own server truncates states longer than 2,048 tokens by default ([CLM README](https://github.com/Contrastive-LM/CLM)).

**Where CLM fails.** The paper gives per-domain numbers but **no explanation of why CLM fails** ([HTML](https://arxiv.org/html/2610.21244)).

| Domain | CLM-v0.1-8B | Qwen3-8B (letter readout) | Gap |
|---|---|---|---|
| Social (subjective judgments) | 22.1 | 46.8 | −24.7 |
| Pro (professional domains) | 28.2 | 59.2 | −31.0 |
| Classification | 35.6 | 56.6 | −21.0 |
| Forecasting | 35.5 | 39.4 | −3.9 |
| Struct (structured data) | 40.7 | 63.5 | −22.8 |
| Batch | 42.2 | 68.7 | −26.5 |
| Safety (CLM's best) | 54.7 | 69.5 | −14.8 |
| Route (CLM's second best) | 49.6 | 54.0 | −4.4 |

- Overall: CLM 41.4 vs Qwen3-8B 59.3 vs Jev 72.6.
- Brier: CLM 0.749 vs Qwen3-8B 0.715.
- **(inference)** CLM loses least where tasks look like retrieval or action matching (Route, Fore, Safety, Know). It loses most where the answer depends on applying a rubric or instruction to the state (Pro, Social, Struct, Batch).

**What correlates with accuracy.** The paper computes **no correlation coefficients**. Its observations ([HTML](https://arxiv.org/html/2610.21244)):
- **Post-training vs base model.** System-1 post-training improves non-agentic accuracy over the base model in 3 of 5 pairs:
  - Kai-0.6B: +11.0 pp and Brier −0.400 over Qwen3-0.6B-Base.
  - MS-Decision-1: +5.6.
  - Vega: +2.9.
  - CLM: −17.8, and it is the only pair whose Brier gets worse (+0.034).
- **Agentic step accuracy.** All five System-1 models lose step accuracy against their base (−1.2 to −19.3). CLM loses the most.
- **Size and loop success.** Open LLMs under 10B reach under 6% loop success.
- **Reasoning.** On Qwen3.8-27B, reasoning raises accuracy from 70.2 to 75.6 but worsens Brier from 0.391 to 0.430, with mean confidence going from 81.4% to 97.4%.
- **Overconfident readouts.** Gemma-4-31B puts ≥0.999 on its chosen option for 87% of questions.

**Agentic results.**
- CLM: step accuracy 40.6%, loop success 1.7%.
- Equal step accuracy can hide large loop differences: Vega, Jev and Qwen3.5-9B all score 67.8% step accuracy but reach 26.5%, 17.4% and 5.7% loop success.
- In the realtime mode, CLM's 41 ms latency did not turn into loop success ([HTML](https://arxiv.org/html/2610.21244)).

**What it means for us.**
1. **(inference)** A 0.6B joint-encoding model (Kai, 50.3%) beats our frozen 8B bi-encoder. The deficit therefore comes from the architecture and the training objective, not from capacity.
2. Our main metric must be the same-backbone letter readout at 59.3%.
3. Report Brier and loop success, not only top-1 accuracy.
4. Lift the 2,048-token cap. Cost: small.

## 2. Jeff-1.1 / "Releasing the Inner Jev" (arXiv 2610.17205)

**Strategy** ([HTML](https://arxiv.org/html/2610.17205)).
- **Readout:** read the decision at the last prompt position, using a chat template with thinking disabled. Options are labeled with letters A–P. A softmax runs over the first K of the 16 letter rows of the LM head.
- **More than 16 options:** split into groups of up to 16 plus a final round, up to 17 passes. The authors say this mode was not evaluated systematically.
- **Head:** initialized from the LM-head letter rows, plus a trained correction A·std(h)+b that starts at zero.
- **LoRA:** r32 / α64 on q/k/v/o and gate/up/down, plus Qwen's in_proj and out_proj. LR 2e-5 for LoRA and 1e-4 for the head. 1,500 steps, WSD schedule, inputs capped at 8,192 tokens, options shuffled during training.
- **Soft targets:** the teacher is Qwen3.8-27B with reasoning, averaged over 2 samples.
  - Public labeled data: 50/50 mix of the dataset label (or human vote distribution) and the teacher.
  - LLM-written questions: teacher only.
  - Programmatic questions: exact targets.
- **DecisionBank (261,561 rows):** 163,732 decider/custom, 9,168 human-vote, 60,219 LLM-written, 28,442 programmatic.
- **Temperature:** one per type (choice, yes/no, score), fit by NLL on held-out task families and constrained to T ≥ 1. Yes/no probabilities strictly inside (0.2, 0.8) are snapped to 0.199 or 0.801.
- **Weight soup:** the average of 3 seeds. Its gain is not isolated.
- **Bidirectional attention:** used for Gemma-4-12B, Qwen3.6-27B and Qwen3.5-4B, but not for Gemma-4-31B.

**Flaws fixed, with ablation numbers** (Gemma 4 31B, reference 84.4) ([HTML](https://arxiv.org/html/2610.17205)):

| Variant | Capability | Δ vs reference |
|---|---|---|
| Zero-shot | 79.6 | −4.8 |
| **Head only, frozen backbone** | **78.9** | **−5.5** |
| LoRA, head frozen at init | 84.9 | +0.6 |
| Base (not instruction-tuned) model + recipe | 70.3 | −14.1 |
| No LM-labeled data | 82.5 | −1.9 |
| Hard labels instead of soft | 84.1 | −0.2 |
| LoRA rank 16 / 64 / 128 | 84.7 / 84.5 / 84.9 | flat |

Other ablations:
- **Temperature scaling alone** took zero-shot Gemma-4-31B from 72.9 to 79.6 Capability.
- **Bidirectional attention:** −0.2 ± 1.1, so no gain.
- **Order-invariant (option-parallel) encoding:** −0.6 to −2.4 against the letter format.
- **Calibration losses** (Brier, focal, soft-ECE, MMCE): none lowers ECE by more than 0.01 after temperature scaling.
- **Expected-utility loss:** +0.9 ± 0.9 accuracy, and the ECE gain disappears after temperature scaling.
- **Zero-shot size curve (Qwen3.5):** 0.8B 48.9 → 4B 76.8 → 9B 82.2 → 27B 86.5.

**What it means for us.**
- **(inference)** "Head only on a frozen backbone" is the closest analog to our CLM recipe, and it scored *below zero-shot readout*. The gain came entirely from adapting the backbone (LoRA).
- Concrete moves:
  - Build the letter-logit readout on Qwen3-8B as both a baseline and a slow path. Cost: about 1 day.
  - Adopt the temperature rules: per type, T ≥ 1, fit on held-out *task families*. Cost: about 1 day.
  - Use DecisionBank's soft targets as training data.
- Letter readout is joint encoding. It cannot reuse cached option embeddings, so it would serve as a second path, not as a replacement.

## 3. Jeva, "System One Models from First Principles" (arXiv 2610.18773)

**Architecture** ([HTML](https://arxiv.org/html/2610.18773)).
- **Encoder:** Qwen3.5-4B, pooled at the last non-padding token (width 2,560), with **LoRA r64 on the backbone**. The encoder is not frozen.
- **Inputs:** the request template holds decision type, question and state. The option template holds **decision type and option text only**.
- **Projections:** bias-free projections give a request vector z₀ and option vectors uᵢ ∈ ℝ⁷⁶⁸. The option vectors are ℓ2-normalized.
- **Decision stack:** one block (L=1), 8 heads, d=768, SwiGLU FFN width 3,072.
  - Queries come from the request state. Keys and values come from the fixed option memory.
  - **Options never attend to each other.** All option interaction passes through the state.
  - Two scaled residual updates: an attention message scaled by α, then an RMSNorm-SwiGLU FFN scaled by L^(−1/2).
  - 13.37M trainable parameters in the projections and core.
- **Scoring:** sᵢ = z_Lᵀuᵢ, then q = softmax(s/τ_dec). This is permutation-equivariant by construction.
- **Coverage head:**
  - ρ = σ((maxᵢ sᵢ − b_cov)/τ_cov), p⊥ = 1 − ρ, pᵢ = ρ·qᵢ.
  - Loss: BCE(logit ρ, c_y) − Σ yᵢ log qᵢ, where c_y says whether any candidate is correct.
- **Noul:** two cached Boolean candidates, "false" and "true".
- **Score:** rubric levels as options, output Σ qᵢvᵢ. Auxiliary losses: squared error plus a margin for Noul; expectation error plus a CDF term for Score.
- **Curriculum:** about 10 stages, each roughly 2k–500k examples.
  - One stage blends 7:3 original:teacher targets, accepted only when the teacher agrees under two shuffled orders.
  - Replay with earlier-model probabilities.
  - No post-hoc temperature scaling is described.
- **Cache:** per option it stores uᵢ plus the per-layer, per-head k and v. The request side is never cached.

**Flaw fixed.**
- With independent scoring, the log-odds between two options are (gᵢ − gⱼ)/τ. Adding or removing other options therefore cannot reverse a preference (Prop. 1). This is the IIA / Luce property.
- One cross-attention block can produce set-dependent choices ([HTML](https://arxiv.org/html/2610.18773)).

**Numbers** (Table 1; CLM-8B given an explicit "None of the above" candidate) ([HTML](https://arxiv.org/html/2610.18773)):

| Model | Typed | BANKING77 K=16 | MASSIVE K=16 | Mind2Web task |
|---|---|---|---|---|
| CLM-8B | 43.05 | 18.15 | 23.23 | 21.73 |
| Jeva-4B | 78.30 | 91.62 | 90.22 | 79.99 |
| InternDecision-4B | 78.70 | – | – | – |
| Kev-4B | 66.00 | 82.47 | – | – |

- **Null detection:** AUROC 98.66%, vs Kev 93.61% and InternDecision 84.27%.
- **Permutation flips:** 0 on 600 items, vs Jev 17.0%.
- **Candidate scaling:** CLM stays correct at every size on only 7 of 83 queries, vs Jeva 58. At K=2048, Jeva scores 69.88%.
- **Reuse:** p50 falls from 2,102 to 75 ms at full option reuse (MLDR). Shared Typed banks avoid 99.2% of candidate encodings, with max probability difference 0.007.
- **No component ablations.** Gains from LoRA, the stack and the data cannot be separated.

**What it means for us: the most direct upgrade path.**
1. Keep the two towers. Put the decision type into the option template.
2. Replace cosine × capped scale with an unnormalized z_L against normalized uᵢ. **(inference)** The norm of the request vector then acts as a per-input temperature.
3. Add one cross-attention block (state queries, cached option k/v) and a coverage head.
4. Add LoRA on the backbone.

- **Caching is preserved:** cache uᵢ and the k/v projections. All option caches must be invalidated after any LoRA update.
- **Effort:** about 1 week for the stack and coverage head on frozen features, plus 2–3 weeks for LoRA and the curriculum **(inference)**.

## 4. Kev (github.com/jaredpalmer/kev)

**Strategy.**
- **Pointer head** ([model.py](https://raw.githubusercontent.com/jaredpalmer/kev/main/kev/model.py)):
  - `q, k = Linear(d,256), Linear(d,256)` with bias and no normalization.
  - z = k(h_{</opt>}) · q(h_{<decide>}) / √256.
  - A temperature is applied only at eval.
  - The `<decide>` token is last, so it sees every option.
- **Masking:** block-causal. Each token sees the state and its own question. Position IDs restart after the state. An optional "option isolation" mode gives option spans shared position IDs, and each option sees only state, instruction and itself.
- **Hybrid DeltaNet bases** cannot use masks. There each question is its own row continuing from a cached state ([README](https://github.com/jaredpalmer/kev)).
- **LoRA:** r16 / α32 on attention, MLP and DeltaNet projections (33.8M parameters). Kev-27B is fully fine-tuned ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- **Curriculum** ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)):

  | Stage | Data | Settings |
  |---|---|---|
  | 1 | decision-v7, 12,576 records | 2 epochs, LR 5e-5 |
  | 2 | Dates / missing evidence, 1,425 | replay 2,000 |
  | 3 | Real documents, 5,219, states ≤7,552 tokens | replay 2,000 |
  | 4 | Skills, 6,000 + 5,320 | replay 4,000 |

- **Augmentation:** shuffled options, random "none of the above" and distractor options. A quarter of choice records also become a **minimal pair**: the same question with the correct option present, and with it removed (`p_none_pair 0.25`).
- **Temperature:** one scalar per checkpoint (T = 2.41 for 4B).

**Flaws found, with evidence.**
- **Buried evidence:** placing the state among 1–4k tokens of unrelated records cost 22–43 pp. Training on such states recovered +17 to +20 pp ([PLAN.md](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).
- **Diminishing data returns:** the first 6k hard-v1 records gave +26 pp; the next 12k gave +5 pp.
- **Soft targets:** ambiguity soft targets beat hard labels (Brier −0.011, accuracy +1.1).
- **Data beats method:** full fine-tuning on Kev's own data gained nothing and raised ECE by about 0.045. Broad data gave +9.1 pp.
- **Temperature fitting:**
  - Fitting T on held-out items of the *training* sources is a declared failure mode.
  - Fitting on held-out *datasets* gave breadth ECE 0.0085.
  - Per-(type, K) temperatures and a "reliability head" were *worse*.
- **Reported negative (no deltas given):** perm_kl, ord_w, option isolation, special embeddings.
- **Confident errors:** Kev-9B's confident-error rate on new sources fell from 8.2% to 2.4% (Jev 3.7%). On unknowable items Kev answers at ≥0.9 confidence 0% of the time; Jev does 9% ([README](https://github.com/jaredpalmer/kev)).
- **Option order** can still change answers.

**What it means for us.**
- The pointer head is a joint, non-cacheable path. Our cacheable analog is Jeva.
- Directly portable:
  - minimal-pair "none" augmentation;
  - buried-evidence training states;
  - staged curriculum with replay (CLM already showed that replay held hard-negative top-1 at 68.5% vs 56.2% without it ([CLM README](https://github.com/Contrastive-LM/CLM)));
  - temperature fit on held-out datasets.
- Cost: small to medium, with no change to the cache.

## 5. Laya (huggingface.co/convaiinnovations/laya, vegaml)

**Encoder model** ([HF](https://huggingface.co/convaiinnovations/laya)).
- ModernBERT-large, fully fine-tuned. Each option gets a `[MASK]` token, followed by 2 transformer layers and an option-marker scorer.
- **RLCD training:**
  - the policy outputs a distribution, with zero-mean Gaussian noise added to the logits (σ not given);
  - rewards are log and spherical scores, plus RPS for ordinal questions;
  - REINFORCE with a group-mean baseline; TD(λ=1) for multi-turn.
- **Per-(type, K) temperature refit:** mean ECE 0.466 → 0.081.
- **Conformal sets:** split conformal at α = 0.10. Abstention keeps the error rate among answered items at or below 10%.

**vegaml frozen reader** ([vegaml repo](https://github.com/NandhaKishorM/vegaml)).
- **Reader:** a frozen LLM reads one joint prompt, `Observation… Measurement… Possible outcomes… Outcome:`. Features come from **two intermediate layers**, and the layers above them are cut off.
- **Pooled vectors:** the state span, question span, each option span and the final token. A linear "port" per layer maps into a shared 1024-d space.
- **Head:** a 64-d "particle" settles among Gaussian wells, one per option. P = softmax(−E/τ).
- **Per-decision temperature:** log τ is predicted from the residual kinetic energy, the distance to the nearest well, and log(#options).
- **Results by reader** (held-out split of 50,676 records):

  | Reader | Held-out acc | ECE | Zero-shot acc |
  |---|---|---|---|
  | Qwen3.5-0.8B | 0.765 | 0.010 | 0.534 |
  | Qwen3.5-4B | 0.791 | 0.017 | 0.568 |
  | Qwen3.8-27B | 0.811 | 0.021 | 0.629 |

- **Per-question learner:** small heads over frozen features, refit every 50 confirmed answers. 84.7% → 96.0% after 120 answers, vs Jev 89.5% → 92.6%.

**Flaws exposed in Laya itself.**
- Zero-shot accuracy is near chance (0.362) ([HF](https://huggingface.co/convaiinnovations/laya)).
- Swapping yes/no names flips 91.3 of 100 answers ([2610.00346](https://arxiv.org/html/2610.00346)).
- Accuracy collapses from 0.852 at 20 options to 0.020 at 50, because option text overflows the input budget ([2610.00346](https://arxiv.org/html/2610.00346)).

**What it means for us.** vegaml is the closest frozen-reader precedent, but it reads options jointly with the state. Ideas we can take while keeping the cache:
- **Intermediate-layer taps.** **(inference)** A cheap ablation that also cuts latency.
- **Per-decision temperature from features including log K.**
- **Conformal sets and abstention** on top of calibrated probabilities. Cost: about 2 days.
- **Per-customer few-shot heads** over cached frozen features.

## 6. Bongard (arXiv 2609.39111) and SanSi (arXiv 2610.07730)

### Bongard

**Strategy** ([HTML](https://arxiv.org/html/2609.39111)).
- Backbone: T5Gemma-2 4B-4B. The state is encoded once by a **bidirectional encoder**, then shared as cross-attention K/V across per-question decoder branches.
- From stage 3 on, the encoder also reads the questions' instructions.
- **Bilinear head:** zᵢ = wᵀhᵢ + (W_c hᵢ)ᵀ(W_g g)/√256, where hᵢ is the hidden state at a candidate-end marker and g at the decision-end marker.

**Training stages.**
1. Supervised judgments.
2. A "JEPA" alignment stage. It uses **pack centring** because raw cosine between unrelated readouts was **0.93–0.996 (saturated)**. λ = 0.004.
3. Sandbox action-outcome training with a proper-scoring cross-entropy.

**Ablations.**
- **Encoder mask (1B probe):** bidirectional 69.8% / NLL 0.677 vs forward-only 64.6% / 2.944.
- **Stage 2:** re-expression accuracy 75.7 → 85.9.
- **Stage 3:** panel accuracy 50.6 → 64.8, Brier 0.482 → 0.289. Table 10, stage 2 → final:
  - hard-style decisions 34.9 → 50.5;
  - frontier-style 45.2 → 73.9;
  - fair-chance KL from uniform 0.173 → 0.0009.
- **Calibration:** per-primitive, per-K temperatures of 1.2–2.15. External ECE 0.141 → 0.030.
- **Encoder cache:** a repeated 1,924-token request goes from 1,165 to 115 ms.

### SanSi

**Strategy** ([HTML](https://arxiv.org/html/2610.07730)).
- A frozen looped LM (Ouro-1.4B, 24 layers), LoRA r64, and a letter readout after **every** loop.
- The loss is CE + Brier averaged over loops.

**Ablations.**
- +13.5 pp over a same-shape non-looped model (72.0 vs 58.4).
- Loss on the last loop only: loop-1 accuracy falls to 35.6.
- CE without Brier: ECE 0.093 → 0.104.
- RL instead of SFT: no gain (71.7).
- Adding loops to a model not pretrained to loop: no gain.
- **Averaging the 8 loops' predictions halves ECE to 0.044.**
- Gains concentrate in multi-step reasoning (+15.1), long documents (+16.7) and knowledge (+17.6).

### What these mean for us
- **Bongard (inference):**
  - Saturated raw cosine argues for centring or whitening the pooled Qwen states, or at least checking the cosine distribution of our 512-d outputs.
  - The bidirectional-encoder gain argues for testing an LLM2Vec-style bidirectional fine-tune of the state tower.
  - Action-outcome data from sandboxes is the best-evidenced data move for agentic decisions.
- **SanSi:** looping does not transfer to Qwen3-8B. What we can test is reading heads at several depths and averaging them, with a loss on each exit. Cost: small, because we already extract hidden states **(inference)**.

## 7. CLM itself (github.com/Contrastive-LM/CLM)

**How typed questions become text** ([schema.py](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/schema.py), [engine.py](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/engine.py)):
- **State side:** `f"{state}\n\n{instructions}"`. The question is appended to the state text.
- **Choice:** each action is the option description, or the bare key. It carries **no question or type context**.
- **Noul:** the actions are `true` / `false`, rendered as "true: desc", or by default "Yes. This is true: {instructions}" and "No. This is false: {instructions}".
- **Score:** the actions are the level texts, with score = Σ i·pᵢ. There is **no ordinal structure in the loss**.

**Scoring** ([heads.py](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/heads.py)).
- Score = exp(logit_scale) clamped at 100, times the cosine of the ℓ2-normalized projections.
- The head defaults are depth 2, GELU, no LayerNorm, projection 512.

**Data** ([README](https://github.com/Contrastive-LM/CLM)).

| Stage | Data |
|---|---|
| Pre-training | ~60M Nemotron DQA pairs |
| Mid-training | ~30M Gemini-2.5-Flash-Lite hard negatives |
| Post-training | ~1M trajectories (ADP, terminal traces), mixed 40% DQA replay / 60% agentic |

- Ablations: 52.1% after pre-training, 69.2% with mid-training, 62.4% peak when hard negatives are used from the start.

**Stated limitations.**
- States longer than 2,048 tokens are truncated.
- Heads are tied to the encoder they were trained on.
- Probabilities are relative to the candidate set.
- The headline verifier numbers come from fine-tuned heads.

**35B variant.** The model card says a multimodal CLM-35B is "coming in early October" ([HF](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)). One news piece calls it CLM-35B-A3B MoE ([pasqualepillitteri](https://pasqualepillitteri.it/en/news/19074/clm-8b-nvidia-stanford-jev-en)). I found **no evidence of a release**.

**What it means for us (inference).**
- CLM was **never trained on typed, rubric or classification decisions**. Its Q&A and trajectory data teach "which answer matches this question", not "apply instruction X to state Y". That matches the S1-Arena domains where it does worst.
- The Noul template makes yes and no near-paraphrases that differ by one word, which a cosine bi-encoder separates poorly. This fits TypedBench's finding that encoders largely ignore negation (§8).

## 8. TypedBench (2610.11392), "DeBERTa vs System One" (2610.00346), Runoff (2610.18233)

**TypedBench** ([HTML](https://arxiv.org/html/2610.11392)). Models evaluated: Jev, Laya, and Kev 0.8B/4B/9B. **CLM was not evaluated.** Findings:
- Jev is **underconfident**: its fitted temperatures are below 1, and on ticket priority its ECE is 2.2× the noise floor.
- **Wording:** Jev's priority accuracy swings from 0.708 to 0.946 across friendly wordings.
- **Negation:** a paraphrased phishing negation costs Jev 23 pp. Encoders answer as if the negation were absent.
- **Score scales:** encoders compress ordinal scales. Laya keeps order on 1 of 4 ladders at 5 and 10 levels.
- **Many options:** both encoders collapse before 100 options.
- **Embedding baseline** (MiniLM nearest description): below the majority prior on anger (0.44 vs 0.71) and priority (0.20 vs 0.34).
- **Decisions vs argmax:** Bayes-cost decisions from Jev's probabilities cost more than argmax (2,300 vs 1,080 per 10⁴ decisions).

**2610.00346** ([HTML](https://arxiv.org/html/2610.00346)).
- **Fine-tuned DeBERTa-v3-large leads** on most labeled tasks:
  - typed-decisions 0.793 vs Jev 0.732;
  - CLINC-150 0.953 vs Jev 0.913;
  - emotion 0.938 vs Jev 0.504.
- **Yes/no rename flips per 100:**

  | Model | Flips per 100 |
  |---|---|
  | Laya | 91.3 |
  | Jev | 50.5 |
  | Generative readouts | ≥16.8 |
  | Fine-tuned decision models | ≤6.5 |

- **Stored temperatures break at large K:** Kev-9B's ECE goes from 0.051 at T=1 to 0.377 as shipped, at 150 options.
- **Jev at scale:** accuracy falls from 0.990 to 0.913 as options go from 5 to 150, and it accepts 0.310 of out-of-scope requests.

**Runoff** ([HTML](https://arxiv.org/html/2610.18233)).
- **How it works:** score each option alone, keep the top set (α = 0.99, at least 3 options), then run a listwise pass on every cyclic shift and average.
- **Order invariance:** 0.0% flips on open models. Listwise Qwen3-8B flips 17.0% at K=20.
- **Where flips happen:** over 90% involve the top 3 pointwise candidates. Flip rates exceed one-third when top pointwise probability is below 0.8.
- **Speed:** 3–4× faster than cyclic averaging at K=20.
- **Weakness:** it trails cyclic averaging on MMLU-Pro, because the shortlist misses the answer.
- **Agents:** OSWorld 27 vs 21 tasks with Qwen3.5-35B-A3B.

**What it means for us.**
- Our bi-encoder is already order-invariant, which is an advantage, but it is likely weak on negation and polarity. Train on polarity-swapped pairs and use neutral option IDs.
- Add an ordinal loss (RPS or CDF) for Score.
- Calibrate per K, or as a function of log K.
- Runoff's two-stage shape fits our design well **(inference)**: the cached bi-encoder becomes the pointwise "consider" stage, and a joint reranker ("choose") runs only on the uncertain shortlist.
- Include a DeBERTa baseline on labeled tasks.

---

## Synthesis: the 8 design moves that matter most, ranked by expected gain per unit cost

**1. Calibrate per type, as a function of K, and fit on held-out datasets. Cost: days. Cache preserved.**
- Temperature scaling alone added +6.7 Capability to zero-shot Gemma ([Jeff](https://arxiv.org/html/2610.17205)).
- Laya's per-(type, K) refit took ECE from 0.466 to 0.081 ([HF](https://huggingface.co/convaiinnovations/laya)). Our own one-temperature fit took ECE from 0.065 to 0.013.
- Stored temperatures fail at K=150 ([2610.00346](https://arxiv.org/html/2610.00346)), and fitting on in-distribution held-out items is a known failure mode ([Kev PLAN](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).
- Recipe:
  - Drop the scale cap of 100.
  - Fit T(type, log K) with T ≥ 1, and validate on unseen datasets.
  - Add conformal sets and abstention ([vegaml](https://github.com/NandhaKishorM/vegaml)).

**2. Retrain the towers on typed-decision data with soft targets, with type-aware templates on both sides. Cost: 1–2 weeks. Cache preserved.**
- CLM has no typed or rubric training data, and its option tower sees bare option text ([schema.py](https://raw.githubusercontent.com/Contrastive-LM/CLM/main/src/clm/schema.py)).
- Jeva puts the decision type into option templates ([Jeva](https://arxiv.org/html/2610.18773)).
- Evidence for soft and LM-labeled data: removing LM-labeled data cost −1.9 ([Jeff](https://arxiv.org/html/2610.17205)); soft targets beat hard labels by +1.1 accuracy and −0.011 Brier ([Kev PLAN](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).
- Data sources: DecisionBank (261k items).
- Additions:
  - Kev-style minimal-pair "none" items ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md));
  - polarity-swap and negation pairs;
  - an RPS or CDF ordinal loss for Score ([Laya](https://huggingface.co/convaiinnovations/laya), [Jeva](https://arxiv.org/html/2610.18773));
  - keep Q&A replay ([CLM](https://github.com/Contrastive-LM/CLM)).

**3. Add a same-backbone letter-logit readout as the slow path, behind a calibrated-confidence cascade. Cost: about 1 day zero-shot, about 1 week with LoRA. The cache is kept for the fast path.**
- Same-backbone readout scores 59.3 vs CLM's 41.4 ([S1-Arena](https://arxiv.org/html/2610.21244)).
- Cascades reach near-teacher accuracy at much lower cost: up to about 54% saving in System One Auto (landscape), and Gemma-level accuracy at about Jev's price ([2610.00346](https://arxiv.org/html/2610.00346)).
- Runoff's "consider, then choose" shape suits a bi-encoder first stage ([Runoff](https://arxiv.org/html/2610.18233)).

**4. LoRA on the backbone, used inside both towers. Cost: 2–3 weeks plus a full re-embed. Cache preserved, but invalidated per checkpoint.**
- Head-only training on a frozen backbone scored *below* zero-shot (78.9 vs 79.6). LoRA scored 84.4 ([Jeff](https://arxiv.org/html/2610.17205)).
- Jeva's LoRA'd dual encoder reaches 78.3 Typed vs CLM's 43.05 ([Jeva](https://arxiv.org/html/2610.18773)), although this is confounded with its stack and data.
- **(inference)** This is probably the largest single accuracy lever. It costs us the "frozen, shared backbone" property.

**5. A Jeva-style late-interaction block plus a coverage head. Cost: about 1 week on frozen features. Cache preserved (cache uᵢ plus k/v).**
- Fixes the IIA limitation of independent scoring (Prop. 1).
- Gives a real null outcome: AUROC 98.66 vs Kev 93.61 ([Jeva](https://arxiv.org/html/2610.18773)).
- Adds a per-request confidence scale through the norm of the unnormalized z_L **(inference)**.
- No isolated ablation exists, so we must run our own.

**6. Raise the context limit to ≥8k and train on buried-evidence states. Cost: days of serving config plus data.**
- CLM truncates at 2,048 tokens ([README](https://github.com/Contrastive-LM/CLM)), and S1-Arena fed it only the last 2,048 tokens ([S1-Arena](https://arxiv.org/html/2610.21244)).
- Kev: burying costs 22–43 pp, and training on it recovers 17–20 pp ([Kev PLAN](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).

**7. Representation hygiene: Matryoshka loss, centring, and multi-layer or multi-exit readouts. Cost: days. Cache preserved and smaller.**
- Our 64-d truncation drop (69 → 51) means no Matryoshka training was done.
- Raw readout cosine is saturated at 0.93–0.996, and centring fixes it ([Bongard](https://arxiv.org/html/2609.39111)).
- Averaging readouts across loops halved ECE ([SanSi](https://arxiv.org/html/2610.07730)).
- Intermediate-layer taps underpin vegaml ([vegaml](https://github.com/NandhaKishorM/vegaml)).

**8. Outcome-grounded agentic data with proper-scoring losses. Cost: weeks (sandboxes).**
- Bongard's stage 3 moved its panel from 50.6 to 64.8 and Brier from 0.482 to 0.289 ([Bongard](https://arxiv.org/html/2609.39111)).
- S1-Arena shows that step accuracy does not predict loop success, and CLM reached 1.7% loop success ([S1-Arena](https://arxiv.org/html/2610.21244)).
- Exotic calibration losses are **not** worth it. After temperature scaling they changed ECE by ≤0.01 ([Jeff](https://arxiv.org/html/2610.17205)). CE plus Brier is enough ([SanSi](https://arxiv.org/html/2610.07730)).

**Deprioritized, on the evidence:**
- **Bidirectional attention in a decoder:** −0.2 ([Jeff](https://arxiv.org/html/2610.17205)). It helped only in Bongard's encoder probe.
- **LoRA rank:** flat across 16 to 128.
- **perm_kl, ord_w, reliability heads:** negative in Kev ([PLAN](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).
- **RL instead of SFT:** no gain in SanSi.
- **Weight soup:** no isolated gain reported.
