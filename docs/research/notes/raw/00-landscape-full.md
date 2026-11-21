# System One landscape study: Jev, "Kev", "Laya", "Inkling" and related decision models

Date of study: 2026-11-20. Scope: the decision-model ("System One") ecosystem around our CLM-8B rebuild. Every factual line has a source link. Lines marked **(inference)** are my reasoning, not sourced fact. Most sources are from Sept to Nov 2026. Many are vendor or secondary write-ups; I note this where it matters.

---

## 1. Executive summary

1. **The names, resolved.**
   - **"Kev" is a real System One model family.** It is not only a typo for Jev. Jared Palmer's **Kev** (Kev-0.8B/4B/9B/27B, Apache-2.0) is an open "Jev-like" decision model family on Qwen3.5/3.8 that speaks TypeSafe's `/v1/systemone` wire format ([repo](https://github.com/jaredpalmer/kev), [Kev-4B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)). A smaller, unrelated `arjun988/Kev` project wraps any Ollama or OpenAI-compatible LLM ([repo](https://github.com/arjun988/Kev)). **Jev** (TypeSafe) is the closed original ([docs](https://docs.typesafe.ai/concepts/system-one.md)).
   - **"Laya" is clearly identified.** It is Convai Innovations' open (Apache-2.0) ModernBERT-based decision model, 421M parameters ([HF card](https://huggingface.co/convaiinnovations/laya)).
   - **"Inkling" was identified as a model, but not as a System One model.** It is Thinking Machines Lab's 975B-total/41B-active open-weight generative MoE, released 2026-07-15 ([TML post](https://thinkingmachines.ai/news/introducing-inkling/)). I found **no** Inkling decision, verifier or reward variant. It is relevant to us only as a possible teacher or labeler, and it appears as generator metadata on the Jeva paper page ([HF papers](https://huggingface.co/papers/2610.18773)).
2. **The field has exploded since Jev launched (2026-09-15).** It now includes:
   - open families: vLLM-SR Decision 2.0 (0.6B to 27B), Kev, Laya, Jeff-1.1, Bongard and SanSi;
   - hosted APIs: Microsoft-Decision-1 and the OpenAI Decisions API;
   - independent benchmarks: System1-Arena, JevBench, the Jev Decision Index, OpenDecisions and TypedBench.
3. **The most important finding for us: CLM-8B is weak zero-shot on typed or rubric-style decisions.** Our rebuild is CLM-8B.
   - **System1-Arena:** CLM-v0.1-8B scored 41.4% accuracy with Brier 0.749, the lowest of 10 System-1 models, though it was also the fastest at 41 ms. It lost 17.8 pp against simple answer-token readout from its own backbone Qwen3-8B ([arXiv 2610.21244](https://arxiv.org/html/2610.21244)).
   - **JevBench v1.5.4:** CLM-8B's composite was 1.51, ranked 77th ([BenchLM](https://benchlm.ai/md/decision-models.md)).
   - **Where CLM is strong:** large reusable action sets (verifiers, games), with 4 to 9x lower latency than Jev ([VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests)).
4. **What the strong open models do differently.** They put state, question and options through **one forward pass with cross-attention between them**, then read probabilities with a pointer or letter-logit head, plus LoRA or fine-tuning and temperature calibration.
   - [Kev](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md): pointer head, LoRA r16.
   - [Inner-Jev/Jeff](https://arxiv.org/html/2610.17205): letter logits, LoRA r32, gets +8.9 Capability over Jev with a 31B model.
   - [Decision 2.0](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B).
   - [Laya](https://huggingface.co/convaiinnovations/laya): `[MASK]`-per-option scorer.
   - The pure bi-encoder (CLM) gives up state-option interaction in exchange for cacheability **(inference)**.
5. **Recommendation.** Keep the CLM bi-encoder as the fast path for big, cached candidate sets. Add three things:
   - a typed-decision path: a pointer head or late-interaction "decision stack" in the style of [Jeva](https://arxiv.org/abs/2610.18773);
   - calibration via per-type temperature or proper-scoring training;
   - a TypeSafe-compatible `/v1/systemone` server, so that Jev, Kev, Decision 2.0, Laya, Microsoft-Decision-1 and OpenAI Decisions all become drop-in baselines on our benchmark.
   Also evaluate on S1-Arena and OpenDecisions, which both have code and data available.

---

## 2. Comparison table

| System | Builder | Open? | Backbone | Scoring style | Training | Latency (reported) | Benchmarks (reported) |
|---|---|---|---|---|---|---|---|
| **CLM-v0.1-8B** (our reference) | Stanford + NVIDIA ([repo](https://github.com/Contrastive-LM/CLM)) | Yes, Apache-2.0 ([HF](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)) | Frozen Qwen3-8B, last-token pooling | Bi-encoder: state head and action head, dot/cosine, softmax | Symmetric InfoNCE: about 60M QA, then about 30M hard negatives, then about 1M trajectories ([repo](https://github.com/Contrastive-LM/CLM)) | T-Rex 16.5 ms vs Jev 149.8 ms ([search summary of model card](https://huggingface.co/sekkit/CLM-v0.1-8B)); about 28 ms per new state on RTX 4090, 0.6–0.7 ms cached ([repo](https://github.com/Contrastive-LM/CLM)) | DeepSWE 81.6% vs Jev 71.1%; TB2.1 87.6% vs 83.1%; BFCL 95.2% vs 99.2%; WikiRacing 26/30 vs 30/30 ([VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests)); S1-Arena 41.4% ([arXiv](https://arxiv.org/html/2610.21244)) |
| **Jev 1.13 / 1.14** | TypeSafe AI ([docs](https://docs.typesafe.ai/models.md)) | No, hosted API | Undisclosed ([Dibia](https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision)); suspected causal MoE **(third-party inference)** ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)) | Single pass over one sequence; typed outputs (Choice/Score/Noul/Box) | "RLCD", RL for calibrated decisions ([primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)) | 70–500 ms end to end ([TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)); OpenRouter p50 0.21 s ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev)); S1-Arena 321 ms | S1-Arena 72.6% / Brier 0.349; JevBench Capability 77.1 (1.13) ([Parallel](https://parallel.ai/articles/jevbench-benchmark.md)); OpenDecisions 80.1 ([arXiv](https://arxiv.org/html/2610.17205)) |
| **Kev** (0.8B/4B/9B/27B) | Jared Palmer ([repo](https://github.com/jaredpalmer/kev)) | Yes, Apache-2.0 | Frozen Qwen3.5-*-Base + LoRA r16 (27B: full fine-tune on Qwen3.8-27B) | One sequence; pointer head (option closing tag vs. question decide token), softmax | Cross-entropy over options, 4 SFT stages, single temperature T=2.41 ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)) | Kev-4B: 18.1 ms (H100, 6 questions, short state); 12.9 ms with cached state; 100.8 req/s | Kev-4B transfer-v4 test 0.838; MMLU-Pro 0.565 vs Jev 0.840 ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)); Kev-27B held-out index 52.3 vs Jev 54.0 ([repo](https://github.com/jaredpalmer/kev)) |
| **Laya** | Convai Innovations ([HF](https://huggingface.co/convaiinnovations/laya)) | Yes, Apache-2.0 | ModernBERT-large 421M (fully fine-tuned); mmBERT-base 322M multilingual | Encoder; `[MASK]` per option; 2 transformer layers + option-marker scorer; softmax per question | RLCD-style: REINFORCE with group-mean baseline, proper scoring-rule rewards; then per-type temperature refit | 32.8–39.5 ms on T4 (1 question) | Fine-tuned typed-decisions 0.766 vs Jev 0.727; zero-shot 0.362 (near chance) ([HF](https://huggingface.co/convaiinnovations/laya)); S1-Arena 44.5% |
| **Decision 2.0** (Kai 0.6B to Vega 27B) | vLLM Semantic Router team ([HF collection](https://huggingface.co/collections/vllm-sr/decision-20)) | Yes, Apache-2.0 | Kai: Qwen3-0.6B-Base ([card](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)); Vega: Qwen3.8-27B adapter ([card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)) | One pass, many questions; probability for every answer; head undisclosed | Undisclosed | Team-reported: Kai 4.9 ms, Vega 71.4 ms per single question ([cards](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)); S1-Arena: Vega 228 ms, Kai 72 ms | S1-Arena: Vega 73.0% (best System-1), Kai 50.3% ([arXiv](https://arxiv.org/html/2610.21244)) |
| **Microsoft-Decision-1** | Microsoft ([blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)) | No; Foundry and OpenRouter | Post-trained Qwen3.5-9B | Single-pass option scoring | Undisclosed | About 35x faster p50 than GPT-6 Sol (vendor); S1-Arena 304 ms | S1-Arena 70.9% / Brier 0.369; flip rate 1.3% across 8 perturbations (vendor) |
| **OpenAI Decisions API** | OpenAI ([Decoder](https://the-decoder.com/openai-launches-decisions-api-that-reduces-complex-evaluations-to-yes-no-or-pick-one/)) | No | GPT-6 Luna, constrained ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev)) | Bounded answer space; predicate/choice/score | Undisclosed | 150 ms vs 1.6 s for Luna (vendor slide); S1-Arena 140 ms | S1-Arena 68.6%; JevBench Capability 73.5 (600-item subset) |
| **Jeff-1.1** ("Releasing the Inner Jev") | Geuter, Cheng, Alvarez-Melis ([arXiv 2610.17205](https://arxiv.org/abs/2610.17205)) | Yes ([HF](https://huggingface.co/collections/jgeuter/jeff-11-6ada6708f3ed49beaf0f97ae)) | Gemma 4 31B/12B, Qwen3.6-27B, Qwen3.5-4B | Letter-token logits at last position (16 slots) | LoRA r32 cross-entropy on soft targets (DecisionBank, 261k items); per-type temperature | About 58 ms per decision on H200 (31B) | OpenDecisions Capability 89.0 vs Jev 80.1 ([arXiv HTML](https://arxiv.org/html/2610.17205)) |
| **Jeva** | Li & Wang ([arXiv 2610.18773](https://arxiv.org/abs/2610.18773)) | Weights not confirmed | Shared encoder (4B per HF page) | Request and options encoded separately; compact decision stack; symmetric aggregation; coverage (null) head | Not described | "Substantially reduced" through option reuse (no number) | Qualitative only |
| **Bongard** | Ding, Jin, Ji ([arXiv 2609.39111](https://arxiv.org/abs/2609.39111)) | "Open-weight" (per paper) | T5Gemma 2 4B-4B encoder-decoder | Encoder reads state once; decoder branch per question; trained head | 3 stages: judgments, then semantic relations, then action outcomes | 36 ms short; 221 ms for 32 questions (RTX PRO 6000) | DecisionBench 78.05% |
| **Inkling** | Thinking Machines Lab ([post](https://thinkingmachines.ai/news/introducing-inkling/)) | Yes (weights on HF) | 975B MoE, 41B active | Generative LLM, not a decision model | Pretrained on 45T tokens; RL with rubric and claims graders, proper-scoring calibration | n/a | GPQA-D 87.2, TB2.1 63.8, etc. |

---

## 3. Per-system sections

### 3.1 Jev (TypeSafe AI)

**What it is.**
- TypeSafe's flagship "System One" model. It returns typed decisions and probabilities and never generates text ([System One docs](https://docs.typesafe.ai/concepts/system-one.md)).
- Closed and hosted only.
- Announced 2026-09-15 after "two years in stealth" ([launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).
- Jev 1.14.0 was released 2026-10-14 ([models](https://docs.typesafe.ai/models.md)).
- Founder Diogo Almeida previously worked at OpenAI ([launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).

**API shape** ([API docs](https://docs.typesafe.ai/api/system-one.md)).
- Request: `POST /v1/systemone` with `model`, `state` (string, object or array), a map of named `questions` and optional image `assets`.
- Question types:
  - `noul`: probability of yes.
  - `choice`: 1–255 options; returns `choice`, `confidence` and `probabilities`.
  - `score`: 1–10 ordered levels; returns the probability-weighted `score`, `probabilities` and `legend`.
  - `box`: image box, 1.14 only.
- Context is 64k per request and 32k per question; 262k behind a "danger zone" header ([models](https://docs.typesafe.ai/models.md)).
- Price is $0.042 per M input tokens; output is free ([models](https://docs.typesafe.ai/models.md)).

**Architecture.**
- Officially undisclosed ([Dibia](https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision)).
- TypeSafe mentions "a new model architecture and a parallel sampler that generates all outputs in one query." Above 255 options it uses two-stage scoring ([launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).
- Third-party reverse-engineering from about 10k API calls ([Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)):
  - Latency grows with state length: 57.5 ms at 360 tokens, 218 ms at 29.8k tokens.
  - Latency is flat from 1 to about 100 questions (72–86 ms) and **independent of option count**: 200 options cost about the same as 2.
  - Additive token accounting suggests the state is encoded once and questions branch from it, which is consistent with prefix caching.
  - Head type (pointer vs 256-slot) is unresolved.
  - The MoE backbone is explicitly "an inference, not a measurement."
  - Tokenizer is closest to Qwen (348/415 probes).
  - Adding an irrelevant option shifts log-odds between existing options, so options interact. **This rules out a pure independent-score bi-encoder like CLM.**
- **(inference)** Jev reads state, questions and options jointly in one causal pass, with question branches sharing a cached state prefix.

**Training.**
- RLCD, "Reinforcement learning for calibrated decisions" ([primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)). Details are unpublished.
- Not fine-tuned on customer data ([models](https://docs.typesafe.ai/models.md)).

**Calibration and confidence.**
- `confidence` for Choice is computed client-side as (p_max − 1/K)/(1 − 1/K) ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)).
- MMLU ECE is 0.0313 on 1,200 items ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)).
- TypedBench finds the hosted model "systematically underconfident" and sensitive to wording ([arXiv 2610.11392](https://arxiv.org/abs/2610.11392)).
- Swapping yes and no flipped 50.5 answers per 100 for Jev ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)).

**Benchmarks.**

| Benchmark | Jev result | Source |
|---|---|---|
| S1-Arena | 72.6% / Brier 0.349 / 321 ms | [arXiv](https://arxiv.org/html/2610.21244) |
| JevBench v1.8.7 Capability | 77.1 (1.13), 75.0 (1.14) | [Parallel](https://parallel.ai/articles/jevbench-benchmark.md) |
| OpenDecisions Capability | 80.1 | [arXiv](https://arxiv.org/html/2610.17205) |
| Agentic step accuracy / loop success (S1-Arena) | 67.8% / 17.4% | [arXiv](https://arxiv.org/html/2610.21244) |
| CLM tasks: BFCL v4 | 99.2% | [VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests) |
| CLM tasks: WikiRacing | 30/30 | same |
| CLM tasks: DeepSWE verifier | 71.1% | same |
| CLM tasks: TB2.1 verifier | 83.1% | same |

**How to include it.**
- **As a baseline: yes.** It is cheap at $0.042/M input. The TypeSafe skill and SDK are available.
- **As a teacher: check TypeSafe's ToS first** ([legal](https://docs.typesafe.ai/legal.md)). Kev explicitly avoided Jev outputs in training ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)), which suggests this is a sensitive point **(inference)**.

**Ideas to adopt.**
- The typed question schema (Choice/Score/Noul).
- Score computed as the expectation over levels.
- One state, many questions, sharing the state computation.
- The 255-option cap, with two-stage scoring for larger sets.

### 3.2 "Kev": Jared Palmer's Kev (most likely), plus `arjun988/Kev`

**Jared Palmer's Kev** ([repo](https://github.com/jaredpalmer/kev); about 10.1k stars per the fetch).
- **Release:** Kev 1.0. Kev-4B was published 2026-09-24 ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- **License:** Apache-2.0.
- **Lineage:** The architecture follows Archer Hume's Jev write-up ([repo](https://github.com/jaredpalmer/kev)).

**Architecture.**
- Frozen Qwen3.5-4B-Base: 32 layers, 24 of them Gated DeltaNet; hidden size 2560.
- LoRA r16/α32: 33.8M parameters, merged at load.
- **Pointer head:** two projections score each option's closing token against the question's final token, followed by a softmax.
- Each question is its own row continuing from a cached state ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- On non-DeltaNet models, an attention mask isolates questions inside one sequence ([repo](https://github.com/jaredpalmer/kev)).
- Context: served up to 65,536 tokens of state; validated at 8k.

**Training.**
- Cross-entropy over options, with shuffled option order and inserted "none of the above" and distractor options.
- Four sequential SFT stages with replay, then a single temperature T=2.41 fit by NLL ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- Data:

  | Set | Size and content |
  |---|---|
  | decision-v7 | 12,576 records from ten public classification sets, policy minimal pairs and rule structures |
  | Dates and missing evidence | 1,425 |
  | CFPB documents | 5,219 |
  | hard-v1 skills | 6,000 |
  | devtools-v1 | 5,320 |

- No Jev outputs were used.

**Serving.**
- Kev-4B on H100: 18.1 ms for a new state and 12.9 ms with a cached state (6 questions, short state). With a 2,200-token state: 89.4 ms new and 22.5 ms cached. Throughput is 100.8 req/s at 64 clients. GPU memory is 14.3 GB ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- Kev implements TypeSafe's `/v1/systemone`, and there is a one-command Modal deploy ([repo](https://github.com/jaredpalmer/kev)).

**Benchmarks.**
- Kev-4B vs Jev:
  - transfer-v4 dev 0.817 vs 0.857;
  - breadth-v1 0.690 vs 0.757;
  - MMLU-Pro 0.565 vs 0.840;
  - ECE about 0.017–0.029 on held-out sets ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- Kev-27B: held-out index 52.3 vs Jev 54.0 ([repo](https://github.com/jaredpalmer/kev)).
- SanSi reports Jev at 83.5%, 11.5 pp above released Kev-4B on its suite (per [search summary](https://arxiv.org/pdf/2610.07730), not verified in full text).

**`arjun988/Kev` ("NotJev : Kev")** ([repo](https://github.com/arjun988/Kev)).
- Apache-2.0, 22 stars.
- Wraps an LLM (Ollama or OpenAI-compatible) using letter-logprob readout, constrained JSON, or micro-scoring. No training.
- Self-reported: Banking77 83%; p50 504 ms for 2 questions with qwen3.5:9b.
- **(inference)** Less relevant: it is a wrapper, not a model.

Other "kev" entries on JevBench ("kev 4B/8B research preview", "Malkuth-4B (Kev post-train)") have unclear provenance ([BenchLM](https://benchlm.ai/md/decision-models.md)).

**Include.**
- Kev-4B is the **best open, small, well-documented baseline**: it fits on a 24 GB GPU and speaks the same API.
- Its pointer head and multi-question masking are directly portable to our Qwen3-8B backbone **(inference)**.

### 3.3 Laya (Convai Innovations)

**Identity.**
- Open (Apache-2.0) "non-autoregressive System 1 decision model" ([HF](https://huggingface.co/convaiinnovations/laya)).
- Weights on HF around 2026-09-18 per secondary sources ([search result: innfactory/eesel/LLMReference](https://www.eesel.ai/blog/laya-ai)).
- Free on Vercel AI Gateway through Oct 31 ([Vercel changelog](https://vercel.com/changelog/laya-decision-model-now-available-on-ai-gateway-free-through-october-31)).
- No other relevant "Laya" in the AI decision space surfaced. All results point to Convai.

**Architecture** ([HF](https://huggingface.co/convaiinnovations/laya)).
- Fully fine-tuned ModernBERT-large (421M, 512 context), or mmBERT-base (322M, 1024 context, up to 8,192).
- Head: 2 transformer layers, then an option-marker scorer that scores each option at its own `[MASK]` token, then a softmax per question, plus an act/escalate head.
- All questions are answered in one forward pass.
- The sequence budget is split into options (`head_max_len` 192/256) and state.
- Version 0.5 adds a "vegaml" engine: a small decision head on a **frozen LLM reader** (0.8B–27B), with conformal prediction sets (α=0.10), abstain and route outputs.

**Training.**
- "RLCD": the policy outputs a distribution, Gaussian logit noise is added, rewards are proper scoring rules (log, spherical, RPS for ordinal), and updates use REINFORCE with a group-mean baseline (GRPO-style). TD(λ) over prefixes for multi-turn.
- Then temperature refit per question type and option count, which takes mean ECE from 0.466 to 0.081 ([HF](https://huggingface.co/convaiinnovations/laya)).

**Serving.**
- T4: 39.5 ms (1 question) and 158.6 ms (10 questions), English model.
- `laya-serve` exposes a Jev-compatible `POST /v1/systemone` ([HF](https://huggingface.co/convaiinnovations/laya)).

**Benchmarks.**
- Fine-tuned typed-decisions 0.766 vs Jev 0.727, but zero-shot only 0.362, below a 0.461 majority baseline ([HF](https://huggingface.co/convaiinnovations/laya)).
- Banking77 0.425 vs Jev 0.870.
- S1-Arena 44.5% (inputs cut to 512 tokens) ([arXiv](https://arxiv.org/html/2610.21244)).
- JevBench v1.5.4 composite about 0 ([BenchLM](https://benchlm.ai/md/decision-models.md)).

**Include.**
- As a cheap encoder baseline and an "after fine-tuning" comparison.
- Its frozen-reader plus small-head "vegaml" mode, with conformal sets, is conceptually closest to our design **(inference)**.

### 3.4 Inkling (Thinking Machines Lab), identified but not a System One model

**Facts.**
- Released 2026-07-15. MoE, 975B total / 41B active, 1M context. Pretrained on 45T tokens of text, image, audio and video.
- Weights on HF (original and NVFP4). Fine-tunable on Tinker ([TML post](https://thinkingmachines.ai/news/introducing-inkling/)).
- Inkling-Small: 276B/12B active, Apache-2.0 ([layer3labs](https://www.layer3labs.io/guides/inkling-explained)).
- Inkling 2 Small: 594B/17B active, released 2026-11-05 on Tinker, with weights promised under Apache-2.0 the following week ([layer3labs](https://www.layer3labs.io/guides/inkling-explained)).
- Post-training used a rubric grader, an agentic claims grader, and RL against proper scoring rules for calibration ([TML post](https://thinkingmachines.ai/news/introducing-inkling/)).
- **No decision, verifier or reward-model variant was found** ([search summary](https://techcrunch.com/2026/07/15/thinking-machines-amps-up-its-bet-against-one-size-fits-all-ai-with-its-first-open-model-inkling/)).

**Why the user may have bundled it.**
- It appears as generator metadata on the Jeva paper page ([HF papers](https://huggingface.co/papers/2610.18773)).
- It is listed as a comparison point on the Jev-Omni card ([HF](https://huggingface.co/akhilaaa3/Jev-Omni)).
- **(inference)** Possibly confused with a System One model.

**Include.**
- Only as a **teacher or labeler** for hard-negative generation and soft targets. Inkling-Small at 12B active is the realistic size.
- Its "RL against proper scoring rules" idea mirrors RLCD.
- Not a baseline on our benchmark: it is generative.

### 3.5 vLLM Semantic Router Decision 2.0 (and System One Auto)

**Release.**
- Six Apache-2.0 models: Kai-0.6B, Eos-0.8B, Sol-2B, Nox-4B, Lux-9B and Vega-27B.
- Announced 2026-10-02 ([ModelSystem.One](https://modelsystem.one/news/vllm-sr-decision-2-family/), [vLLM tweet](https://x.com/vllm_project/status/2106193438098256191)).

**Architecture.**
- Kai is fine-tuned from Qwen3-0.6B-Base with 8,192 context ([card](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)).
- Vega is based on Qwen3.8-27B with 32,768 context ([card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)).
- "Answers them all at once and returns a probability for every answer." The head is undisclosed.
- Loading: `AutoModel.from_pretrained(trust_remote_code)` and `model.system_one(...)`.

**Serving** ([docs](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/)).
- Shared context: up to 3.4x faster for 128 questions on one input (Vega).
- Fused Triton kernels: 11–48% faster with no decision changes.
- Batched rows on ROCm changed some decisions.
- Kai: about 0.2 s for 2 questions on 16 CPU cores.

**Benchmarks.**
- S1-Arena: Vega 73.0% / Brier 0.343 / 228 ms (best System-1, ahead of Jev); Kai 50.3% / 72 ms ([arXiv](https://arxiv.org/html/2610.21244)).
- Team-reported: JevArena 74.0 (Vega).

**System One Auto (cascade)** ([ModelSystem.One](https://modelsystem.one/news/vllm-sr-decision-2-family/)).
- Kai first, escalating to Vega when the acceptance gate fails.
- Accuracy 82.25% vs Kai 65.80% vs Vega 87.88% on 231 JevBench requests.
- About 54% estimated cost saving (author-run pilot).

**Include.**
- As baselines (Kai and Nox are cheap to run).
- Cascade idea: CLM-0.6B, then CLM-8B **(inference)**.

### 3.6 Microsoft-Decision-1

**Release and access.**
- Released 2026-10-09. Post-trained Qwen3.5-9B for "single-pass decision scoring."
- Hosted on Foundry and OpenRouter at $0.042/M input, free output ([MS blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)).
- Open weights are not stated (hosted only per [search summary](https://wilsonwu.me/blog/2026/microsoft-decision-1/)).
- Context is 32,768 tokens ([AICoder](https://aicoder.com/news/news-20261010-microsoft-decision-1-fast-decision-model-foundry-openrouter)).

**Vendor claims.**
- Highest accuracy on 36 held-out benchmarks.
- Flip rate 1.3% across 8 perturbation types, and 0 under option shuffle or reversal.
- About 35x lower p50 than GPT-6 Sol.

**Independent results.**
- S1-Arena: 70.9% / Brier 0.369 / 304 ms ([arXiv](https://arxiv.org/html/2610.21244)).

**Include.**
- As an API baseline via OpenRouter.
- Its perturbation-flip protocol, with 8 perturbation types, is a good robustness test to copy.

### 3.7 OpenAI Decisions API

**Release.**
- Announced at DevDay on 2026-09-29 ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev)).
- GA on 2026-10-22, with gpt-6-luna as the only model ([Digital Applied](https://www.digitalapplied.com/blog/openai-decisions-api-ga-pricing-400-questions); primary: [OpenAI guide](https://developers.openai.com/api/docs/guides/decisions), not fetched directly).

**Pricing and limits.**
- $0.10/M input; cached input $0.01/M; output not billed.
- Up to 400 questions per request.
- Returns predicate probabilities, choice with per-option probabilities and confidence, and score as an expectation ([Digital Applied](https://www.digitalapplied.com/blog/openai-decisions-api-ga-pricing-400-questions)).

**Results.**
- Vendor: 150 ms vs 1.6 s for a normal Luna call.
- S1-Arena: 68.6% / 140 ms ([arXiv](https://arxiv.org/html/2610.21244)).
- JevBench Capability: 73.5 ([Parallel](https://parallel.ai/articles/jevbench-benchmark.md)).

**Include.** API baseline. Its explicit cached-prefix billing confirms that prefix caching is the key industry optimization **(inference)**.

### 3.8 Jeff-1.1 / "Releasing the Inner Jev" (Geuter et al., Harvard)

**Key result** ([arXiv 2610.17205](https://arxiv.org/abs/2610.17205)).
- Instruction-tuned LLMs read via **option-letter logits at the last token** match Jev zero-shot: Gemma 4 31B 79.6 vs Jev 80.1 on OpenDecisions.
- LoRA SFT adds 7.8–15.7 points.

**Recipe** ([HTML](https://arxiv.org/html/2610.17205)).
- Head: 16 letter slots. Larger option sets use groups of up to 16 plus a final round (up to 17 passes).
- Cross-entropy on soft teacher targets.
- Teacher: Qwen3.8-27B with reasoning, averaging 2 samples.
- LoRA r32/α64.
- Per-type temperature with T≥1; noul probabilities in [0.2, 0.8] are pushed outward.
- Three-seed weight soup.
- **Bidirectional attention** for most models.

**Data and code.**
- DecisionBank: 261,561 items ([HF](https://huggingface.co/datasets/jgeuter/DecisionBank/viewer/DecisionBank-260k)).
- OpenDecisions: 2,352 validation items ([HF](https://huggingface.co/datasets/jgeuter/OpenDecisions)).
- Code: [github.com/j-geuter/Jeff](https://github.com/j-geuter/Jeff).

**Include.**
- **Zero-shot Qwen3-8B letter-logit readout is the minimum baseline CLM must beat.** S1-Arena shows that CLM currently loses to it by 17.8 pp.
- DecisionBank is ready-made training data for a typed-decision head.

### 3.9 Jeva ("System One Models from First Principles")

**Design** ([arXiv 2610.18773](https://arxiv.org/abs/2610.18773)).
- A shared encoder represents the request and **each option separately**, so options are cacheable as in CLM.
- A compact decision stack refines **only the request state against fixed option representations**: late interaction.
- Symmetric aggregation gives **permutation-equivariant** probabilities.
- A separate **coverage estimate** gives a null outcome ("none fits").
- Claims: scales to thousands of candidates, no permutation flips, latency cut through cross-request reuse. No numbers in the abstract.

**Relevance.** **(inference)** This is the closest published blueprint for upgrading CLM: it keeps the action cache and adds state-option interaction plus a null head.

### 3.10 Others worth knowing

**Bongard** ([arXiv 2609.39111](https://arxiv.org/abs/2609.39111)).
- T5Gemma 2 4B-4B encoder-decoder: the state is encoded once and each question gets a decoder branch.
- Three-stage training, ending in "action outcomes" from sandbox rollouts, which moved accuracy from 50.6% to 64.8%.
- 36 ms per request; 221 ms for 32 questions.

**SanSi** ([arXiv 2610.07730](https://arxiv.org/abs/2610.07730)).
- A looped LM with a typed readout after each loop, and a proper scoring rule at every loop, so compute is adjustable at test time (1–8 loops).
- +13.5 pp over a non-looped model of the same shape.

**Decision-model routing and robustness research.**
- "Consider, Then Choose" (Runoff) ([arXiv 2610.18233](https://arxiv.org/abs/2610.18233)): score options individually, shortlist, then compare in all positions. Order-invariant, two passes, 3–4x faster than cyclic averaging at 20 options.
- "Reading Right, Acting Wrong" ([arXiv 2610.18579](https://arxiv.org/abs/2610.18579)): most wrong agent actions follow correct reading of the state. Composing fact-probe answers in code repairs this.

**Classic routers and cross-encoders (2024–25).**
- Arch-Router-1.5B, a preference-aligned query router ([arXiv 2506.16655](https://arxiv.org/abs/2506.16655)).
- Qwen3-Reranker 0.6B/4B/8B cross-encoders ([arXiv 2506.05176](https://arxiv.org/abs/2506.05176)).
- A fine-tuned DeBERTa-v3-large still wins most labeled benchmarks against System One models ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)). This is a reminder to include a supervised classifier baseline.

### 3.11 Benchmarks we can run

| Benchmark | Contents | Access |
|---|---|---|
| **System1-Arena** ([arXiv](https://arxiv.org/abs/2610.21244)) | 92k questions, 105 tasks, 138 closed-loop environments | [code](https://github.com/TianyuCodings/S1-Arena), [data](https://huggingface.co/datasets/C-Tianyu/S1-Arena), [leaderboard](https://tianyucodings.github.io/S1-Arena). CLM is already evaluated there, with inputs cut to the last 2,048 tokens ([HTML](https://arxiv.org/html/2610.21244)). |
| **OpenDecisions** | 2,352 items, Capability = accuracy + calibration | [HF](https://huggingface.co/datasets/jgeuter/OpenDecisions) |
| **JevBench** | 1,500 decisions (300 public / 1,200 sealed); Capability and Composite | [benchmarkheaven.com/jev-models](https://parallel.ai/articles/jevbench-benchmark.md) |
| **Jev Decision Index** | 132,422 requests, 37 benchmarks | [HF Space](https://huggingface.co/spaces/multimodalart/jev-decision-index) ([AI Weekly](https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models)) |
| **TypedBench** | Calibration vs noise floor, framing sensitivity, cost-matrix utility | [arXiv 2610.11392](https://arxiv.org/abs/2610.11392); data release not confirmed |

**Comparability with CLM's reported benchmarks.** CLM's own suite (tricky Q&A, DeepSWE, T-Rex, WikiRacing, Mario, BFCL, TB) is not included in these boards. Only S1-Arena's SWE-bench, games and Wikispeedia environments overlap thematically. Numbers are **not directly comparable** **(inference)**.

---

## 4. Serving-optimization notes for our architecture

| Technique | Evidence | Applicability to CLM-style model |
|---|---|---|
| **Cache action/option embeddings** | CLM: revisited-state p50 from about 1.7–2.0 ms to 0.6–0.7 ms; new state about 28 ms on RTX 4090 ([repo](https://github.com/Contrastive-LM/CLM)). 13x faster than Jev at about 1k candidates with caching ([HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)). | Already core to us. Key cache entries by (backbone version, head version, text hash). Jeva shows that option caching survives adding a decision stack ([arXiv](https://arxiv.org/abs/2610.18773)). |
| **Prefix / KV caching of shared state** | Jev: flat latency to about 100 questions ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)). Kev-4B: 89.4 ms new vs 22.5 ms cached for a 2.2k-token state ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)). Decision 2.0: 3.4x for 128 questions ([docs](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/)). vLLM APC reuses the KV of shared prefixes, prefill only ([vLLM docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)). | Matters if we add a joint (state+question+options) path. For the pure bi-encoder, put the shared agent context first so that APC hits across steps of one trajectory **(inference)**. Whether APC applies to pooling models is not documented. |
| **vLLM pooling runner** | `--runner pooling`; default `embed` pooler uses LAST pooling with normalization (`use_activation`); hidden states via `token_embed`; poolers use final hidden states ([vLLM docs](https://docs.vllm.ai/en/latest/models/pooling_models.html)). CLM serves this way ([repo](https://github.com/Contrastive-LM/CLM)). | Keep it. Set `use_activation=false` if the heads expect raw states. Intermediate-layer pooling is not supported out of the box, so it needs a truncated model or a custom pooler **(inference)**. |
| **Smaller backbones** | Qwen3-Embedding offers 0.6B/4B/8B; 8B gets MTEB-multilingual 70.58 ([HF](https://huggingface.co/Qwen/Qwen3-Embedding-8B)). Decision 2.0 shows steep size scaling: Kai-0.6B 50.3% vs Vega-27B 73.0% (S1-Arena), with latency 72 vs 228 ms ([arXiv](https://arxiv.org/html/2610.21244)). Kev-0.8B lags Kev-4B badly (0.648 vs 0.817 on new sources) ([repo](https://github.com/jaredpalmer/kev)). | Train heads on Qwen3-0.6B/1.7B/4B embeddings as a size sweep. Expect a big drop below about 2–4B on typed decisions, and a smaller one on retrieval-like action matching **(inference)**. |
| **Early-layer exits** | Intermediate layers often give better embeddings than the last across 32 tasks ([Layer by Layer, arXiv 2502.02013](https://arxiv.org/abs/2502.02013)). 2DMSE trains for elastic layer count and dimension ([arXiv 2402.14776](https://arxiv.org/abs/2402.14776)). | Cheap experiment: extract hidden states at layers {L/2, 3L/4, L}, train the same heads, and compare top-1 vs. latency. Truncating layers cuts prefill roughly in proportion **(inference)**. |
| **Matryoshka head outputs** | MRL: up to 14x smaller embeddings at the same accuracy (ImageNet) ([arXiv 2205.13147](https://arxiv.org/abs/2205.13147)). Qwen3-Embedding supports user-set dimensions 32–4096 ([HF](https://huggingface.co/Qwen/Qwen3-Embedding-8B)). | Add an MRL loss over prefixes of the head output (e.g. 64/128/256/full) to shrink option caches and dot-product cost at about 1k+ candidates. |
| **Quantize cached embeddings** | int8 keeps about 97–99.3% retrieval quality with 4x memory savings; binary keeps about 92.5–96% (with rescoring) at 32x ([HF blog](https://huggingface.co/blog/embedding-quantization)). | int8 option cache is near-free. Binary plus fp rescoring of the top-k suits very large action sets. |
| **Quantize backbone** | Official Qwen3-8B-FP8 (block-128 fine-grained fp8) works in vLLM ≥0.8.5 ([HF](https://huggingface.co/Qwen/Qwen3-8B-FP8)). No quality numbers on the card. | The frozen backbone is easy to swap. **Must retrain or verify heads on fp8 hidden states**, since the head was trained on bf16 features **(inference)**. Heads (about 10–20M parameters) do not need quantization. |
| **Cascades** | System One Auto: Kai-first, then Vega, reached 82.25% vs Vega 87.88% at about 54% estimated cost saving ([ModelSystem.One](https://modelsystem.one/news/vllm-sr-decision-2-family/)). Intent-trained first stage plus Gemma-4-31B escalation matched Gemma accuracy at about Jev's price ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)). | Small-backbone CLM, then 8B CLM (or LLM readout) gated on calibrated max-probability. |
| **Order robustness** | Runoff: order-invariant in two passes ([arXiv 2610.18233](https://arxiv.org/abs/2610.18233)). | Our bi-encoder is already permutation-invariant by construction. That is a selling point to measure, e.g. with Microsoft's flip-rate protocol ([MS blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)). |

---

## 5. Prioritized "adopt / evaluate next"

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Implement a TypeSafe-compatible `POST /v1/systemone` adapter for our model, plus a client harness that calls Jev, Kev, Laya, Decision 2.0, MS-Decision-1 and OpenAI Decisions** | CLM, Kev and Laya all already expose this shape ([CLM](https://github.com/Contrastive-LM/CLM), [Kev](https://github.com/jaredpalmer/kev), [Laya](https://huggingface.co/convaiinnovations/laya)), so baselines become config entries | S (2–3 days) |
| 2 | **Add a zero-shot Qwen3-8B letter-logit baseline** (Inner-Jev style) on our benchmark | Same backbone, no training; S1-Arena shows it beats CLM by 17.8 pp on typed decisions ([arXiv](https://arxiv.org/html/2610.21244), [recipe](https://arxiv.org/html/2610.17205)) | S (1–2 days) |
| 3 | **Run S1-Arena (non-agentic subset) and OpenDecisions** on our checkpoints, without the 2,048-token truncation | Public code and data; direct comparison to 10 System-1 models ([S1-Arena code](https://github.com/TianyuCodings/S1-Arena), [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions)) | M (1 week incl. compute) |
| 4 | **Calibration layer:** per-question-type and per-K temperature (NLL fit), report Brier/ECE; optionally a proper-scoring-rule fine-tune | Laya ECE went from 0.466 to 0.081 by refit; Kev and Jeff use the same; CLM had the worst Brier (0.749) in S1-Arena | S (2 days) |
| 5 | **Typed-decision head on the same frozen backbone:** a Kev-style pointer head (state+question+options in one sequence, LoRA optional), trained on DecisionBank and Kev's public sets | Kev-4B reaches 0.838 OOD with about 30k records ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)); DecisionBank is 261k ([HF](https://huggingface.co/datasets/jgeuter/DecisionBank/viewer/DecisionBank-260k)) | M–L (2–3 weeks) |
| 6 | **Jeva-style late-interaction stack over cached option embeddings, plus a null/coverage head** | Keeps the CLM cache advantage while adding state-option interaction and "none fits" ([arXiv](https://arxiv.org/abs/2610.18773)) | L (3–4 weeks, research) |
| 7 | **Serving sweep:** backbone size (0.6B/1.7B/4B/8B), layer exit (L/2, 3L/4, L), FP8 backbone, int8 option cache, MRL head dims | Cheap ablations once the embedding pipeline is parameterized; cited gains above | M (1–2 weeks) |
| 8 | **Robustness suite:** option shuffle, yes/no swap, paraphrase, irrelevant-option insertion; report flip rates | MS-Decision-1 publishes 1.3%; Jev flips 50.5/100 on yes/no swap ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)); the bi-encoder should shine | S (2–3 days) |
| 9 | **Cascade:** small CLM, then 8B CLM or LLM readout, gated on calibrated confidence | System One Auto evidence | S–M (after 4 and 7) |
| 10 | **Teacher labels for hard negatives and soft targets** from an open model (Qwen3.8-27B-reasoning as in Jeff, or Inkling-Small) instead of Gemini Flash-Lite | Open, license-clean; avoid Jev outputs pending ToS review | M (data pipeline + compute) |
| 11 | Add a supervised DeBERTa-v3-large classifier baseline on labeled tasks | Wins most labeled benchmarks vs System One models ([arXiv 2610.00346](https://arxiv.org/abs/2610.00346)) | S |

### Feasibility of including each system

| System | Baseline | Backbone | Teacher | Notes |
|---|---|---|---|---|
| Jev | API, $0.042/M | – | ToS check needed | Not open |
| Kev-4B/9B | Local, Apache, 24 GB GPU | Pointer-head recipe portable | – | Best open small baseline |
| Decision 2.0 | Local, Apache, Kai on CPU | Could warm-start from Kai/Nox weights | – | Head undisclosed (custom code) |
| Laya | Local, Apache, T4 | – | – | Zero-shot weak; fine-tune first |
| MS-Decision-1 | OpenRouter, $0.042/M | – | ToS | Hosted only |
| OpenAI Decisions | API, $0.10/M | – | ToS | No published calibration |
| Jeff-1.1 | Local, open | Recipe on Qwen3-8B | Its DecisionBank data | 31B model about 58 ms on H200 |
| Inkling / Inkling-Small | Not applicable (generative) | – | Open; 12B-active Small is feasible on multi-GPU | Not a decision model |

---

## 6. Caveats

- Almost all numbers are from 2026 preprints, vendor posts or secondary aggregators. Many are self-reported and not reproduced (e.g. Decision 2.0 team numbers; MS-Decision-1 vendor claims).
- Leaderboards with the same name ("JevBench") use different protocols and are not comparable ([Parallel](https://parallel.ai/articles/jevbench-benchmark.md)).
- CLM head size: the CLM README says about 20M parameters per head ([repo](https://github.com/Contrastive-LM/CLM)), while our project notes say 9.44M each. Worth reconciling.
- CLM-35B(-A3B) was announced for early October 2026, but the Contrastive-LM HF org shows only CLM-v0.1-8B and DeepSWE heads as of this study ([HF org](https://huggingface.co/Contrastive-LM)). **Not found as released.**
- Not found: Microsoft-Decision-1 open weights; Decision 2.0 head and training details; Jeva weights; per-model ECE in S1-Arena; Jev's architecture or training data.
