# Inkling (Inkling, Inkling-Small, Inkling 2 Small, Inkling 2 Mini)

> **Family:** generative mixture-of-experts LLMs. They are not System One decision models, but they are relevant to us as a readout baseline, a frozen reader and a teacher · **Team:** Thinking Machines Lab (TML) · **Released:** Inkling 2026-07-15; Inkling-Small 2026-07-30; Inkling 2 Small 2026-11-05 (weights posted 2026-11-12); Inkling 2 Mini 2026-11-20 · **License:** Apache-2.0 for all weights, per the HF tags and cards · **Code/weights:** HF org `thinkingmachines`: Inkling(-NVFP4), Inkling-Small(-NVFP4), Inkling-2-Small(-MXFP8), Inkling-2-Mini, `TML-d24-varloop-research` and tokenizers ([HF API](https://huggingface.co/api/models?author=thinkingmachines)); hosted on Tinker · **Sources read:** the TML posts [Introducing Inkling](https://thinkingmachines.ai/news/introducing-inkling/) and [Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/); HF cards and `config.json`; the TML news index; Tinker SDK docs ([samplingclient.md](https://github.com/thinking-machines-lab/tinker/blob/main/docs/api/samplingclient.md)) and [pricing](https://thinkingmachines.ai/tinker/pricing/); the [OpenRouter endpoints API](https://openrouter.ai/api/v1/models); vLLM recipes; transformers model directories and docs; `chat_template.jinja`; TML blog posts on LoRA and on-policy distillation. Secondary (landscape notes): layer3labs and TechCrunch summaries. **Not found:** any decision, verifier, reward or System One variant; any decision API; any third-party use of Inkling as a System One backbone or vegaml reader.

## Summary

Inkling is TML's family of open-weight generative MoE models. The flagship has 975B total and 41B active parameters, a 1M-token context, and was pretrained on 45T tokens ([post](https://thinkingmachines.ai/news/introducing-inkling/); [@inkling]).

It is **not** a System One model. The HF org, the TML news index and the Tinker SDK show no decision, classifier or calibration-model variant. The Tinker SDK offers only sampling, logprobs and training. The name surfaced in this survey for two reasons:
- it appears as generator metadata on the Jeva paper's HF page;
- it is listed as a comparison point on the Jev-Omni card.

Its relevance comes from its post-training. TML "trained for calibration with RL against proper scoring rules on a large corpus of resolved real-world questions" and used abstention-aware rewards. That mirrors the proper-scoring idea behind RLCD. However, **no ECE is published for any Inkling model**, and the scoring rule, reward weights and corpus size are not disclosed.

For our project, Inkling is useful in three roles, each examined in [Inkling as ablation arms for our project](#inkling-as-ablation-arms-for-our-project):
1. A zero-shot letter-readout baseline.
2. A frozen reader (Inkling 2 Mini only).
3. A soft-target teacher.

Three access facts decide which role each model can play:
- **Logprobs:** Tinker serves them for all four models. OpenRouter providers serve them only for the two v2 models.
- **Hidden states:** these require self-hosting through transformers.
- **Hardware:** only Inkling 2 Mini (about 20 GB) fits on one GPU.

## Architecture

All figures are from each repo's `config.json` and the linked cards or posts.

| Model | Release | Total / active params | Layers × hidden | Experts | Context | Source |
|---|---|---|---|---|---|---|
| Inkling | 2026-07-15 | 975B / 41B | 66 × 6144 | 256 routed + 2 shared, top-6 | 1M | [post](https://thinkingmachines.ai/news/introducing-inkling/), [config](https://huggingface.co/thinkingmachines/Inkling) |
| Inkling-Small | 2026-07-30 | 276B / 12B | 42 × 4096 | 256 + 2, top-6 | 1M (OpenRouter serves 524k) | [card](https://huggingface.co/thinkingmachines/Inkling-Small) |
| Inkling 2 Small | 2026-11-05; weights posted 11-12 | 594B / 17B | 48 × 4608 | 384, top-8, no shared experts | 1M | [card](https://huggingface.co/thinkingmachines/Inkling-2-Small), [post](https://thinkingmachines.ai/news/inkling-2-small/) |
| Inkling 2 Mini | 2026-11-20 | 26.4B / 3.4B (5.4B effective) | 24 × 2304; layers 6–18 run 3× | 128, top-8 | 1M | [card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) |

**Shared design**
- Attention mixes sliding-window layers and global layers at a 5:1 ratio. The window is 512 tokens in v1 and 1024 in v2.
- Positions use relative position embeddings instead of RoPE, plus short convolutions.
- There are 8 multi-token-prediction (MTP) heads.
- Inkling 2 Mini is **looped**: its core layers 6–18 run three times. That is why its effective active size (5.4B) exceeds its nominal active size (3.4B). transformers ships a dedicated `inkling2_loop` model directory, and the HF org also hosts `TML-d24-varloop-research`.
- Inkling's pretraining covered text, image, audio and video (landscape notes, from the TML post).

**Discrepancy.** Inkling is announced at 975B total parameters. The safetensors total of the BF16 repo is 952B (about 1.9 TB of storage).

**Passes per request.** These models generate text, so a decision readout needs one prefill per prompt. On Tinker, `sample(..., max_tokens=1, target_prompt_logprobs=…)` scores chosen candidate token ids at chosen prompt positions **within a single prefill**. `prompt_logprobs_last_n` reuses the prefix cache. *(inference)* A state-first prompt whose question and options come last can therefore share the state's KV across the questions of one request.

| Property | Inkling | Inkling-Small | Inkling 2 Small | Inkling 2 Mini |
|---|---|---|---|---|
| Type | Generative MoE | Generative MoE | Generative MoE | Generative MoE, looped core |
| Active params | 41B | 12B | 17B | 3.4B (5.4B effective) |
| Attention | SWA 512 : global = 5:1 | SWA 512 : global = 5:1 | SWA 1024 : global = 5:1 | SWA 1024 : global = 5:1 |
| Context | 1M | 1M (524k on OpenRouter) | 1M | 1M |
| Logprobs: Tinker / OpenRouter | yes / none | yes / none | yes / yes | yes / yes |
| Hidden states | self-host only | self-host only | self-host (transformers: MXFP8 only) | self-host |
| Smallest documented footprint | NVFP4 about 594 GB | NVFP4 about 171 GB | NVFP4 ≥500 GB | about 20 GB |
| Cacheable | Prefix KV (Tinker `prompt_logprobs_last_n`) | same | same | same |

## Training

**Inkling** ([post](https://thinkingmachines.ai/news/introducing-inkling/); [@inkling]):
- **Pretraining:** 45T tokens, with the Muon optimizer.
- **RL:** more than 30M rollouts.
- **Graders:** a rubric grader, plus a claims grader that checks each claim with agentic web search.
- **Calibration**, in the post's words:
  - "We trained for calibration with RL against proper scoring rules on a large corpus of resolved real-world questions."
  - "short-form factual QA with abstention-aware rewards: answering only pays off when the model is likely to be right".
  - Prompts that encourage or forbid hedging.
- **Not disclosed:** the scoring rule, the reward weights and the corpus size. TML's blog index has no post on calibration or proper scoring rules; the material appears only inside the Inkling post.

**Inkling 2 Small** ([post](https://thinkingmachines.ai/news/inkling-2-small/); [@inkling2small]):
- **Pretraining:** NVFP4-native, in four phases: H1 8.4T tokens, H2 3.8T at 512K context, H3 about 320B, H4 about 34B.
- **RL:** 1,075 steps, more than 17M rollouts, 2,048 GPUs, StableReinforce. "Forecasting tasks in RL used proper scoring rules."
- **Alignment:** a phase using on-policy self-distillation.

**Inkling-Small and Inkling 2 Mini:** the notes record no training details.

## Calibration

The proper-scoring and abstention-aware RL described under [Training](#training) is the only calibration method TML discloses. The published numbers below contain **no ECE for any model**.

| Metric | Inkling | Inkling 2 Small | Inkling 2 Mini |
|---|---|---|---|
| ForecastBench Brier Index (no search / with search) | 61.1±0.79 / 63.7±0.82 | — | — |
| Prophet Arena Brier | 0.1617 | — | — |
| AA-Omniscience score | 2.1 (post) / 1.0% (HF card): inconsistent | 16.1 | −8.6 |
| AA-Omniscience non-hallucination rate | — | 73.3% | 63.4% |

Caveats:
- **Different checkpoint.** The forecasting numbers were measured on "a different checkpoint of Inkling than the one released" ([post](https://thinkingmachines.ai/news/introducing-inkling/)).
- **Inkling 2 Small reports none.** "No Brier score, calibration error, or abstention rate is reported" ([post](https://thinkingmachines.ai/news/inkling-2-small/)).
- **Inconsistent score.** The post and the HF card disagree on Inkling's AA-Omniscience score (2.1 vs 1.0%).

**Context from other systems.** These findings bear on any LLM readout, although none was measured on Inkling:
- **Readouts are overconfident.** Gemma-4-31B puts ≥0.999 on its chosen option for 87% of S1-Arena questions ([arXiv 2610.21244](https://arxiv.org/html/2610.21244); [@s1arena]).
- **Reasoning helps accuracy but hurts calibration.** On Qwen3.8-27B, reasoning raises accuracy from 70.2 to 75.6 but worsens Brier from 0.391 to 0.430. Mean confidence rises from 81.4% to 97.4%.
- **Temperature scaling alone helps a lot.** It took zero-shot Gemma-4-31B from 72.9 to 79.6 Capability ([arXiv 2610.17205](https://arxiv.org/html/2610.17205); [@jeff]).

*(inference)* An Inkling readout should be temperature-scaled on held-out data before any comparison, whatever its RL calibration training.

## Results

| Benchmark | Inkling | Source |
|---|---|---|
| GPQA-D | 87.2 | TML [post](https://thinkingmachines.ai/news/introducing-inkling/) / [card](https://huggingface.co/thinkingmachines/Inkling) (the notes do not say which) |
| Global-MMLU-Lite | 88.7 | TML post / card (the notes do not say which) |
| Terminal-Bench 2.1 | 63.8 | landscape notes, from the post |
| ForecastBench Brier Index | 61.1±0.79 / 63.7±0.82 (different checkpoint) | post |
| Prophet Arena Brier | 0.1617 | post |

- The notes record no general benchmark numbers for Inkling-Small. For Inkling 2 Small and Inkling 2 Mini they record only the AA-Omniscience rows above.
- **No System One benchmark evaluates any Inkling model.** S1-Arena does not include it, and no third-party result uses it as a vegaml reader or System One backbone (GitHub and web searches).

## Serving and efficiency

### Hosted access and log-probabilities

| Route | Inkling | Inkling-Small | Inkling 2 Small | Inkling 2 Mini |
|---|---|---|---|---|
| **Tinker** (prefill / sample, $ per M tokens) | 1.87 / 4.68 | 0.58 / 1.44 | 0.30 / 1.00 | 0.15 / 0.50 |
| Tinker logprobs | yes | yes | yes | yes |
| OpenRouter providers with `logprobs`/`top_logprobs` | **none** (Together only, without logprobs) | **none** (DeepInfra fp4 only) | PrimeIntellect $0.15; Modal / Nebius / Together / Cloudflare $0.30 (input) | Io Net $0.08; PrimeIntellect $0.10 |
| Serverless input / output ($ per M) | 1.00 / 4.05 | 0.30 / 1.20 | 0.30 / 1.00 | — |

- Tinker prices are from [tinker/pricing](https://thinkingmachines.ai/tinker/pricing/) [@tinker-pricing] as of 2026-10-19. "Forward-only requests from a training client are charged at the training rate."
- **Tinker logprob tools:**
  - `sample(..., max_tokens=1, target_prompt_logprobs=…)` scores chosen candidate token ids at chosen prompt positions in one prefill;
  - `topk_prompt_logprobs` returns the top-$k$ tokens;
  - `prompt_logprobs_last_n` reuses the prefix cache.
- **Thinking effort.** The chat template injects a thinking-effort system message, in which "none" maps to 0.0 ([chat_template.jinja](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/main/chat_template.jinja)). The 2 Mini card warns that minimum effort "does not fully disable reasoning".

### Self-hosting footprint

| Model | Footprint | Notes |
|---|---|---|
| Inkling | BF16 about 1.9 TB (952B params); NVFP4 about 594 GB, `vram_minimum_gb` 651 | Verified on GB200/B200/H200; Hopper falls back to BF16 dequantization ([vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling.yaml)) |
| Inkling-Small | NVFP4 about 171 GB; VRAM minimum 180 GB | TP1 on B300/GB300, TP2 on B200 ([recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-Small.yaml)) |
| Inkling 2 Small | NVFP4 (native) ≥500 GB (2×B300 or 4×H200); MXFP8 ≥800 GB | vLLM ≥0.35.0, VRAM minimum 473 ([recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-2-Small.yaml)) |
| Inkling 2 Mini | about 20 GB, mixed NVFP4/MXFP8 | One DGX Spark or a Mac with ≥32 GB; H100/H200 verified; vLLM needs the `inkling-2-mini` docker image (no wheel yet) |

**Other engines.** SGLang supports Inkling v1 and v2. llama.cpp has an `inkling2` converter (found by code search).

### Hidden states

- **Tinker has no hidden-state API.** A search of the SDK for `hidden_states` returns nothing.
- **transformers** has native `inkling`, `inkling2` and `inkling2_loop` model directories ([transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)). The repos contain no remote code.
  - For Inkling 2 Small, "Currently only the `Inkling-2-Small-MXFP8` checkpoint is supported" ([inkling2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/inkling2.md)).
  - *(inference)* `output_hidden_states=True` should work as for any HF model.

## Flaws and failure modes

### (a) Inkling's own flaws, as they affect decision use

| Issue | Evidence |
|---|---|
| Not a decision model | No decision variant, API or System One benchmark entry found |
| Calibration claims cannot be verified | No ECE for any model; the scoring rule, weights and corpus are undisclosed; Inkling 2 Small reports no Brier, calibration error or abstention rate |
| Calibration evidence is from a different checkpoint | The ForecastBench and Prophet Arena numbers come from "a different checkpoint … than the one released" |
| Inconsistent reporting | AA-Omniscience 2.1 (post) vs 1.0% (HF card); 975B announced vs 952B in safetensors |
| Small-model hallucination | Inkling 2 Mini: AA-Omniscience −8.6; non-hallucination rate 63.4% |
| Reasoning cannot be switched off on 2 Mini | "does not fully disable reasoning" at minimum effort |
| v1 logprobs unavailable through OpenRouter | No provider exposes `logprobs` for Inkling or Inkling-Small |
| Hidden states need self-hosting | No Tinker hidden-state API; transformers supports only the MXFP8 checkpoint of Inkling 2 Small |
| Tooling immaturity | Inkling 2 Mini on vLLM needs a docker image; on Hopper, Inkling's NVFP4 falls back to BF16 dequantization |

### (b) Flaws of other systems it claims to fix

- **Claim.** TML positions proper-scoring RL and abstention-aware rewards as a fix for poorly calibrated, over-answering LLMs: "answering only pays off when the model is likely to be right".
- **Evidence.** Within the notes, the only evidence is forecasting Brier numbers on an unreleased checkpoint, plus AA-Omniscience scores.
- **No head-to-head.** No System One flaw (order flips, yes/no swaps, option-count collapse) has been measured on Inkling, so the notes contain no evidence that it fixes any of them.

## Inkling as ablation arms for our project

Our model is a CLM-style bi-encoder rebuilt on Qwen3-8B. Each arm below gives a reference point that the bi-encoder does not depend on. All costs are *(inference)*: arithmetic on Tinker list prices, with stated token assumptions.

### Arm 1: zero-shot readout baseline

**Why this arm matters**
- On S1-Arena, a same-backbone answer-token readout from Qwen3-8B scores 59.3, against CLM's 41.4 ([@s1arena]).
- Jeff-1.1 shows that instruction-tuned LLMs read through option-letter logits match Jev zero-shot: Gemma 4 31B scores 79.6 vs Jev's 80.1 on OpenDecisions.
- Zero-shot accuracy scales steeply with size: on Qwen3.5, 48.9 at 0.8B, 76.8 at 4B, 82.2 at 9B and 86.5 at 27B ([@jeff]).
- *(inference)* An Inkling arm extends that size curve with a frontier-scale open model trained with proper-scoring RL. That tests whether RL calibration training shows up in readout calibration.

**Readout** *(inference)*. The readout follows Jeff-1.1's letter readout and Intern-Decision's Eq. 3:

$$\ell_k = \log p_\theta\big(\mathrm{tok}(c_k)\mid x\big),\qquad P(c_k\mid x) = \frac{\exp(\ell_k/T_t)}{\sum_{j=1}^{K}\exp(\ell_j/T_t)}$$

- $x$ is a raw-completions prompt that ends by forcing `<|message_model|><|content_text|>`. This bypasses the thinking-effort system message, which cannot fully disable reasoning on 2 Mini.
- $T_t$ is a per-type temperature fit on held-out datasets.
- **Order debiasing.** Average $P$ over 2–4 option permutations to reduce order effects.

**Access**
- Inkling and Inkling-Small: Tinker or self-hosting only.
- Inkling 2 Small and 2 Mini: any of the OpenRouter providers that expose `logprobs`.

**Cost.** At about 1k prompt tokens per item, the 78k non-agentic S1-Arena questions are about 78M prefill tokens. At Tinker prefill rates:

| Model | Approximate cost |
|---|---|
| Inkling 2 Mini | about $12 |
| Inkling 2 Small | about $23 |
| Inkling-Small | about $45 |
| Inkling | about $146 |

### Arm 2: frozen reader / backbone

- **Only Inkling 2 Mini is practical.** At about 20 GB it fits one 32–80 GB GPU under a vegaml-style head. The larger models are impractical as per-request readers.
- **Extraction route.** Self-host with transformers and `output_hidden_states=True` *(inference)*. Tinker exposes no hidden states.
- **Loop choice.** On 2 Mini, forward hooks such as vegaml's `find_layers` fire three times on the looped layers 6–18, so one must choose which pass to tap. *(inference)* Treat the pass index as a hyperparameter, alongside layer depth.
- **No precedent.** No vegaml port and no third-party System One use of any Inkling model exist.
- **Evidence to weigh** *(inference for the implication)*:
  - Jeff-1.1's head-only frozen-backbone arm scored *below* zero-shot readout (78.9 vs 79.6), while the full LoRA recipe reached 84.4 ([@jeff]).
  - Small Heads finds scale dominates, with a cosine head at 82.0/78.0 ID/OOD on 0.6B vs 85.8/89.8 on 8B, and finds that base, instruct and thinking checkpoints tie ([@smallheads]).
  - So a 2 Mini reader is a test of a looped MoE backbone at roughly 3–5B active, not a guaranteed upgrade over Qwen3-8B.

### Arm 3: teacher

- **Reasoning teacher.**
  - The precedent is Jeff-1.1, whose teacher was Qwen3.8-27B with reasoning, averaged over 2 samples and mixed 50/50 with dataset labels for public data.
  - Cost on Inkling 2 Small: 261k items × (1k prompt + 2k thinking tokens) ≈ $78 prefill + $522 sampling. *(inference)* The 261k matches DecisionBank's size (261,561 rows).
- **Readout teacher.** Average the $K$-candidate readout logprobs over 2–4 option permutations. This yields distributions directly, at roughly prefill cost only.
  - *(inference)* This also avoids the Brier penalty that reasoning introduced on Qwen3.8-27B in S1-Arena.
- **Licensing** *(inference)*. The weights are Apache-2.0, so Inkling labels avoid the terms-of-service question raised by distilling from Jev outputs. The landscape notes judged Inkling-Small (12B active) the realistic teacher size.
- **Caveat.** The teacher's own calibration is unmeasured (no ECE). Its distributions should be checked against held-out human labels before use as soft targets *(inference)*.

### Recipe sources and training costs

- **LoRA Without Regret** ([TML blog](https://thinkingmachines.ai/blog/lora/), 2025-09-29; [@tml-lora]):
  - use a LoRA learning rate of about 10× the full-fine-tune rate;
  - apply LoRA to all layers, especially MLP/MoE;
  - α=32, with A initialised at $1/\sqrt{d_{\text{in}}}$ and B=0;
  - LoRA dislikes large batches;
  - rank 1 suffices for RL.
- **On-Policy Distillation** ([TML blog](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025-10-27; [@tml-opd]):
  - the per-token advantage is $-\mathrm{KL}_{\text{rev}}(\text{student}\,\Vert\,\text{teacher})$ at discount 0;
  - about 9–30× cheaper than SFT and about 50–100× cheaper than RL in their AIME setup;
  - recipe at `tinker_cookbook/recipes/distillation`.
- **Tinker training:** $0.75/M tokens (2 Mini) and $1.50/M (2 Small) at 64K.

### Summary of arms

| Arm | Model | Route | Needs | Approximate cost | Main caveat |
|---|---|---|---|---|---|
| Zero-shot readout | 2 Mini / 2 Small (any route); Inkling / -Small (Tinker or self-host) | Tinker `target_prompt_logprobs`; OpenRouter `logprobs` for v2 | Forced answer prefix; per-type $T$; permutation averaging | $12–$146 per 78M prefill tokens | Not same-backbone; no published ECE |
| Frozen reader | 2 Mini | Self-host, transformers | One 32–80 GB GPU; choice of loop pass | Hardware only | No precedent; head-only frozen arms underperform in Jeff-1.1 |
| Teacher | 2 Small (reasoning) or any v2 (readout) | Tinker / OpenRouter | Sampling budget or permutation readout | About $600 (reasoning, 261k items) or prefill only | Teacher calibration unmeasured |

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **An Inkling 2 Mini or 2 Small letter-readout arm** next to the same-backbone Qwen3-8B readout. It is cheap (about $12–$23 for the S1-Arena non-agentic set) and needs no training.
- **Permutation-averaged readout logprobs as soft targets.** Use these before paying for reasoning traces, and calibrate the teacher on held-out labels first.
- **TML's LoRA guidance** (a learning rate about 10× the full-fine-tune rate, all layers including MLP) when we add LoRA to the backbone.

**Avoid**
- **OpenRouter for v1 models**, which offers no logprobs.
- **Trusting thinking-effort "none"** on 2 Mini. Force the answer slot instead.
- **Citing Inkling's calibration as evidence for proper-scoring RL.** No ECE is published, and the forecasting numbers come from a different checkpoint.
- **The larger models as per-request frozen readers.**

## References

- TML posts:
  - Introducing Inkling [@inkling]: [thinkingmachines.ai/news/introducing-inkling](https://thinkingmachines.ai/news/introducing-inkling/)
  - Inkling 2 Small [@inkling2small]: [thinkingmachines.ai/news/inkling-2-small](https://thinkingmachines.ai/news/inkling-2-small/)
  - News index: [thinkingmachines.ai/news](https://thinkingmachines.ai/news/)
- Model cards:
  - [Inkling](https://huggingface.co/thinkingmachines/Inkling)
  - [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small)
  - [Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
  - [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) [@inkling2mini]
  - HF org listing: [HF API](https://huggingface.co/api/models?author=thinkingmachines)
- Chat template: [chat_template.jinja](https://huggingface.co/thinkingmachines/Inkling-2-Small/raw/main/chat_template.jinja)
- Tinker:
  - pricing [@tinker-pricing]: [thinkingmachines.ai/tinker/pricing](https://thinkingmachines.ai/tinker/pricing/)
  - SDK: [samplingclient.md](https://github.com/thinking-machines-lab/tinker/blob/main/docs/api/samplingclient.md)
- OpenRouter endpoints API: [openrouter.ai/api/v1/models](https://openrouter.ai/api/v1/models)
- vLLM recipes:
  - [Inkling](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling.yaml)
  - [Inkling-Small](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-Small.yaml)
  - [Inkling-2-Small](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-2-Small.yaml)
- transformers:
  - [model directories](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
  - [inkling2.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/inkling2.md)
- TML recipes:
  - LoRA Without Regret [@tml-lora]: [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)
  - On-Policy Distillation [@tml-opd]: [thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Context evidence:
  - S1-Arena [@s1arena]: [arXiv 2610.21244](https://arxiv.org/html/2610.21244)
  - Jeff-1.1 [@jeff]: [arXiv 2610.17205](https://arxiv.org/html/2610.17205)
  - Small Heads [@smallheads]: [arXiv 2611.15592](https://arxiv.org/html/2611.15592)
  - Intern-Decision readout [@interndecision]
  - DecisionBank [@decisionbank]
  - Jeva HF paper page (Inkling as generator metadata): [huggingface.co/papers/2610.18773](https://huggingface.co/papers/2610.18773)
  - Jev-Omni card: [huggingface.co/akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni)
