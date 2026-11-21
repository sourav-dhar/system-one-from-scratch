# Hosted decision APIs: Jev, Microsoft-Decision-1 and the OpenAI Decisions API

> **Family:** decoder-readout per the research index, but the architecture of all three is undisclosed · **Team:** TypeSafe AI (Jev; founder Diogo Almeida, previously at OpenAI); Microsoft (Microsoft-Decision-1); OpenAI (Decisions API) · **Released:** Jev announced 2026-09-15 after "two years in stealth", Jev 1.14.0 on 2026-10-14; Microsoft-Decision-1 on 2026-10-09; OpenAI Decisions API announced at DevDay 2026-09-29, GA 2026-10-22 · **License:** closed, hosted only · **Code/weights:** none. Microsoft-Decision-1 open weights are not stated (hosted only per a search summary) · **Sources read:** TypeSafe [System One concepts](https://docs.typesafe.ai/concepts/system-one.md), [API](https://docs.typesafe.ai/api/system-one.md), [models](https://docs.typesafe.ai/models.md), [launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md); Archer Hume's ["Jev's Architecture Unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked/); Victor Dibia's [newsletter](https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision); the [Microsoft blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/); the [OpenAI guide](https://developers.openai.com/api/docs/guides/decisions) (read in the decoder-family dossier; the earlier landscape note had relied on [Digital Applied](https://www.digitalapplied.com/blog/openai-decisions-api-ga-pricing-400-questions) and noted the guide was "not fetched directly"); secondary write-ups (Firecrawl, Parallel, AICoder); and every third-party evaluation in the notes that includes these systems. Not available: weights, readouts and training for all three; Microsoft's full list of 8 perturbation types; OpenAI internals and calibration.

## Summary

These are the three closed, hosted decision APIs that the open systems are measured against. **Jev** (TypeSafe) is the original "System One" model: it returns typed decisions and probabilities and never generates text ([docs](https://docs.typesafe.ai/concepts/system-one.md)) [@jev-docs]. **Microsoft-Decision-1** is a post-trained Qwen3.5-9B for "single-pass decision scoring" ([blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)) [@msdecision]. The **OpenAI Decisions API** exposes `gpt-6-luna` with predicate, choice and score question types and explicit prefix-cache billing ([guide](https://developers.openai.com/api/docs/guides/decisions)) [@openai-decisions]. The most important evidence about how such systems work is Archer Hume's black-box probing of Jev (about 10k API calls): latency grows with state length but is flat in the number of questions and independent of the number of options, and adding an irrelevant option shifts the log-odds between existing options — consistent with a joint pass over state, questions and options that branches questions from a cached state prefix, and inconsistent with a pure independent-score bi-encoder ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)) [@jev-hume]. Kev's architecture was built from this write-up [@kev-repo]. In System1-Arena the three score 72.6% (Jev), 70.9% (Microsoft) and 68.6% (OpenAI) ([S1-Arena](https://arxiv.org/html/2610.21244)) [@s1arena].

## Architecture

### Jev: what TypeSafe discloses

**API shape** ([API docs](https://docs.typesafe.ai/api/system-one.md)) [@jev-api].
- Request: `POST /v1/systemone` with `model`, `state` (string, object or array), a map of named `questions`, and optional image `assets`.
- Question types:
  - `noul`: probability of yes;
  - `choice`: 1–255 options; returns `choice`, `confidence` and `probabilities`;
  - `score`: 1–10 ordered levels; returns the probability-weighted `score`, `probabilities` and `legend`;
  - `box`: image box, Jev 1.14 only.
- Context: 64k tokens per request and 32k per question; 262k behind a "danger zone" header ([models](https://docs.typesafe.ai/models.md)).

**Architecture statements.** Officially undisclosed ([Dibia](https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision)). TypeSafe mentions "a new model architecture and a parallel sampler that generates all outputs in one query", and two-stage scoring above 255 options ([launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)) [@jev-launch].

### Jev: Archer Hume's third-party probing

From about 10k API calls ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)) [@jev-hume]:

| Probe | Finding |
|---|---|
| State length | Latency grows with state length: 57.5 ms at 360 tokens, 218 ms at 29.8k tokens |
| Number of questions | Latency flat from 1 to about 100 questions (72–86 ms) |
| Number of options | Latency independent of option count: 200 options cost about the same as 2 |
| Token accounting | Additive token accounting suggests the state is encoded once and questions branch from it, consistent with prefix caching |
| Option interaction | Adding an irrelevant option shifts the log-odds between existing options, so options interact. **This rules out a pure independent-score bi-encoder like CLM** |
| Head type | Pointer vs 256-slot head: unresolved |
| Backbone | A causal MoE is suggested, explicitly "an inference, not a measurement" |
| Tokenizer | Closest to Qwen (348 of 415 probes) |
| Confidence field | `confidence` for Choice is computed client-side as $(p_{\max} - 1/K)/(1 - 1/K)$ |

The landscape note's synthesis *(inference)*: Jev reads state, questions and options jointly in one causal pass, with question branches sharing a cached state prefix.

### Microsoft-Decision-1

- Post-trained Qwen3.5-9B for "single-pass decision scoring"; the readout is not described ([blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)) [@msdecision].
- Context 32,768 tokens ([AICoder](https://aicoder.com/news/news-20261010-microsoft-decision-1-fast-decision-model-foundry-openrouter)).
- Hosted on Foundry and OpenRouter; open weights are not stated ([search summary](https://wilsonwu.me/blog/2026/microsoft-decision-1/)).

### OpenAI Decisions API

- Model: `gpt-6-luna`, the only model at GA; described by Firecrawl as "GPT-6 Luna, constrained" ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev)).
- Question types: predicate, choice and score; 1–400 questions per request ([guide](https://developers.openai.com/api/docs/guides/decisions)) [@openai-decisions].
- Returns predicate probabilities, choice with per-option probabilities and confidence, and score as an expectation ([Digital Applied](https://www.digitalapplied.com/blog/openai-decisions-api-ga-pricing-400-questions)).
- **Caching:** an implicit cache breakpoint sits at the end of the shared `input`, with a ≥1,024-token prefix; **question content is not part of the reusable prefix**.

| Property | Jev | Microsoft-Decision-1 | OpenAI Decisions API |
|---|---|---|---|
| Passes per request | "All outputs in one query" (vendor); latency flat in #questions and #options (Hume) | "Single-pass decision scoring" (vendor) | Not disclosed |
| State prefix reusable | Consistent with prefix caching (Hume, inferred from token accounting) | Not disclosed | Yes: implicit breakpoint at the end of the shared input (≥1,024 tokens) |
| Option embeddings reusable | No evidence; options interact (Hume) | Not disclosed | No: question content is not in the reusable prefix |
| Max K | 255; two-stage scoring above | Not stated | Not stated in the notes |
| Context | 64k per request, 32k per question; 262k "danger zone" | 32,768 | Not stated in the notes |

## Training

- **Jev:** "RLCD", "Reinforcement learning for calibrated decisions" ([primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)) [@jev-primer]; details are unpublished. Jev is not fine-tuned on customer data ([models](https://docs.typesafe.ai/models.md)).
- **Microsoft-Decision-1:** post-trained Qwen3.5-9B; the training method is undisclosed.
- **OpenAI Decisions API:** undisclosed.

## Calibration

### Jev

| Source | Finding |
|---|---|
| [Hume](https://archerhume.com/posts/jevs-architecture-unmasked/) [@jev-hume] | MMLU ECE 0.0313 on 1,200 items |
| [TypedBench](https://arxiv.org/html/2610.11392) [@typedbench] | **Underconfident:** fitted temperatures T = 0.66 and 0.21 (< 1); ticket-priority ECE 2.2× the noise floor; Bayes-cost decisions from Jev's probabilities cost more than argmax (2,300 vs 1,080 per 10⁴ decisions, i.e. value of calibration −1,220) |
| [StartLux](https://arxiv.org/html/2610.17309) [@startlux] | **Overconfident:** claims 13.8 pp calibration error on DecisionPivot ("systematic overconfidence") |
| [Bongard](https://arxiv.org/html/2609.39111) [@bongard] | DecisionBench ECE 0.128 (Bongard 0.063) |
| [Jeva](https://arxiv.org/html/2610.18773) [@jeva] | Noul Brier 0.0342 (Jeva 0.0184) |
| [Probability axioms](https://arxiv.org/html/2609.33209) [@axioms] | Negation-complement MAD 0.064 (Qwen3.8-27B first-token 0.293, verbalized 0.122); partition sum 1.14, exceeding 1 on 159 of 160 items; noise floor 0.013 |
| [ProbabilityBench](https://arxiv.org/html/2610.17016) [@probabilitybench] | Expected KL to exact $p^*$ improves with demonstrations, 0.248 → 0.203 |
| [Rafe & Das](https://arxiv.org/html/2610.00346) [@rafedas] | Yes/no swap moves AUC from 0.89 to 0.31 |

**The direction of Jev's miscalibration is contested in the notes:** TypedBench finds it underconfident (T < 1) on its suites, while StartLux claims systematic overconfidence on DecisionPivot; Hume measures a low MMLU ECE (0.0313). The suites differ, and the notes do not reconcile them.

### Microsoft-Decision-1 and OpenAI Decisions

- Microsoft-Decision-1: no calibration claims recorded; S1-Arena Brier 0.369.
- OpenAI Decisions API: "No calibration claims" in the guide; S1-Arena Brier 0.424. ProbabilityBench reports that "Luna's" KL worsens with demonstrations (0.199 → 0.411); the notes do not say whether this was measured through the Decisions API or as a generative model.

## Results

### Head-to-head: System1-Arena ([2610.21244](https://arxiv.org/html/2610.21244)) [@s1arena]

Jev is called through the Vercel gateway and Microsoft-Decision-1 through OpenRouter; hosted latencies include network time.

| Model | Accuracy | Brier | p50 | Agentic step acc. / loop success |
|---|---|---|---|---|
| Decision-2.0-Vega-27B (open, reference) | 73.0 | 0.343 | 228 ms | 67.8 / 26.5 |
| **Jev 1.13** | 72.6 | 0.349 | 321 ms | 67.8 / 17.4 |
| **Microsoft-Decision-1** | 70.9 | 0.369 | 304 ms | – |
| **OpenAI Decisions** | 68.6 | 0.424 | 140 ms | – |
| Opus 5.5 (best closed LLM) | 81.1 | 0.232 | – | – |

- System-1 vs its own backbone (Fig. 5): Microsoft-Decision-1 +5.6; OpenAI Decisions −2.6.
- No-answer rates: 13.2–13.7% of single steps for Jev, Vega and Microsoft-Decision-1, which depresses their accuracy.

### Jev in other evaluations

| Evaluation | Jev result | Comparator(s) | Source |
|---|---|---|---|
| JevBench v1.8.7 Capability | 77.1 (1.13), 75.0 (1.14) | OpenAI Decisions 73.5 (600-item subset) | [Parallel](https://parallel.ai/articles/jevbench-benchmark.md) |
| OpenDecisions Capability / JevBench-Open | 80.1 / 86.6% | Jeff-1.1-Large 89.0 / 93.1% | [Jeff](https://arxiv.org/html/2610.17205) [@jeff] |
| Knowledge (Jeff) | 92.9 | Jeff-1.1-Large 89.5 | [Jeff](https://arxiv.org/html/2610.17205) |
| DI 0.3 public | 57.96 | StartLux 27B AutoThink 73.37 | [StartLux](https://arxiv.org/html/2610.17309) [@startlux] |
| DecisionPivot pair accuracy | 31.0 | StartLux 27B AutoThink 86.9 | [StartLux](https://arxiv.org/html/2610.17309) |
| Kev suites | breadth-v1 0.757; breadth index 54.0; MMLU-Pro 0.840; transfer-v4 dev 0.857; unanswerable at p ≥ 0.9 0.09; `deadline` 0.95; confident errors 3.7% | Kev-4B / Kev-9B | [Kev](https://github.com/jaredpalmer/kev) [@kev-repo] |
| Jeeves suites | test 0.857; JevBench public 0.866; hard 0.730 | Jeeves v2 0.905 / 0.896 / 0.793 | [Jeeves](https://github.com/PostHog/jeeves) [@jeeves] |
| SanSi | 78.9 overall; 83.5 on sources none of their models trained on | SanSi-2.6B 71.9 on that subset | [SanSi](https://arxiv.org/html/2610.07730) [@sansi] |
| AnyS1 Dev | 78.4 | AnyS1 83.8 | [AnyS1](https://arxiv.org/html/2611.23751) [@anys1] |
| Jeva permutation flips (Choice) | 17.00% | Jeva 0.00% | [Jeva](https://arxiv.org/html/2610.18773) [@jeva] |
| Runoff flips | 5.3 → 0.9 when wrapped by Runoff | – | [Runoff](https://arxiv.org/html/2610.18233) [@runoff] |
| Rafe & Das, labeled tasks | typed-decisions 0.732; CLINC-150 0.913; emotion 0.504 | fine-tuned DeBERTa-v3-large 0.793 / 0.953 / 0.938 | [2610.00346](https://arxiv.org/html/2610.00346) [@rafedas] |
| Rafe & Das, option count | 0.990 → 0.913 from 5 to 150 options | – | same |
| Rafe & Das, out-of-scope | accepts 0.310 | Kev-9B 0.015 | same |
| Rafe & Das, yes/no swap | 50.5 flips per 100 | Laya 91.3; Gemma 17.2; fine-tuned decision models ≤6.5 | same |
| Rafe & Das cascade | this-that → Jev on CLINC: 0.927 at 0.43 of Jev's cost | – | same |
| TypedBench | priority 0.874 [0.708, 0.946] across friendly wordings; paraphrased phishing negation −23 pp; out-of-scope AUROC 0.77 | – | [TypedBench](https://arxiv.org/html/2610.11392) [@typedbench] |
| Reading Right, Acting Wrong | DD 0.763; 73.1% of errors follow correct fact reading; fact probes composed in code 0.791 → 0.996; ALFWorld next-sub-goal +16.3 | – | [2610.18579](https://arxiv.org/html/2610.18579) [@readingright] |
| JevAdvBench | ASR 46.3 (task override H1 92.0; flawed reasoning C3 77.2); 23.5% risk at the 0.8 gate; noise floor 0.17%; raising the gate to 0.99 still leaves 7.9% wrong | open models ASR 34.0–51.7; CLM 34.5 | [2609.31142](https://arxiv.org/html/2609.31142) [@jevadvbench] |
| Evidence audit | trails strong models by 14.6 on JudgeBench and 19.8 on hard RM-Bench | – | [2609.32160](https://arxiv.org/html/2609.32160) [@tdmaudit] |
| Bongard Table 12 (24 benchmarks) | Jev wins 17, ties 1, Bongard wins 6; Banking77 79.8, JudgeBench 79.7, prompt injection 75.9 | Bongard 93.1 / 55.4 / 56.9 | [Bongard](https://arxiv.org/html/2609.39111) [@bongard] |
| Beveridge | Beveridge-Bench 87.9; Banking77 79.2; automation at 95% accuracy 59% | Beveridge 93.0 / 63.6 / 76% | [Beveridge](https://arxiv.org/html/2611.18655) [@beveridge] |
| this-that-model-1.0 | 68 questions: 0.765 / Brier 0.133; 2,250 questions: 0.803 | this-that 0.941 / 0.042; 0.844 | [this-that](https://arxiv.org/html/2609.23886) [@thisthat] |
| Laya card | typed-decisions 0.727; Banking77 0.870 | Laya fine-tuned 0.766; Banking77 0.425 | [Laya](https://huggingface.co/convaiinnovations/laya) [@laya-card] |
| vegaml per-question learner | 89.5% → 92.6% after 120 answers | vegaml 84.7% → 96.0% | [vegaml](https://github.com/NandhaKishorM/vegaml) [@vegaml] |
| SemIf | TypeSafe-subset agreement 0.883 | SemIf letter logits 0.845 | [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md) [@semif] |
| NanoJev | ViZDoom Basic 56/128 | NanoJev 128/128 | [NanoJev](https://github.com/TianyuCodings/NanoJev) [@nanojev] |
| CLM README, zero-shot | T-Rex 5/5 (149.8 ms); BFCL v4 99.2% (125.5 ms); WikiRacing 30/30 (225 ms); Super Mario 5/5 (132.6 ms) | CLM 5/5 (16.5 ms); 95.2% (76.8 ms); 26/30 (79.8 ms); 5/5 (33.5 ms) | [CLM](https://github.com/Contrastive-LM/CLM) [@clm-repo] |
| CLM README, fine-tuned verifier | DeepSWE Bo4 71.1% (449 ms); Terminal-Bench 2.1 Bo5 83.1% (131 ms) | CLM 81.6% (79 ms); 87.6% (32 ms) | same |
| CLM T-Rex result JSONs | agreement with planner 0.987; shield interventions 28; model p50 131.9 ms | CLM 0.658; 4,883; 2.6 ms | same |

**Inconsistencies across these sources.**
- **Banking77:** 0.870 (Laya card), 79.8 (Bongard), 79.2 (Beveridge). Different harnesses and splits; not comparable.
- **Order sensitivity:** 17.00% Choice flips under permutation (Jeva) vs 5.3 native flips (Runoff). Different suites and protocols.
- **"JevBench":** Capability 77.1 (Parallel) vs JevBench-Open 86.6% (Jeff) vs JevBench public 0.866 (Jeeves). The landscape note warns that leaderboards with the same name use different protocols and are not comparable.
- **Decision Index:** Jev 57.96 (StartLux Table 4), 57.91 (StartLux repo), 57.9 (pplx-decider card).
- **JevAdvBench caveat:** 68% of its labels are Jev's own stable answers, and adaptive search was run only on Jev.

### Microsoft-Decision-1

- Vendor claims ([blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)): highest accuracy on 36 held-out benchmarks; flip rate 1.3% averaged across 8 perturbation types. The decoder-family dossier records 0 flips "under option paraphrase, reversal or shuffle"; the landscape note records 0 "under option shuffle or reversal" (without paraphrase).
- Independent: S1-Arena 70.9% / Brier 0.369 / 304 ms; +5.6 over its base.

### OpenAI Decisions API

- Vendor: 150 ms vs 1.6 s for a normal Luna call ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev), vendor slide).
- Independent: S1-Arena 68.6% / Brier 0.424 / 140 ms (−2.6 vs base); JevBench Capability 73.5 on a 600-item subset ([Parallel](https://parallel.ai/articles/jevbench-benchmark.md)).

## Serving and efficiency

| | Jev | Microsoft-Decision-1 | OpenAI Decisions API |
|---|---|---|---|
| Price | $0.042 per M input tokens; output free ([models](https://docs.typesafe.ai/models.md)) | $0.042 per M input tokens; output free | $0.10 per M uncached input; $0.01 per M cached input; output not billed |
| Vendor latency | 70–500 ms end to end ([launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)) | p50 about 35× faster than GPT-6 Sol | 150 ms vs 1.6 s for a normal Luna call |
| Independent latency | OpenRouter p50 0.21 s ([Firecrawl](https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev)); S1-Arena 321 ms; Jeff App. J API p50 0.189 s; Hume 57.5 ms (360 tokens) to 218 ms (29.8k tokens) | S1-Arena 304 ms | S1-Arena 140 ms |
| Scaling | Flat 72–86 ms from 1 to ~100 questions; independent of option count (Hume) | Not reported | 1–400 questions per request; cached shared input ≥1,024 tokens |
| Access | TypeSafe API, Vercel gateway, OpenRouter (as used by evaluations) | Foundry and OpenRouter | OpenAI API |

Third-party comparisons: CLM claims 4–9× lower latency than Jev on its own tasks ([VentureBeat](https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests)) and "13× faster than Jev" at about 1k candidates with caching ([HF card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)) [@clm-card].

## Flaws and failure modes

**(a) Flaws of these systems**

*Jev*
- **Label-name and wording sensitivity:** 50.5 flips per 100 on yes/no swap (AUC 0.89 → 0.31) [@rafedas]; priority accuracy ranges 0.708–0.946 across friendly wordings; a paraphrased negation costs 23 pp [@typedbench].
- **Order sensitivity:** 17.00% Choice flips under permutation [@jeva]; 5.3 native flips in Runoff [@runoff].
- **Out-of-scope acceptance:** 0.310 at a held-out threshold [@rafedas]; out-of-scope AUROC 0.77 [@typedbench].
- **Adversarial state injection:** ASR 46.3, task override 92.0; 23.5% risk at the 0.8 gate [@jevadvbench].
- **Incoherent probabilities across formats:** partition sums of 1.14, exceeding 1 on 159 of 160 items [@axioms].
- **Acting wrong after reading right:** 73.1% of errors follow correct fact reading [@readingright].
- **Calibrated probabilities that do not pay off:** Bayes-cost decisions cost more than argmax (−1,220 per 10⁴) [@typedbench].
- **Supervised baselines win on labeled tasks:** fine-tuned DeBERTa-v3-large beats Jev on typed-decisions, CLINC-150 and emotion [@rafedas].
- **Weaknesses on intent routing and judging:** Bongard beats Jev on Banking77 (93.1 vs 79.8); Jev trails strong models by 14.6 on JudgeBench [@tdmaudit].
- **Agentic loop success** 17.4% despite 67.8% step accuracy (Vega 26.5% at the same step accuracy) [@s1arena].
- **Opacity:** architecture, training data and RLCD details unpublished.

*Microsoft-Decision-1*
- Undisclosed readout, training and weights; the full list of 8 perturbation types is not published; no-answer rate 13.2–13.7% of single steps in S1-Arena.

*OpenAI Decisions API*
- Lowest accuracy and worst Brier of the three in S1-Arena (68.6%, 0.424), and −2.6 vs its own base.
- "Choice value changes can alter results even when descriptions stay the same" (guide).
- No published calibration.

**(b) Flaws of other systems these vendors claim to address**
- **Jev:** typed outputs instead of free text, one state shared across many questions, and score as an expectation over levels (docs). Hume's probing shows options interact, which the landscape note reads as ruling out a pure independent-score bi-encoder such as CLM.
- **Microsoft-Decision-1:** robustness to perturbations (1.3% average flip rate; 0 under shuffle and reversal), a claim aimed at order and wording sensitivity of LLM readouts.
- **OpenAI Decisions API:** latency and cost of full generative calls (150 ms vs 1.6 s for Luna).

## Implications for our design

All bullets are *(inference)*, grounded in the evidence above and the landscape and methods notes.

**Adopt**
- **A TypeSafe-compatible `POST /v1/systemone` server and client harness**, so that Jev, Kev, Decision 2.0, Laya, Microsoft-Decision-1 and OpenAI Decisions all become config entries on one benchmark (landscape note, top priority).
- **Jev as a cheap baseline** ($0.042 per M input tokens). Do not use it as a teacher without checking TypeSafe's [terms](https://docs.typesafe.ai/legal.md); Kev explicitly avoided Jev outputs, and the catalog warns that AutoTrust's JEV-27B / JEV-9B are distilled from Jev 1.13 outputs.
- **Jev's typed schema** (Choice / Score / Noul), score as an expectation over levels, one state shared by many questions, and a 255-option cap with two-stage scoring for larger sets.
- **Microsoft's 8-perturbation flip-rate protocol** and Rafe & Das's yes/no swap as standing robustness tests; a bi-encoder is order-invariant by construction, which should be measured and reported.
- **Prefix caching as the key serving optimisation:** OpenAI's explicit cached-prefix billing and Hume's flat latency in the number of questions both point to it. Note that OpenAI caches only the shared input, not question content.
- **Report agentic loop success, not only step accuracy:** equal step accuracy hid loop-success differences of 26.5% vs 17.4%.

**Avoid**
- **Treating Jev's probabilities as a calibration gold standard:** its miscalibration direction is contested (underconfident in TypedBench, overconfident on DecisionPivot), and its partition sums exceed 1.
- **A pure independent-score design if option interaction matters:** Hume's irrelevant-option probe shows the market leader is not one; a late-interaction block over cached option vectors (Jeva) is the notes' proposed middle ground [@jeva].
- **Treating the state as trusted input:** no evaluated defence against JevAdvBench-style injection exists in the notes.

## References

- [@jev-docs] https://docs.typesafe.ai/concepts/system-one.md
- [@jev-api] https://docs.typesafe.ai/api/system-one.md
- [@jev-primer] https://docs.typesafe.ai/introduction/machine-learning-primer.md
- [@jev-launch] https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe models and legal pages: https://docs.typesafe.ai/models.md, https://docs.typesafe.ai/legal.md
- Other TypeSafe posts listed in the catalog (content not summarised in the notes): https://typesafe.ai/blog/dont-call-it-a-decision-model, https://typesafe.ai/blog/jev-1-14, https://typesafe.ai/blog/antibenchmaxxing, https://typesafe.ai/blog/bitterest-lesson
- [@jev-hume] https://archerhume.com/posts/jevs-architecture-unmasked/
- Dibia: https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision
- [@msdecision] https://commandline.microsoft.com/microsoft-decision-1-model-foundry/ (also https://aicoder.com/news/news-20261010-microsoft-decision-1-fast-decision-model-foundry-openrouter, https://wilsonwu.me/blog/2026/microsoft-decision-1/)
- [@openai-decisions] https://developers.openai.com/api/docs/guides/decisions (also https://www.digitalapplied.com/blog/openai-decisions-api-ga-pricing-400-questions, https://www.firecrawl.dev/blog/openai-decisions-api-vs-jev, https://the-decoder.com/openai-launches-decisions-api-that-reduces-complex-evaluations-to-yes-no-or-pick-one/)
- Parallel (JevBench): https://parallel.ai/articles/jevbench-benchmark.md; [@jevbench] https://github.com/fstandhartinger/jevbench
- [@s1arena] https://arxiv.org/html/2610.21244
- [@typedbench] https://arxiv.org/html/2610.11392
- [@rafedas] https://arxiv.org/html/2610.00346
- [@runoff] https://arxiv.org/html/2610.18233
- [@readingright] https://arxiv.org/html/2610.18579
- [@axioms] https://arxiv.org/html/2609.33209
- [@jevadvbench] https://arxiv.org/html/2609.31142
- [@probabilitybench] https://arxiv.org/html/2610.17016
- [@tdmaudit] https://arxiv.org/html/2609.32160
- [@jeva] https://arxiv.org/html/2610.18773
- [@jeff] https://arxiv.org/html/2610.17205
- [@startlux] https://arxiv.org/html/2610.17309
- [@bongard] https://arxiv.org/html/2609.39111
- [@beveridge] https://arxiv.org/html/2611.18655
- [@thisthat] https://arxiv.org/html/2609.23886
- [@sansi] https://arxiv.org/html/2610.07730
- [@anys1] https://arxiv.org/html/2611.23751
- [@kev-repo] https://github.com/jaredpalmer/kev
- [@jeeves] https://github.com/PostHog/jeeves
- [@semif] https://github.com/TheoLeeCJ/SemIf-OpenJev
- [@nanojev] https://github.com/TianyuCodings/NanoJev
- [@laya-card] https://huggingface.co/convaiinnovations/laya
- [@vegaml] https://github.com/NandhaKishorM/vegaml
- [@clm-repo] https://github.com/Contrastive-LM/CLM
- [@clm-card] https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- VentureBeat: https://venturebeat.com/ai/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests
