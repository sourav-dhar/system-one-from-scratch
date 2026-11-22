# ADR 0003: Re-sequenced architecture: the typed-decision model on a forked state first, the cached-option path behind a go/no-go

- **Status:** accepted, 2026-11-22. **Supersedes** [ADR 0001](0001-architecture-late-interaction-two-path.md).
- **Deciders:** @sourav-dhar
- **Evidence:**
  - the independent reviews [01](../research/reviews/01-technical-review.md), [02](../research/reviews/02-fact-check.md), [03](../research/reviews/03-architecture-review.md) and [04](../research/reviews/04-new-evidence-for-the-lock.md), and the [response tracker](../research/review-response.md);
  - [survey](../research/survey.md) §3, §8–§12;
  - [start-here guide](../research/README.md) §2 and §12.

## Context

ADR 0001 chose "B designed for C":
- question-blind state and question towers, fused late;
- a late-interaction block over cached option keys and values, with a coverage gate;
- a joint `prefix_bidir` slow path later;
- an Inkling 2 Mini frozen-reader arm.

The independent reviews found five problems with it:
- **The towers misread the evidence.**
  - NDM's question tokens attend to the state through its prefix-tree mask, and Jeva encodes type, instructions and state in one request template.
  - Question-blind towers drop the state-question interaction that typed decisions depend on ([01](../research/reviews/01-technical-review.md) issue 1).
- **The headline motivation is not a fair comparison.**
  - It set Jeva's 78.30 against a zero-shot CLM.
  - InternDecision-4B, a plain letter readout trained on 17,880 items, scores 78.70 on the same suite ([03](../research/reviews/03-architecture-review.md) §1.1).
- **Frozen late interaction has no supporting evidence** ([03](../research/reviews/03-architecture-review.md) §2.1).
- **The slow path's mask does not fit the backbone.**
  - `prefix_bidir` cannot be honoured on Gated DeltaNet layers.
  - Bidirectional attention cost 3.5–6.0 points in distribution and 5.4–11.8 out of distribution in Small Heads ([01](../research/reviews/01-technical-review.md) issue 19).
- **Inkling 2 Mini does not fit the target GPUs as specified.** It needs about 53 GB in BF16, and its shipped FP4/MX formats are Blackwell-only ([03](../research/reviews/03-architecture-review.md) §5).

Eight newer papers, read in full ([04](../research/reviews/04-new-evidence-for-the-lock.md)), separate the design's two caching bets:
- **Exact state reuse is solved, including on hybrid backbones.** Flide forks each question from the state's exact KV and recurrent state on Gated DeltaNet hybrids. It has open code and is up to 9.5× faster than vLLM/SGLang when many questions share a state.
- **Options need not see each other.** Prism-1's per-option blocks read the state but not other options. They cost at most 0.9 points against a trained letter readout at 8–12B, with zero order flips. Letting options attend to each other *lowered* accuracy.
- **Staying causal is cheap.** Removing the causal mask gained about 1.4 points in one seed (GLiDE), and forking requires a causal state.
- **State-independent cached options remain unproven.** No paper tests them with a late block. Their closest relatives trail:
  - MetaEncoder, where the cross-encoder leads at every size;
  - a frozen-LM rebuild of the CLM recipe, at 46% on MedMCQA.

State reuse and option caching are therefore separable. The first is solved; the second carries the accuracy risk.

## Decision

1. **M0, harness and baselines first.**
   - Reproduce two anchors on a pinned S1-Arena: CLM-v0.1-8B at 41.4%, and the Qwen3-8B answer-token readout at 59.3%.
   - Run the calibrated same-backbone readout, which is the bar.
   - Run a linear probe (seen and unseen labels), fine-tuned DeBERTa-v3-large, BM25, and reference models on one harness.
   - Add a W2 quality suite and the robustness battery. Details are in [plan v3](../research/plan.md).
2. **M1, the typed-decision model (W1).**
   - **Backbone and adaptation:** Qwen3.5-4B with LoRA at a conservative learning rate. Frozen Qwen3-8B stays the reproduction anchor.
   - **Causal attention throughout.** The state is prefilled once, and each question forks from the state's exact KV and recurrent state (Flide-style).
   - **Option scoring:** each option is scored in its own block, which reads the state and question but not other options, at shared positions. This gives exact permutation equivariance.
   - **Ablations:**
     - an in-context bilinear head over candidate endpoints (Decision 1.0, Eq. 3);
     - a comparison head over the final option states.
   - **Readouts:**
     - *choice:* a softmax over option scores;
     - *noul:* a head that ignores option text, initialised from the LM-head rows of the answer words;
     - *score:* an ordinal head, reporting the expectation;
     - *null:* a coverage gate trained only on none-pairs built from hard-label choice items, and evaluated only on open-set protocols.
   - **Calibration:** a post-hoc temperature per question type, fitted leave-source-out ([ADR 0004](0004-data-calibration-and-decontamination-amendments.md)). It is orthogonal to the architecture ablations.
3. **M2, the cached path for agent loops (W2).**
   - Options are encoded once, without the state. A late-interaction block runs over the forked state's pooled representation, followed by the coverage gate.
   - It is kept only if it passes the go/no-go below.
4. **Inkling.**
   - It serves as a readout baseline and distillation teacher through the Tinker API.
   - It also gets an off-target frozen-reader arm on a rented H100. Its cost is estimated and approved before it runs.
   - Inkling is not a serving backbone.
5. **Dropped or deferred.**
   - Question-blind towers (an M2 ablation only).
   - `prefix_bidir` (an ablation on attention-only backbones only).
   - Matryoshka and int8 option caches, until catalogs exceed about 1M options.
   - The Jev API reference: its published 72.6% suffices.

## Acceptance criteria (pre-registered)

- **Statistics.** Every trained configuration gets 3 seeds. An effect counts only if it exceeds 2× the seed standard deviation under a task-cluster paired bootstrap; item-level McNemar overstates significance on about 78k items.
- **M0 passes** when both anchors reproduce within 1.0 pp on the pinned S1-Arena version.
  - If either still misses after two fix rounds, we report only against our pinned version.
- **M1 is a success** if both hold:
  - On S1-Arena accuracy, it beats the calibrated zero-shot readout of its own backbone. The cluster-bootstrap CI must exclude zero; the target is +3 points.
  - It is no worse than that readout on Brier.
- **M1's guardrail.** Held-out-source accuracy is no more than 15 points below seen-source accuracy.
- **M2 go/no-go.** Compare the same head on cached options and on in-context options. The cached path is kept only if both hold:
  - its S1-Arena accuracy is within 3 points of the in-context head (cluster-bootstrap CI);
  - its W2 latency is at least 5× lower, at K ≥ 50 or on revisits.

  Otherwise W2 is served by the M1 model with state forking.
- **Serving.**
  - FP8 (W8A8 on Ada) is adopted only if it changes at most 0.5% of decisions, with max |Δp| ≤ 0.02. Otherwise we serve bf16 on the L40S.
  - The coverage gate is disabled for any type whose closed-set Brier worsens by more than 0.01 with it on.

## Consequences

- **The first trained model competes directly on typed decisions.** Option caching becomes a measured result rather than an assumption.
- **Serving needs forked KV and recurrent states.** Flide's open implementation is evaluated before any kernel is written.
- **The README's architecture section and graphic** move to the M0/M1/M2 view.
- **Budget:**
  - M0 is ≤ $60 (approved).
  - M1 is about $200–400, and the H100 Inkling arm is extra. Both are estimated, and approved at milestone start.
- **Time:** about 3 months to the M1 decision for one engineer *(estimate, [03](../research/reviews/03-architecture-review.md) §6)*.

## Alternatives considered

- **"B modified" first** (review 03's recommendation before review 04). It is not the first step, because its central bet (state-independent cached options) has no published evidence. It survives as M2.
- **Joint readout only.** This would give up the agent-loop latency story. M1 already captures most of its benefit.
- **ADR 0001 as written.** Superseded for the reasons above.
