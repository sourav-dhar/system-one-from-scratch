# ADR 0001: A two-tower late-interaction System One model, designed for a joint slow path

- **Status:** accepted, 2026-11-22
- **Deciders:** @sourav-dhar
- **Evidence:** [`docs/research/survey.md`](../research/survey.md) §3, §7, §8, §10–§11; per-system notes for [CLM](../research/systems/clm.md), [Jeva](../research/systems/jeva.md), [NDM](../research/systems/ndm.md), [Kev](../research/systems/kev.md), [Jeff-1.1](../research/systems/jeff.md), [Strands Decider](../research/systems/strands-decider.md), [Inkling](../research/systems/inkling.md)

## Context

We must serve two workloads from one model family on **one mid-range GPU** (L4, A10 or L40S):
- **typed decisions:** classification, yes/no, rubric scores, routing;
- **cached action selection:** agent loops over large reusable candidate sets.

The survey establishes four things:

1. **Our reproduction target, CLM-8B (a cached bi-encoder), is the fastest System-1 model but the least accurate on typed decisions.** System1-Arena: 41.4% at 41 ms, vs 59.3% for its own backbone's answer-token readout. The causes are specific:
   - state collapse from appending the question to the state;
   - context-free short options;
   - an uncalibrated scale saturated at its cap;
   - no null outcome;
   - independent (IIA) scoring.

   A linear probe on the same frozen embeddings beats its fine-tuned head (0.900 vs 0.795).
2. **Late interaction over cached option representations** gives set-dependence, a null outcome and exact permutation equivariance while keeping option caching. Jeva reports Typed 78.30 vs CLM 43.05, 0% order flips, and null AUROC 98.66. Jeva reports no component ablation and no frozen-backbone results.
3. **Joint decoders lead on typed accuracy**, but have no option caching, are order-sensitive (exact fixes cost 0.6–5.8 pp), and full bidirectional attention prevents prefix reuse. Strands v2's `prefix_bidir` mask keeps state reuse.
4. **Adapting the backbone (LoRA) beats a better head on a frozen backbone** (Jeff: 84.4 vs 78.9). Backbone scale matters up to about 4B and levels off after (Small Heads; Jeff).

## Decision

Build **Architecture B**, designed so that **Architecture C** can be added without rework.

**B: two-tower late interaction** (milestone 1)
- **Option tower.** Encodes each option *with its decision type but without the request*, so option representations and their per-layer K/V are cacheable across requests.
- **State and question towers.** Encode the state and the question **separately** and fuse them late. This removes CLM's suffix collapse. The state embedding is reusable across questions.
- **Decision block.** One Jeva-style block: the request vector attends over cached option keys and values; options never attend to each other. This gives set-dependence and exact permutation equivariance.
- **Outputs:**
  - scores $s_i = z_L^\top u_i$;
  - a **coverage gate** for the null outcome;
  - a **Noul head** that does not depend on option text;
  - an expectation over levels for Score.
- **Calibration.** A post-hoc procedure: per type, with $K$ as an input, fitted on held-out *datasets*; plus a per-workload refit tool.
- **Representation regularizers.** Matryoshka prefixes (for 128-d option caches) and batch centring.
- **CLM++ is not a separate model.** The fixes that turn CLM into "CLM++" are the first rungs of B's ablation ladder, each toggled and measured.

**C: two paths** (milestone 2, designed for now)
- A **joint slow path** on the *same* backbone: state, question and rendered options in one sequence under a `prefix_bidir` mask, LoRA, and a readout initialized from LM-head rows.
- A **calibrated-confidence gate** routes uncertain fast-path decisions to the slow path.
- Interfaces in milestone 1 (encoder registry, calibration, `/v1/systemone` server, harness) must already allow a second scorer and a router.

**Backbone arms for milestone 1.** All arms fit one mid-range GPU:

| Arm | Role |
|---|---|
| Frozen Qwen3-8B (last layer) | Anchor: reproduces CLM and isolates head and layout effects |
| Frozen Qwen3-8B intermediate layers (L/2, 3L/4) | Latency arm |
| Qwen3.5-4B + LoRA | Adaptation arm (the Jeva/Kev class) |
| Inkling 2 Mini, frozen reader (26.4B total / 3.4B active, about 20 GB) | Frontier open-model reader |
| Inkling via API: zero-shot letter readout baseline; optionally soft-target teacher | Reference, not deployed |

## Consequences

- **Benefits:**
  - keeps CLM's sub-millisecond cached path for agent loops;
  - order invariance is exact by construction;
  - a principled null outcome;
  - per-component ablations produce new evidence (frozen-backbone late interaction is untested in the literature).
- **Costs and risks:**
  - LoRA arms invalidate option caches per checkpoint, though not per request;
  - two calibrations to maintain once C lands;
  - with no published Jeva ablation, B's gain over CLM++ may be smaller than the headline numbers suggest. That is exactly what the ladder measures.
- **Out of scope for milestone 1:**
  - agent post-training on trajectories (#14);
  - adversarial robustness defences (none exist in the literature);
  - reasoning escalation.

## Alternatives considered

- **A only (CLM++).** Cheapest. Leaves IIA and null coverage partly unsolved. Kept as B's ablation baseline.
- **C from the start.** Highest ceiling, but the most engineering before the first comparable number.
- **Joint decoder only (Kev / Jeff / Strands style).** Strongest typed accuracy in the literature. It gives up option caching, which is central to the agent workload and to our latency target.
