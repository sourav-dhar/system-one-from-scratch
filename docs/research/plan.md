# Research and build plan, v2

- **Status:** accepted, 2026-11-22
- **Supersedes:** the Tier 1 plan in #31, written before the survey
- **Architecture:** [ADR 0001](../adr/0001-architecture-late-interaction-two-path.md)
- **Data:** [ADR 0002](../adr/0002-data-licensing-and-decontamination.md)
- **Evidence:** [survey](survey.md)

## Ground rules (carried over)

- **Noise floor.** Every trained configuration gets 3 seeds. An effect counts only if it exceeds 2× the seed standard deviation, tested with McNemar or a paired bootstrap.
- **One factor at a time** against a stated baseline.
- **Negative results are published.**
- **Every reported number** comes from a file in `results/`, produced by a committed config.
- **Calibration** is reported as ECE, Brier and NLL, on suites never used for fitting.
- **Budget gates.** Each milestone states its GPU budget and needs approval before it starts.

## Milestone M0: benchmark harness and reference implementations (approved, ≤ $60 Modal)

| Step | Deliverable | Done when |
|---|---|---|
| M0.1 | Typed-decision schema (`choice`/`noul`/`score`, null), loaders for System1-Arena (official adapter), OpenDecisions, typed-decisions, Jeva's BANKING77/MASSIVE K = 8/16 protocol | Unit tests on fixtures; loaders stream from the Hub |
| M0.2 | Metrics and robustness battery: accuracy, Brier, ECE, NLL per type; option shuffle/reversal flips; yes/no identifier swap; K-sweep 5–255; null detection (gold removed); latency with and without caches | Tests against hand-computed cases |
| M0.3 | `/v1/systemone`-compatible server and client, so references and our model share one harness | Contract tests |
| M0.4 | **Harness validation:** reproduce S1-Arena's CLM-v0.1-8B **41.4%** (non-agentic) within noise | Number in `results/references/` |
| M0.5 | Reference runs on the same harness: official CLM-8B; Kev-4B; Laya; Decision 2.0 Kai/Nox; Jeff-style zero-shot letter readout on Qwen3-8B; Inkling zero-shot readout via API (Inkling 2 Mini / Small); Jev via API if budget allows | `results/references/*` + a comparison table |

## Milestone M1: Architecture B (budget estimated at M1 start; needs approval)

**M1.1 Encoder registry and embedding pipeline.**
- Backbones: frozen Qwen3-8B (last layer, L/2, 3L/4); Qwen3.5-4B + LoRA; frozen Inkling 2 Mini.
- One causal pass over a trajectory or a state yields every needed embedding.
- Embeddings are cached on the volume, keyed by backbone, layer and text hash.

**M1.2 Data.**
- Build the decontamination registry (ADR 0002).
- Typed data: DecisionBank, kev-suites and typed-decisions train splits.
- Q&A replay from the CLM vectors/text.
- Augmentation:
  - option shuffling;
  - none-pairs;
  - polarity/negation twins;
  - many questions per state;
  - descriptive option rendering.

**M1.3 Model.** B's ablation ladder, each rung measured with 3 seeds on the M0 harness:

| Rung | Change | Hypothesis (survey §10) |
|---|---|---|
| 0 | CLM heads as published (anchor) | — |
| 1 | + calibration procedure (per type, K-aware, held-out-dataset fit) | H3 |
| 2 | + separate state/question encoding, late fusion | H1 |
| 3 | + descriptive, type-conditioned option rendering; Noul head without option text | H1 |
| 4 | + coverage gate (null outcome) | — |
| 5 | + Matryoshka + centring | H8 |
| 6 | + late-interaction decision block over cached option K/V (**B**) | H2, H4 |
| 7 | backbone arms: 3L/4 and L/2 taps; Qwen3.5-4B + LoRA; Inkling 2 Mini frozen | H6, H7 |

**M1.4 Serving.**
- Latency on one L4/L40S: new state vs revisited state; K = 2–255 cached options.
- int8 option cache; FP8 backbone validation.

## Milestone M2: Architecture C (designed in M1, built after M1 results)

- Joint `prefix_bidir` slow path with LoRA and an LM-head-initialized readout on the same backbone.
- A calibrated-confidence cascade between the two paths (H9).
- Serving work: CUDA graphs, fused kernels, prefix/KV reuse.

## Milestone M3: release

- README overhaul with real results (#13).
- Model card listing data licences.
- `v0.2.0` release.

## Parked

- Tier 1 CLM-recipe sweeps (`configs/sweeps/`, local branch `exp/tier1-sweeps`). The pre-training recipe ablations remain useful for the contrastive replay stage. They will be re-cut into M1.2 once the data mix is fixed.
- Agent post-training on trajectories (#14).
- DeepSWE and T-Rex downstream tasks (#15).
