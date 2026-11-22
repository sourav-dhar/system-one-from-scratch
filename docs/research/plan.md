# Research and build plan, v3

- **Status:** accepted, 2026-11-22
- **Supersedes:** plan v2, which was written for ADR 0001 before the independent reviews
- **Architecture:** [ADR 0003](../adr/0003-architecture-resequenced.md), which supersedes [ADR 0001](../adr/0001-architecture-late-interaction-two-path.md)
- **Data:** [ADR 0002](../adr/0002-data-licensing-and-decontamination.md), as amended by [ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md)
- **Evidence:** the [survey](survey.md), the [start-here guide](README.md) and the [independent reviews](reviews/)

## Ground rules

- **Noise floor.**
  - Every trained configuration gets 3 seeds.
  - An effect counts only if it exceeds 2× the seed standard deviation under a task-cluster paired bootstrap.
  - Item-level McNemar would call 0.3 pp "significant" on System1-Arena's ~78k items, so it is not used.
- **Baseline-anchored factorials with drop-one ablations.** We do not use cumulative ladders, whose effects depend on rung order.
- **Calibration is a procedure, not a rung.**
  - The same calibrator is applied to every configuration.
  - Raw and calibrated results are both reported.
  - Fit sources are frozen before scoring ([ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md)).
- **Metrics.**
  - ECE is reported with both estimators (15 equal-width and 10 equal-mass bins), plus multi-class Brier and NLL, with CIs.
  - Latency is reported with hardware, batch size and cache state.
- **Negative results are published.**
- **Every number comes from a file in `results/`**, produced by a committed config.
- **Budget gates.** Each milestone states its GPU budget and needs approval before it starts.

## M0: harness, anchors and baselines (approved, ≤ $60 on Modal)

| Step | Deliverable | Done when |
|---|---|---|
| M0.1 | **Typed-decision schema and loaders.**<br>• `choice`/`noul`/`score` plus null.<br>• Pinned System1-Arena (dataset and harness commits), OpenDecisions, typed-decisions, Jeva's BANKING77/MASSIVE and nested-set protocols, ToolRet K-sweeps.<br>• S1-Arena's agentic static tasks and one closed-loop environment (T-Rex) as the **W2 quality suite**. | Unit tests on fixtures; loaders stream from the Hub |
| M0.2 | **Metrics and probe battery.**<br>• Accuracy, Brier, both ECE estimators and NLL per type.<br>• Flips under shuffle and reversal; yes/no identifier swap; option renaming.<br>• Negation twins; question sensitivity; burying; K-sweep with distractors; gold-removed null.<br>• Latency with and without caches. | Tests against hand-computed cases |
| M0.3 | **`/v1/systemone`-compatible server and client**, so references and our models share one harness | Contract tests |
| M0.4 | **Two anchors on the pinned version:**<br>• CLM-v0.1-8B at 41.4%, through the official adapter *and* through our encoder (fixed truncation, PR #41), plus an embedding-parity test (cosine > 0.999 on ~1k Nemotron rows);<br>• the Qwen3-8B answer-token readout at 59.3%. | Both within 1.0 pp, in `results/references/` |
| M0.5 | **Baselines and references on one harness:**<br>• calibrated readouts of Qwen3-8B and Qwen3.5-4B (the M1 bar);<br>• linear probe with seen and unseen label splits;<br>• fine-tuned DeBERTa-v3-large on the suites with train splits;<br>• BM25 for catalogs;<br>• Qwen3-Embedding-8B zero-shot;<br>• Jeva-4B, Kev-4B, Jeff-Qwen-4B, Decision 2.0 Kai and Nox, Laya;<br>• the Inkling readout via Tinker. | `results/references/*` plus a comparison table with contamination flags and clean subsets |

**M0 cost:** about $40–70 *(estimate, [review 03](reviews/03-architecture-review.md) §6)*. Staying under $60 needs three things:
- the battery runs on a 10k stratified subset;
- full-set runs are limited to the anchors and the calibrated readouts;
- Jev and Inkling 2 Small are dropped.

## M1: the typed-decision model (budget ≈ $200–400, approved at start)

**M1.1 Registry and data** ([ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md)).
- Build the decontamination registry first.
- Train on the typed-data train splits (DecisionBank, kev-suites, typed-decisions) under the release allow-list rules.
- Mine hard negatives with open models.
- Augment with:
  - option shuffling;
  - none-pairs from hard-label choice items;
  - negation and polarity twins;
  - many questions per state;
  - descriptive option rendering.

**M1.2 Backbone and forking.**
- Qwen3.5-4B, with frozen Qwen3-8B as the anchor.
- Adopt or adapt Flide's exact state forking for the Gated DeltaNet and full-attention layers.
- Validate forking with a parity test: forked logits vs standalone logits (total-variation distance, flips).

**M1.3 Model factorial,** with 3 seeds per cell on the M0 harness:

| Arm | What it is | Answers |
|---|---|---|
| R0 | Calibrated zero-shot letter readout of the same backbone | The bar |
| R1 | LoRA letter readout (all options in context) | What training alone buys |
| R2 | **LoRA, per-option blocks that read the forked state** (Prism-1 style) with choice / noul / score heads and the coverage gate | The M1 model |
| R2 − x | Drop-ones: −comparison head, −Noul head, −LM-row initialisation, −gate, −descriptive options | Attribution |
| R3 | In-context bilinear head over candidate endpoints (Decision 1.0 Eq. 3) | Alternative scoring |
| Pilot | LoRA learning rates {2e-5, 5e-5, 1e-4} on ¼ of the data, 1 seed | Stability (2e-4 diverged in Kev and Small Heads) |
| Data | Questions per state, none-pair share, descriptive rendering | "Fixing data shortcuts mattered more" (Liquid) |
| Calibration | Global vs per-type vs per-type × $\log K$, leave-source-out (post-hoc, free) | H3 |

**M1 success** follows [ADR 0003](../adr/0003-architecture-resequenced.md):
- R2 beats R0 on S1-Arena accuracy, with a cluster-bootstrap CI excluding zero (target +3 points);
- R2 is no worse than R0 on Brier;
- held-out-source accuracy is within 15 points of seen-source accuracy.

**M1.4 Serving on one GPU.**
- Measure p50 and p95 on an L4 and an L40S: new state vs forked questions, and K from 2 to 255.
- FP8 W8A8 on Ada is adopted only if it changes ≤ 0.5% of decisions with max |Δp| ≤ 0.02. The A10 runs bf16.

**Inkling arms.**
- The Tinker readout and teacher (M0.5, and distillation in M1.1).
- The off-target frozen-reader arm on a rented H100, after a cost estimate is approved.

## M2: the cached path for agent loops (designed in M1, built after the M1 decision)

- Options are encoded once without the state. A late-interaction block runs over the forked state's pooled representation, followed by the coverage gate.
- **Go/no-go ([ADR 0003](../adr/0003-architecture-resequenced.md)).** Compare the same head on cached and on in-context options. The cached path is kept only if both hold:
  - accuracy is within 3 points;
  - W2 latency is at least 5× lower at K ≥ 50 or on revisits.
- **Ablations:**
  - question-blind towers vs a state-aware question;
  - question-conditioned option-cache keys;
  - a CSLS hubness control.

## M3: release

- A README with real results.
- A model card listing every data source and its licence.
- A public checkpoint only from an allow-listed training run ([ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md)).
- `v0.2.0`.

## Time and risks

- **Time:** about 3 months to the M1 decision for one engineer *(estimate)*:
  - M0: 3–4 weeks;
  - M1 data, forking and model: 5–7 weeks;
  - serving: 1–2 weeks.
- **Risks and kill criteria:** we adopt the [architecture review's risk register](reviews/03-architecture-review.md#6-budget-time-and-risk-register). Two of its risks are specific to this plan:
  - Forked logits must match standalone logits; otherwise we serve one sequence per question and re-measure latency.
  - Per-option blocks must not fall more than 3 points below R1; otherwise M1 ships R1, with sorting for order invariance.

## Parked

- Tier 1 CLM-recipe sweeps (local branch `exp/tier1-sweeps`). They are reusable for the contrastive replay stage.
- Agent post-training on trajectories (#14).
- DeepSWE and T-Rex downstream tasks (#15). T-Rex re-enters as part of M0's W2 suite.
