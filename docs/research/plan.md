# Research plan: ablations, architectures, serving

Status: **proposed** · Owner: @sourav-dhar · Tracking: #1

This plan fixes *what we will run, why, and how a result is judged* before any compute is spent,
so that findings cannot be cherry-picked after the fact. Every row maps to a sweep file in
`configs/sweeps/` and lands in `results/<study>/<run>/` through a PR.

## 0. Ground rules

**Baselines (scaled recipe).**

| | Data | Budget | Selected by | Wall time (L40S) |
|---|---|---|---|---|
| `pre/baseline` | 24 chunks (2.4M pairs) over shards 01-04 | 3,000 steps × 16,384 = 49M pairs | general loss | ~6 min train + ~4 min load/eval |
| `mid/baseline` | 6 hard-negative chunks (600k pairs, ≤10 negatives) | 400 steps × 16,384 | tricky-Q&A loss | ~5 min |

The recipe itself (authors' configs) is the default in `configs/base.yaml`. The scale-down is the
only deliberate departure, and it applies equally to every arm.

**Benchmark.** Everything is scored by one code path (`systemone.eval`), validated by reproducing
the authors' heads exactly (51.73 / 69.16 / 68.63 % tricky top-1 on n = 191,161).

**Decision metrics.**
* pre-training studies: `general/top1` and `hard/top1` (tricky Q&A) at the end of training;
* mid-training studies: `hard/top1` (primary), `hard/loss` (NLL);
* serving studies: quality at a stated cost (latency, memory, bytes per cached option);
* every table also reports `hard/ece` because the product output is a probability.

**Noise floor.** The baseline is run with **3 seeds** first. A difference counts as an effect only
if it exceeds **2 × the seed standard deviation**; smaller differences are reported as "no effect
detected", not as wins or losses.

**One factor at a time.** Each arm changes exactly one thing against its baseline; anything else
that must change with it (e.g. steps when batch size changes, to hold *pairs seen* fixed) is stated.

**Negative results are published** in the same tables.

## 1. Study A: pre-training recipe (#11)

| ID | Question | Arms (vs baseline) | Held fixed | Hypothesis |
|---|---|---|---|---|
| A0 | How noisy is a run? | seeds 1234, 1, 2 | all | defines the noise floor |
| A1 | Does contrastive batch size matter at fixed data? | batch 4,096 / 8,192 (LR by recipe √batch rule) | pairs seen (49M) | larger batch → more negatives per step → better tricky top-1 |
| A2 | Is the learned temperature necessary? | fixed τ=0.07; fixed τ=0.02; learned with no cap (max scale 1000) | all | learned+cap ≥ fixed; uncapped hurts calibration |
| A3 | Does same-document masking help? | masking off | all | off hurts: same-doc answers are false negatives |
| A4 | Is the symmetric loss needed? | state→action only | all | symmetric helps the action head generalize |
| A5 | How does quality scale with data? | 3 / 6 / 12 chunks | steps | log-linear gains; tells us what the full 1 TB buys |
| A6 | Is the constant LR optimal? | 0.5× LR; 2× LR; cosine + 5% warmup | all | cosine helps at a fixed budget |

## 2. Study B: head architecture (#28)

| ID | Question | Arms | Notes |
|---|---|---|---|
| B1 | Depth | depth 2 (no hidden layer); depth 4 + residual; depth 6 + residual | residual needed beyond depth 3 |
| B2 | Width | 768; 3,072 (LR by recipe width rule) | params 4.7M / 22M per head |
| B3 | Normalization | no LayerNorm | |
| B4 | Embedding size | out dim 128 / 256 / 1,024 | smaller = cheaper option cache |
| B5 | Tied heads | one shared (siamese) head | halves params; states and actions are different text types, so expected to hurt |
| B6 | Matryoshka | train prefixes {64, 128, 256} jointly | hypothesis: 128-d keeps ≥ 95% of 512-d tricky top-1 (the authors' head keeps only 82% when truncated) |

## 3. Study C: mid-training (#11, #12)

All arms warm-start from `pre/baseline/best.pt`.

| ID | Question | Arms | Hypothesis |
|---|---|---|---|
| C1 | Do the gains come from negatives or from extra steps? | k = 0 (control), 1, 3, 10 negatives | k=0 gains little; most of the gain arrives by k=3 |
| C2 | Are some mined negatives false negatives? | filter negatives with cos(neg, gold) > 0.95 / 0.90 | light filtering is neutral-to-positive on top-1, improves NLL |
| C3 | Negative weighting | w = 0.5, 2 | w > 1 trades general top-1 for tricky top-1 |
| C4 | Mid-training LR | 3e-4, 3e-3 | |
| C5 | Transfer check | warm start from the **authors'** pre head | isolates our mid stage from our (scaled) pre stage |

## 4. Study D: serving and efficiency (#29)

| ID | Question | Method | Cost |
|---|---|---|---|
| D1 | Calibration at decision time | post-hoc scale fit on chunk 0028, test on 0029 | done for the authors' heads (ECE 0.065 → 0.013 for mid); repeat for ours, CPU-cheap |
| D2 | Cheaper option caches | Matryoshka prefix quality vs dims (B6), bytes/option | free with B6 |
| D3 | Head quantization | fp16 / bf16 / int8 dynamic heads: tricky top-1 delta + CPU/GPU latency | CPU-only |
| D4 | Latency budget | heads-only latency vs K options, batch size, device; cached vs uncached option embeddings | CPU + one short GPU run |
| D5 | Smaller / earlier backbones | Qwen3-0.6B / 1.7B / 4B and intermediate layers of 8B: re-embed a 1M-pair subset + eval sets, retrain heads | **needs text + embedding pipeline; gated (see §6)** |
| D6 | Distillation | teach a small-backbone head from the 8B head's option distributions | after D5 |

## 5. Study E: related systems (#30)

A landscape study of other System One-style decision models (Jev and others) covers their
architecture, training, serving, and whether weights or an API are available. Outcome: which
ideas to adopt, and which systems can be scored on our benchmark as external baselines.
Results go to `docs/research/landscape.md`.

## 6. Budget and gating

Measured pilot: 134k pairs/s training, 1.8 min to load 39 GB, ~10-12 min per pre-training run
including evals, on an L40S container (8 CPU, 80 GB RAM).

| Tier | Contents | Runs | Est. cost | Status |
|---|---|---|---|---|
| 1 | A0-A6, B1-B6, C1-C5, D1-D4 | ~38 | **~$20-25** | proposed |
| 2 | D5 smaller/earlier backbones (embedding pipeline, ~1M pairs × 3 backbones) | - | ~$40-80 | gated on Tier 1 + landscape study |
| 3 | E: external baselines on our benchmark; D6 distillation; post-training on agent data (#14) | - | depends on API/weights; post-training embedding alone ~$100+ | gated |

Volume storage: ~121 GB staged (eval sets, 24 pre-training chunks, 6 hard-negative chunks).

## 7. Reporting

Each study PR adds: the sweep file, `results/<study>/*`, a table and figure generated by
`scripts/`, and an entry in `docs/research/log.md` (hypothesis → result → decision). The README
summarizes only results that cleared the noise floor, and links to everything else.
