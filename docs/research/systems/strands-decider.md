# Strands Decider

> **Family:** decoder-readout · **Team:** Strands Labs / AWS (blog by Brooker, Chambers, Nonato de Paula) · **Released:** repo created 2026-09-29; blog 2026-10-01; v1 2026-10-09 (Qwen3.5-2B plus four Gemma-4 sizes); v2 tagged 2611 (five Gemma-4 sizes) ([CHANGELOG](https://github.com/strands-labs/strands-decider/blob/main/CHANGELOG.md)) · **License:** Apache-2.0 · **Code/weights:** training code present; weights on the StrandsAgents HF org (the catalog lists 2B plus Gemma-4 E2B, E4B, 12B and 26B-A4B). v1 data builders for 4 of 9 input files are marked "NOT REPRODUCIBLE"; the v2 data mix of **17.48M rows** "is not in this repository" ([docs/training.md](https://github.com/strands-labs/strands-decider/blob/main/docs/training.md)) · **Sources read:** `docs/architecture.md`, `docs/training.md`, `docs/models.md`, `CHANGELOG.md`, v1 and v2 configs, `research/history.md`, `evaluation/results.md`, `designs/inference.md`, the launch blog. Not available: the v2 data mix and v1's four non-reproducible inputs.

## Summary

Strands Decider is AWS Strands Labs' open decision-model family, with two readout generations ([repo](https://github.com/strands-labs/strands-decider)) [@strands] [@strands-blog]. v1 puts a randomly initialised, ~1M-parameter pointer head on Qwen3.5-2B-Base with LoRA; v2 replaces it with a 255-code token readout initialised from Gemma-4's LM-head rows. The idea that matters most is v2's **`prefix_bidir` mask**: state tokens attend bidirectionally within the state but never to the question, and question tokens see the state and their own question, so "the torso encodes the state once, and each question reads that encoding" — bidirectional state encoding without losing state reuse. The research history also documents a failure mode that accuracy alone hides: an earlier checkpoint gave "the same answer to a changed question 94–99% of the time".

## Architecture

### v1: pointer head ([docs/architecture.md](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md))

- **Prompt shape:** `<state>…</state><question type=…>…<options>1. … </options></question><answer>`.
- **Torso:** Qwen3.5-2B-Base, 18 of 24 layers Gated DeltaNet. LoRA r16 / α32 on attention, MLP and DeltaNet projections.
- **Readout:** a pointer between the answer position and each option,

$$\mathrm{logit}_k = \frac{\langle q(h_{\text{answer}}),\, k(h_{\text{option}_k}) \rangle}{\sqrt{\mathrm{dim}}}, \qquad \mathrm{dim} = 256,$$

  about 1M parameters, random initialisation, head LR 1e-3.
- **Mask:** causal; the state KV is broadcast to question suffixes (decoder-family taxonomy).

### v2: 255-code token readout

- **Readout:** one linear map with 255 rows (codes A…IV, skipping GZ, each a single Gemma token), initialised from the corresponding LM-head rows, with the Gemma soft-cap, masked past $K$; one global temperature. In symbols, $z_k = \mathrm{softcap}(w_k^\top h)$ for $k \le K$ with $w_k$ initialised from the LM-head row of code $k$. The notes do not state the exact read position or the soft-cap constant.
- **`prefix_bidir` mask:** "A state token sees all of the state, in both directions, and never the question. A question token sees the state and all of its own question." Hence "the torso encodes the state once, and each question reads that encoding."
- **Backbones:** five Gemma-4 sizes (v2 table lists 31B and E2B results).

| Property | v1 (Qwen3.5-2B, pointer) | v2 (Gemma-4, 255-code) |
|---|---|---|
| Passes per request | One prompt; state KV broadcast to question suffixes | State once, then question suffixes |
| State prefix reusable | Yes | Yes, by construction (`prefix_bidir`) |
| Option embeddings reusable | No (options are inside each question) *(inference)* | No *(inference)* |
| Max K | Not stated in the notes for the pointer | 255 |
| Context | Not stated for v1 | `max_length` 16,384 in the 31B training config |

## Training

### v1 (v19 recipe)

**Loss.**
- Cross-entropy with 10% ordinal smoothing on Score.
- KL to the frozen torso's option-number readout (weight 0.3); the v1 release anchors to the released v19 instead.
- KL to a teacher (weight 1.0), applied only where the teacher beats the student.
- Schematically, $\mathcal{L} = \mathrm{CE}_{\text{ord-smooth}} + 0.3\,\mathrm{KL}_{\text{anchor}} + 1.0\,\mathbb{1}[\text{teacher beats student}]\,\mathrm{KL}_{\text{teacher}}$; the notes do not give the KL directions.
- "Option order is re-shuffled on every example, every epoch"; Score levels are only reversed (p 0.5).

**Data** ([v1 config](https://github.com/strands-labs/strands-decider/blob/main/configs/v1/strands-decider-2B-qwen3.5-v1-2610.yaml)).
- 246,678 rows. The teacher is gemma-4-31B-it, used where it agrees with gold.
- The decoder-family adopt list separately credits Strands v12/v14 with "use a teacher only where it beats the student".
- The release is a three-seed soup, stored as one rank-48 adapter with stacked heads.

**Pointer vs slot head** ([research/history.md](https://github.com/strands-labs/strands-decider/blob/main/research/history.md)).

| Head | Params | Score |
|---|---|---|
| Linear slot | 53k | 0.6619 |
| Slot + mean(option states) | 2.12M | 0.7056 |
| **Pointer, dim 16** | **70k** | **0.7200** |
| Pointer, dim 256 | 1.05M | 0.7250 |

"Roughly half the gain is having option representations at all, half is addressing them per option."

### v2 31B recipe ([config](https://github.com/strands-labs/strands-decider/blob/main/configs/v2/strands-decider-31B-gemma4-v2-2611.yaml))

- LoRA r64 / α128; LR 1e-4 for adapters and readout; 2,680 rows per step; 1 epoch.
- `max_length` 16,384; no teacher, no KL, no ordinal smoothing.
- Three-run average, then one temperature.
- Data: the 17.48M-row mix, not published.

## Calibration

**v19 (v1 line)** ([evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md) "Calibration").
- Per-primitive temperatures fitted against ECE: noul 0.91, choice 0.77, score 0.96. All are below 1, i.e. the model is **underconfident**.
- Held-out ECE 0.054 overall, but **noul 0.162**.
- "Calibrating score changes its answer."
- "No single temperature per primitive serves both" short classification and long documents. v19 is underconfident on multi-step documents: right 0.83 of the time at a stated 0.64.
- v1 2B soup: ECE 0.072 and Brier 0.280 (results.md "v1").

**v2.** One global temperature after a three-run average; no ECE values for v2 appear in the notes.

The decoder-family cross-table uses Strands as the example that LoRA'd small models can be *under*confident (T < 1), in contrast to instruct models and post-reasoning answers, which are overconfident.

## Results

**v1** ([results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md) "v1"), 2B soup:

| Metric | Value |
|---|---|
| JevBench | 180/231 |
| Brier | 0.280 |
| ECE | 0.072 |
| HotpotQA (never trained on) | 0.832 |
| Held-out short tasks | 0.653 |

**v2** ([docs/models.md](https://github.com/strands-labs/strands-decider/blob/main/docs/models.md)):

| Model | DI 0.3 | JevBench | OpenDecisions | DecisionPivot | H100 median latency |
|---|---|---|---|---|---|
| 31B | 71.5 | 93.1 | 92.2 | 84.3 | 99 ms (1 question) / 614 ms (30 questions) |
| E2B | 57.1 | – | 77.7 | – | 62 ms |

No third-party evaluation of Strands Decider appears in the notes. For reference only, other systems' self-reported numbers on the same named suites: StartLux reports DI 0.3 public 73.37 (27B AutoThink) and DecisionPivot pair accuracy 86.9 (AutoThink); Jeff-1.1-Large reports OpenDecisions 89.0 ([StartLux](https://arxiv.org/html/2610.17309), [Jeff](https://arxiv.org/html/2610.17205)) [@startlux] [@jeff]. Whether these runs share protocols is not established in the notes, and the landscape note warns that leaderboards named "JevBench" use different protocols.

## Serving and efficiency

- **Shared-prefix KV broadcast:** 1 vs 6 questions on a ~640-token state take 219.4 vs 217.4 ms ([designs/inference.md](https://github.com/strands-labs/strands-decider/blob/main/designs/inference.md)).
- **Hybrid cache fork:** copies the DeltaNet states "following decider-2b's `shared_prefix.py`". (The notes do not resolve which repository "decider-2b" refers to; Kev also ships a `kev/shared_prefix.py`.)
- **Latency:** v19 p50 115 ms on an RTX 3090; v2 31B median 99 ms for one question and 614 ms for 30 questions on an H100; v2 E2B 62 ms.
- **Deploy targets:** Lambda, SageMaker, MLX and vLLM.

## Flaws and failure modes

**(a) Flaws of Strands Decider itself**
- **Question-ignoring:** v7 "gives the same answer to a changed question 94–99% of the time". The fixes v11a and v11b both lost on JevBench, and v11b "answers yes on 60 of 74 yes/no tasks" ([history.md](https://github.com/strands-labs/strands-decider/blob/main/research/history.md) v11). The decoder-family note attributes the cause to one fixed question per task in the data.
- **Pointer order sensitivity:** reversing options moves the distribution by ~0.016 ([architecture.md](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md)).
- **Poor transfer of score and noul:** the v1 card says "score and noul transfer poorly"; noul ECE is 0.162 even after fitting.
- **Underconfidence and type-dependent calibration:** all v19 temperatures < 1; one temperature per primitive cannot serve both short classification and long documents; calibrating Score changes its answer.
- **Reproducibility gaps:** 4 of 9 v1 input builders are "NOT REPRODUCIBLE"; the 17.48M-row v2 mix is unpublished. (The catalog's "Yes: weights, data, configs and scripts" overstates this relative to the dossier.)

**(b) Flaws of other systems this design addresses**
- **Loss of state reuse under full bidirectional attention** (StartLux, pplx-decider, Jeff): `prefix_bidir` keeps the state bidirectional only within itself, which preserves caching. This is the decoder-family note's comparison *(inference)*; Strands does not report a matched causal vs `prefix_bidir` ablation in the notes.
- **Slot-head readouts** that ignore per-option representations: the 70k-parameter pointer beats a 2.12M slot+mean head (0.7200 vs 0.7056).

## Implications for our design

All bullets are *(inference)*, grounded in the evidence above and the decoder-family adopt/avoid list.

**Adopt**
- **Per-option addressing beats bigger slot heads:** a 70k-parameter pointer (dim 16) outperformed a 2.12M slot head, and half the gain came simply from having option representations. A small head is enough when options are represented in context.
- **`prefix_bidir`** if we want a bidirectional state encoder that still caches: the state never attends to the question.
- **Teacher KL only where the teacher beats the student** (and/or agrees with gold), rather than blanket distillation.
- **Probe question sensitivity** explicitly (same state, changed question) as a standard regression test; accuracy alone hid a 94–99% question-ignoring rate.
- **Reshuffle options every example, every epoch**; reverse (not shuffle) ordinal levels.

**Avoid**
- **One fixed question per task** in training data (the root cause attributed to question-ignoring).
- **A single temperature per primitive across very different workloads** (short classification vs long documents).
- **Assuming LoRA'd small models are overconfident:** v19 needed T < 1; fit T rather than clamp it to ≥ 1 for such models.

## References

- [@strands] https://github.com/strands-labs/strands-decider — [architecture.md](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md), [training.md](https://github.com/strands-labs/strands-decider/blob/main/docs/training.md), [models.md](https://github.com/strands-labs/strands-decider/blob/main/docs/models.md), [CHANGELOG.md](https://github.com/strands-labs/strands-decider/blob/main/CHANGELOG.md), [history.md](https://github.com/strands-labs/strands-decider/blob/main/research/history.md), [results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md), [inference.md](https://github.com/strands-labs/strands-decider/blob/main/designs/inference.md), [v1 config](https://github.com/strands-labs/strands-decider/blob/main/configs/v1/strands-decider-2B-qwen3.5-v1-2610.yaml), [v2 config](https://github.com/strands-labs/strands-decider/blob/main/configs/v2/strands-decider-31B-gemma4-v2-2611.yaml)
- [@strands-blog] https://strandsagents.com/blog/introducing-strands-decider
- [@startlux] https://arxiv.org/html/2610.17309
- [@jeff] https://arxiv.org/html/2610.17205
