# decider (Mapika)

> **Family:** decoder-readout · **Team:** Mark Marosi (Mapika) · **Released:** repo created 2026-09-16 (1,036 stars as of 2026-11-20 per the catalog) · **License:** Apache-2.0 · **Code/weights:** models 0.8B / 2B / 4B (Qwen3.5-Base), 35B-A3B, Gemma-4 12B and 31B, plus stock-weight "decider-chat" configs (the catalog lists 0.8B, 2B, 4B, 35B-A3B and a vision model); full SFT data builders for ~95 datasets (`teacher_data/`, scripts); the RL code lives in a separate repository · **Sources read:** README ("How it works", "Train your own", "Temperatures", "Limits"), [decider/model.py](https://github.com/Mapika/decider/blob/main/decider/model.py), [decider/train.py](https://github.com/Mapika/decider/blob/main/decider/train.py), [decider/data/mixture.py](https://github.com/Mapika/decider/blob/main/decider/data/mixture.py), [teacher_labels.py](https://github.com/Mapika/decider/blob/main/decider/data/teacher_labels.py), [docs/RL.md](https://github.com/Mapika/decider/blob/main/docs/RL.md). Only lightly read: `docs/RESULTS.md` and `HISTORY.md`.

## Summary

decider is an open decision-model family that adds **no new head**: each question gets an answer slot in the prompt, and the option logits are the LM head's own letter rows applied to that slot's hidden state ([repo](https://github.com/Mapika/decider)) [@decider]. Many questions can share one prompt (one slot each) or be scored in separate rows, and a second, "schema-first" layout lets the question-and-options prefix be cached per schema. The idea that matters most is that **the base model dominates and the readout can be nearly free**: stock Gemma-4-31B weights plus this readout ("decider-chat") ranked #2 of 70 on Decision Index 0.2.1 (57.33), above the trained 35B / 4B / 2B models. decider's question builders also became the largest component of Jeff's DecisionBank (163,732 items) [@jeff] [@decisionbank], and it adds an RL stage with exact-law belief scoring and rendering-consistency KL. Its README is also explicit that one released calibration setting (Noul T = 0.05) was "chosen for JevBench" at a large cost in ECE.

## Architecture

**Readout** ([decider/model.py](https://github.com/Mapika/decider/blob/main/decider/model.py)). No new parameters:

$$z_k = \mathrm{softcap}\big(e_{\ell_k}^\top h_{\text{slot}}\big), \quad k \le K, \qquad p = \mathrm{softmax}(z/T),$$

where $e_{\ell_k}$ = `lm_head.weight[letters]` row for the $k$-th label and $h_{\text{slot}}$ is the hidden state at the question's answer slot; logits are soft-capped and masked past $K$. Labels run A–J, then K–Z, then two-letter tokens up to 255.

**Layouts** (README "How it works").
- **Multiple answer slots in one prompt**, one per question. "Every question can also be scored in its own row."
- Two layouts, trained 50/50:
  - **state-first** (default);
  - **schema-first**, whose question-and-options prefix is cached per schema; it "trades accuracy for speed".
- **Mask:** causal (decoder-family taxonomy).

| Property | Value |
|---|---|
| Passes per request | One pass with many answer slots, or one row per question |
| State prefix reusable | Optional (state-first layout with per-question rows) |
| Option embeddings reusable | Not as embeddings; in schema-first layout the question-and-options *prefix* is cached per schema |
| Max K | 255 (single letters, then two-letter tokens) |
| Context | Not stated in the notes; long JSON arrays degrade (see Flaws) |

## Training

**SFT** ([train.py](https://github.com/Mapika/decider/blob/main/decider/train.py)).
- **Full-parameter** fine-tune; LR 1e-5, cosine schedule; AdamW β = (0.9, 0.95).
- Loss: cross-entropy, $\mathcal{L} = -\log p_y$, plus optional Brier or label smoothing.
- One epoch = 1.47M examples / 455M tokens, 5.3 h on a GH200 (README "Train your own").

**Mixture** ([decider/data/mixture.py](https://github.com/Mapika/decider/blob/main/decider/data/mixture.py)).

| Component | Count |
|---|---|
| wide (full label sets ≤255) | 8,000 per task |
| padded with unrelated labels | 25,000 |
| described options (Qwen3.5-27B-written descriptions) | 40,000 |
| JSON states | 32,000 + 20,000 indexed |
| single-question | 30,000 |
| isolated yes/no per Score level | 26,000 |
| programmatic rules with state and rule twins | 90,000 |
| contrastive pairs | ×3 |
| replay | 200,000 |

- Plus "none" augmentation.
- Teacher data comes from a local Qwen3.5-27B ([teacher_labels.py](https://github.com/Mapika/decider/blob/main/decider/data/teacher_labels.py)), filtered because the "checker shares the bias being fixed".
- Downstream: Jeff's DecisionBank-260k contains 163,732 "decider public and custom questions (79 tasks, ≤2,094 per task)" with 50/50 label and teacher targets ([Jeff](https://arxiv.org/html/2610.17205)) [@jeff].

**RL stage v8 → v10** ([docs/RL.md](https://github.com/Mapika/decider/blob/main/docs/RL.md)). The objective combines four weighted terms:

$$\mathcal{L}_{\mathrm{RL}} = 0.1\,\mathcal{L}_{\mathrm{PPO}} + 0.2\,\mathcal{L}_{\mathrm{belief}} + 1.0\,\mathrm{KL}_{\mathrm{render}} + 0.7\,\mathrm{KL}_{\mathrm{retain}}$$

- PPO actor (0.1) on live MiniWoB++ and games.
- **Belief log-score** against exact laws (0.2), reading options in two orders.
- **Rendering-consistency KL** across reversed order and schema-first (1.0).
- Retention KL to v8 (0.7), with a hard budget.
- LR 1e-6. Belief excess fell from 0.473 to 0.219 nats.

*(Inference: the weighted-sum form is our reading of the listed weights; the notes do not give the exact equation or the sign conventions.)*

## Calibration

- **Per-type temperatures.** `temperature_by_options` makes T a function of the option count: $T(n) = \max(T_{\min},\, a + b\ln n)$; decider-chat-gemma4-31b uses $a = 10.124$, $b = -1.633$.
- **decider-12b v2 sets Noul T = 0.05, "chosen for JevBench"** (README "Temperatures"):

| Noul T | AUROC | ECE |
|---|---|---|
| 0.05 | 0.878 | 0.151 |
| 1.0 | 0.934 | 0.038 |

  At T = 0.05, "most answers round to exactly 0 or 1."
- **Hard-tier ECE 0.31** (2B v10) (README "Limits").

## Results

| Model | Benchmark | Score | Source |
|---|---|---|---|
| decider-chat-gemma4-31b (stock weights + readout) | Decision Index 0.2.1 | 57.33 (#2 of 70) | README |
| decider 35B (trained) | Decision Index 0.2.1 | 47.11 | README |
| decider 4B (trained) | Decision Index 0.2.1 | 40.70 | README |
| decider 2B (trained) | Decision Index 0.2.1 | 28.97 | README |
| decider-31b v2 | DI 0.3 public | 63.90 | README |
| decider 4B v2 | JevBench | 71.28 | catalog ([repo](https://github.com/Mapika/decider)) |

The README's own reading: "the base model sets most of the score."

**Third-party.**
- Rafe & Das yes/no swap: decider is in the group with 4.2–6.5 flips per 100 (with Kev-9B, Nimble and this-that), vs Jev 50.5 and Laya 91.3 ([2610.00346](https://arxiv.org/html/2610.00346)) [@rafedas].
- **Derivative:** this-that-model-1.0 (FLock.io) is built on decider-2b and uses a schema-first layout whose prefix is reused across states; it reports accuracy 0.941 / Brier 0.042 vs Jev 0.765 / 0.133 on 68 questions (30.9 ms; the gap rests on about 12 questions) and 0.844 vs Jev 0.803 on 2,250 in-distribution questions ([this-that](https://arxiv.org/html/2609.23886)) [@thisthat].

## Serving and efficiency

- **Schema-first caching:** the question-and-options prefix is cached per schema, at a stated accuracy cost ("trades accuracy for speed").
- **GGUF / quantisation:** multi-row packing moves probabilities by up to 0.16 at Q4_K_M (README "Limits"); the decoder-family note lists this among batch and runtime nondeterminism issues.
- The Strands Decider hybrid cache fork says it copies DeltaNet states "following decider-2b's `shared_prefix.py`" ([Strands inference.md](https://github.com/strands-labs/strands-decider/blob/main/designs/inference.md)) [@strands]; the notes do not confirm that this file is in the Mapika repo.
- No latency figures for decider itself appear in the notes (this-that, a derivative, reports 30.9 ms).

## Flaws and failure modes

**(a) Flaws of decider itself** (README "Limits" unless noted)
- **Calibration bent to a benchmark:** Noul T = 0.05 "chosen for JevBench" raises ECE from 0.038 (T = 1) to 0.151 and lowers AUROC from 0.934 to 0.878.
- **Hard-tier ECE 0.31** for 2B v10.
- **"Rules written into the question are not followed at this size."**
- **Long structured states:** accuracy 0.51 with 64 JSON records vs 0.70 with one.
- **Quantisation sensitivity:** GGUF multi-row packing moves probabilities by up to 0.16 at Q4_K_M.
- **Training adds little over a strong base:** trained 35B / 4B / 2B (47.11 / 40.70 / 28.97) score below stock-weight decider-chat-gemma4-31b (57.33).
- **Schema-first layout costs accuracy** (magnitude not given in the notes).

**(b) Flaws of other systems it targets**
- **Order and rendering sensitivity:** the RL stage's rendering-consistency KL (reversed order and schema-first) and the two-order belief log-score; the decoder-family cross-table counts decider's "optional reversed second reading" as a partial fix for order sensitivity.
- **Yes/no label-name bias:** third-party flips of 4.2–6.5 per 100 vs Jev's 50.5.
- **Teacher bias:** teacher labels are filtered because the "checker shares the bias being fixed".

## Implications for our design

All bullets are *(inference)*.

**Adopt**
- **The zero-parameter LM-head letter readout as a baseline**, including on stock instruct weights: decider-chat-gemma4-31b (57.33) shows the base model sets most of the score.
- **A K-dependent temperature of the form $a + b\ln n$**, which directly addresses the stored-temperature failure at large K reported for Kev ([Rafe & Das](https://arxiv.org/html/2610.00346)); validate it on unseen datasets, since Kev found grouped per-(type, K) temperatures harmful.
- **Rendering-consistency KL** (order reversal, layout change) as a training-time invariance regulariser.
- **Mixture components that target known failures:** padded label sets, described options, isolated yes/no per Score level, state/rule twins and contrastive pairs.

**Avoid**
- **Benchmark-tuned temperatures** (Noul T = 0.05).
- **Assuming packed multi-row inference is numerically safe under quantisation** (up to 0.16 probability shifts).
- **Long JSON-array states without dedicated training and evaluation** (0.51 vs 0.70).

## References

- [@decider] https://github.com/Mapika/decider — [model.py](https://github.com/Mapika/decider/blob/main/decider/model.py), [train.py](https://github.com/Mapika/decider/blob/main/decider/train.py), [mixture.py](https://github.com/Mapika/decider/blob/main/decider/data/mixture.py), [teacher_labels.py](https://github.com/Mapika/decider/blob/main/decider/data/teacher_labels.py), [docs/RL.md](https://github.com/Mapika/decider/blob/main/docs/RL.md)
- [@jeff] https://arxiv.org/html/2610.17205
- [@decisionbank] https://huggingface.co/datasets/jgeuter/DecisionBank
- [@rafedas] https://arxiv.org/html/2610.00346
- [@thisthat] https://arxiv.org/html/2609.23886
- [@strands] https://github.com/strands-labs/strands-decider
- [@jdi] https://huggingface.co/spaces/multimodalart/jev-decision-index (Decision Index)
