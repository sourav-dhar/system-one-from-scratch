# GLiNER2.5-Decide

> **Family:** encoder (schema-prompted multi-task encoder in the GLiNER2 line) · **Team:** Fastino · **Released:** 2026-09-23 · **License:** Apache-2.0 · **Code/weights:** HF only, at [huggingface.co/fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) [@gliner-decide] (67.7k downloads); variants `-1B` (Ettin encoder-from-decoder, 1B) and `-multi` (287M); no training code · **Sources read:** HF card, configs, and the GLiNER2 paper ([arXiv 2507.18546](https://arxiv.org/html/2507.18546); [@gliner2]), which the HF card tags. **Not disclosed:** the Decide-specific training data and calibration. **Not found:** any third-party evaluation.

## Summary

GLiNER2.5-Decide adapts the GLiNER2 information-extraction architecture to decisions. It uses a 340M DeBERTa-v3-large encoder:
- **Input.** Each task is a prompt of the form `[P] task ([L] label₁ [L] label₂ …) [SEP] text`.
- **Scoring.** Each label is scored by an MLP on the hidden state of its `[L]` marker. A softmax gives single-label decisions and a sigmoid gives multi-label ones.
- **Cost.** Several tasks share one encoder pass.

On Fastino's fast-decisions benchmark it reports 60.2%, against JevK5 57.6% and "Laya Router" 46.6%. On CPU it runs in 130 ms with 5 labels and 208 ms with 50.

Three caveats apply:
- The Decide-specific training data and calibration are undisclosed.
- Ordinal scales are trained "as plain classes".
- No independent benchmark in the notes includes it.

## Architecture

**Backbone.** DeBERTa-v3-large, 340M parameters, 24 layers × 1024, 512 positions.

**Input format** (GLiNER2 mechanism, [arXiv 2507.18546](https://arxiv.org/html/2507.18546)):

```
[P] task ([L] ℓ1 [L] ℓ2 …) [SEP] text
```

**Scoring.** With $h_{[L]_i}$ the encoder state at the $i$-th label marker:

$$z_i = \mathrm{MLP}\big(h_{[L]_i}\big),\qquad p = \mathrm{softmax}(z)\ \text{(single-label)},\qquad p_i = \sigma(z_i)\ \text{(multi-label)}$$

**Passes per request.** Several tasks are scored in **one encoder pass**. This is the paper's mechanism; the notes do not record how many tasks fit.

**Cacheability** *(inference)*. The encoder is bidirectional, and the text follows the task and label prompt. Neither the text encoding nor the label encodings can be reused across requests with different label sets.

**Context** *(inference)*. The 512 positions are shared by the task prompts, the labels and the text. Many labels or long texts therefore compete for one budget, as in Laya. The notes record no truncation policy.

**Type mapping.** The notes do not record how System One's choice / noul / score types map onto the single-label and multi-label heads. The card does say that ordinal scales are trained "as plain classes".

| Property | Value |
|---|---|
| Backbone | DeBERTa-v3-large, 340M (24 × 1024); `-1B` uses Ettin 1B; `-multi` is 287M |
| Context | 512 positions |
| Option encoding | `[L]` marker per label inside a task prompt, before the text |
| Head | MLP on each `[L]` state; softmax or sigmoid |
| Passes per request | One encoder pass for several tasks |
| Cacheable | No *(inference)* |
| Order invariance | Not reported in the notes |
| Ordinal handling | Plain classes |
| Calibration | Not disclosed |

## Training

- **GLiNER2 base:**
  - 254,334 examples, GPT-4o-annotated or synthetic;
  - 5 epochs;
  - learning rates 1e-5 and 2e-5 ([arXiv 2507.18546](https://arxiv.org/html/2507.18546)).
- **Decide-specific training data and procedure:** not disclosed.
- **Ordinal scales:** trained "as plain classes", which the card admits. *(inference)* No ordinal loss (RPS or CDF) is used, so adjacent levels are as different as distant ones during training.

## Calibration

**Not disclosed.** The notes record no temperature, ECE or Brier for any GLiNER2.5-Decide variant.

*(inference)* With sigmoid multi-label outputs, per-label probabilities need not sum to one. The probability-axioms study warns against treating per-option yes/no probabilities as a distribution ([@axioms]).

## Results

| Benchmark | GLiNER2.5-Decide | Comparison | Source |
|---|---|---|---|
| fast-decisions (17 domains × 300 held out) | 60.2% | JevK5 57.6%; "Laya Router" 46.6% | [card](https://huggingface.co/fastino/GLiNER2.5-Decide) |
| GLiNER2 zero-shot Banking77 | 0.70 | GPT-4o 0.78 | [arXiv 2507.18546](https://arxiv.org/html/2507.18546) |
| GLiNER2 zero-shot average | 0.72 | GPT-4o 0.84 | same |

- The fast-decisions result is self-reported, and the notes do not say whether it is in-distribution for Decide's own training.
- The card does not specify which Laya checkpoint "Laya Router" refers to.
- The GLiNER2 zero-shot rows describe the base system, not the Decide fine-tune.

## Serving and efficiency

| Setting | Latency |
|---|---|
| CPU, 5 labels | 130 ms |
| CPU, 50 labels | 208 ms |

- *(inference)* Going from 5 to 50 labels adds only 78 ms, so labels are cheap relative to the text.
- That gentle scaling holds only while the labels and text fit in the 512 positions.
- The notes record no GPU latency or serving stack.

## Flaws and failure modes

### (a) GLiNER2.5-Decide's own flaws

| Flaw | Evidence |
|---|---|
| Opaque training and calibration | Decide-specific data and calibration not disclosed |
| No ordinal structure | Scales trained "as plain classes" |
| Short context | 512 positions shared by the labels and the text |
| No independent evaluation | Absent from S1-Arena, TypedBench, Rafe & Das, Decision Index and JevBench rows in the notes |
| Robustness untested | Option-order, yes/no-swap and negation behaviour not reported |

### (b) Flaws of other systems it claims to fix

The notes record no explicit claim to fix another system's flaw. The card positions the model against JevK5 and "Laya Router" on accuracy (60.2% vs 57.6% and 46.6%), and stresses CPU latency.

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **Many tasks in one pass** over one text, as a reference design for a cheap CPU tier in a cascade.
- **Marker-based label scoring.** It is the encoder analogue of option-marker heads, and needs very few parameters.

**Avoid**
- Training ordinal score types as plain classes. Use an RPS or CDF term instead, as Laya, Jeva and NDM do.
- Sigmoid per-label outputs where the question is a partition.
- Undisclosed calibration. Our cards should report ECE and Brier before and after temperature scaling.
- Accepting self-reported benchmarks without an external harness. Running it through a `/v1/systemone` adapter on S1-Arena or JevBench would be the test.

## References

- Model card and configs: [huggingface.co/fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) [@gliner-decide]
- GLiNER2 paper: [arXiv 2507.18546](https://arxiv.org/html/2507.18546) [@gliner2]
- Laya (the "Laya Router" comparison point): [@laya-repo]
- Probability-axioms study: [@axioms]
