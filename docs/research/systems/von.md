# Von

> **Family:** encoder (Laya-style option-marker model with order-invariant masks) · **Team:** wfzyx · **Released:** release date not recorded in the notes; the repo was **archived** after its last commit on 2026-10-19, which points users to `LiquidAI/d1-omni-600M` · **License:** Apache-2.0 · **Code/weights:** [github.com/wfzyx/von](https://github.com/wfzyx/von) [@von] (922 stars); weights at [huggingface.co/wfzyx/von](https://huggingface.co/wfzyx/von) (395M, ModernBERT-large, 8192 context); the code is complete, including training scripts (`training/`) · **Sources read:** [hf/MODEL_CARD.md](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md), [models/option_marker.py](https://github.com/wfzyx/von/blob/main/src/von/models/option_marker.py), `train_option_marker.py`, `train_rlcd.py`, the README, and the [docs/von-2.md](https://github.com/wfzyx/von/blob/main/docs/von-2.md) post-mortem. **Not recorded in the notes:** the corpus sources beyond category shares, and latency methodology.

## Summary

Von is a 395M ModernBERT-large decision encoder. Like Laya, it scores each option at a `[MASK]` marker. Its distinguishing feature is an **order-invariant attention mask with reset position ids**. This took option-order flips on JevBench-hard from 49.5% (Von 1.1) to 0 (Von 1.2).

It was trained on a balanced corpus of about 290k items with cross-entropy plus 0.5·Brier, and calibrated with an input-conditioned temperature. It reports a JevBench v1.4 sealed ECE of 0.107, against 0.172 for Laya on the same card, but a lower composite score (27.5 vs 30.3).

The project's own post-mortem is unusually candid:
- seven encoder-side interventions all came back null under a paired McNemar test;
- the encoder ranked #101 of 114 on the Decision Index;
- Von 2 moved to a Qwen3.5-0.8B decoder with LoRA and Clef's JointSchemaHead, gaining 6.5 pp pooled;
- the repository was then archived.

## Architecture

**Input** (from [MODEL_CARD.md](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md)):

```
"{question} {state} [SEP] [MASK] opt1 [MASK] opt2 …"
```

Forged `[MASK]`/`[SEP]` literals in user text are neutralised with a zero-width joiner. Unlike Laya, the question and state come first and the options last.

**Scorer** (`OptionMarkerScorer` in [option_marker.py](https://github.com/wfzyx/von/blob/main/src/von/models/option_marker.py)). With $h_{m_k}$ the encoder state at option $k$'s `[MASK]`:

$$z_k = w_2^{\top}\,\mathrm{LN}\Big(\mathrm{GELU}\big(W_1\,\mathrm{LN}(h_{m_k})\big)\Big) + b_2,\qquad W_1\in\mathbb{R}^{(d/2)\times d},\qquad p=\mathrm{softmax}(z/T)$$

**Order-invariant masks** (`build_independent_option_masks`, `build_option_invariant_position_ids`):
- Option tokens see only the prefix and themselves.
- Position ids restart at the end of the prefix.

Let $P$ be the prefix (question and state) and $B_k$ option $k$'s block. Then:

$$M_{ij} = \mathbb{1}\big[j\in P\big] \;\lor\; \mathbb{1}\big[i\in B_k \wedge j\in B_k\big],\qquad \mathrm{pos}(B_k[r]) = |P| + r$$

So every option block is encoded identically wherever it appears in the list. On JevBench-hard, Von 1.1 flipped 49.5% of answers across 4 orderings, and Von 1.2 flips 0.

The notes record the mask only for option tokens. *(inference)* Whether prefix tokens can attend to option tokens is not stated. If they could, the options would leak back into the shared prefix, so strict invariance implies they cannot.

**Passes per request.** The README says several questions are answered in one call. *(inference)* As in Laya, these are batched rows, one encoder row per question (the encoder dossier's cross-system synthesis says the same).

**Cacheability** *(inference)*. The encoder is bidirectional and the question precedes the state, so nothing is reusable across questions or requests. The invariant masks would in principle let option blocks be appended to a cached prefix, but the notes record no such cache.

| Property | Value |
|---|---|
| Backbone | ModernBERT-large, 395M; 8192 context |
| Attention | Bidirectional; options isolated from each other; shared prefix |
| Option encoding | `[MASK]` marker per option, with position ids reset at the prefix end |
| Head | LN → Linear(d, d/2) → GELU → LN → Linear(·, 1) |
| Passes per request | One row per question, batched *(inference)* |
| Cacheable | No cache recorded |
| Order invariance | Yes, by construction (Von 1.2: 0 flips) |
| Calibration | Input-conditioned $T$, clamped to [0.3, 12]; noul "band" rule |

## Training

- **Corpus:** a balanced corpus of about 290k items, followed by synthetic two-hop and numeric continue-training.

  | Category | Share |
  |---|---|
  | enterprise | 25% |
  | security | 20% |
  | safety | 15% |
  | linguistic | 15% |
  | triage | 10% |
  | ANLI/WANLI | 15% |

- **Loss:** $\mathcal{L} = \mathrm{CE} + 0.5\cdot\mathrm{Brier}$, in `train_option_marker.py`.
- **Optimisation:** learning rate 3e-5, with the encoder at ×0.5 and the scorer at ×2.5; 3 epochs; batch 8 × accumulation 2.
- **Initialisation:** from a `von-modernbert-rlcd` checkpoint. That checkpoint was trained by `train_rlcd.py`, which, despite its name, also uses CE + Brier. *(inference)* So Von's "RLCD" stage is a supervised proper-score fit, consistent with Laya's #741 finding that the policy-gradient term adds nothing (see [laya.md](laya.md)).

## Calibration

**Input-conditioned temperature.** $H_{\text{norm}}$ is the normalised entropy:

$$T = \mathrm{clip}\Big(b + w_H H_{\text{norm}} + w_{\text{len}}\,\tfrac{\log_{10}(\text{state\_tokens})}{4} + w_K\,\tfrac{K}{8},\ 0.3,\ 12\Big)$$

**Noul "band" rule:**

$$p' = 0.8 + 0.1\,(p - 0.5)$$

*(inference)* Applied to the full range $p\in[0,1]$, this maps every probability into $[0.75, 0.85]$. The notes therefore probably omit a condition on when the rule applies. It cannot be interpreted further from the sources.

**Scope.** The map is fit in-sample on JevBench public, and the card says it "does not carry to other distributions".

## Results

| Benchmark | Von | Comparison |
|---|---|---|
| JevBench v1.4 ([card](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md)) | I 34.5, C 75.7, composite 27.5, sealed ECE 0.107 | Laya: I 36.1, C 63.7, composite 30.3, sealed ECE 0.172 |
| Decision Index v0.2.1 | 13.74; p50 32.8 ms on A10G | — |
| Decision Index Full ([von-2.md](https://github.com/wfzyx/von/blob/main/docs/von-2.md)) | 8.2 (#101 of 114) | — |
| jabr v2 | 72.0% | — |
| Latency | p50 0.096 s on 4 vCPU (OpenVINO); 0.023 s on A10G | — |

- The two A10G p50 figures, 32.8 ms (Decision Index) and 23 ms (card), differ. The notes do not give the protocols.
- **Von's notes also report on other systems:**
  - Clef-flash at Decision Index 57.1 vs Kev-9B at 38.5 ("recipe and head beat parameter count");
  - Intern-Decision's sub-1B model at Decision Index Full 11.8.

## Serving and efficiency

- **Latency.** p50 is 0.096 s on 4 vCPU with OpenVINO, and 0.023 s (card) or 32.8 ms (Decision Index) on an A10G.
- **Context.** The 8192-token context is 16× Laya's 512 for the English checkpoint.
- **Status.** The repository is archived. Its last commit (2026-10-19) redirects users to `LiquidAI/d1-omni-600M`.

## Flaws and failure modes

### (a) Von's own flaws

| Flaw | Evidence |
|---|---|
| The encoder plateau is weak | Decision Index Full 8.2 (#101 of 114); JevBench v1.4 composite 27.5, below Laya's 30.3 |
| Encoder-side changes do not help | Seven interventions all null under a paired McNemar test ([von-2.md](https://github.com/wfzyx/von/blob/main/docs/von-2.md)) |
| Persistent noul yes-bias | It picked "yes" on 224 of 400 held-out noul items, against an expected 195. This was traced to data: there are no same-state negated-criterion twins |
| Calibration does not transfer | The map is fit in-sample on JevBench public and "does not carry to other distributions" |
| The noul band rule is unclear | As recorded, it compresses all probabilities into [0.75, 0.85] *(inference)* |
| Abandoned | Archived; Von 2 left the encoder family |

### (b) Flaws of other systems it claims to fix

- **Option-order flips.**
  - Laya's sequential layout and Von 1.1 (49.5% flips) are order-sensitive. Von 1.2's masks give 0 flips on JevBench-hard.
  - Laya ships an equivalent "parallel layout" disabled, with no checkpoint trained on it ([laya.md](laya.md)). Von is the evidence that the idea works when trained.
- **Reporting discipline.** Every claim is gated by a paired McNemar test.
- **Noul label-word bias** (Laya #156, and its own yes-bias).
  - The encoder dossier lists the proposed fixes as same-state negated twins and neutral option keys.
  - The evidence is Von's polarity data and an "r4 hinge loss", for which **no result is reported**.

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **Isolated option blocks with reset position ids** whenever options share a sequence with the state, for example in a joint slow path. This is the cleanest evidenced fix for order flips: 49.5% → 0. Our bi-encoder path is already order-invariant.
- **Paired McNemar gates** (or a paired bootstrap) for every claimed improvement.
- **Same-state negated-criterion twins** in noul training data. The yes-bias was traced to their absence.
- **The post-mortem's conclusion.** The top Decision Index tier is held by decoders of 9B or more, and Von's own move to a 0.8B decoder with LoRA gained 6.5 pp. Do not invest in small encoder backbones for reading-heavy tasks.

**Avoid**
- Calibration maps fit in-sample on the evaluation set's public split.
- Probability-band post-processing rules that are not documented with their domain.

## References

- Von code, card and post-mortem: [github.com/wfzyx/von](https://github.com/wfzyx/von) [@von]
  - [hf/MODEL_CARD.md](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md)
  - [option_marker.py](https://github.com/wfzyx/von/blob/main/src/von/models/option_marker.py)
  - [docs/von-2.md](https://github.com/wfzyx/von/blob/main/docs/von-2.md)
- Weights: [huggingface.co/wfzyx/von](https://huggingface.co/wfzyx/von)
- Laya, the architecture Von follows: [@laya-repo], [@laya-card]
- Clef (JointSchemaHead, used by Von 2): [@clef], [@clef-blog]
- Jev Decision Index: [@jdi]; JevBench harness: [@jevbench]
