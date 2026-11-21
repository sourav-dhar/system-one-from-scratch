# Phocinae-Largha-150M-v1

> **Family:** encoder (a fine-tune with Laya's architecture on a small multilingual backbone) · **Team:** Phocinae Project · **Released:** 2026-10-08 · **License:** Apache-2.0 · **Code/weights:** [github.com/Phocinae/Phocinae-Largha-150M-v1](https://github.com/Phocinae/Phocinae-Largha-150M-v1) [@phocinae] (790 stars); 144.3M parameters, fp16 weights 288.6 MB. The repo ships calibration and datasets; the catalog did not confirm full training code · **Sources read:** [technical report](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/docs/technical-report.md), [config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/config.json), [rl_agent_config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/rl_agent_config.json). **No third-party evaluation** was found.

## Summary

Phocinae-Largha is a 144.3M-parameter English/Chinese decision encoder built on mmBERT-small. Its configuration uses Laya's schema, so it is *(inference)* a fine-tune with Laya's architecture: a `[MASK]` per option, a 2-layer head and an act/escalate head.

**Training** took 300 updates (0.044 hours) on three sources:
- the typed-decisions train split;
- option reorderings, as flip augmentation;
- about 3,800 Chinese rows.

**Reported results:**
- typed-decisions accuracy of 0.791 (English) and 0.848 (Chinese). The report calls these in-distribution and "not a like-for-like head-to-head" with Laya or Jev.
- an option-order flip rate of 0.0283;
- 21.0 ms per case on an RTX 5090.

**Calibration** is a contradiction. The shipped temperatures *sharpen* the outputs. An opt-in recalibration cuts ECE from 0.2519 to 0.0168 but raises Brier from 0.0207 to 0.1218.

## Architecture

**Backbone** ([config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/config.json)). mmBERT-small: 22 layers × 384 hidden, 6 heads, a 256k vocabulary and a local attention window of 128.

**Head configuration** ([rl_agent_config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/rl_agent_config.json)). The file uses Laya's schema:

| Key | Value |
|---|---|
| `head_layers` | 2 |
| `max_len` | 512 |
| `head_max_len` | 192 |
| `act_costs` | `{escalate: 0.5}` |

*(inference)* If Laya's `DecisionModel` is used unchanged, the scoring path matches [laya.md](laya.md):

$$z_k = W_2\,\mathrm{GELU}\big(W_1\,\mathrm{LN}(\tilde h_{m_k})\big),\qquad \tilde H = \mathrm{TransformerEncoder}_2\big(\mathrm{Enc}(x) + e_{\text{type}}\big),\qquad p = \mathrm{softmax}(z/T_t)$$

Under that reading:
- The head has $d=384$ and $n_{\text{head}} = d/64 = 6$.
- Each option is cut to $\max(4, \lfloor 176/K\rfloor)$ tokens once the 192-token option block overflows.
- The `act_costs` entry implies the act/escalate head is present. Laya's act head was found saturated (#185), and the notes do not say whether Phocinae's is affected.

**Passes and cacheability** *(inference, from Laya)*. One encoder row per question, batched. Nothing is cacheable across questions.

**Context contradiction.** The configuration sets `max_len` to 512, yet the report includes 16k and 32k "context probes" (0.453 / 0.387). The notes do not explain how inputs that long are fed to the model.

| Property | Value |
|---|---|
| Backbone | mmBERT-small, 22 × 384, 6 heads, 256k vocabulary, local window 128 |
| Parameters | 144.3M (fp16 288.6 MB) |
| Head | Laya schema: 2 head layers, `head_max_len` 192, escalate cost 0.5 |
| Context | `max_len` 512 in config; 16k/32k probes reported |
| Passes per request | One row per question *(inference)* |
| Order invariance | Not by construction; reduced by augmentation (flip rate 0.0283) |
| Languages | English, Chinese |

## Training

- **Data:**
  - typed-decisions train split, 1,200 cases [@typed-decisions];
  - flip-augmented option reorderings;
  - about 2,400 machine-translated rows;
  - about 1,400 native Chinese rows.
- **Loss:** CE + Brier. The weighting is not recorded.
- **Optimiser:** Adafactor, learning rate 2e-5 (encoder) and 1e-4 (head).
- **Budget:** **300 updates, 0.044 hours**, or about 2.6 minutes *(inference)*.
- **Comparison with Laya.** Laya's #963 found that small fine-tunes (72 updates) can collapse to the prior. Phocinae's 300 updates on a 144M backbone report no such collapse. *(inference)* The notes do not record a seed study.

## Calibration

**Shipped temperatures** are 0.866 / 0.808 / 0.662, one per type. All are below 1, so they **sharpen** the outputs.

| Setting | ECE | Brier |
|---|---|---|
| Shipped | 0.2519 | 0.0207 |
| With the opt-in `calib/` column | 0.0168 | 0.1218 |

**Contradiction.** ECE falls 15-fold while Brier rises about 6-fold. *(inference)* A monotone recalibration that lowers ECE this much would normally also lower Brier, or leave it roughly unchanged. The pair suggests different evaluation sets or metric definitions. The notes do not resolve this.

## Results

| Metric | Value | Caveat |
|---|---|---|
| typed-decisions accuracy | 0.791 (en) / 0.848 (zh) | In-distribution; "not a like-for-like head-to-head" with Laya / Jev |
| JevBench public | 0.5455 | — |
| Option-order flip rate (any of 3 reorders) | 0.0283 | Laya: 3.7% in-domain, 19.4% out of domain, per this report |
| 16k / 32k context probes | 0.453 / 0.387 | See the `max_len` contradiction |
| Latency | 21.0 ms on an RTX 5090; 1.64 s per case on one CPU thread | — |

**Points of reference** *(not like-for-like)*:
- `laya-typed-decisions` (421M) reports 0.766 on typed-decisions, with no committed result file. Jev 1.13 reports 0.727 ([laya.md](laya.md)).
- Laya's flip rates in this report differ from those in other sources: 0.150 in Laya's own BENCHMARKS.md, 9.8% in sbert2s1 and 30.6 in Runoff. Each protocol differs.

## Serving and efficiency

- **GPU:** 21.0 ms per case on an RTX 5090.
- **CPU:** 1.64 s per case on a single thread.
- **Size:** fp16 weights are 288.6 MB. The model has about one third of Laya's 421M parameters *(inference)*.
- **Gaps:** the notes record no serving stack. *(inference)* Laya's `laya-serve` should load it, since the config schema matches.

## Flaws and failure modes

### (a) Phocinae's own flaws

- **In-distribution headline.** The typed-decisions result is in-distribution, by the report's own statement.
- **Contradictory calibration.** The shipped temperatures sharpen, and the recalibrated column trades ECE 0.2519 → 0.0168 for Brier 0.0207 → 0.1218.
- **Unexplained context.** It reports 16k/32k context probes on a 512-token configuration.
- **Long contexts degrade.** Accuracy is 0.453 at 16k and 0.387 at 32k.
- **No external evaluation.** None was found in the notes, and full training code was not confirmed.

### (b) Flaws of other systems it claims to fix

- **Option-order flips in Laya.** The report puts Laya at 3.7% in-domain and 19.4% out of domain. Phocinae reports 0.0283 using flip augmentation, with no architectural change.
- **Assessment** *(inference)*. This is augmentation, not invariance by construction as in Von 1.2's 0 flips. The residual 2.8% may grow out of domain, as Laya's did, but the notes record no out-of-domain flip rate for Phocinae.

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **Option-permutation augmentation** as a low-cost baseline fix in any joint path. It cut the flip rate to 0.0283 here.
- **Bilingual data from translated plus native rows.** It is a cheap route to a second language, but only with native-language test items.
- **Both ECE and Brier for every calibration variant.** Phocinae shows the two can move in opposite directions.

**Avoid**
- Headline accuracy on the training distribution.
- Shipping temperatures below 1 without held-out evidence. Laya's 0.1006 temperature is the cautionary precedent (#394).

## References

- Phocinae repository: [github.com/Phocinae/Phocinae-Largha-150M-v1](https://github.com/Phocinae/Phocinae-Largha-150M-v1) [@phocinae]
  - [technical report](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/docs/technical-report.md)
  - [config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/config.json)
  - [rl_agent_config.json](https://github.com/Phocinae/Phocinae-Largha-150M-v1/blob/main/rl_agent_config.json)
- Laya, the architecture it follows: [@laya-repo], [@laya-card]
- typed-decisions: [HF](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) [@typed-decisions]
- Comparison flip rates:
  - sbert2s1 [@sbert2s1]
  - Runoff [@runoff]
  - Von [@von]
