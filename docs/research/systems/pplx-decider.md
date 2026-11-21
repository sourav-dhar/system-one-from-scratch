# pplx-decider (Perplexity; AutoJev lineage)

> **Family:** decoder-readout · **Team:** Perplexity · **Released:** v1 (= AutoJev-27B) and v1.1; release dates not recorded in the notes; Jeff's Table 3 also lists a "pplx-decider-v1.3" · **License:** Apache-2.0 (catalog) · **Code/weights:** HF only ([v1.1 card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b); 3.7k downloads, 182 likes per the catalog); inference code at `source/src/autojev/model.py`; no training code; no technical report found · **Sources read:** HF card and inference code, the [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Perplexity/PPLX-Decider-v1.1-27B.md), Kev PLAN.md "Against Jev" (for v1).

## Summary

pplx-decider is Perplexity's 27B decision model on Qwen3.8-27B ([card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)) [@pplx]. It reads a 255-row decision head (shape `[255, 5120]`) mapped to vocabulary token ids. The idea that matters most is v1.1's **removal of the causal mask in the 16 full-attention layers** (the 48 DeltaNet layers stay causal): it is meant to be served noncausally, and serving it causally changes the top answer on 101 of 2,167 items. The cost is that nothing can be reused across requests: SGLang disables the radix cache and chunked prefill for it.

## Architecture

- **Backbone:** Qwen3.8-27B; 16 full-attention layers and 48 DeltaNet layers.
- **Readout:** a `[255, 5120]` matrix mapped to vocabulary token ids, i.e. $z_k = r_k^\top h$ for $k \le K$, $p = \mathrm{softmax}(z / T)$. The decoder-family adopt list groups it with Strands v2 as a 255-code readout initialised from LM-head rows; the notes do not state the read position.
- **Mask (v1.1):** causal mask removed in the 16 full-attention layers; the 48 DeltaNet layers stay causal.
- **Passes:** one pass.

| Property | Value |
|---|---|
| Passes per request | One |
| State prefix reusable | No (radix cache disabled; "nothing is reused across requests") |
| Option embeddings reusable | No |
| Max K | 255 |
| Context | Trained to 8,192 tokens (decoder-family cross-table) |

## Training

- **v1 = AutoJev-27B:** full-weight SFT on 73k synthetic rows (Kev PLAN.md "Against Jev") [@kev-repo].
- **v1.1:** 1 epoch on 626,033 rows, of which 530,103 come from tasksource ([SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Perplexity/PPLX-Decider-v1.1-27B.md)).
- Objective, LR and other hyperparameters are not recorded in the notes.

## Calibration

- A single temperature: v1.1 T = 1.0087 (v1 T = 2.2076).

## Results

| Model | Benchmark | Value | Source |
|---|---|---|---|
| v1.1 | Decision Index (version not stated) | 61.56 (vs v1 56.4) | SGLang cookbook |
| v1.1 | Decision Index, card comparison | 61.56 vs Jev 57.9 | card (catalog) |
| v1.1 ("Perplexity Decider v1.1") | DI 0.3 public | 62.25 | [StartLux Table 4](https://arxiv.org/html/2610.17309) [@startlux] |
| v1.3 | OpenDecisions Capability | 85.1 | [Jeff Table 3](https://arxiv.org/html/2610.17205) [@jeff] |
| v1 (AutoJev) | Kev breadth index | 50.0 (Kev-27B v2 52.3, Jev 54.0) | [Kev-27B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) [@kev-repo] |
| v1.1 served causally | top answer changed | 101 of 2,167 items | card |
| v1.1 | WinoGrande, causal vs noncausal | 88.87% vs 92.42% | card |

- The card's 61.56 and StartLux's 62.25 differ; the notes do not say whether they use the same DI version.

## Serving and efficiency

- SGLang disables the radix cache and chunked prefill: "nothing is reused across requests".
- GB300: 53 ms for a short question; four questions totalling 4,540 tokens: 183 ms.

## Flaws and failure modes

**(a) Flaws of pplx-decider itself**
- **No reuse:** bidirectional full-attention layers rule out radix/prefix caching and chunked prefill.
- **Mask-dependence:** serving causally changes 101 of 2,167 top answers and costs about 3.6 points on WinoGrande (88.87% vs 92.42%), so it cannot fall back to a standard causal cache path without loss.
- **Context:** trained to 8,192 tokens.
- **Opacity:** no technical report; recipe details beyond row counts are not public.

**(b) Flaws of other systems it addresses**
- None claimed explicitly in the notes. Its DI improvement over v1 (61.56 vs 56.4) came alongside the mask change and a larger SFT set; the notes do not isolate the mask's contribution.

## Implications for our design

All bullets are *(inference)*.

- **Avoid full-prompt bidirectional attention** in a model that must serve many questions per state: the decoder-family note groups pplx with StartLux and Jeff as giving small gains while killing reuse.
- **If bidirectionality is wanted, prefer a state-only bidirectional mask** (Strands' `prefix_bidir`) that keeps the state cacheable [@strands].
- **Test mask/serving parity explicitly:** report how many decisions change between the training-time and serving-time attention patterns (here 101 / 2,167).

## References

- [@pplx] https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b
- SGLang cookbook: https://lmsysorg.mintlify.app/cookbook/autoregressive/Perplexity/PPLX-Decider-v1.1-27B.md
- [@kev-repo] https://github.com/jaredpalmer/kev (PLAN.md "Against Jev"; Kev-27B card)
- [@startlux] https://arxiv.org/html/2610.17309
- [@jeff] https://arxiv.org/html/2610.17205
- [@strands] https://github.com/strands-labs/strands-decider
