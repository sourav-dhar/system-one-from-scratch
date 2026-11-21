# Liquid d1

> **Family:** decoder-readout · **Team:** Liquid AI · **Released:** 2026-10-07 per the [d1-open blog](https://www.liquid.ai/blog/d1-open) · **License:** LFM Open License (HF tag "other"; not OSI-approved) per the catalog; the decoder-family dossier did not read the licence text · **Code/weights:** HF only, no training code. The dossier names d1-3B and d1-omni-600M; the catalog lists [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (20.3k downloads), d1-2B (16.2k) and d1-Omni-2B (6.1k) — the two notes disagree on the variant names and sizes · **Sources read:** d1-open blog, [prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py) on HF. The catalog additionally lists a [d1 blog](https://www.liquid.ai/blog/d1-decision-model) and [docs](https://docs.liquid.ai/lfm/models/decision-models). Not disclosed: training objective.

## Summary

Liquid d1 is Liquid AI's small decision-model line on its LFM2 architecture; d1-3B is LFM2.5-VL-3B weight-averaged with LFM2.5-2.6B ([blog](https://www.liquid.ai/blog/d1-open)) [@liquid-d1] [@lfm2]. Its readout scores each option by the **maximum LM logit over that option's token forms** (for example "A" and " A") at the answer slot. The idea that matters most is the blog's claim that "long inputs, shuffled answer options, and fixing data shortcuts mattered more" than advanced techniques. It is fast (8 ms per question on an RTX 4090) and scores 48.57 on Decision Index v0.2.1 public.

## Architecture

- **Backbone:** causal LFM2; d1-3B = LFM2.5-VL-3B averaged with LFM2.5-2.6B. The d1-3B card tags the LFM2 technical report (arXiv 2511.23404) [@lfm2].
- **Readout** ([prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py)): single-token codes A–Z, then 00…; each option's score is
$$z_k = \max_{t \in \mathcal{F}_k} \ell_t,$$
  where $\mathcal{F}_k$ is the set of token forms for option $k$ (e.g. "A" and " A") and $\ell_t$ the LM logit at the answer slot. Noul uses yes/no forms; Score uses digits (2–10 levels).
- **Passes:** one pass; packed states (decoder-family taxonomy).

| Property | Value |
|---|---|
| Passes per request | One |
| State prefix reusable | Not stated in the notes |
| Option embeddings reusable | No |
| Max K | Not stated (codes run A–Z, then 00…) |
| Context | Not stated; a 3.4k-token state is timed |

## Training

- Seed and mixture merges.
- The blog: "long inputs, shuffled answer options, and fixing data shortcuts mattered more" than advanced techniques.
- The objective is not stated.

## Calibration

No calibration procedure or values are recorded in the notes.

## Results

| Benchmark | d1-3B | Source |
|---|---|---|
| Decision Index v0.2.1 public | 48.57 | [blog](https://www.liquid.ai/blog/d1-open) |
| DI 0.3 public | 48.99 | [StartLux Table 4](https://arxiv.org/html/2610.17309) [@startlux] |

For context, StartLux's Table 4 places d1-3B below Jev (57.96) and above Decision 2.0 Sol (30.05).

## Serving and efficiency

- RTX 4090: 8 ms per question; 102 ms for a 3.4k-token state.

## Flaws and failure modes

**(a) Flaws of d1 itself**
- **Lower accuracy than larger systems:** 48.57 (DI v0.2.1) / 48.99 (DI 0.3) vs Jev 57.96 on DI 0.3.
- **Undisclosed objective and non-OSI licence** (LFM Open License).
- **Max-over-forms readout** depends on which token forms are enumerated *(inference)*.

**(b) Flaws of other systems it addresses**
- None claimed explicitly. The blog's emphasis on shuffled options and fixing data shortcuts targets order sensitivity and shortcut learning generally *(inference)*.

## Implications for our design

All bullets are *(inference)*.

- **Adopt:** score options over all tokenisation variants of a label (max over "A" / " A"), which is cheap and removes a tokenisation artefact from letter readouts.
- **Adopt:** prioritise data hygiene (long inputs, option shuffling, shortcut removal) before readout or loss novelties, per the blog's own ranking.
- **Note:** sub-10 ms per question at 3B makes d1 a candidate cheap first stage for a cascade, if its calibration can be established.

## References

- [@liquid-d1] https://www.liquid.ai/blog/d1-open
- d1-3B: https://huggingface.co/LiquidAI/d1-3B ([prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py))
- Other Liquid pages listed in the catalog: https://www.liquid.ai/blog/d1-decision-model, https://docs.liquid.ai/lfm/models/decision-models
- [@lfm2] (LFM2 Technical Report, arXiv 2511.23404)
- [@startlux] https://arxiv.org/html/2610.17309
