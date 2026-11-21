# Cloudflare Clef / Clef-flash

> **Family:** decoder-readout · **Team:** Cloudflare · **Released:** HF weights 2026-09-30; blog 2026-10-01; `clef-omni` added 2026-10-09 · **License:** Apache-2.0 · **Code/weights:** HF only, no repo and no training code ([Cloudflare/clef](https://huggingface.co/Cloudflare/clef), Qwen3.8-27B, 78.2k downloads / 2,380 likes; [clef-flash](https://huggingface.co/Cloudflare/clef-flash), Qwen3.5-9B, 33.4k downloads / 933 likes, per the catalog); served on Workers AI · **Sources read:** HF card, `joint_schema_model.py`, `joint_head_config.json`, the [blog](https://blog.cloudflare.com/clef-decision-models). **Source caveat:** most facts below come from the encoder-family dossier (`06-encoder-family-laya-inkling.md`, §6), which is not one of the source files assigned for this set of notes; it is used here because the assigned files mention Clef only in passing (catalog row, Small Heads, Reading Right). No arXiv paper exists.

## Summary

Clef is Cloudflare's open decision model, a decoder despite being grouped with encoder-style systems ([card](https://huggingface.co/Cloudflare/clef)) [@clef] [@clef-blog]. One causal prefill reads the system prompt, the state and a schema of all questions, and a 4-layer "joint schema" head then routes option queries over the sequence memory, letting questions ("fields") attend to each other and to the memory. The idea that matters most is the **lexical prior**: each option query and the final logit include a term built from the mean of the LM's *output embeddings* of the option's tokens, so the label text itself contributes a prior *(inference: this should help unseen labels; the notes report no ablation)*. Clef leads Jev on BANKING77 (94.2 vs 79.7) but trails it on knowledge and reasoning suites (MMLU-Pro 65.9 vs 82.7; BBH 73.7 vs 92.9), in Cloudflare's own Decision Index run.

## Architecture

(`joint_schema_model.py`, `joint_head_config.json`; encoder-family dossier §6)

- **Backbones:** Clef = Qwen3.8-27B; Clef-flash = Qwen3.5-9B.
- **Sequence:** one causal prefill over `system + "STATE:" + state + schema + "<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"`. **All questions of a record share one pass.**
- **Head config:** width 1024, 2 routing layers, 4 layers total, 16 heads.
- **EvidenceRoutingLayers:** option queries = proj(option-span mean) + proj(mean of the LM **output embeddings** of the option tokens, the "lexical" vector) + proj(question mean). These cross-attend to the projected sequence memory.
- **Fields** (questions) cross-attend to each other and to memory.
- **Logit** for option $k$ of field $f$:
$$z_{f,k} = \pi_k + \sigma(g)\Big(s\cdot\cos(f, o_k) + \mathrm{MLP}\big[f;\, o_k;\, f \odot o_k;\, |f - o_k|\big]\Big), \qquad \pi_k = e^{\rho}\cos\big(\mathrm{lex}_k,\; q + h_{\text{last}}\big),$$
  where $\mathrm{lex}_k$ is the lexical vector, $q$ the question representation and $h_{\text{last}}$ the last-token state.
- **Small Heads' description:** the "joint schema (Clef)" readout, re-implemented on Qwen3-0.6B, is "cross-attention to a token memory plus 4 decoder layers" with 27.3M parameters ([Small Heads](https://arxiv.org/html/2611.15592)) [@smallheads].

| Property | Value |
|---|---|
| Passes per request | One causal prefill for all questions of a record, then the head |
| State prefix reusable | Possible in principle: the causal "state first, schema after" layout is the only one in its group that allows prefix-KV reuse *(inference, as in the encoder-family note)*; no measured reuse is reported |
| Option embeddings reusable | No (option queries are pooled from the joint sequence) |
| Max K | Not stated in the notes |
| Context | Not stated in the notes |

## Training

(blog, as recorded in the encoder-family dossier)
- The routing head is trained jointly with **rank-256 LoRA**. The blog says the backbone is "frozen", but the card says `base_model_relation: finetune` *(the dossier's inference: the LoRA is merged into the weights)*.
- **Loss:** label-smoothed cross-entropy + Brier.
- **"RLCD"** with ordinal partial credit, an exact-record reward and a reference penalty.
- **Data:** synthetic, with permuted fields and schemas; sizes not given.

## Calibration

No ECE is reported. Cloudflare's Decision Index run reports a ForecastBench Brier (below).

## Results

**Decision Index 0.2.1, Cloudflare's internal run** (encoder-family dossier §6).

| Benchmark | Clef | Clef-flash | Jev |
|---|---|---|---|
| BANKING77 | 94.2 | 90.9 | 79.7 |
| MMLU-Pro | 65.9 | 65.3 | 82.7 |
| BBH | 73.7 | 68.9 | 92.9 |
| ForecastBench Brier | 13.9 | 10.6 | 17.4 |
| Median latency (ms) | 209.3 | 38.8 | 524.1 |

- Von's notes put Clef-flash at DI 57.1 vs Kev-9B at 38.5 ("recipe and head beat parameter count") (encoder-family dossier).

**Third-party.**
- **Reading Right, Acting Wrong:** asking facts as narrow questions and composing the action in code lifts Clef to 1.000 (Jev 0.791 → 0.996; Qwen3.5-9B → 0.965) ([2610.18579](https://arxiv.org/html/2610.18579)) [@readingright].
- **Small Heads:** with a joint prompt on Qwen3-0.6B, the Clef-style joint-schema readout scores 81.9 in-distribution and 76.1 out-of-distribution, vs 82.0 / 78.0 for a 0.53M cosine head; it costs 13.6 ms per row vs 6.4 ms, and 8.7 vs 6.9 GPU-hours to train (Tables 3–4; head and prompt regime are confounded) ([2611.15592](https://arxiv.org/html/2611.15592)) [@smallheads].

## Serving and efficiency

- SGLang `/v1/systemone` on an H200; Workers AI model `@cf/cloudflare/clef`.
- Median latency in the internal run: Clef 209.3 ms, Clef-flash 38.8 ms (Jev 524.1 ms).

## Flaws and failure modes

**(a) Flaws of Clef itself**
- **Knowledge and reasoning gap vs Jev:** MMLU-Pro 65.9 vs 82.7; BBH 73.7 vs 92.9 (internal run).
- **Head cost without clear gain at small scale:** in Small Heads' controlled study, the 27.3M joint-schema head did not beat a 0.53M cosine head (81.9 vs 82.0 ID; 76.1 vs 78.0 OOD) and was about twice as slow per row.
- **Questions interact within a record:** fields cross-attend to each other, so one question can influence another *(inference from the architecture)*.
- **Undisclosed data sizes; no ECE; a self-run benchmark;** and a blog/card inconsistency about whether the backbone is frozen.

**(b) Flaws of other systems it addresses**
- **Option-count collapse in encoders with fixed per-option token budgets** (Laya): the encoder-family note cites Clef's 94.2 on BANKING77 as evidence that avoiding a fixed per-option budget fixes it.
- **Jev's intent-classification weakness:** BANKING77 94.2 vs 79.7.

## Implications for our design

All bullets are *(inference)*.

- **Adopt:** a **lexical prior from the LM's output embeddings** of option tokens, which the encoder-family note lists among its adopt items; it gives unseen labels a sensible starting score and fits a cached-option design, since it depends only on the option text.
- **Avoid:** assuming a large joint-schema head is needed; Small Heads found a 0.53M head as good in a joint prompt.
- **Avoid:** letting questions attend to each other if per-question isolation and caching are requirements.

## References

- [@clef] https://huggingface.co/Cloudflare/clef (also https://huggingface.co/Cloudflare/clef-flash)
- [@clef-blog] https://blog.cloudflare.com/clef-decision-models
- [@smallheads] https://arxiv.org/html/2611.15592
- [@readingright] https://arxiv.org/html/2610.18579
