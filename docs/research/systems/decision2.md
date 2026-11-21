# vLLM-SR Decision 2.0 (and System One Auto)

> **Family:** decoder-readout · **Team:** vLLM Semantic Router team · **Released:** 2 Oct 2026 (six models: Kai-0.6B, Eos-0.8B, Sol-2B, Nox-4B, Lux-9B, Vega-27B) · **License:** Apache-2.0 · **Code/weights:** weights on HF ([collection](https://huggingface.co/collections/vllm-sr/decision-20)); the head is readable in HF remote code (`decision2/_vendor/dev2model/decision_model.py` in [Kai-0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/main)); the router framework repo ([vllm-project/semantic-router](https://github.com/vllm-project/semantic-router), 6,512 stars per the catalog) has no Decision 2.0 training code; **training data and loss are undisclosed** (the code shows only a `teacher_probs` replay field) · **Sources read:** [docs page](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/), Kai remote code (`decision_model.py`, `calibration.py`, `score_bias.json`, `shared_ctx.py`), Kai and Vega cards, the [ModelSystem.One news post](https://modelsystem.one/news/vllm-sr-decision-2-family/). Only lightly read: vLLM-SR blog posts beyond the docs page. Not found: training data, loss, and the System One Auto gate rule.

## Summary

Decision 2.0 is a six-model open family (0.6B to 27B) from the vLLM Semantic Router team, served through `AutoModel.from_pretrained(trust_remote_code)` and `model.system_one(...)` ([docs](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/)) [@decision2] [@decision2-kai]. Each question is read in a causal row whose options are wrapped in `<option>` tags, and a `CandidateHead` scores each option's endpoint against the final `Decision:` position with a bilinear pointer plus an MLP interaction term. The idea that matters most is **opt-in shared-context serving with an exactness guard**: the common prefix is computed once (with DeltaNet continuing from the prefix state), and near-tie decisions fall back to the exact path. Vega-27B is the best System-1 model in System1-Arena (73.0%, ahead of Jev's 72.6%), and Kai-0.6B gains the most over its own backbone (+11.0 pp) ([S1-Arena](https://arxiv.org/html/2610.21244)) [@s1arena].

## Architecture

**Backbones.**
- Kai: fine-tuned from Qwen3-0.6B-Base, 8,192 context ([card](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B)).
- Vega: Qwen3.8-27B adapter, 32,768 context ([card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)).
- The Kai config is "base-random-head", a **uniform soup of 6 arms**.

**Prompt.** `Context:{state} Task type … Question … Options: <option>{json}</option>… Select the single option… Decision:`.

**Readout** (`CandidateHead`, fp32, causal mask, one row per question):

$$s_k = \frac{\big(K\,\mathrm{LN}(h_{\mathrm{end},k})\big) \cdot \big(Q\,\mathrm{LN}(h_{\mathrm{Decision}})\big)}{\sqrt{256}} + w \cdot \mathrm{GELU}\big(M_c\,\mathrm{LN}(h_k) + M_q\,\mathrm{LN}(h_q)\big),$$

where $h_{\mathrm{end},k}$ is the hidden state at option $k$'s endpoint and $h_{\mathrm{Decision}}$ the state at `Decision:`. The decoder-family taxonomy describes this as an "endpoint pointer: bilinear plus MLP". The notes write the MLP term's inputs as $h_k$ and $h_q$ without further definition.

**Shared context** (`shared_ctx.py`).
- The longest common prefix is cut before the first option.
- "Tree" mode packs the prefix and the per-question suffixes into one row, with two attention kernels merged by log-sum-exp; DeltaNet layers continue from the prefix state.
- Near-ties (margin < τ) fall back to the exact path.

| Property | Value |
|---|---|
| Passes per request | One row per question by default; opt-in "tree" mode packs prefix + suffixes into one row |
| State prefix reusable | Opt-in, with a near-tie fallback to the exact path |
| Option embeddings reusable | No (options follow the shared prefix inside each question) |
| Max K | 255 (decoder-family note: "one pass up to 255") |
| Context | Kai 8,192; Vega 32,768 |

**Earlier note.** The landscape note (2026-11-20) described the head as "undisclosed" and the model as "one pass, many questions" (quoting "answers them all at once and returns a probability for every answer"). The later decoder-family read of the remote code supersedes the first point and refines the second: the default is one row per question, with shared context opt-in.

## Training

- **Undisclosed.** Training data and loss are not published; the code shows only a `teacher_probs` replay field.
- Kai's config "base-random-head" and its **uniform soup of 6 arms** indicate a randomly initialised head and a six-run weight average *(inference from the config names, as recorded)*.

## Calibration

- Per-type temperatures in $[0.05, 20]$ (`calibration.py`).
- A per-level **score bias** for 5-level scores, fitted with class-balanced NLL (`score_bias.json`).
- Third-party: Reading Right, Acting Wrong reports Vega's ECE as 26.6% (and CLM's as worse) ([2610.18579](https://arxiv.org/html/2610.18579)) [@readingright].

## Results

**Team-reported.**

| Model | Metric | Value | Source |
|---|---|---|---|
| Vega-27B | JevArena | 74.0 | [Vega card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B) |
| Vega-27B | Decision Index | 56.5 | Vega card |
| Vega-27B | Latency per single question | 71.4 ms | Vega card |
| Kai-0.6B | Latency per single question | 4.9 ms | cards (landscape note) |

**System1-Arena** ([2610.21244](https://arxiv.org/html/2610.21244)) [@s1arena].

| Model | Accuracy | Brier | p50 |
|---|---|---|---|
| Vega-27B | 73.0 | 0.343 | 228 ms |
| Jev 1.13 | 72.6 | 0.349 | 321 ms |
| Microsoft-Decision-1 | 70.9 | 0.369 | 304 ms |
| OpenAI Decisions | 68.6 | 0.424 | 140 ms |
| Kai-0.6B | 50.3 | 0.537 | 72 ms |

- System-1 post-training vs base model (Fig. 5): Kai +11.0 pp and Brier −0.400 over Qwen3-0.6B-Base; Vega +2.9.
- Agentic: Vega, Jev and Qwen3.5-9B all reach 67.8% step accuracy but 26.5%, 17.4% and 5.7% loop success. All five System-1 models lose step accuracy against their base (−1.2 to −19.3).
- No-answer rates of 13.2–13.7% of single steps for Jev, Vega and MS-Decision-1 depress their accuracy.

**StartLux Table 4 (DI 0.3 public):** Decision 2.0 Sol 30.05, vs StartLux 27B AutoThink 73.37 and Jev 57.96 ([StartLux](https://arxiv.org/html/2610.17309)) [@startlux].

**System One Auto** ([ModelSystem.One](https://modelsystem.one/news/vllm-sr-decision-2-family/)) [@s1auto].
- Kai first, escalating to Vega when the acceptance gate fails; the gate rule is not stated.
- JevBench public (231 requests): Kai 65.80%, Auto 82.25%, Vega 87.88%.
- The "54.11%" cost saving is estimated from an author-run pilot, not billed.

## Serving and efficiency

- Shared context: "up to 3.4× faster for 128 questions on Vega-27B" ([docs](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/)).
- Fused Triton kernels: 11–48% faster with no changed decision.
- **ROCm merged rows changed decisions** (docs).
- Kai: about 0.2 s for 2 questions on 16 CPU cores (landscape note).
- Size vs latency in S1-Arena: Kai 72 ms vs Vega 228 ms.

## Flaws and failure modes

**(a) Flaws of Decision 2.0 itself**
- **Opaque training:** data and loss undisclosed; only the head and calibration code are readable.
- **Batch and runtime nondeterminism:** merged rows on ROCm changed decisions; the shared-context path needs a near-tie fallback to stay exact (decoder-family cross-table).
- **Steep size scaling:** Kai-0.6B 50.3% vs Vega-27B 73.0% in S1-Arena.
- **Agentic step accuracy falls after post-training** (all five System-1 models, −1.2 to −19.3), and Vega leaves 13.2–13.7% of single steps unanswered.
- **Calibration under shift:** Vega ECE 26.6% in Reading Right.
- **Cascade claims are estimates:** System One Auto's gate is undisclosed and the 54.11% saving is not billed.

**(b) Flaws of other systems it addresses**

The notes record no explicit claim by the Decision 2.0 team to fix a named flaw of another system. What the evidence shows:
- **Cost of running a large model on every request:** System One Auto escalates from Kai to Vega only when the gate fails (82.25% vs Vega's 87.88% on 231 JevBench public requests; savings estimated).
- **Post-training that hurts the backbone (CLM in S1-Arena):** Kai gains +11.0 pp and −0.400 Brier over Qwen3-0.6B-Base, the largest gain among the study's System-1/base pairs (Kai +11.0, MS-Decision-1 +5.6, Vega +2.9, OpenAI Decisions −2.6, CLM −17.8) ([S1-Arena](https://arxiv.org/html/2610.21244)). The methods note infers that a 0.6B joint-encoding model beating the frozen 8B bi-encoder CLM (50.3 vs 41.4) points to architecture and objective, not capacity *(inference, as in the note)*.

## Implications for our design

All bullets are *(inference)*.

**Adopt**
- **Shared-prefix serving with an exactness guard:** compute the common prefix once and fall back to the exact path on near-ties; measure decision changes per kernel/backend.
- **A bilinear pointer plus a small MLP term** as a cheap readout; Strands' evidence that a 70k-parameter pointer beats larger slot heads points the same way [@strands].
- **A cascade from a small to a large model gated on calibrated confidence**; the methods and landscape notes propose a small-then-large CLM cascade on this evidence.
- **Per-level score bias** fitted with class-balanced NLL for ordinal outputs.
- **Use Kai/Nox as cheap local baselines** (the landscape note's feasibility table).

**Avoid**
- **Reporting cascade savings without billing data or a stated gate.**
- **Assuming kernels and batched rows are decision-preserving across hardware** (ROCm).
- **Judging agentic readiness by step accuracy:** equal step accuracy (67.8%) hid loop-success differences of 26.5% vs 17.4% vs 5.7%; report loop success too.

## References

- [@decision2] https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/
- [@decision2-kai] https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B (tree: https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/main)
- Vega card: https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B; HF collection: https://huggingface.co/collections/vllm-sr/decision-20
- [@s1auto] https://modelsystem.one/news/vllm-sr-decision-2-family/
- [@s1arena] https://arxiv.org/html/2610.21244
- [@startlux] https://arxiv.org/html/2610.17309
- [@readingright] https://arxiv.org/html/2610.18579
- [@vllmsr], [@semcache] (vLLM Semantic Router background papers)
- Router repo: https://github.com/vllm-project/semantic-router
- [@strands] https://github.com/strands-labs/strands-decider
