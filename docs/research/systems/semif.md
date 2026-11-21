# SemIf (formerly OpenJev)

> **Family:** decoder-readout · **Team:** Theo Lee (TheoLeeCJ) · **Released:** not recorded in the notes · **License:** MIT · **Code/weights:** inference technique only, no training and no weights of its own; reads option logits from existing open models; 4,970 stars (catalog) · **Sources read:** [docs/METHOD.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/METHOD.md), [docs/RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md).

## Summary

SemIf is a training-free decision readout: a frozen Qwen3.5-4B is prompted with the evidence, the criterion and lettered options, and the probabilities are its letter logits for 2–16 options ([repo](https://github.com/TheoLeeCJ/SemIf-OpenJev)) [@semif]. It matters mainly as a baseline and as an ancestor: its byte-identical prompt became the prompt used by Jeff-1.1 [@jeff]. On its own workloads it beats a dedicated reranker and comes within 0.038 of Jev's agreement on a TypeSafe subset, and it shows that a shared state prefill with parallel question suffixes gives about 8.6× throughput (our arithmetic: 20.03 / 2.33).

## Architecture

- **Backbone:** frozen Qwen3.5-4B; no training ([docs/METHOD.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/METHOD.md)).
- **Readout:** letter logits for 2–16 options, $p_k = \mathrm{softmax}_{k \le K}(z_{\ell_k})$ where $z_{\ell_k}$ is the next-token logit of letter $k$.
- **Prompt:** the prompt Jeff later adopted byte-for-byte: a system message ("Apply the supplied criterion… Respond with only its uppercase letter") plus a JSON object `{evidence, criterion, options: [{letter, description}]}` ([Jeff prompting.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py)) [@jeff-repo].
- **Serving layout:** one state prefill, then parallel question suffixes ([RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md)).

| Property | Value |
|---|---|
| Passes per request | One state prefill plus parallel question suffixes (or a fresh prompt per question) |
| State prefix reusable | Yes (parallel suffixes after one state prefill) |
| Option embeddings reusable | No |
| Max K | 16 |
| Context | Not stated in the notes |

## Training

None.

## Calibration

No calibration procedure or ECE values are recorded in the notes.

## Results

([docs/RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md))

| Workload | Direct letter logits | Qwen3-Reranker-4B | Jev |
|---|---|---|---|
| Authored (balanced accuracy) | 0.813 | 0.625 | – |
| TypeSafe subset agreement | 0.845 | – | 0.883 |

- The catalog also records 68.66 on the BenchLM JevBench list.

## Serving and efficiency

- RTX 3090, parallel suffixes after one state prefill: 20.03 decisions/s vs 2.33 with fresh prompts.
- Argmax drift reported alongside this measurement: 6 of 777.

## Flaws and failure modes

**(a) Flaws of SemIf itself**
- **Order sensitivity:** option reversal caused 10 flips of 36 for direct letter logits.
- **Small K ceiling:** 2–16 options.
- **Numerical drift with prefix sharing:** 6 of 777 argmax changes.

**(b) Flaws of other systems it addresses**
- **Retrieval-style rerankers for decisions:** letter logits 0.813 vs Qwen3-Reranker-4B 0.625 balanced accuracy on authored workloads [@qwen3emb].

## Implications for our design

All bullets are *(inference)*.

- **Adopt:** SemIf-style direct letter logits on our own backbone as a zero-training baseline; this is the "label log-probability baseline" that the evidence audit found in only 5 of 26 papers ([2609.32160](https://arxiv.org/html/2609.32160)) [@tdmaudit].
- **Adopt:** measure argmax drift whenever a shared-prefix fast path is introduced (6 / 777 here).
- **Avoid:** using raw letter logits where order robustness matters (10 / 36 flips) without a debiasing step.

## References

- [@semif] https://github.com/TheoLeeCJ/SemIf-OpenJev — [METHOD.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/METHOD.md), [RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md)
- [@jeff] https://arxiv.org/html/2610.17205
- [@jeff-repo] https://github.com/j-geuter/Jeff
- [@qwen3emb] (Qwen3-Embedding / Reranker)
- [@tdmaudit] https://arxiv.org/html/2609.32160
