# Small Heads, Strong Decisions

> **Family:** decoder readout (joint-prompt LoRA backbone; controlled comparison of five decision heads) · **Team:** Rastogi · **Released:** arXiv [2611.15592](https://arxiv.org/html/2611.15592) (submission date not recorded in the notes we read) · **License:** not disclosed / not read · **Code/weights:** the promised repository `Ritvik19/DecideAnything` returns HTTP 404; no weights recorded · **Sources read:** the arXiv HTML paper, including Tables 3–10, 15 and 17. **Gaps:** no code is available; the exact attention mask between options in the joint prompt is not specified in what we read.

## Summary

Small Heads is a controlled study, not a product. It fixes a joint-prompt backbone (Qwen3-0.6B with LoRA, all questions and options in one sequence) and swaps only the decision head. Five readouts are compared, from a 0.53M-parameter cosine head to a 27.3M "joint schema" head.

The headline finding is that **the readout barely matters when options are read in context**: the cosine head ties the 27.3M head in-distribution (82.0 vs 81.9) and beats it out of distribution (78.0 vs 76.1). Backbone scale dominates: moving from 0.6B to 4B adds about 11 OOD points. Retrieval-tuned checkpoints add nothing, and bidirectional attention hurts.

For a CLM-style design, the study isolates the variable that matters. A small cosine head is not the problem; encoding options without context may be.

## Architecture

**Backbone and prompt.**

- Qwen3-0.6B with LoRA r32 (α 64) on all projections.
- A **joint prompt** puts the state, all questions and all options in one sequence.
- One forward pass serves the whole request; every head reads from it.

**Readouts compared:**

| Readout | Parameters | Mechanism |
|---|---|---|
| Cosine | 0.53M | last-token pooling of the question and of each option (inside the joint sequence), projected to 256-d, RMS-normalised, scaled by $\min(e^{\theta},100)$ with the scale initialised at 10 |
| Joint schema (Clef-style) | 27.3M | cross-attention to a token memory plus 4 decoder layers |
| Bilinear-MLP | 1.05M | not detailed in our notes |
| Learned letter | 0.26M | not detailed in our notes |
| LM letter | 0 | frozen LM-head rows |

The cosine head therefore computes

$$\ell_i=\min(e^{\theta},100)\cdot\cos\!\big(\mathrm{RMS}(P h_q),\,\mathrm{RMS}(P h_{o_i})\big),$$

where $h_q$ and $h_{o_i}$ are last-token hidden states of the question and option spans *inside the joint sequence*. Whether the projections are shared between the two sides is not recorded *(rendering of the description)*. The scale cap of 100 is the same as CLM's.

**Passes per request.** One joint forward pass. Because options sit in context with the state, no option representation is request-independent, and nothing beyond a shared prefix can be cached *(inference)*.

| Property | Value |
|---|---|
| Backbone | Qwen3-0.6B (scale study up to 14B), LoRA r32 / α 64 on all projections |
| Encoding | joint prompt (state + all questions + options) |
| Heads compared | cosine 0.53M; joint schema 27.3M; bilinear-MLP 1.05M; learned letter 0.26M; LM letter 0 |
| Best pooling | last/last |
| Attention | causal; bidirectional variants worse |
| Calibration | two-level temperature: global in [0.05, 20] × per-type in [0.5, 5] |
| Cached | nothing request-independent *(inference)* |

## Training

- Data: 15 sources, 220,264 rows per epoch.
- Loss: soft cross-entropy averaged over questions.
- Learning rate: LoRA at 2e-4 diverged on Qwen3-8B with the cosine head; 1e-4 trains normally.
- Evaluation: single seed, with paired bootstrap for comparisons.

## Calibration

- The temperature has two levels: a global temperature clamped to [0.05, 20], multiplied by a per-type factor clamped to [0.5, 5].
- Calibrated ECE is about 0.033–0.034 in-distribution and 0.057–0.077 out of distribution.

## Results

**Readout comparison (Table 3), Qwen3-0.6B:**

| Readout | ID | OOD |
|---|---|---|
| Cosine | 82.0 | 78.0 |
| Joint schema | 81.9 | 76.1 |
| Bilinear-MLP | 82.0 | n.r. |
| LM letter | n.r. | 79.4 |

The LM-letter readout's OOD lead comes entirely from Belebele, and it depends on the letter codes (Table 5).

**Cost (Table 4):** the cosine head runs at 6.4 ms per row vs 13.6 ms for the joint head, and costs 6.9 vs 8.7 GPU-hours to train. The head and the prompt regime are confounded in this comparison.

**Scale (Table 7), cosine head:**

| Backbone | ID | OOD |
|---|---|---|
| 0.6B | 82.0 | 78.0 |
| 4B | 85.9 | 88.9 |
| 8B | 85.8 | 89.8 |
| 14B | 86.6 | 90.8 |

**Other controlled findings.**

- *Checkpoint flavour (Tables 8–9):* base, instruct and thinking checkpoints tie within 0.25 points in-distribution.
- *Retrieval-tuned checkpoints (Tables 10 and 17):* out of distribution, Qwen3-Embedding-8B scores 89.2 and Qwen3-Reranker-8B 88.7, against 89.8 for the general 8B [@qwen3emb].
- *Pooling and attention (Tables 6 and 15):* last/last pooling is best (78.0 OOD). **Bidirectional attention costs 3.5–6.0 points ID and 5.4–11.8 OOD.**

No independent evaluation of this study's models appears in the sources we read.

## Serving and efficiency

- The cosine head is about 2× cheaper per row than the joint-schema head (6.4 vs 13.6 ms) and cheaper to train (6.9 vs 8.7 GPU-hours).
- Every request still needs one joint backbone pass over the state plus all options.
- The study does not test request-independent option caching *(inference)*.

## Flaws and failure modes

### (a) The study's own limitations, with evidence

- **Single seed.** Small differences, such as 82.0 vs 81.9, are within what a second seed could move. The paper uses paired bootstrap but no seed replication.
- **Confounded cost comparison.** The head and the prompt regime vary together in Table 4.
- **Fragile LM-letter result.** Its OOD lead comes only from Belebele and depends on the letter codes.
- **Unavailable code** (404), and an **unspecified option-to-option attention mask** in the joint prompt.
- **Training instability at 8B.** LoRA at 2e-4 diverged.

### (b) Flaws of other systems it addresses

- **The belief that large decision heads are needed.** A 0.53M cosine head matches a 27.3M head (Table 3).
- **The belief that retrieval-tuned or bidirectional backbones help decisions.** Neither does: retrieval tuning gives 89.2 / 88.7 vs 89.8, and bidirectional attention costs 3.5–11.8 points.
- **Letter-readout brittleness.** The LM-letter OOD advantage is an artefact of one benchmark and of the letter codes.

## Implications for our design

*(All items are inference unless cited.)*

- **Head size is not the lever.** A small cosine head is fine *when options are encoded in context with the state*. CLM's cosine head encodes options context-free, and CLM's failures (#3, #15) point to the encoding, not to the head. The variable to test is **context-free vs in-context option encoding**, holding the head fixed.
- **Backbone scale is the lever.** 0.6B → 4B adds about 11 OOD points, while 4B → 8B → 14B adds only 0.9 and then 1.0. Our 8B backbone is not the bottleneck.
- **Do not expect Qwen3-Embedding to help** as a decision backbone (89.2 vs 89.8).
- **Keep causal attention and last-token pooling.** Bongard's bidirectional-encoder gain [@bongard] does not carry over to decoders here.
- **Use 1e-4 if we add LoRA on Qwen3-8B.** 2e-4 diverged.
- **Adopt the two-level temperature** (global × per type) with clamps, as a cheap calibration layer.

## References

- [@smallheads] Rastogi, "Small Heads, Strong Decisions", arXiv 2611.15592: https://arxiv.org/html/2611.15592
- Promised code (404): https://github.com/Ritvik19/DecideAnything
- [@qwen3emb] Qwen3-Embedding / Reranker, arXiv 2506.05176
- [@clef] Cloudflare Clef (the joint-schema head family): https://huggingface.co/Cloudflare/clef
- [@bongard] Bongard, arXiv 2609.39111: https://arxiv.org/html/2609.39111
- [@clm-repo] CLM issues #3 and #15: https://github.com/Contrastive-LM/CLM
