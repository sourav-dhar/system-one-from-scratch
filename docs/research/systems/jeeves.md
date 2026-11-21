# Jeeves

> **Family:** decoder-readout · **Team:** PostHog (Nicholas Waltz) · **Released:** repo created 2026-09-29 (461 stars as of 2026-11-20 per the catalog) · **License:** MIT · **Code/weights:** 9B weights on [HF](https://huggingface.co/PostHog/jeeves); full training code and data prep (`train.py`, `calibrate.py`, CISPO trainer, diffusion drafter); the SFT reasoning chains are "not published yet" · **Sources read:** README ("How it works", results table, ablations), [trainer.py](https://github.com/PostHog/jeeves/blob/master/trainer.py) `train_cispo`, [calibrate.py](https://github.com/PostHog/jeeves/blob/master/calibrate.py). Not available: the reasoning chains.

## Summary

Jeeves is PostHog's open decision model, "inspired by Kev": it reuses Kev's delimiter tokens and pointer head on Qwen3.5-9B with LoRA ([repo](https://github.com/PostHog/jeeves)) [@jeeves] [@kev-repo]. The idea that matters most is **reason-then-point**: when the direct answer is not confident, the model writes a reasoning trace, then **repeats the question and options** before the `<decide>` token, so the pointer reads a fresh question representation conditioned on the trace. This raises accuracy above Jev on its reported suites, but answers after reasoning are badly overconfident (a separate temperature of 5.42), and reasoning makes latency jump from 22 ms to over a second at p50.

## Architecture

- **Backbone:** Qwen3.5-9B with LoRA r16 on all projections.
- **Delimiters and readout:** Kev's delimiter tokens and pointer: query at `<decide>`, key at `</opt>` (README "How it works"). Following Kev's form, $z_k = (W_k h_{\langle/\mathrm{opt}\rangle_k}) \cdot (W_q h_{\langle\mathrm{decide}\rangle}) / \sqrt{d_p}$ with a softmax over options; the notes do not restate Jeeves' head dimension.
- **Sequence:** `<state> … <q> instr <opt>…</opt> … <think>` → reasoning → `</think>` → **the question and options repeated** → `<decide>`.
- **Mask:** causal (decoder-family taxonomy).
- **README ablations:** plain-text tags instead of special tokens, or not repeating the question after reasoning, "worsened performance".

| Property | Value |
|---|---|
| Passes per request | One prefill and direct readout; if confidence < threshold, an autoregressive reasoning trace, the repeated question, and a second readout |
| State prefix reusable | Not stated in the notes |
| Option embeddings reusable | No (options are in the joint sequence, and repeated after reasoning) *(inference)* |
| Max K | Not stated in the notes |
| Context | Not stated in the notes; training chains capped at 1,024 tokens |

## Training

**SFT.**
- 1 epoch, 16,867 steps on 4 GPUs; 453,531 questions from ~175 public sources.
- About 15% of examples carry a self-generated reasoning chain, kept only if it closes within 1,024 tokens, does not loop and reaches the right answer.

**CISPO (v1)** ([trainer.py](https://github.com/PostHog/jeeves/blob/master/trainer.py) `train_cispo`).
- Group size $G = 8$; 8 prompts per step.
- Reward: the pointer head's probability of the gold option after reasoning, times a length factor:
  $r = p_{\text{correct}} \cdot (1 - \lambda_{\text{len}})$, with a hinge length penalty from 2,048 to 1,024 tokens, capped at 0.1.
- Advantage: $A = r - \bar r_{\text{group}}$.
- Importance weight: $\mathrm{sg}\big[\min(\exp(\log\pi - \log\pi_{\text{old}}),\, 1 + 0.2)\big]$ (detached).
- Plus rollout cross-entropy (weight 0.5) and anchor cross-entropy (weight 0.5).
- "CISPO did not improve v2."

## Calibration

Two temperatures ([calibrate.py](https://github.com/PostHog/jeeves/blob/master/calibrate.py)): **1.52** for direct answers and **5.42** after reasoning. The stated reason: "Every training chain reaches the right answer, so answers after reasoning come out overconfident." The decoder-family cross-table groups this with StartLux's post-reasoning calibration error (8.1 vs 3.7 default) as evidence that post-reasoning answers are overconfident.

## Results

README table; Jeeves reasons when confidence < 0.9.

| Suite | Jeeves v2 | Jeeves v1 | Jev | Kev-9B |
|---|---|---|---|---|
| Test overall | 0.905 | – | 0.857 | 0.822 |
| JevBench public | 0.896 | 0.913 | 0.866 | – |
| JevBench hard | 0.793 | 0.820 | 0.730 | – |
| MMLU-Pro | 0.770 | – | 0.840 | – |

- The catalog records the README as reporting "JevBench-231 0.935 vs Jev 0.866". That 0.935 does not match either 0.896 (v2) or 0.913 (v1) in the dossier's reading of the same README; the notes do not explain the difference *(inference: possibly a different README revision or setting)*.
- v1 scores higher than v2 on both JevBench rows.

## Serving and efficiency

- **Without thinking:** 22 ms p50.
- **With `nothink_threshold` 0.9:** 1.2 s p50 / 8.2 s p90 (H100, FP8).
- **Diffusion drafter** (Orthrus-style) for Gated DeltaNet: block size 4 gives 176 vs 109 chain tokens/s [@orthrus].

## Flaws and failure modes

**(a) Flaws of Jeeves itself**
- **Confidence gate misfires:** v2 "is overconfident without reasoning on some long-policy and multi-hop items, so it skips reasoning on them."
- **Post-reasoning overconfidence:** requires T = 5.42, a consequence of training only on chains that reach the right answer.
- **Latency tail:** reasoning lifts p50 from 22 ms to 1.2 s and p90 to 8.2 s.
- **Knowledge gap:** MMLU-Pro 0.770 vs Jev 0.840.
- **RL did not help v2:** "CISPO did not improve v2"; v1 (with CISPO) beats v2 on JevBench public and hard.
- **Reproducibility:** the SFT reasoning chains are unpublished.

**(b) Flaws of other systems it addresses**
- **One-pass readouts cannot deliberate:** Jeeves adds optional reasoning before the pointer readout and reports higher JevBench public/hard than Jev (0.896 / 0.793 vs 0.866 / 0.730).

## Implications for our design

All bullets are *(inference)*.

- **Adopt:** if we add a reasoning slow path, **repeat the question and options after the trace** before reading out; the README reports that omitting the repeat hurt.
- **Adopt:** a **separate temperature for post-reasoning answers**; the decoder-family avoid list names "reasoning escalation without its own post-reasoning temperature" explicitly.
- **Avoid:** gating escalation on raw confidence alone when the direct path is overconfident on known item types (long policies, multi-hop); the gate silently skips the items that need reasoning.
- **Avoid:** training only on correct reasoning chains without correcting the resulting confidence.

## References

- [@jeeves] https://github.com/PostHog/jeeves — [trainer.py](https://github.com/PostHog/jeeves/blob/master/trainer.py), [calibrate.py](https://github.com/PostHog/jeeves/blob/master/calibrate.py); weights https://huggingface.co/PostHog/jeeves
- [@kev-repo] https://github.com/jaredpalmer/kev
- [@orthrus] (Orthrus, arXiv 2605.12825, cited by Jeeves)
