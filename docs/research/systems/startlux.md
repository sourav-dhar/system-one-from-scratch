# StartLux-Decision

> **Family:** decoder-readout · **Team:** Zhuang, Su, Pan, Wong, Chen and Kuo (StartLux; HKU; SUSTech) · **Released:** arXiv 2610.17309; the catalog dates it 2026-10-12, and the decoder-family dossier read v2 of 20 Oct 2026 · **License:** code Apache-2.0; **weights CC BY-NC 4.0** (catalog); paper under the arXiv non-exclusive licence · **Code/weights:** [repo](https://github.com/StartLuxLabs/StartLux-Decision) is **inference only** (vLLM plugin, llama.cpp bidirectional patches, MLX); weights 0.8B–27B in Instant and AutoThink variants (the catalog also lists 35B-A3B); training code not public; blog at [decision.startlux.com](https://decision.startlux.com/) · **Sources read:** arXiv HTML v2 (§3, §5.2, §6, Eqs. 1 and 3–6, Table 4, Fig. 4, App. D.3), repo README, `docs/none_probability.md`. Not found: RLCD² hyperparameters (G, ε, β, LR, batch) and the SFT/LoRA hyperparameters.

## Summary

StartLux-Decision is a family of letter-logit decision models on a hybrid linear-plus-full-attention backbone, shipped in two modes ([arXiv](https://arxiv.org/html/2610.17309)) [@startlux] [@startlux-repo]. **Instant** makes some full-attention layers bidirectional and reads A–Z letter logits in one pass; **AutoThink** stays causal and, when the first-pass confidence is below a gate, reasons for up to 4,096 tokens and fuses the post-reasoning distribution with the direct one. The idea that matters most is **RLCD²**: an RL reward defined as the *improvement* in a decision-weighted proper score from reasoning, minus a length penalty, trained only on items the direct pass is unsure about. StartLux's 27B models top its own Table 4 on DI 0.3 public (73.37 / 72.83), and the paper quantifies the serving cost of its bidirectional mask: reusing document representations would raise 27B throughput from 2.8 to 27 decisions/s, and "bidirectional attention prevents that reuse."

## Architecture

**Backbone and mask** (paper §3).
- Hybrid linear-plus-full-attention backbone. At 4B and 9B, 8 of 32 layers are made bidirectional.
- **Instant** is bidirectional (in those layers); **AutoThink** is causal.

**Readout.** Letter logits A–Z through the LM head; Eq. 1 is a temperature softmax:

$$p_k = \frac{\exp(z_k / T)}{\sum_{j=1}^{K} \exp(z_j / T)}, \qquad z_k = \text{LM-head logit of letter } k.$$

**Passes and large K.** One prompt per question. Option sets larger than the letter range are handled as a tournament: groups of ≤25 options, with the top 3 of each group advancing.

**AutoThink.**
- If $\max_k p^{(0)}_k \ge 0.9$ (paper), return $p^{(0)}$.
- Otherwise reason (≤4,096 tokens), read $p^{(1)}$ and fuse: $p = 0.25\,p^{(0)} + 0.75\,p^{(1)}$; $\alpha$ was chosen by held-out Brier (§5.2).
- The repo's serving default gate is **0.6**, not 0.9 ([README](https://github.com/StartLuxLabs/StartLux-Decision)).

| Property | Value |
|---|---|
| Passes per request | One per question (Instant); plus up to 4,096 reasoning tokens and a second readout when the AutoThink gate fails; plus tournament rounds when K exceeds a group |
| State prefix reusable | Prevented by the bidirectional mask in Instant (paper §6); prefix caching changed 2.6% of readouts by > 0.05 (App. D.3) |
| Option embeddings reusable | No (letters in a joint prompt) |
| Max K | 26 letters per pass; groups of ≤25 with top-3 advancing beyond that (151 options evaluated) |
| Context | Not stated in the notes |

## Training

**SFT.**
- Stage 1: full SFT with soft cross-entropy, $\mathcal{L} = -\sum_k t_k \log p_k$, on ~2.7M examples (0.8–9B) or ~0.8M (27B), with hash-based option permutation.
- Stage 2: adapters on ~1.7M examples, then a 3–4-seed average.
- Stage 3: a reasoning adapter on 4k–21k examples.
- Teachers are unnamed.

**RLCD²** (Eqs. 3–6).
- Reward:
$$r = S(p^{(1)}, t) - S(p^{(0)}, t) - \lambda \max\big(0, |\tau| - B\big), \qquad \lambda = 10^{-4},\; B = 3{,}072,$$
  where $|\tau|$ is the reasoning-trace length *(inference from the penalty form; B is in tokens by the same reading)*.
- $S$ is a decision-weighted proper score, $s(q, z) = z\,W(q) - C(q)$, with weighting distribution $w = \tfrac12\,\mathrm{Uniform} + \tfrac12\,\mathrm{Beta}(8, 2)$.
- GRPO-style: group-centred advantages clipped to $[-5, 5]$, **not** std-normalised, plus a KL penalty.
- Trained only on items with $\max p^{(0)} < 0.9$.
- G, ε, β and LR are **not reported**.

## Calibration

- One temperature per question type; "all fitted temperatures exceed 1" (overconfident).
- DecisionPivot calibration error, AutoThink / Instant: 4B 2.4 / 7.3; 27B 5.2 / 5.1.
- Post-reasoning answers are overconfident: post-reasoning calibration error 8.1 vs 3.7 by default (decoder-family cross-table).
- **None-of-the-above signal** ([docs/none_probability.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/none_probability.md)): an untrained, appended "None of the above" letter gives AUROC 0.920 for detecting that no option is correct; a threshold of 0.2 flags 66.3% at 2.2% false alarms.

## Results

**Table 4, DI 0.3 public.**

| System | DI 0.3 |
|---|---|
| StartLux 27B AutoThink | 73.37 |
| StartLux 27B Instant | 72.83 |
| Perplexity Decider v1.1 | 62.25 |
| Jev | 57.96 |
| Liquid d1-3B | 48.99 |
| Decision 2.0 Sol | 30.05 |

**Other results.**
- DecisionPivot pair accuracy: 27B Instant 75.4, AutoThink 86.9, **Jev 31.0**.
- Bidirectional attention: up to +2.4 DI points at 4 of 5 scales (Fig. 4).
- Large K: 151-option intent 86.7% vs 95.7% at 26 options; tournament "elimination 0.2–0.7%" (decoder-family cross-table; the notes do not define the rate further).
- RL hurts long computations: −5.9 points on 2,048–4,095-token traces.
- Self-reported Decision Index 63.88 vs Jev 57.91 (catalog, from the repo). This differs from Table 4 (73.37 / 57.96); the notes do not say which DI version or protocol the 63.88 refers to.
- **Reported by Jeff:** StartLux-27B Instant scores 88.5 OpenDecisions Capability, vs Jeff-1.1-Large 89.0 and Jev 80.1 ([Jeff](https://arxiv.org/html/2610.17205)) [@jeff].

## Serving and efficiency

- H200 / vLLM, Instant: 27B 51.9 ms median; 4B 27.4 ms (§6).
- "Prefix caching changed 2.6% of readouts by more than 0.05" (App. D.3).
- On a 3,200-token document with 32 questions, 27B throughput goes from 2.8 to 27 decisions/s when document representations are reused. "Bidirectional attention prevents that reuse." (§6)
- Inference stack: a vLLM plugin, llama.cpp bidirectional patches, and MLX ([repo](https://github.com/StartLuxLabs/StartLux-Decision)).

## Flaws and failure modes

**(a) Flaws of StartLux itself**
- **No state reuse in Instant:** the bidirectional mask prevents document reuse, forgoing the 2.8 → 27 decisions/s gain (about 9.6×, our arithmetic) on multi-question documents; even with caching, 2.6% of readouts move by > 0.05.
- **Large K degrades:** 151-option intent 86.7% vs 95.7% at 26 options.
- **RL hurts long reasoning:** −5.9 points on 2,048–4,095-token traces.
- **Overconfidence:** all fitted temperatures > 1; post-reasoning error 8.1 vs 3.7; Instant 4B DecisionPivot calibration error 7.3.
- **Gate inconsistency:** paper gate 0.9 vs repo serving default 0.6.
- **Reproducibility and licence:** training code not public; RL and SFT hyperparameters unreported; teachers unnamed; weights CC BY-NC 4.0.

**(b) Flaws of other systems StartLux claims to expose or fix**
- **Jev "systematic overconfidence":** StartLux claims Jev has a 13.8 pp calibration error on DecisionPivot and 31.0 pair accuracy. **This conflicts with TypedBench**, which finds Jev *underconfident* (fitted T = 0.66 and 0.21) on its suites ([TypedBench](https://arxiv.org/html/2610.11392)) [@typedbench], and with Archer Hume's MMLU ECE of 0.0313 ([Hume](https://archerhume.com/posts/jevs-architecture-unmasked/)) [@jev-hume]. The suites differ, so the claims need not be contradictory, but they are not reconciled in the notes.
- **Missing "none fits" outcome:** an appended, untrained "None of the above" letter yields AUROC 0.920.
- **One-pass readouts that cannot deliberate:** AutoThink plus RLCD² rewards only the improvement over the direct answer.

## Implications for our design

All bullets are *(inference)*.

**Adopt**
- **Reward reasoning by improvement over the direct answer** (score difference, not raw correctness), and train RL only on uncertain items; this aligns the slow path with what the fast path lacks.
- **Fuse rather than replace** the direct and post-reasoning distributions, with the mixing weight chosen by held-out Brier.
- **A cheap "none fits" signal** via an appended untrained letter as a baseline (AUROC 0.920) before training a dedicated head.
- **Separate calibration for post-reasoning outputs** (8.1 vs 3.7 error).

**Avoid**
- **Full-prompt bidirectional attention** where many questions share a document: the reported gain (≤ +2.4 DI) is small relative to the 2.8 → 27 decisions/s reuse it forfeits. Strands' `prefix_bidir` keeps bidirectional state encoding *with* reuse [@strands].
- **Letter tournaments for large K** without measuring the elimination rate (reported 0.2–0.7%) and the accuracy drop at 151 options.
- **Gates whose serving default diverges from the paper** (0.6 vs 0.9); report the shipped gate.

## References

- [@startlux] https://arxiv.org/html/2610.17309 (abs: https://arxiv.org/abs/2610.17309)
- [@startlux-repo] https://github.com/StartLuxLabs/StartLux-Decision — [docs/none_probability.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/none_probability.md)
- Blog: https://decision.startlux.com/
- [@jeff] https://arxiv.org/html/2610.17205
- [@typedbench] https://arxiv.org/html/2610.11392
- [@jev-hume] https://archerhume.com/posts/jevs-architecture-unmasked/
- [@strands] https://github.com/strands-labs/strands-decider
