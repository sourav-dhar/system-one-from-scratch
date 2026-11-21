I identified all three names: Kev and Laya are System One models, but Inkling is not. The finding that matters most is that CLM-8B, which we're rebuilding, does badly zero-shot on typed decisions. The report is written; most of its numbers come from 2026 preprints or vendor posts, and many are self-reported.

**The three names**
- **"Kev" is a real model family, not only a typo for Jev.** Jared Palmer's Kev (0.8B, 4B, 9B, 27B; Apache-2.0; Kev-4B published 2026-09-24) is an open copy of the Jev idea. It is a LoRA r16 adapter plus a pointer head on frozen Qwen3.5 and uses TypeSafe's `/v1/systemone` API. Kev-4B runs in 18 ms on an H100 (13 ms with the state cached) and scores 0.838 out-of-domain. It trails Jev on harder sets (MMLU-Pro 0.565 vs 0.840). A small unrelated project, `arjun988/Kev`, only wraps an existing LLM. Jev itself (TypeSafe, closed, $0.042 per million input tokens) has no published architecture. Third-party probing suggests the state is processed once and options influence each other, unlike CLM.
- **"Laya" is Convai Innovations' open decision model:** ModernBERT-large, 421M parameters, Apache-2.0. It scores each option at its own `[MASK]` token, is trained with proper scoring rules, and runs in about 33–40 ms on a T4. Zero-shot it is near chance (0.362); only after fine-tuning does it beat Jev (0.766 vs 0.727).
- **"Inkling" is Thinking Machines Lab's open generative model** (975B total, 41B active, released 2026-07-15), plus Inkling-Small. I found no decision, verifier or reward version. It is only useful to us as a teacher for generating labels.

**Other related systems**
- vLLM Semantic Router's Decision 2.0 (0.6B to 27B, Apache). Vega-27B is the best System One model on the System1-Arena benchmark at 73.0%.
- Microsoft-Decision-1 (hosted only, built on Qwen3.5-9B).
- OpenAI Decisions API (GA 2026-10-22, $0.10 per million input tokens, up to 400 questions per request).
- Jeff-1.1 / "Releasing the Inner Jev": an open LLM read through option-letter scores matches Jev with no training; adding LoRA tuning beats it by 8.9 points. It comes with a 261k-item training set (DecisionBank) and a 2,352-item eval set (OpenDecisions).
- Jeva, Bongard and SanSi (research models).

**What this means for us**
- On System1-Arena, CLM-v0.1-8B is the fastest model (41 ms) but the least accurate: 41.4% with a Brier score of 0.749. Plain answer-letter readout from its own Qwen3-8B base beats it by 17.8 points.
- It scores about zero on one JevBench leaderboard (composite 1.51, ranked 77th).
- It does win where large action sets are cached and reused: verifiers and games, at 4–9x Jev's speed.

**Recommendations, in priority order**
1. Expose our model through the `/v1/systemone` API so Jev, Kev, Laya, Decision 2.0 and the hosted APIs become drop-in baselines.
2. Add a zero-shot Qwen3-8B letter-readout baseline that we must beat.
3. Run System1-Arena and OpenDecisions without CLM's 2,048-token truncation.
4. Calibrate with a temperature per question type and option count; Laya's calibration error fell from 0.466 to 0.081 this way.
5. Add a Kev-style pointer head, or a Jeva-style decision layer over cached option embeddings with a "none fits" output.
6. Sweep serving options: backbone size, using an earlier layer's output, FP8 backbone, int8 option cache, shorter head outputs.
7. Measure flip rates when options are reordered; our design should be immune to reordering, which is a selling point.

**Not found:** a released CLM-35B; open weights for Microsoft-Decision-1; head and training details for Decision 2.0; Jev's architecture. CLM's README says each head is about 20M parameters, while our notes say 9.44M, which is worth reconciling.

The report is at C:\Users\dhars\AppData\Local\Temp\claude\e3bad9d5fd4b315f\777241e1-0588-4c74-940a-4d72bf5420d9\scratchpad\research\landscape.md. It has the comparison table, a section per system, serving notes, and the prioritized plan with effort estimates.
