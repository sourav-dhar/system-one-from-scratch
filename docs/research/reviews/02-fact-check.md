> **Primary-source fact-check of 40 survey claims.** Independent review commissioned on 2026-11-22 before locking the architecture. Reviewers received the documents, not the author's conclusions. Published verbatim; how each finding was addressed is tracked in [`../review-response.md`](../review-response.md).

## Fact-check of `C:\system-one-from-scratch\docs\research\survey.md`

I checked 40 claims against primary sources and left out everything already listed in the survey's Appendix A:
- **Papers:** the local arXiv PDFs in `research\papers\pdf\`, extracted with pdftotext.
- **GitHub:** read-only `gh api` GET calls. Kev's research log came from tag `research-archive-2026-09-24`.
- **Hugging Face:** model and dataset cards fetched with plain GET requests.

No project files were changed and no git commands were run. Extracted texts and downloads are in `C:\Users\dhars\AppData\Local\Temp\claude\e3bad9d5fd4b315f\777241e1-0588-4c74-940a-4d72bf5420d9\scratchpad\` (`txt\`, `dl\`).

| # | Survey location (short quote) | Claim | Primary source checked | Verdict | Correct value / note |
|---|---|---|---|---|---|
| 1 | Table 9: "78,362 non-agentic questions (87 tasks, 13 domains); 3,481 agentic steps; 138 closed-loop environments" | System1-Arena (S1A) composition | arxiv.org/abs/2610.21244 §3.2–§4 | VERIFIED | Exact. The full benchmark is 92,125 questions, 105 tasks, 16 domains. Loop success is averaged over 127 of the 138 environments (App. B). |
| 2 | Table 9: "CLM queried via its official library keeping the last 2,048 tokens" | How CLM was run in S1A | 2610.21244 App. B ("Inputs", "Services") | VERIFIED | Source: "for CLM we keep the last 2,048 tokens of each text". The same paragraph says the Laya library cuts inputs to 512 tokens. |
| 3 | Table 11 rows: Kai-0.6B 72 / 50.3 / 0.537; Laya 136 / 44.5 / 0.660; OpenAI Decisions 140 / 68.6 / 0.424; MS-Decision-1 304 / 70.9 / 0.369 | p50, accuracy, Brier | 2610.21244 Table 1 | VERIFIED | All 12 values exact. |
| 4 | Table 11: Vega 228 ms, Jev 321 ms; caption "Local models ran on an idle H100. Hosted models include network time." | Latency and setup | Table 1; App. B | VERIFIED | Exact. The H100 is for non-agentic runs; agentic runs used an A100. |
| 5 | FM10: "Vega, Jev and Qwen3.5-9B share 67.8% … 26.5%, 17.4% and 5.7% loop success" | Step accuracy vs loop success | Table 2; §5 | VERIFIED | Exact. |
| 6 | FM10: "CLM reaches 40.6% step accuracy and 1.7% loop success" | CLM agentic results | Table 2 | VERIFIED | Exact. |
| 7 | §9.2: "empty region is the upper left: above 65% accuracy below 100 ms. No published system occupies it on S1-Arena" | Frontier gap | 2610.21244 Table 1 (open-LLM block) | MISMATCH | Four rows sit there: Gemma-4-31B 73.9% at 60 ms, Gemma-4-26B-A4B 69.9% at 47 ms, DiffusionGemma 69.7% at 68 ms, Qwen3.5-9B 65.3% at 77 ms. All are answer-token readouts, which the survey's own F8 family covers, and all were left out of Table 11 and Fig. 2. The claim is only true for the nine System-1-native rows. |
| 8 | Abstract: "weakest system … is also the fastest"; §1.1 "fastest, at 41 ms" | CLM speed | Table 1 | MINOR MISMATCH | True only among System-1 rows. Llama-3.1-8B (21 ms), Qwen3-0.6B (27), Gemma-4-E4B (29) and Qwen3-8B (30) are faster. Qwen3-8B, CLM's own frozen backbone read at the answer token, is both faster (30 vs 41 ms) and more accurate (59.3 vs 41.4). |
| 9 | §3.2: "L=1, d=768, 8 heads, SwiGLU 3,072; 13.37M trainable … with the projections" | Jeva decision block | arxiv.org/abs/2610.18773 §3; App. B.7 | VERIFIED | 13,370,117 parameters. Table 10's "≈9.4M" for the block alone is consistent (13.37M minus 2×2560×768). |
| 10 | §3.2: "zero order flips (Jev 17.0%) and null-detection AUROC 98.66 [Tables 1–2]"; §4.4 "vs Kev's 93.61" | Jeva order stability and null detection | Table 2; §4.2 | VERIFIED | Values exact. The AUROC is in §4.2 / Fig. 6b, not Tables 1–2 (citation slip). |
| 11 | FM3: "CLM … only 7 of 83 queries, vs Jeva's 58" | Nested candidate sets | §4.3; Fig. 7 | VERIFIED | Exact (JevK5: 48). |
| 12 | §9.1: "2,102 ms to 75.1 ms on MLDR (28×) … 99.2% … 93.1 → 46.6 ms … 0.007" | Jeva cache speedup | §4.4; App. C.5 | VERIFIED | 27.98× at p50 (median paired 27.86×); maximum probability difference 0.00736. |
| 13 | §3.3: "Jeff-1.1-Large 89.0 … vs. Jev 1.13 at 80.1 [Table 3]" | OpenDecisions Capability | arxiv.org/abs/2610.17205 Table 3; §5.2 | VERIFIED | Exact. |
| 14 | Table 5: "Large: 1.49/1.57/1.11"; "72.9 to 79.6"; §5.1 "Gemma-4 needs 2.7–6.0 … pretrained close to calibrated [App. I]" | Temperatures | Tables 17, 18; App. I | VERIFIED | Exact. The 2.7–6.0 range is with the plain prompt. With the chat prompt Jeff actually uses, zero-shot Gemma-4-31B fits 5.96 / 7.24 / 7.24 (Table 19). |
| 15 | Table 7: DecisionBank 163,732 / 9,168 / 60,219 / 28,442; "50/50 label + teacher; exact for programmatic" | Training data | Jeff Table 2 | VERIFIED | Exact; total 261,561. |
| 16 | FM1: "Exact fixes in joint models cost accuracy … Jeff −0.6 to −2.4"; H4: "Joint decoders pay 0.6–5.8 pp" | Cost of order invariance | Jeff App. G | MISMATCH (context) | Numbers exact, but App. G says "with one run per variant these differences are within noise". It also says sorting options by text "gives the same invariance at no cost" (+1.0 / −0.4). The source contradicts the inference that exact invariance must cost accuracy. |
| 17 | §3.3 mask table: "Full bidirectional … Jeff (Qwen variants) … −0.2 ± 1.1" | Which Jeff models are bidirectional | Jeff §4 "Training"; §5.4 | MINOR MISMATCH | −0.2 ± 1.1 is exact. Bidirectional attention is used for every model except Gemma-4-31B, so Gemma-4-12B (Jeff-1.1-Medium) is included too, not only Qwen variants. |
| 18 | Mask table: "Costs 5.8 pp at 4B; flip rate 0.0 at 0.6B with no accuracy gain" | Kev option isolation | github.com/jaredpalmer/kev PLAN.md @ research-archive-2026-09-24 | VERIFIED | Source: "option isolation at low lr 0.729 (−5.8 pp, significant)"; at 0.6B "flip rate of exactly 0.0", transfer at parity. These are earlier Qwen3-generation runs. |
| 19 | Table 2/5: Kev-4B "LoRA r16 (33.8M) … 0.838; breadth 0.690 vs Jev 0.757"; "T 2.41 … ECE 0.017 / 0.084 / 0.101" | Kev-4B headline numbers | kev `docs/model-cards/kev-4b.md` | VERIFIED | All exact (test / locked partitions). |
| 20 | Table 7: Kev-27B "145,840 / 337,130 … 8-gram Jaccard decontamination removed 6,388" | Kev-27B corpus | kev `docs/model-cards/kev-27b.md` | MINOR MISMATCH | Counts and teacher list exact. The 6,388 records were removed by exact match, OR 8-gram Jaccard > 0.2, OR containment ≥ 0.5, against 102 evaluation partitions. Not 8-gram Jaccard alone. |
| 21 | FM6 "22–43 pp … recovered 17–20 pp"; Table 10 #10 lists "+17–20 pp" as fix evidence | Long-context burying | kev PLAN.md, "What we have learned" #11 | MINOR MISMATCH (context) | Numbers exact, but Kev's own heading is "Buried synthetic states are not real documents". On real CFPB documents, the long-state training moved documents-v1 by only ±1 pp. |
| 22 | §3.4: "capped at 48 tokens … 192-token head budget (256 multilingual) … max(4,(192−16)/K)" | Laya option budget | github.com/NandhaKishorM/laya `laya/common.py` `build_head`; README | VERIFIED | Code matches. The 256 budget also applies to laya-typed-decisions. The budget includes the question text, and the cut triggers when fewer than 16 tokens remain. |
| 23 | §4.3: "Laya's own three-arm ablation … (issue #741)" plus its table; Table 10 #14 "soft-CE best accuracy and Brier" | RL term vs soft-CE | laya issue #741 and its comment on merged PR #1012 | MISMATCH (attribution / replication) | Values exact, but it was a single-seed run by an external user (PengyiZhang), not the maintainers. The merged three-seed follow-up (PR #1012) found the arms tied: 0.7763 vs 0.7762, per-seed −0.0145 / +0.0135 / +0.0015, and says the single-seed soft-CE advantage "did not replicate". |
| 24 | §5.4: "choice:11+ temperature of 0.1006, a 10× sharpener … later clamped to [0.5, 5]" | Laya calibration bug | laya issue #394 | VERIFIED | Shipped value 0.10058…; clamp to [0.5, 5] came from PR #42. "Coin flips" paraphrases right and wrong answers both near 1.0 confidence. |
| 25 | FM6: "falls from 0.85 to 0.40 as evidence moves to the end of a 7,000-token filler" | Laya evidence position | laya `research/README.md` | VERIFIED | 7,000-token row: 0.850 → 0.400. Caveats: laya-multilingual, n = 20 per cell (±0.22), majority-class rate 0.45. At 4,000 tokens the end position is the best cell (0.95). |
| 26 | §3.1: "θ = 4.6132, i.e. λ sits at the cap [issue #15]" | CLM scale | github.com/Contrastive-LM/CLM issue #15; `src/clm/heads.py` | VERIFIED | CLM_v0.1-8B.pt rev e939398d; e^θ = 100.8, clamped to 100. Appendix A's 4.6423 is a different head and also above ln 100. |
| 27 | §1.1: "DeepSWE (31/38 vs 27/38) at 79 ms vs 449 ms" | CLM as verifier | CLM README and `assets/agentic.png` | VERIFIED | 81.6% vs 71.1%, 5.7×, latency measured on an H100. Jev scores below pass@1 (73.7%). |
| 28 | FM6: "Since vLLM 0.13 truncation keeps the head … −0.0006 [#6, #29]"; Table 10 #9 "two lines" | CLM truncation bug | CLM issues #6, #29 | VERIFIED | Exact. #6 measured on an MLX emulation; #29 on real vLLM 0.30.0 gives −0.1169. Both issues are still open. |
| 29 | P6/FM7: "Strands v7 … 94–99%"; §4.7 "60 of 74 yes/no tasks" | Question-ignoring | github.com/strands-labs/strands-decider `research/generations.md` | VERIFIED | Exact. The same docs say every model from v7 to v19 still gives the same answer about 92–95% of the time, so this is not a v7-only flaw. |
| 30 | FM3 "86.7% at 151 vs 95.7% at 26"; FM12 "2.6% of readouts by >0.05"; mask table "2.8 → 27 decisions/s [§6]" | StartLux serving | arxiv.org/abs/2610.17309 App. D.4, Table D.2 | VERIFIED | Exact (4B model; vLLM 0.19.1). The figures are in App. D.4, not §6. |
| 31 | §5.4: decider-12b v2 "Noul T = 0.05 'chosen for JevBench'… AUROC 0.878 / ECE 0.151 vs 0.934 / 0.038" | Leaderboard-tuned temperature | github.com/Mapika/decider README; huggingface.co/Mapika/decider-12b card | VERIFIED | Exact; 92% of answers are exactly 0 or 1 at T = 0.05. |
| 32 | §5.1: "Jev's fitted temperatures … 0.66 and 0.21 on two TypedBench suites" | Jev under-confidence | arxiv.org/abs/2610.11392 Table 8 | MINOR MISMATCH | Both values come from Suite A (tickets.priority 0.66, phish.is_phishing 0.21), not two suites. Jev's other fits are degenerate. |
| 33 | FM2: "50.5 … (AUC 0.89 → 0.31) … 91.3 for Laya. Fine-tuned checkpoints flip 4.2–6.5"; Table 10 #7 | Yes/no swap | arxiv.org/abs/2610.00346 §5.2; abstract | MINOR MISMATCH | 50.5, 0.89 → 0.31 and 91.3 are exact. The 4.2–6.5 range covers only four checkpoints (Kev-9B, Nimble-9B, decider-2b, this-that); fine-tuned Kev-0.8B flips 14.3. The source never ties robustness to "randomized identifiers / polarity twins". |
| 34 | FM1 "Runoff 5.3", "Laya 30.6%"; §3.6 "99% mass (≥3) … 3–4× … 63.3 vs 69.0" | Runoff | arxiv.org/abs/2610.18233 Tables 2–5; §3 | VERIFIED | Exact (τ = 0.99, kmin = 3; 2.9–4.3×). Closed APIs keep 0.9–1.4% flips from API nondeterminism. |
| 35 | FM9: "46.3% … 34.0–51.7% … 0.99 still 7.9% wrong … CLM 34.5%, 0.0, 60.6%"; "no defence evaluated" | Adversarial robustness | arxiv.org/abs/2609.31142 Table 2; §4.5; Limitations | VERIFIED | All exact. |
| 36 | FM8: "73–85% … Jev 0.791 → 0.996 … CLM 0.123; 79.6%" | Reading right, acting wrong | arxiv.org/abs/2610.18579 Table 2; §6 | VERIFIED | Exact (73.1–85.2%). The 85% end is GPT-6.1, the reasoning reference; System One models span 73–78%. |
| 37 | Table 10 #16: fix "next-sub-goal block", evidence "ALFWorld +16.3" | Sub-goal fix | 2610.18579 §7, Fig. 9b | MISMATCH (context) | +16.3 is JEV given the correct (oracle) sub-goal. JEV's own sub-goal answers are right in only 66.8% of states and add 6.4. For Clef and Gemma, their own answers change nothing (−1.2 to +0.3). |
| 38 | §3.2/§4.4/Table 10 #5: "adequacy … recall 0.66 vs 0.25 for an explicit 'none' option" | NDM null handling | arxiv.org/abs/2610.25553 §10.4 | MISMATCH (context) | Recall exact, but false alarms are 0.29 vs 0.025 and AUROC is 0.76 vs 0.79 (explicit option better). The paper presents it as a trade-off, not an improvement. |
| 39 | §5.5: "42.0% of 282 settings … 51.7% … LTT 7.8% … 129 … PCA-reduced Qwen3-8B embeddings competitive, echoing CLM #13" | Risk budgets | arxiv.org/abs/2610.19811 Table 3; PCA-Ridge baseline appendix | MISMATCH | The rates are 105/250, 108/209 and 12/153 (settings with defined risk or a policy); 105/282 would be 37.2%. The baseline is built on qwen/qwen3-embedding-8b, a retrieval-trained embedding model, not Qwen3-8B states. |
| 40 | §3.6: "K = 20 … 0.328 to 0.138 … adaptive stopping uses about 7 of 18 [Table 1]" | AnyJev rotations | arxiv.org/abs/2610.00831 Table 1; §5 / Table 10; §7 / Table 3 | MINOR MISMATCH | 0.328 → 0.138 is exact. "7 of 18" is the serving run (7.3), whose threshold was selected and bounded on the same states. Verified on disjoint data it reads 10.6 of 18. It is also a different (18-option) task, not in Table 1. |

**Counts:** 27 VERIFIED, 7 MINOR MISMATCH, 6 MISMATCH, 0 UNVERIFIABLE. Every source I picked was reachable.

**Systematic problems**
- **Numbers are transcribed accurately.** I found no wrong digits in the 40 claims, nor in about 45 further spot-checks.
- **Context is stripped where the survey argues for fixes.** Table 10's "Evidence" column is where most problems sit. It presents:
  - an oracle-information gain (#37);
  - an in-distribution synthetic gain (#21);
  - a single-seed result that did not replicate (#23);
  - one favourable metric out of a stated trade-off (#38);
  - fix evidence for #6 that the cited source contradicts (#16).
- **Scope is narrowed without saying so.** The S1A frontier and the "fastest" claim silently restrict to System-1-native rows, which hides answer-token readouts (the survey's own F8 family) that occupy the supposedly empty region (#7, #8).
- **Attribution.** A third-party issue result is presented as "Laya's own" (#23).
- **Loose wording.** Denominators and scope words drift: "of 282 settings", "two suites", "about 7 of 18", "8-gram Jaccard".
- **Citation locators are sometimes off.** Jeva's AUROC is in §4.2, StartLux's throughput is in App. D.4, and AnyJev's figure is in Table 3 rather than Table 1.

**The 5 most important corrections**
1. **§9.2, Fig. 2 and Abstract conclusion 1.** The ">65% accuracy, <100 ms" region is not empty: Gemma-4-31B scores 73.9% at 60 ms, and Gemma-4-26B-A4B, DiffusionGemma and Qwen3.5-9B also qualify. CLM is fastest only among System-1 rows; its own backbone read at the answer token is faster (30 ms) and more accurate (59.3%). Add these rows to Table 11, or qualify the claim.
2. **FM1, H4 and Table 10 #6.** Jeff reports its −0.6 to −2.4 pp order-invariance cost as within noise, and shows that sorting options gives exact invariance at no cost. "Joint decoders pay 0.6–5.8 pp" overstates; Kev's −5.8 pp is a single 4B run, with no cost at 0.6B.
3. **§4.3 and Table 10 #14.** Issue #741 was a single-seed, third-party run, and the merged three-seed PR #1012 found soft-CE and RLCD tied. "No gain from RL" still holds; "soft-CE best" and "Laya's own" do not.
4. **Table 10 #16 and #5.** Reading Right's +16.3 requires the correct sub-goal; JEV's own sub-goal answers give +6.4. NDM's adequacy recall of 0.66 comes with 0.29 false alarms and a lower AUROC (0.76 vs 0.79) than the explicit "none" option.
5. **§5.5.** The risk-budget baseline uses Qwen3-Embedding-8B, not Qwen3-8B, which weakens the "echoes CLM #13" link. The 42.0%, 51.7% and 7.8% rates are over 250, 209 and 153 settings, not 282.

**Further spot-checks, all VERIFIED**
- **S1-Arena:** Gemma-4-31B puts ≥0.999 on its chosen option for 87% of questions. Licences are Apache-2.0 for code and CC BY 4.0 for generated requests, and only test splits exist.
- **Jeva and Jeff:**
  - Jeva's curriculum is about 1.13M items, including the 7:3 teacher blend.
  - Jeff App. F gives 48.9 / 57.6 / 76.8 / 82.2 / 86.5; this is accuracy on seven held-out sets, and the paper itself says it "levels off".
  - Jeff's LoRA rank results are flat, and no calibration loss lowers ECE by more than 0.01 after temperature scaling.
- **Kev:**
  - The hard-v1 refit matches (0.137 / 0.073 / 0.047 → 0.067 / 0.034 / 0.041).
  - −11 pp MMLU, −14 pp PAWS, and +4.7 pp from lowering the learning rate all check out.
  - Kev-4B curriculum stages 12,576 / 1,425 / 5,219 / 11,320 (the last is 6,000 + 5,320).
  - Kev-4B MMLU-Pro is 0.565 and it uses 14.3 GB.
  - Latencies 18.1 / 12.9 and 41.5 / 27.7 ms match; the 89.4 / 22.5 ms pair is for 5 questions.
  - Kev-27B uses a 0.85 / 0.15 weight blend, and its breadth index is 52.3 vs 50.2.
- **Laya:**
  - ECE 0.466 → 0.081, Banking77 0.425 on both checkpoints, and T4 latencies 39.5 / 84.5 / 158.6 / 771 ms all match.
  - The RLCD trainer defaults match, and issue #156 confirms the negative label is returned at confidence 1.0.
- **CLM:** issue #15 BoolQ (498/500 "no") and Emotion (460/500 "sadness"); README ablation 52.1 / 69.2 / 62.4 / 68.5 / 56.2; 28 ms for a new state and 0.6–0.7 ms for a revisit; the cache key format.
- **Strands, StartLux and decider:** Strands v2-31B scores 71.5 on Decision Index 0.3 and 92.2 on OpenDecisions; StartLux scores 73.37 on Decision Index 0.3, and all its fitted temperatures are above 1; decider's PPO stage cuts belief excess from 0.473 to 0.219 nats.
- **TypedBench:** Bayes cost −1,220; Laya ladders 3/1/1 for the English checkpoint (2/2/2 for typed-decisions); negation accuracy 0.10–0.28; −23 pp on the phishing paraphrase.
- **Rafe & Das:** large-K drops (Laya 0.852 → 0.020, Jev 0.990 → 0.913), Kev-9B ECE 0.051 → 0.377, out-of-scope acceptance 0.310 vs 0.015, and the this-that cascade (0.927 at 0.43 of cost).
- **NDM:** set head 1.181 → 0.799 NLL; ECE 0.188 → 0.033; state encoding 6.5 / 59 ms.
- **Evidence audit:** 5/26, 1/27, 3/20 and 12/27 all match. The 5/26 counts "ordinary or same-backbone" baselines.
- **Probability axioms:** 0.064 (noise floor 0.013), 1.14, and 159 of 160.
- **Others:**
  - SanSi: +13.5 points at 7.7× compute, and ECE 0.093 → 0.044.
  - Bongard: 2.3% → 56.6% retrieval.
  - sbert2s1: −0.6 (not significant) / −9.4, estimator scale 3.6–14.6×.
  - Small Heads: 0.53M ≈ 27.3M head, and +10.9 held-out from 0.6B to 4B.
