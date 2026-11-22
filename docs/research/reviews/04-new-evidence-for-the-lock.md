> **New evidence for the architecture lock: eight papers from the library sweep, read in full.** Commissioned on 2026-11-22 after the three reviews, to test the open questions behind the recommended design (per-option vs joint scoring, frozen vs adapted backbones, exact state reuse on hybrid backbones). Published verbatim; *(inference)* marks the reader's interpretation. Response: [`../review-response.md`](../review-response.md).

I read all eight papers in full. B modified's two halves are separable. Reusing the state's KV cache is proven practical. Caching options independently of the state is the untested part, and it is where the accuracy risk sits.

The local copies hadn't arrived, so I downloaded the PDFs from arXiv to a local scratch directory (not in the repo), one `dl_<arXiv id>` folder per paper, with text versions in its `txt` subfolder. That folder may be cleared. Tables that came out scrambled were re-extracted and checked against each paper's own text. Licences are from the arXiv abstract pages. No repo files were edited and no git was run.

**1. Joint or Per-Option? (2610.30970)**
- **Identity:** 22 Oct 2026; T. R. C. Stekel, Federal Institute of São Paulo; arXiv non-exclusive licence; Crivo weights CC BY-NC 4.0.
- **Method:** Crivo joint encoders (17M–1B, Ettin rerankers, [OPT] option markers; options see each other). They are compared with a per-option reader that re-encodes instr⊕state⊕option once per option, i.e. a cross-encoder. Controlled grid: 32M, 3 seeds, 150k questions.
- **Numbers:**
  - Per-option minus compute-matched joint (Table 3): pub15 skill +0.028 [+0.015, +0.042]; unseen S1-60 tasks +0.011 [−0.009, +0.031]; pub15 ECE +0.063 worse.
  - From the raw (non-reranker) encoder: −0.011 [−0.038, +0.015] (C1).
  - Per-option needs 5.2–13.1× more sequences. A100 mean 59.0 vs 13.7 ms, median 10.7 vs 10.4 ms (§5.3, Table 7).
  - Reordering options flips 12–23% of joint answers and none of the per-option answers (§6.3).
  - Crivo-1B scores 0.539 on the paper's own 209-task S1-Arena set, which is not the leaderboard's item set (Table 4).
- **Limitations:** one size and one data subset; the compute-matched arm repeats rows; no masked-attention joint arm; English only.
- **Implication *(inference)*:** this "per-option" reader keeps full state–option attention. It shows options need not see each other at 32M. It does not show that cached, state-independent options are free. There is no evidence at ≥1B.

**2. MetaEncoder (2610.11316)**
- **Identity:** v1 8 Oct, v2 6 Nov 2026; Meta AI; CC BY 4.0.
- **Method:** a controlled study of bi-encoder vs cross-encoder vs letter readout ("constrained decoder") at 0.6B–9B on 10M pairs. Then a fully fine-tuned 30B causal bi-encoder: last-token embeddings, cosine score, InfoNCE, the option list inside the request, candidates cached.
- **Numbers:**
  - The cross-encoder is best at every size. The bi-encoder beats the letter readout at 0.6B/1.3B and converges with it above 1.3B (§3, Fig. 3; the gaps are only plotted, with no numbers in the text).
  - Dropping the option list from the request: −0.197 MMLU, −0.189 MMLU-Pro (Fig. 5).
  - Bare-label candidates 0.4760 vs label+description 0.8360 (MMLU, v1; Table 3).
  - Permutation s.d. ≤0.003 (§5.4.3).
  - JevBench-hard 0.685 vs 0.766 for a 27B cross-encoder (Table 2).
- **Limitations:** a reasoning ceiling compared with chain of thought (§6). MMMU rises from 0.57 to 0.71 when CoT traces are added (Fig. 7).
- **Implication *(inference)*:** this is the nearest relative of B modified's option side.
  - Option texts belong in the question tokens, so h_Q is option-aware.
  - Cached option encodings should hold label plus description; short labels (yes/no, score levels) are the weak case.
  - The late-interaction block has to recover the cross-encoder gap.

**3. Frozen-LM MedMCQA (2611.23832)**
- **Identity:** v1 18 Nov 2026; P. K. Mantha, Antz.ai; CC BY 4.0; code not yet released.
- **Method:** a rebuild of the CLM recipe. Frozen Qwen3.5-0.8B (a Gated DeltaNet hybrid), Granite-4.2-3B and Qwen3-8B encode the question and each option separately (last token, one-word prompt). Two heads plus a temperature are trained with InfoNCE, hard negatives and a mask for repeated answers.
- **Numbers (3,842 test questions, chance 25.04):**
  - Plain vectors: 33.13 ± 0.93 (3B), 33.00 ± 0.42 (0.8B).
  - Final recipe: 41.74 ± 0.76 (0.8B), 42.48 ± 0.49 (3B), 45.99 ± 0.20 (8B) (Table 3).
  - Negated questions 34.87 vs 48.74 for the rest (8B).
  - Validation gains were 1.7–2× the test gains (§4.2).
- **Limitations:** single-seed ablations; depth confounded with learning rate; public test set; one dataset.
- **Implication *(inference)*:**
  - A frozen, head-only bi-encoder is weak on knowledge-heavy choices. The authors blame never seeing the question and options together, and propose LoRA plus light interaction (§5).
  - There is no letter-readout baseline, so the paper cannot answer Q2 against our bar.
  - Practical tip for GDN hybrids: right-pad and read the last non-pad token (§3.3).

**4. GLiDE (2611.26509)**
- **Identity:** 19 Nov 2026; Fastino Labs; CC BY 4.0; weights licence not stated.
- **Method:** a Qwen3.8-27B hybrid (48 GDN + 16 full-attention layers) with the causal mask removed in the full-attention layers. Context, question and coded options form one sequence, read by a 255-way readout initialised from LM-head rows. Training uses soft-target distillation from a reasoning teacher and others, plus weight averaging and parameter-free recurrent depth.
- **Numbers:**
  - Decision Index 0.4.1: Full 71.52, ECE 2.0%, 92.9 ms (Table 1).
  - Family (Table 8): 27B 73.37 at 122 ms; 9B 62.48 at 86 ms; 2B 43.72 at 37 ms; 0.8B 36.36 at 37 ms.
  - Single-seed ablation (Table 3), described as "within single-seed variability":

    | Variant | Dev | Acc |
    |---|---|---|
    | causal + gold labels | 60.42 | 87.01% |
    | non-causal + gold | 61.78 | 88.38% |
    | non-causal + soft targets | 62.59 | 89.16% |

  - Drex Max (same backbone and readout, different training) scores 4.92 lower.
- **Limitations (no dedicated section):** one benchmark; option-order equivariance not guaranteed; context re-encoded for every question (App. C.3).
- **Implication *(inference)*:** the bidirectional gain (~1.4 points, one seed) is small next to the training recipe. Staying causal, which state reuse requires, looks cheap. Scores drop steeply below 9B.

**5. Decision 1.0 (2611.18629)**
- **Identity:** submitted 22 Sep 2026; vLLM Semantic Router Project, AMD, Microsoft, MBZUAI, McGill; arXiv licence; artifacts Apache 2.0.
- **Method:** causal Qwen3.5 hybrids (3 GDN : 1 full attention) over state → question → candidate blocks → suffix. Each candidate's endpoint h_i is paired with the suffix query h_q by a bilinear-plus-MLP head (Eq. 3). Backbones are adapted on ~24k-example mixtures (Eos: head warm-up, then full model); temperature is fitted afterwards.
- **Numbers (Table 2):**
  - Lux-9B Overall 77.22 vs 69.73 for untuned Qwen3.5-9B answer-letter logits. Decisions 84.17 vs 73.91; Reading 89.69 vs 89.84.
  - Nox-4B 73.09 vs 67.29; Sol-2B 66.32 vs 57.24; Jev 81.05.
  - Lux ECE 5.56% vs Jev 3.35% (Table 3).
- **Limitations:** benchmark seen during development; Overall weights chosen after the results; no order invariance; differing recipes confound attribution.
- **Quoted:** the implementation "does not reuse state activations across questions; eliminating output-token generation alone does not establish a latency advantage over a one-token generative classifier" (§3.2). §7 proposes reusing KV and GDN states.
- **Implication *(inference)*:** Eq. 3 is B modified's late interaction computed in context. Keeping the head and swapping in-context endpoints for cached option encodings would isolate Q1's cost.

**6. Flide (2610.42148)**
- **Identity:** 30 Oct 2026; NUS and Oxford; arXiv licence; code github.com/OmniJev/flide.
- **Method:**
  - Each batch round runs as one flat token stream.
  - ForkAttention lets each question see only the shared state and itself.
  - GDN layers run in two calls, so each question resumes from the state's exact recurrent and convolution state.
  - Also: CUDA-graph replay, incremental checkpoints and a batch-invariant mode.
- **Numbers:**
  - Up to 9.5× faster than vLLM/SGLang (abstract).
  - Qwen3.5-0.8B, all QuALITY questions: 37.7 ms vs 362 ms for vLLM (Table 3).
  - Without forks on 4B, all QuALITY questions: 135.6 → 1868.5 ms (Table 5).
  - Mean total-variation distance from fp32 is 0.30–0.55, vs 3.64–6.69 for delimiter-based multi-item scoring. That method flips up to 28.9% of answers vs ≤2.3% for Flide (Table 11).
  - The bitwise-deterministic mode costs ≤10% throughput at 64 clients (App. E).
- **Limitations (no dedicated section):** exactness is proven only when the state never reads the question (Prop. 1). Models that read the whole prompt bidirectionally run one sequence per question (App. H).
- **JevCascade / prefill-only:** neither is measured; PrefillOnly is only cited.
- **Implication *(inference)*:** Q3 is answered yes for Qwen3.5-class GDN hybrids, with open code. GLiDE-style unmasking gives this up.

**7. Prism-1 (2610.22066)**
- **Identity:** v2 16 Oct 2026; Nanyang Technological University; CC BY 4.0.
- **Method:** the state⊕question context is computed once. Each option is a causal block at a shared position that never sees other options. Its score is log p(yes) − log p(no) at the end of the block, with a softmax over blocks. LoRA fine-tune.
- **Numbers:**
  - Accuracy vs a trained letter readout (SCQ): 79.5 vs 80.4 on Gemma-4-12B, 78.0 vs 77.9 on Qwen3-8B (Table 1).
  - One shuffle flips 7.1%/7.4% of SCQ answers and none of Prism-1's.
  - Held-out JevBench: −3.6/−4.1 points (§7).
  - Letting options attend to each other: 74.4/74.8 vs 79.6/77.7 (Table 2).
  - A comparison head over all final option states adds +0.9/+0.7 (Table 4).
  - 58 vs 60 ms (App. Table 5).
- **Limitations:** needs fine-tuning (28.5%/32.5% untrained); cannot express "none of the above"; ≤20 options; 1–3 seeds.
- **Implication *(inference)*:** the best LLM-scale evidence for Q1. Scoring each option independently costs ≤0.9 points on average, but 3.6–4.1 on held-out JevBench. Its options still read the state, unlike B modified's. The comparison-head gain hints that a late-interaction block helps.

**8. Hitchhiker's Guide (2610.33775) vs our survey**
- **Identity:** v2 29 Oct 2026; Haggai Roitman, no affiliation stated; CC BY-SA 4.0.
- **Scope:**
  - Theirs is a broad teaching guide: reward models, BERT/ColBERT, extraction, 19 application areas, System 1/System 2 cascades. Its text never mentions KV/prefix caching, Gated DeltaNet, Flide or letter/answer-token readout.
  - Ours covers about 30 post-Jev decision systems at code level, with failure modes, a who-fixed-whom matrix, single-GPU serving and a benchmark protocol.
- **Taxonomy:** theirs has five overlapping families by "historical lineage and typical deployment role" (§3). Ours has eight families defined by scoring equation and cacheability.
- **Conclusions:**
  - On calibration training, theirs calls RLCD/proper scoring "the correct incentive" (§18). Ours finds no controlled evidence that RL beats direct proper-score losses.
  - Theirs treats dual vs cross-encoders qualitatively and calls the 50–200 ms regime "largely unexplored" (§16.1). Ours quantifies the gap (41.4% vs 59.3%) and proposes late interaction.
- **Reliability:** it calls System1-Arena (2610.21244) a "500+ scenarios" meta-decision benchmark and lists Prism-1 as "8 tasks". Prism-1 itself reports five sets, and 2610.30970 describes the same arXiv id as a typed-decision benchmark from 268 public sources.

**Synthesis**

*For B modified first:*
- Flide: exact state forking on GDN hybrids works, with open code (Decision 1.0 listed it as future work).
- Prism-1: at 8–12B, options need not see each other (≤0.9 points).
- GLiDE: staying causal costs about 1.4 points (one seed).
- MetaEncoder: cached candidates match the letter readout above 1.3B when the options are in the request; cross-encoders still lead.

*Against, or for another first step:*
- No paper tests state-independent cached options plus a late block. The per-option systems that match joint scoring let options read the state (Prism-1, Crivo P). The state-independent ones trail (MetaEncoder; frozen MedMCQA at 46.0%).
- Q2 is untested: no paper compares a frozen-backbone head with that backbone's own letter readout. The strong systems adapt the backbone (GLiDE, Decision 1.0, Prism-1 via LoRA), and the training recipe dominates (GLiDE vs Drex +4.92).
- Untrained, the letter readout beats per-option reading (Prism-1 §7, Gemma: 75.9 vs 69.9).

*(inference)* State reuse and option caching can be separated. A causal joint readout with the Decision 1.0 layout already forks at the state, so only option caching carries the accuracy risk. That suggests:
- **M0:** the letter readout plus a trained in-context Eq. 3 head on a forked state.
- **M1:** the same head over cached options (B modified).
- Prism-1-style blocks as the middle arm.

*Experiments by paper:*
1. Crivo: reader × (frozen vs LoRA), compute-matched, 3 seeds, unseen tasks.
2. MetaEncoder: option list in h_Q on/off; label vs label+description.
3. MedMCQA: frozen head vs letter readout of the same backbone; negation slice.
4. GLiDE: causal vs prefix-bidirectional; teacher soft targets.
5. Decision 1.0: the same Eq. 3 head on in-context vs cached option encodings.
6. Flide: forked vs standalone logits (TV, flips) on Qwen3.5; latency vs questions per state.
7. Prism-1: the middle arm; shuffle and add-option tests.
8. Hitchhiker: none.
