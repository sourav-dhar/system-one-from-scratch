> **Devil's-advocate review of ADR 0001 (architecture B designed for C).** Independent review commissioned on 2026-11-22 before locking the architecture. Reviewers received the documents, not the author's conclusions. Published verbatim; how each finding was addressed is tracked in [`../review-response.md`](../review-response.md).

# Stress test: ADR 0001 (build B, designed for C), ADR 0002 and plan v2

I modified no files and ran no git commands.

**Short names used below** (every file sits under `C:\system-one-from-scratch\`):
- [ADR1] `C:\system-one-from-scratch\docs\adr\0001-architecture-late-interaction-two-path.md`
- [ADR2] `C:\system-one-from-scratch\docs\adr\0002-data-licensing-and-decontamination.md`
- [PLAN] `C:\system-one-from-scratch\docs\research\plan.md`
- [SURVEY] `C:\system-one-from-scratch\docs\research\survey.md`
- [sys/X] means `C:\system-one-from-scratch\docs\research\systems\X.md`. [raw/NN] means `C:\system-one-from-scratch\docs\research\notes\raw\NN-*.md`.
- Code: `C:\system-one-from-scratch\src\systemone\` and `C:\system-one-from-scratch\cloud\modal_app.py`.

**Provenance.** This session I checked these against primary sources:
- the CLM #13 results table, including its unseen-label rows;
- the S1-Arena README, schema and adapter docs;
- the Jeva-4B model card;
- Qwen3-8B's config;
- the L4, A10 and L40S datasheets;
- Modal's pricing.

Everything else about the 2026 systems comes from the repo's notes. Numbers marked *(est.)* are my own estimates. They are there to help set thresholds and should not be quoted as results.

---

## Executive verdict: modify and re-sequence, do not replace

**B belongs in the design, but not as the first bet for W1.** A cached-option fast path is right for W2 and a sound end state: it gives exact permutation equivariance and a coverage gate, and frozen late interaction is a genuinely open research question. The evidence does not support making it the first W1 milestone, for three reasons:
- **The request side misreads NDM.** B encodes state and question in separate towers, citing NDM. But in NDM's prefix-tree mask the question tokens attend to the state. Separate towers drop the question-aware reading that every W1 leader has.
- **The headline motivation is not a fair comparison.** "Jeva 78.30 vs CLM 43.05" compares a trained model with a zero-shot one. A plain letter readout trained on 17,880 items (InternDecision-4B) scores 78.70 on the same suite.
- **Frozen late interaction has no supporting evidence.** The closest analogs point the other way: vegaml's engine vs a plain classifier, and Jeff's head-only frozen arm.

Before locking, six changes:
1. **Move the cheap, decisive baselines into M0:**
   - the same-backbone answer-token readout, reproduced at 59.3 and calibrated;
   - Jeva-4B (Apache-2.0) and the other Qwen3.5-4B references;
   - a Qwen3-Embedding zero-shot arm;
   - probes with an unseen-label split.
2. **Add a W2 *quality* benchmark to M0.** Today M0 measures only W2 latency.
3. **Rebuild B's request side and test the block properly:**
   - one causal state⊕question pass with state-KV reuse;
   - question-conditioned option caching as an ablation;
   - the block tested as a 2×2: frozen vs LoRA × block on vs off.
4. **Make C causal, sharing the state KV with the fast path.**
5. **Drop the Inkling 2 Mini reader and Matryoshka from M1.**
6. **Pre-register a go/no-go** for frozen B against the calibrated same-backbone readout.

---

## 1. Is B the right first milestone?

### 1.1 Where the ADR's evidence is weaker than it reads

1. **NDM does not encode state and question separately.**
   - In [sys/ndm] "Prefix-tree mask", the input is a tree, state → questions → options. A token attends to its own block and its ancestors, so question tokens attend to the state.
   - NDM pools h_S and h_Q separately, but h_Q is state-conditioned. [ADR1 §Decision "State and question towers"] and [sys/clm "Avoid"] cite NDM for two *independent* encodings, which NDM does not do.
   - The only cited precedent with a truly question-agnostic state is sbert2s1-B. It loses 9.4 points on held-out tasks (23.4 vs 32.8) [sys/sbert2s1 RQ2].
   - [SURVEY §1.1] itself says W1 accuracy "depends on reading the state *in light of the question*".
   - Reading Right shows CLM fails at *reading*: 79.6% of items wrong on both the fact probe and the decision [sys/clm].
   - Single-vector, query-agnostic encodings also have a proven capacity limit ([Weller et al., LIMIT](https://arxiv.org/abs/2508.21038)).

2. **The Jeva-vs-CLM gap is mostly training data, not architecture.**
   - CLM's 43.05 in [sys/jeva] Table 1 is a zero-shot score, with an appended "None" option.
   - InternDecision-4B scores 78.70 on the same column [sys/intern-decision]. It is a plain causal letter readout trained on 17,880 samples for 156 updates, with Typed Decisions in its training mix and options never shuffled.
   - On typed-decisions (very likely the same suite):

     | System | Score |
     |---|---|
     | CLM zero-shot | 0.355 |
     | CLM fine-tuned head | 0.686 |
     | Logistic regression on the same frozen features | 0.737 |
     | Teacher ceiling (Laya notes) | 0.735 |

     Sources: [CLM #13](https://github.com/Contrastive-LM/CLM/issues/13); teacher ceiling from [sys/laya].
   - The suite saturates around 0.74–0.79 and cannot separate architectures. The share of Jeva's lead that comes from architecture is at most about 4–5 points, and it is confounded with LoRA and a 1.1M-item curriculum. Jeva has no ablations.

3. **#13 cuts the other way on unseen labels.** The PR table (re-read this session) gives:

   | Test set | Zero-shot CLM | Fine-tuned CLM heads | LogReg | Zero-shot letter readout (SemIf) |
   |---|---|---|---|---|
   | Banking77, seen intents | 0.051 | 0.775 / 0.795 | **0.900** | 0.767 |
   | Banking77, unseen intents | 0.061 | 0.440 / 0.397 | n/a | **0.675** |
   | Bitext, unseen intents | 0.015 | 0.406 / 0.463 | n/a | **0.880** |

   S1-Arena has no train split, so it is the unseen regime. [ADR1 Context 1] quotes only the seen number.

4. **"Levels off after about 4B" is overstated.**
   - Jeff App. F zero-shot scores for Qwen3.5 are 76.8 (4B), 82.2 (9B) and 86.5 (27B): still +5.4 and +4.3 [sys/jeff].
   - Small Heads' finding that 4B ≈ 8B is for a joint LoRA model out of distribution [sys/small-heads]. It says nothing about frozen bi-encoder readers.
   - H6 has to be tested, not assumed.

5. **`prefix_bidir` for C is the ADR's weakest-evidenced choice.**
   - Strands reports no matched causal vs `prefix_bidir` ablation [sys/strands-decider, flaws (b)].
   - Jeff measures −0.2 ± 1.1 for bidirectional attention [sys/jeff].
   - Small Heads finds bidirectional attention costs 3.5–6.0 points in distribution and 5.4–11.8 out of distribution [sys/small-heads].
   - On Qwen3.5 hybrids, 24 of 32 layers are Gated DeltaNet, and these cannot honour block or bidirectional masks [sys/kev; sys/jeff on `bidir.py`].
   - Plain causal attention already allows state reuse (Kev, Decision 2.0, Intern-Decision).

6. **The latency rationale is a W2 argument, not a W1 one.**
   - For a new ~1k-token state with K ≤ 10 short options, a joint decoder adds about 150 option tokens: ≤ 20% more prefill *(est.)*.
   - B's advantage is real only for large or changing candidate sets and for revisits.
   - Speed alone did not win W2: CLM's 41 ms gave 1.7% loop success (FM10, [SURVEY §7.2]).
   - M0 has no W2 quality benchmark at all, only latency [PLAN M0.2].

7. **"All arms fit one mid-range GPU" is unverified for Inkling 2 Mini.** See §5.

### 1.2 Steelmen of the alternatives

**(a) Joint decoder first: causal, with state-prefix caching** (the Kev / Decision 2.0 / Intern-Decision pattern), on Qwen3.5-4B + LoRA with a pointer or an LM-head-initialised readout.
- This has the strongest W1 evidence:
  - Kai-0.6B, a joint model, scores 50.3 on S1A against 41.4 for the frozen-8B CLM.
  - Jeff's LoRA adds 7.8–15.7 points over zero-shot.
  - Kev-4B runs at 41.5 ms (new state) / 27.7 ms (cached state) on an L40S [sys/kev].
- Its W2 weakness is smaller than the ADR implies. Put a *fixed action catalog before the state*: the catalog's KV and its option-end hidden states then become a reusable prefix.
  - Each step pays only for the state and question tokens plus K pointer dot products.
  - The decision token attends to every option at every layer. That is a deep version of Jeva's one-block late interaction.
- Limits:
  - no per-option reuse across varying subsets (Jeva's Hybrid mode gave 28×);
  - fixed-order position bias;
  - K ≤ 255 per pass;
  - catalog KV of about 0.5 GB at K = 255 on Qwen3-8B *(est.)*;
  - knowledge erosion (MMLU −11 pp at 8B).
- Kev's code (Apache-2.0) can be reused: `shared_prefix.py`, `cuda_graphs.py`, `calibrate.py`.

**(b) Poly-encoder-style cached design** ([Humeau et al.](https://arxiv.org/abs/1905.01969)).
- Keep options cached. Represent each state by m learned "codes": attention-pooled from the frozen token states at the tapped layer and cached per state (16 × 4096 × 2 B = 128 KB).
- The question or each option attends over these codes.
- This targets the documented failure (reading, and the single-vector bottleneck) rather than IIA. It keeps both caches and costs K·m dot products.
- vegaml's 16-chunk MemoryReader is the nearest System One analog [sys/vegaml]. The approach is untested on frozen LLM features.

**(c) Instruction-conditioned embedding backbone: Qwen3-Embedding-8B** ([report](https://arxiv.org/abs/2506.05176)).
- Same architecture and footprint as Qwen3-8B. It is contrastively trained with task instructions on the query side, has MRL built in (which gives H8 for free), and led MTEB multilingual at release.
- Its companion Qwen3-Reranker is a yes/no cross-encoder: a ready-made slow path for boolean and choice questions.
- Small Heads tested it only as a *joint LoRA backbone* (89.2 vs 89.8), so frozen bi-encoder use is an open question.
- Cost: one encoder-registry entry plus re-embedding. Risks: label-word options still fail, and relevance is not the same as decision.

**(d) Linear or logistic probe on frozen features.**
- Not a System One product: it is closed-set, and #13 reports it as n/a on unseen labels.
- It is the cheapest information-ceiling diagnostic (minutes on cached features).
- As a per-deployment adapter for fixed label sets (routing, classification) it is strong: 0.900 and 0.997 on seen labels. This is vegaml's "Learner" pattern.
- It must be in M0, reported next to the unseen split.

**(e) Letter readout + calibrated cascade.**
- The zero-shot same-backbone answer-token readout is *measured* at 59.3 on S1A, 18 points above CLM [SURVEY §9.2].
- Per-type temperature fixes much of its miscalibration: temperature alone took Gemma from 72.9 to 79.6 Capability [sys/jeff].
- It takes about a day to build [raw/03, line 384], and it is also C's initialisation.
- Weaknesses: about 6% flips on option reversal at 4B, K ≤ 16–26 letters per pass, no option caching, and overconfident instruct models.

### 1.3 Comparison

| Option | W1 accuracy, S1A non-agentic (%) | W2 latency on L40S: 1k-token new state / revisit, K = 50 cached | Calibration | Build cost | Main risk |
|---|---|---|---|---|---|
| B as specified (frozen Qwen3-8B, separate towers, block) | 48–56 *(est.)* | 90–130 ms bf16 / ≤ 2 ms; flat in K | Needs post-hoc T (raw CLM Brier 0.749); principled null | High | Frozen late interaction untested; question-agnostic state; unseen-label generalisation |
| **B modified** (causal S⊕Q with state-KV reuse; LM-row-initialised boolean/score heads; question-conditioned option cache as ablation) | 54–61 *(est.)* | Same; extra questions packed into the same pass | Same | Medium-high | Medium |
| B + LoRA, Qwen3.5-4B (Jeva class) | 58–66 *(est.)* | 45–70 ms / ≤ 2 ms; caches rebuilt per checkpoint (seconds) | Same | High: end-to-end two-tower training | Cost; Jeva not yet reproduced |
| Joint decoder first (causal, 4B LoRA, prefix cache) | 60–68 *(est.)* | 80–110 ms including about 750 option tokens; catalog-first prefix gives state-only cost for fixed catalogs | Per-type T; 1–6% flips | Medium (Kev code reusable) | Large or varying K |
| Poly-encoder cached (frozen 8B) | 52–59 *(est.)* | ≈ B | ≈ B | Medium | No precedent on frozen LLM features |
| Qwen3-Embedding-8B bi-encoder | 42–52 zero-shot; 50–58 trained *(est.)* | ≈ CLM | ≈ B | Low | Label words; relevance ≠ decision |
| Linear probe | n/a on S1A (no train split); ceiling diagnostic only | ≈ CLM | Good after T | Trivial | Not a general model |
| Letter readout (LoRA later) + cascade | **59.3 measured** zero-shot; 63–70 with LoRA *(est.)* | ≈ joint; ≤ 26 letters per pass | Overconfident raw; T fixes most | About 1 day | Flips; no option cache |

The estimates are anchored on measured S1A points (CLM 41.4; Qwen3-8B readout 59.3; Kai-0.6B 50.3; Vega-27B 73.0), on the seen/unseen gap in #13, on sbert2s1's −9.4 held-out, and on Jeff's LoRA gains.

### 1.4 Answer

- **B is the right W2 component and a fair research bet, but the wrong first bet for W1 as specified.** The same-backbone readout already sits 18 points above CLM at zero training cost, so M1 has to be judged against it.
- **Pre-register the bar:** frozen B (or A) must come within about 3 points of the calibrated readout on S1A, *and* deliver at least 5× lower W2 latency at K ≥ 50 or on revisits.
- **If it misses,** W1 goes to the readout/joint path and B stays as the W2 fast path. That is still C, just reordered.

---

## 2. Technical risks in B

### 2.1 Does late interaction help when the encoder is frozen?

- **There is no direct evidence.** Jeva reports neither ablations nor frozen results; `tune_mode="frozen"` exists in its code but is unreported [sys/jeva].
- **The closest analogs are negative:**
  - vegaml's authors concede that "a plain classifier on the same features matches or beats the engine" [sys/vegaml];
  - Jeff's head-only frozen arm (78.9) scores below its own zero-shot readout (79.6) [sys/jeff];
  - Small Heads finds the head barely matters [sys/small-heads];
  - NDM's set-head gain (NLL 1.181 → 0.799) was on a fine-tuned encoder with state-conditioned option vectors [sys/ndm].
- **Mechanistically, the block computes a function of {z0, u1…uK}.** It can add set-relative normalisation, hubness suppression and null evidence. It cannot recover information that the frozen request vector lacks, and CLM's documented failure is reading.
- **The coverage gate does not need the block** (rung 4 adds it first), so the block's *unique* contribution is breaking IIA.
- **What to expect:** small typed-accuracy gains, with the real gains on nested sets, gold-removed null detection and large K. Measure it there.
- **Add a parameter-free control:** a CSLS/hubness correction ([Conneau et al.](https://arxiv.org/abs/1710.04087), not re-checked this session). CLM #15 shows hubness: one candidate appears in the top 10 for 9 of 20 unrelated queries [sys/clm].
- **Implementation note:** Jeva's per-option K/V are presumably projections of u_i (inference, in [sys/jeva]). Cache only u_i (1.5 KB at d = 768, bf16) and recompute K/V (2d² multiply-adds per option). Fork Jeva's Apache-2.0 `matrix_decision.py` and `cache.py` rather than rewriting them.

### 2.2 Do option embeddings that exclude the question lose information?

Yes, for several kinds of option:
- label words: yes/no, true/false;
- bare numerals and level labels (S1A score levels are bare strings with ids "0"…"n−1");
- opaque ids with empty descriptions;
- relational options: "both", "neither", "all/none of the above", "the first one";
- levels whose meaning is set by the question's rubric.

The size of the effect (FM14):
- label-word cosine gaps are about 1e-3 [sys/sbert2s1];
- in CLM #3, state sensitivity was 0.000 and rose to 1.059 when a trailing period was added [sys/clm].

The survey itself recommended question context on the option side "where caching allows" ([SURVEY §10 H1; §11 Candidate A]). ADR1 dropped it. In practice caching usually *does* allow it:
- W2 loops and W1 routing or classification reuse the same question across requests, so a cache keyed on (type, question hash, option) keeps its hit rate;
- only one-off questions pay K short, batched passes.

**Ablate** u(t, o) against u(t, q, o), and report S1A choice accuracy bucketed by description length and by whether options are label words.

### 2.3 How should state and question be fused?

**Recommended default:**
1. **One causal pass** over `render(state) ⊕ instructions ⊕ answer cue`.
   - Tap h_S at the state's last token: it is question-independent and cacheable.
   - Tap h_{S⊕Q} at the end of the question; h_Q alone is optional.
   - Pack all questions of a request as branches over one state pass. On an attention-only backbone like Qwen3-8B, use Kev's packed mode with block masks and position restarts; on DeltaNet hybrids, use rows that continue from the cached state.
   - [PLAN M1.1] already states this principle ("one causal pass … yields every needed embedding").
2. **Standardise each tapped vector per dimension** before projecting.
   - Massive-activation dimensions dominate raw last-token and intermediate states ([Sun et al.](https://arxiv.org/abs/2402.17762)).
   - vega-U standardises per reader for the same reason [sys/vegaml].
3. **Project with an interaction term:**
   - z0 = P[LN(h_S); LN(h_{S⊕Q}); LN(h_Q)] + e_type, plus FiLM or an elementwise product of the projected h_S and h_Q;
   - not plain concatenation into a linear layer, which cannot express "does the state satisfy the question".
4. **Ablate four request sides:**
   - the CLM layout (h_{S⊕Q} only);
   - separate towers;
   - causal [h_S; h_{S⊕Q}];
   - causal plus multi-vector state codes (the poly-encoder variant).

If separate towers are kept so that multi-question revisits cost nothing, add a multi-vector state memory for h_Q to attend over. Single-vector state plus MLP fusion is the configuration most exposed to the LIMIT bound.

Note also that CLM's state collapse (mean cosine 0.958 between different states) comes from the anisotropy of raw last-token states plus a retrieval-trained head. It is not evidence against question-aware pooling. Centring and typed training address it.

### 2.4 Long states

- **Burying is expensive even with adaptation.** Hiding the evidence among unrelated records cost Kev 22–43 pp even with a LoRA'd joint decoder. Training on buried states recovered 17–20 pp, but synthetic burying did not transfer to real documents [sys/kev]. Laya fell from 0.85 to 0.40 with filler [sys/laya].
- **A frozen, question-agnostic single vector is the worst case.** In a causal continuation, the question tokens attend to every state token at every layer.
- **S1A gives no shared input budget.** Over-length input counts as a refusal: accuracy 0, uniform distribution for Brier ([S1-Arena README](https://github.com/TianyuCodings/S1-Arena), [adapter docs](https://github.com/TianyuCodings/S1-Arena/blob/main/docs/models.md)).
- **Context policy:**
  - Qwen3-8B's `max_position_embeddings` is 40,960; use at least 8k and ideally 32k;
  - keep the tail for conversations and trajectories, head plus tail for documents;
  - log truncation rates per suite.
- **Add to M0:** a burying probe (evidence at positions 0…1 inside 1–8k tokens of filler) and a real long-document suite.

### 2.5 Yes/no questions

- **S1A boolean questions have no `criteria` at all.** The statement is in `instructions`, and the option ids are fixed to ("true", "false") ([schema.py](https://github.com/TianyuCodings/S1-Arena/blob/main/s1arena/schema.py)). A Noul head is therefore mandatory, not optional.
- **Initialise it from LM-head rows.** Use w0 = W_U["true"] − W_U["false"] applied to the last-layer h_{S⊕Q} (with an answer cue), plus a zero-initialised correction, following Jeff's `head.py`. Training then starts at the zero-shot answer-token readout instead of from scratch.
  - This only works for last-layer taps, so it conflicts with the L/2 and 3L/4 latency arms.
- **Route 2-option label-word choice questions through the same path.**
- **Train with polarity and negation twins.** Test with the yes/no identifier swap and with P(A) + P(¬A) coherence.

### 2.6 Score and ordinal questions

- **S1A score levels are an ordered list of 2–10 strings** with ids "0"…"n−1".
- **Existing systems handle these poorly:**
  - Laya keeps level order on only 1 of 4 ladders;
  - vegaml scores about 0.61 on score questions with every reader;
  - in Strands, "calibrating score changes its answer";
  - in Jeff, temperature scaling cut score competence from 64.5 to 54.1;
  - Kev's CDF penalty was a negative result.
- **Use a dedicated ordinal head on z:**
  - a cumulative-link head with K−1 thresholds that are a function of K, or a positional prior over levels;
  - level-text embeddings added as anchors when the levels are descriptive;
  - log-loss plus RPS, as NDM does;
  - never shuffle levels, but test reversal;
  - a per-level bias, as Decision 2.0 does;
  - report expectation MAE, RPS, Spearman and order preservation.

### 2.7 Is the coverage gate's training signal in the planned data?

Mostly no; it has to be manufactured.
- **No natural negatives.** The gate is trained with BCE on c_y = "some listed option is correct" [sys/jeva]. DecisionBank, kev-suites and typed-decisions almost always contain the answer, so c_y ≈ 1. DecisionBank's teacher distributions put no mass on "none", so c_y is undefined for soft targets.
- **None-pairs are the only signal** [PLAN M1.2]. Kev's none-pairs moved none_present from 0.85 to 0.93 [sys/kev]. Restrict them to hard-label or single-dominant-answer *choice* items. Never apply them to score, boolean, multi-label or ambiguous soft-target items, where neighbouring options are partly right.
- **Replay data gives noisy nulls.** About 1 in 20 near-binary Nemotron "negatives" paraphrase the answer [sys/clm], so gold-removed sets built from them contain false nulls.
- **Keep p_⊥ out of closed-set suites.** S1A answers are distributions over the listed ids only. Fold p_⊥ onto a none-like option if one exists; otherwise renormalise it away. Evaluate the gate only on Jeva's gold-removed protocol (BANKING77, K = 16: 3,080 requests, 588 with the answer absent).
- **The gate and the scale are entangled.** The gate reads the post-block maximum score, and ‖z_L‖ acts as a per-request temperature. Standardise the gate's inputs: Laya's act head died from a ~300× scale mismatch [sys/laya #185].

### 2.8 Smaller risks

- The CLM replay vectors are Qwen3-8B last-layer, last-token vectors. Only that arm can use them. The layer taps, the 4B LoRA arm and Qwen3-Embedding all need re-embedding, and LoRA training cannot use precomputed vectors at all.
- Intermediate-layer taps conflict with LM-row initialisation and with sharing KV between the fast and slow paths.
- Rebuilding the LoRA option cache per checkpoint is cheap: re-encoding a 10k-option catalog is about 300k tokens, a few seconds.

---

## 3. The ablation ladder (plan.md M1.3)

**Problems with the current ladder:**
- **Rungs 0 → 2 change data, objective and architecture at once.**
  - Rung 0's heads were trained on Nemotron retrieval and agent data, never on typed decisions.
  - Typed training alone moves CLM from 0.355 to 0.686 on typed-decisions (#13).
  - Add **rung 1b: CLM layout and head, retrained on the M1.2 data with soft-CE.** That is the honest F1 baseline for H1.
- **Rung 1 (calibration) is post-hoc and orthogonal.** It cannot change argmax accuracy.
  - Apply the same calibrator to every rung and report raw and calibrated results.
  - Test H3 as a separate factorial (global vs per-type vs T(type, log K)) with leave-one-source-out fitting.
- **Rung 3 bundles a choice-side change** (option rendering) **with a boolean-side change** (Noul head). Split it, or at least report per type.
- **Rung 5 bundles centring** (an expected gain; cosines of 0.947–0.958) **with Matryoshka** (an expected loss at 128-d: untrained truncation dropped 69.16 → 56.69 [SURVEY §4.5]). Split them, and move Matryoshka to serving, since memory is not binding (§5).
- **Rung 7 swaps backbones after every head choice was tuned on frozen 8B.** That confounds head with backbone, and H2 and H6 are exactly interaction questions.

**Interactions that can confound results:**
- block × calibration: ‖z_L‖ is an input-dependent temperature and can absorb K-dependence;
- block × gate: the gate reads the post-block maximum;
- block × backbone (frozen vs LoRA);
- option conditioning × Noul routing × each suite's share of label-word questions;
- request side × state length;
- centring × Matryoshka × K;
- data mix (questions per state, none-pairs) × everything.

**What is missing:**
- the same-backbone readout (reproduce 59.3, then calibrate);
- the probe, *with an unseen-label split*;
- Qwen3-Embedding zero-shot;
- Jeva-4B as released, plus Jeff-Qwen-4B, Kev-4B and Intern-Decision-4B if its weights are public (four Qwen3.5-4B readouts on one harness);
- the request-side variants and question-conditioned options;
- the CSLS control;
- a data ablation (Liquid: "fixing data shortcuts mattered more"; Kev: diminishing returns);
- a LoRA learning-rate sweep (Kev: 2e-4 → 5e-5 gave +4.7 pp, and 2e-4 collapsed; Small Heads: 2e-4 diverged on Qwen3-8B);
- W2 metrics for every configuration.

**Minimum defensible set of runs:**

| Stage | Runs | Seeds | What it answers |
|---|---|---|---|
| M0 references (no training) | CLM official; Qwen3-8B answer-token readout (+T); Jeva-4B; Kev-4B; Jeff-Qwen-4B; Qwen3-Embedding-8B zero-shot; LogReg probe (seen and unseen) | 1, plus test-retest on 2k items | The bar; whether the published B reproduces |
| Frozen Qwen3-8B heads (cached features, minutes per run) | F1r (CLM layout retrained); A (best request side + Noul/score heads + options + centring + gate); B = A + block | 3 | H1; H2 on the frozen arm |
| Drop-one from B | −gate; −centring; −Noul head; −question-conditioned options; separate towers instead of causal | 3 | Attribution |
| Layer taps | B at 3L/4 and at L/2 | 3 | H7 |
| Post-hoc calibration | global / per-type / per-type × log K, leave-one-source-out | — (free) | H3 |
| Qwen3.5-4B LoRA | Pilot: 3 learning rates, 1 seed, ¼ of the data; then A_LoRA and B_LoRA | 3 | H2 on the LoRA arm; H6 |

That is about 30 cheap head runs, 3 pilots and 6 LoRA runs.

**Decision rules (pre-register):**
- **Keep the block** if B − A exceeds 2× the seed SD on at least one of {S1A accuracy, nested-set all-sizes count, gold-removed AUROC, ToolRet at K ≥ 256}, in either arm, with no loss larger than 1 SD elsewhere.
- **Statistics:** use a task-cluster paired bootstrap. S1A domain accuracy is a mean over tasks, and on ~78k items, item-level McNemar will call 0.3 pp "significant".

---

## 4. Is the M0 harness complete?

### 4.1 Gaps

1. **There is no W2 quality suite.** Add:
   - S1A's agentic static tasks (README: 48 sources, 8,973 requests);
   - Jeva's nested-set and ToolRet K-sweep protocols;
   - one closed-loop environment scored on *loop success*: T-Rex ships in the CLM repo, and S1A has 138.
2. **Add a second validation anchor:** reproduce the Qwen3-8B answer-token score of 59.3. It validates the baseline that C starts from.
3. **Our own code path must reproduce CLM's 41.4.** Run the published CLM heads through *our* encoder and server.
   - Today it would fail. In `C:\system-one-from-scratch\src\systemone\inference.py`, `LastTokenEncoder` sets `padding_side="left"` but not `truncation_side`, then tokenises with `truncation=True, max_length=2048`.
   - Hugging Face tokenizers default to right truncation, so the end of the text (the question) is dropped. That is exactly CLM #6/#29.
   - Also add an embedding-parity test: re-embed about 1k Nemotron rows from their text and match the published vectors (cosine > 0.999) before re-embedding any new layer or backbone.
4. **Pin the S1A version.**
   - Today's README reports 52,199 static requests, and 43,226 requests / 78,362 questions / 13 domains for the non-agentic set.
   - [raw/04] recorded 49,592 requests, about 92,125 questions and 16 domains two days ago. The set is moving.
   - Pin the dataset and harness commits, and rebuild the 14 locally built sources reproducibly.
5. **Adjust the reference list:**
   - add Jeva-4B (Apache-2.0, Qwen3.5-4B; [card](https://huggingface.co/LIQIIIII/Jeva-4B)), Jeff-Qwen-4B and Qwen3-Embedding;
   - drop the Jev API (its published 72.6 suffices) and Inkling 2 Small, to protect the $60.
6. **Contamination reporting:** flag references whose training mixes overlap S1A sources, and report a clean subset.
7. **Add a probe battery:**
   - question sensitivity (Strands v7 gave the same answer to a changed question 94–99% of the time);
   - burying;
   - negation twins and coherence;
   - option renaming (only 1 of 27 papers tested it);
   - yes/no identifier swap;
   - K-sweep with distractors;
   - gold-removed null;
   - latency with and without caches.
8. **Freeze the calibration-fit sources** (leave-source-out) before any model is scored.

### 4.2 How a bi-encoder should consume S1-Arena items

Schema per [schema.py](https://github.com/TianyuCodings/S1-Arena/blob/main/s1arena/schema.py):

| | `choice` | `boolean` | `score` |
|---|---|---|---|
| S1A fields | `criteria`: dict mapping id → description, 2–255 options | `criteria` must be None; statement in `instructions`; ids ("true", "false") | `criteria`: ordered list of 2–10 level strings; ids "0"…"n−1" |
| Request side | `render(state dict) ⊕ instructions ⊕ answer cue`, one causal pass with h_S and h_{S⊕Q}; all questions of a request packed over one state pass (about 1.8 questions per request) | same | same |
| Option side | `T_o(type, [instructions], id if meaningful, description)`; cache key = backbone/layer/template/adapter/text hash, plus question hash in the conditioned variant; empty description → id only, flagged as short-label | none: Noul head with LM-row init | ordinal head over K positions; level strings as anchors |
| Output | softmax over the ids; p_⊥ folded onto a none-like option or renormalised away | {true: p, false: 1 − p} | distribution over ids; expectation reported separately |
| Tests | renaming; K-sweep; ids that leak order or the answer | identifier swap; negation twins | level reversal (remap ids); ordinal metrics |

S1A states are dicts. Fix one canonical renderer (sorted-key compact JSON, or CLM-style prose), include it in the cache key, and measure the other renderer on a subset. Rendering is a data shortcut that a frozen encoder cannot adapt to.

### 4.3 Evaluation pitfalls to design in

1. **Failures are scored, not skipped.** An unanswered question scores 0 for accuracy and counts as a uniform distribution for Brier and ECE; over-length input is a refusal. The server must never error, must log refusals and truncations, and must never cut off the question.
2. **ECE definition.** S1A uses top-label ECE with 10 equal-mass bins, with tied confidences sharing their mean correctness. `C:\system-one-from-scratch\src\systemone\eval\calibration.py` uses 15 equal-width bins. Implement the official estimator for comparability. Internally, also report per-type and per-K-bucket ECE, [smooth ECE](https://arxiv.org/abs/2309.12236), and ECE/null.
3. **Variable K.** Top-label confidence has a floor of 1/K, so pooling K = 2 with K = 255 hides K-dependent miscalibration (Kev-9B's ECE went from 0.051 to 0.377 at K = 150). Always stratify.
4. **Use metrics suited to the type.** For boolean: binary reliability of P(true) and Brier. For score: RPS and the MAE of the expectation.
5. **Aggregation.** Domain accuracy is the mean over tasks. Report both macro and micro averages, and bootstrap over tasks.
6. **Soft targets.** S1A computes Brier against one-hot or soft targets. Accuracy is against the gold argmax, so define tie handling.
7. **Contamination.**
   - The registry must cover S1A's locally built sources.
   - DecisionBank contains CLINC, Banking77, Civil Comments and ChaosNLI; kev-suites overlaps S1A [raw/04].
   - OpenDecisions is marked "do not train".
8. **Version drift.** Pin everything (see 4.1).
9. **Determinism.** Repeat runs change up to 1.33% of decisions [SURVEY §7.1]. Run a test-retest per reference, and parity tests for FP8 and for batch composition.
10. **Latency.** S1A's latency is the median per *request* on an idle H100. Measure ours on L4/L40S at batch 1, cold and warm, with and without caches, tokenisation included.
11. **Ties.** `summarize_hard` in `C:\system-one-from-scratch\src\systemone\eval\metrics.py` takes `p.max(1)` with the gold in column 0, so exact ties count as correct in ECE while top-1 is strict. Fix this before moving to variable K.
12. **Report accuracy by state-length bucket.**

### 4.4 Code reuse and hazards

- **Reusable:**
  - config, YAML and overrides (`config.py`);
  - `ProjectionHead`;
  - `fit_scale` (L-BFGS on log s) and `reliability_bins`;
  - the Modal volume, sweep and `results/` pattern;
  - the CLM checkpoint loader (rung 0);
  - `RemoteNpy`.
- **To build:**
  - the typed schema and loaders;
  - a text-encoding pipeline with KV continuation and multi-layer taps;
  - a typed trainer: variable-K masks, soft-CE, the gate's BCE, Noul and ordinal heads, the block;
  - the option cache;
  - the server and an S1A adapter;
  - the probe battery and the decontamination registry.

  The current trainer is pair InfoNCE over precomputed vectors (`losses.py`, `train.py`, `PairSet`); it is reusable only for replay.
- **Hazards:**
  - the `inference.py` truncation bug (4.1, item 3);
  - the ECE binning mismatch;
  - the Modal image pins torch 2.5.1 with no transformers, vLLM or fla kernels, and Qwen3.5's DeltaNet layers will need a newer stack.

---

## 5. Serving feasibility on one GPU

**Hardware and model facts:**

| GPU | Memory | Bandwidth | BF16 dense | FP8 dense |
|---|---|---|---|---|
| [L4](https://www.nvidia.com/en-us/data-center/l4/) | 24 GB | 300 GB/s | 121 TFLOPS | 242.5 TFLOPS |
| [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) | 24 GB | 600 GB/s | 125 TFLOPS | none |
| [L40S](https://www.nvidia.com/en-eu/data-center/l40s.md) | 48 GB | 864 GB/s | 362 TFLOPS | 733 TFLOPS |

[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) has 36 layers, 8 KV heads and head_dim 128. That gives:
- 16.4 GB of weights in bf16;
- KV = 2 × 36 × 8 × 128 × 2 B = **144 KiB per token**: 1.2 GB for an 8k-token state, 4.7 GB for 32k;
- a prefill of about 16.4 TFLOP per 1k tokens.

**Floors and prefill times:**
- **Weight-streaming floor per forward pass:**
  - L4: 55 ms in bf16, 27 ms in FP8;
  - A10: 27 ms;
  - L40S: 19 ms in bf16, 9.5 ms in FP8.
- **1k-token prefill at about 50% MFU** *(est.)*:
  - L4: about 270 ms in bf16, 135 ms in FP8;
  - A10: about 265 ms;
  - L40S: about 90 ms in bf16, 45–60 ms in FP8.
- **The survey's "~30 ms new state" on an L40S [SURVEY §11 Table 12] is a short-state number.** CLM's 28 ms was an RTX 4090 with short states, and S1A's 41 ms was an idle H100.

| Arm | Weights | Room left for KV and caches (L4 / L40S) | 1k-token new state p50, L4 / L40S *(est.)* | Notes |
|---|---|---|---|---|
| Frozen Qwen3-8B, bf16 | 16.4 GB | about 5–6 GB / about 28 GB | 250–330 / 90–130 ms | On an L4, only one 32k state fits at a time |
| Qwen3-8B FP8, W8A8 per-channel | about 8.5 GB | about 13 / 36 GB | 140–180 / 55–80 ms | The official Qwen3-8B FP8 checkpoint is block-wise; on Ada that reportedly falls back to Marlin weight-only kernels, saving memory but not compute (third-party [write-up](https://levelup.gitconnected.com/blockwise-fp8-on-ada-gpus-why-your-l4-falls-back-to-marlin-measured-e74a0dfa3024); I could not fetch it, so treat as unverified). vLLM lists Ada as supported for [W8A8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/). The A10 has no FP8 |
| Qwen3.5-4B + LoRA | about 8 GB | about 13 / 36 GB | 130–170 / 45–70 ms | KV only in the 8 attention layers, plus a fixed recurrent state; Kev-4B measured 41.5 / 27.7 ms on an L40S (short state) |
| Inkling 2 Mini | about 20 GB, mixed NVFP4/MXFP8 | ≤ 3 GB on 24 GB cards | not runnable as specified | Card verified only on H100/H200; vLLM needs a custom image; hidden states require transformers. Ada has no FP4 tensor cores and the MXFP8 kernels are Blackwell-targeted; the A10 has no FP8. Dequantised to BF16 it is about 53 GB, more than 24 or 48 GB [sys/inkling] |

**Option cache:**
- Per option:
  - u_i: 1.5 KB at d = 768 in bf16;
  - the block's K/V: 3 KB, but these can be recomputed;
  - the raw 4096-d fp16 vector: 8 KB, if kept for retraining heads.
- Totals: 0.15–1.25 GB per 100k options, and 1.5–12.5 GB per 1M.
- **Memory is not the binding constraint** at one-GPU catalog sizes. Defer Matryoshka and int8 caches until catalogs exceed about 1M options.

**The C slow path:**
- **Memory:** the same base model with an unmerged LoRA adapter costs only +0.1–0.3 GB.
- **Without shared KV:** if the fast path is frozen and the slow path is LoRA'd, an escalated request must re-prefill the whole state. That costs a full new-state latency: +90–130 ms on an L40S, +270 ms on an L4.
- **With shared KV:** if the fast path's state pass is a causal prefix computed with the slow path's weights (a shared multi-task adapter, or both frozen with a trained readout), the slow path just continues. It then pays only for the question plus about 10 rendered options, about 200 tokens: +35–45 ms on an L40S and about +110 ms on an L4 *(est.)*.
- **Adapting only the question side is not free:** in Kev it cost 3.4–6.0 pp.
- **Fix this choice in the M1 interfaces.**

**Starting latency targets** *(est.; replace them with M1.4 measurements)*:

| Path | L40S p50 (p95) | L4 p50 (p95) |
|---|---|---|
| 1k-token new state, 1–3 questions, K ≤ 10 (8B in FP8 W8A8, or 4B in bf16) | 70 (120) ms | 180 (300) ms |
| Same, 8B in bf16 | 120 (200) ms | 320 (500) ms |
| Revisited state with cached options, K ≤ 255 | 2 (5) ms | 2 (5) ms |
| W2 step: 200 new trajectory tokens, K = 50 cached | 45 (80) ms | 120 (200) ms |
| C escalation with shared state KV / without | +45 / +120 ms | +120 / +320 ms |

---

## 6. Budget, time and risk register

**M0 cost against the $60** (Modal prices: L4 $0.80/h, A10 $1.10/h, L40S $1.95/h, H100 $3.95/h; [pricing](https://modal.com/pricing)). For S1A non-agentic, I assume about 45–90M prefill tokens per model, and about 11k tokens/s for an 8B model on an L40S *(est.)*.

| Item | Estimated cost |
|---|---|
| CLM official | $2–4 |
| Qwen3-8B readout (+ 2-rotation variant on a subset) | $3–6 |
| Jeva-4B | $2–4 |
| Kev-4B | $1.5–3 |
| Jeff-Qwen-4B | $1.5–3 |
| Kai-0.6B + Nox-4B | $1.5–3 |
| Laya | < $0.5 |
| Qwen3-Embedding zero-shot | $2–4 |
| Inkling 2 Mini readout via API | $4–14 |
| Robustness battery on a 10k stratified subset | $8–12 |
| Overhead (test-retest, debugging, cold starts) | $10–15 |
| **Total** | **about $40–70** |

$60 is feasible only if Jev and Inkling 2 Small are dropped, the battery runs on subsets, and full-set runs are limited to the anchor models.

**M1 cost** *(est.)*:
- **Frozen arm:**
  - embed the typed data once per backbone and template, with all taps in one pass: about 120–150M tokens, about 3–4 hours on an L40S, $6–8 each;
  - about 35 head runs: $20–40.
- **LoRA arm:** about 6 × parameters × tokens.
  - DecisionBank-260k plus options is about 150M tokens per epoch.
  - That is about 3.6 EFLOP, plus a third more for gradient checkpointing.
  - Each epoch-run takes about 9–13 hours on an L40S, $20–25.
  - Pilots plus 6 runs of 1–2 epochs: $150–300.
  - Jeva-scale data (1.1M items) would cost about 4× that.
- **M1 total:** about $200–400. Write this into the plan now.

**Time, one engineer** *(est.)*:

| Stage | Duration |
|---|---|
| M0 | 3–4 weeks |
| M1 frozen stage (registry, data, typed trainer, block, gate) | 4–5 weeks |
| M1 LoRA arm | +3–4 weeks |
| Serving | 1–2 weeks |
| **Total to an M1 decision** | **about 3 months** |

The methods note's "about 1 week for the block on frozen features" covers only the head.

**Risk register:**

| # | Risk | Likelihood | Impact | Mitigation | Kill criterion |
|---|---|---|---|---|---|
| 1 | Frozen B stays below the calibrated same-backbone readout on W1 | High | High | Make the readout M1's bar; causal S⊕Q; LM-row-initialised heads; run the LoRA pilot in parallel | Best frozen config is more than 5 pp below the calibrated readout (cluster-bootstrap CI excludes −3) → frozen B becomes W2-only |
| 2 | The block adds nothing measurable (H2 is null) | Medium-high | Medium | Measure nested sets, null AUROC and ToolRet K-sweep; 2×2 with LoRA; CSLS control | Gain below 2× seed SD on every set-dependence metric in both arms → drop the block and ship A + gate |
| 3 | Poor generalisation to unseen option sets (S1A has no train split) | High | High | Leave-source-out dev metric; question-conditioned options; LM-row initialisation; many questions per state | Held-out-source accuracy more than 15 pp below seen-source accuracy after these fixes → move W1 to LoRA or the joint path |
| 4 | The harness fails to reproduce the anchors (S1A drift) | Medium | High | Pin commits; two anchors (41.4 and 59.3); per-domain diffs | Gap above 1.0 pp on both anchors after two fix rounds → report only against our pinned version |
| 5 | Calibration does not transfer across sources or K | Medium | Medium | Leave-one-source-out fitting; K as a smooth, shrunk covariate; a refit tool | T(type, log K) worse than per-type on at least half of held-out sources → ship per-type only |
| 6 | The LoRA arm overruns on cost or schedule, or is fragile to learning rate | Medium-high | Medium | ¼-data learning-rate pilot; deduplicate options per batch; cap K during training | Pilot B_LoRA does not beat frozen B by at least 3 pp on held-out sources → do not scale it |
| 7 | Contamination or licensing compromises comparisons | High (overlap is documented) | Medium-high | Registry first; source-level exclusion; clean subset; flags on references | Any leak above 0.5% of a suite's items found after the fact → invalidate and rerun before publishing |
| 8 | Serving targets miss, or arms are infeasible | High (against the implied 30–40 ms) | Medium | Per-GPU targets; W8A8 FP8 with parity tests; the 4B arm; drop Inkling 2 Mini | FP8 changes more than 0.5% of decisions, or max \|Δp\| > 0.02 → serve bf16 and target the L40S |

A further risk: the coverage gate fires falsely on closed-set suites. Kill criterion: if closed-set Brier worsens by more than 0.01 with the gate on, disable it for those types.

---

## 7. Recommended changes to ADR 0001 and plan.md

| | Change | Justification |
|---|---|---|
| **Keep** | The two-path end state, and router/second-scorer interfaces in M1 | Cascade evidence [SURVEY §9.4]: System One Auto 82.25 vs 87.88 (Vega) and 65.80 (Kai); sbert2s1 routing 59.5 → 68.4 |
| **Keep** | A cached-option fast path with exact equivariance for W2 | Free for F1/F2; Jeva Hybrid caching gave 28× |
| **Keep** | The coverage gate as the null mechanism | AUROC 98.66 vs 93.61; Prop. 2 |
| **Keep** | A Noul head that ignores option text | S1A boolean questions have no criteria at all |
| **Keep** | Centring; calibration as a procedure with a refit tool; 3 seeds and the 2×SD rule; registry-first ([ADR2]) | Measured gains (Bongard 2.3% → 56.6%; our ECE 0.065 → 0.013) and good hygiene |
| **Change** | Request side: separate towers → one causal S⊕Q pass with state-KV reuse, fusing h_S and h_{S⊕Q}; separate towers become an ablation | §1.1 item 1, §2.3; NDM was misread; the LIMIT bound |
| **Change** | Option tower: add an ablation with question-conditioned cache keys | The survey's own H1 and Candidate A; FM14; real reuse patterns |
| **Change** | Specify the boolean and score heads: LM-row initialisation; an ordinal head; label-word choice questions routed to them | Jeff's initialisation; the S1A schema; FM5 |
| **Change** | C: causal by default; `prefix_bidir` only as an H5 ablation on attention-only backbones; a contract for sharing KV between the fast and slow paths | Jeff −0.2 ± 1.1; Small Heads −3.5 to −11.8; DeltaNet layers; escalation latency |
| **Change** | Ladder → a baseline-anchored factorial with drop-one ablations; calibration orthogonal; split rungs 3 and 5; start the LoRA pilot in parallel | §3 |
| **Change** | Correct [ADR1] Context: Jeva vs InternDecision; #13 unseen rows; the "levels off at 4B" wording; NDM's mask | §1.1 |
| **Change** | Backbone table: remove "all arms fit one mid-range GPU"; state per-GPU footprints | §5 |
| **Change** | [ADR2] item 5: fit temperatures on held-out *source datasets or task families*, not DecisionBank `group_id` | Kev lists fitting T on held-out items of training sources as a failure mode (PLAN #5); [SURVEY §5.3]; Jeff fits on held-out task families |
| **Change** | [ADR2] item 3: specify shingles (character 5-grams for short fields; 13-grams cannot fire on ~10-token intents); consider a stricter Jaccard (Kev used 8-gram Jaccard > 0.2); hash label sets for a seen/unseen split | Short typed items slip through the current rule |
| **Change** | [PLAN M1.4]: per-GPU targets; FP8 means W8A8 on Ada; the A10 runs bf16 only; [ADR1] status becomes "conditional on the M0 gate" | §5 |
| **Add** | M0 references: Jeva-4B, Jeff-Qwen-4B, Qwen3-Embedding zero-shot, probes with an unseen split; reproduce 59.3 | §1.2, §3 |
| **Add** | A W2 quality suite in M0 | §1.1 item 6, §4.1 |
| **Add** | M0.4b: our own path reproduces CLM's 41.4; an embedding-parity test; fix the truncation bug | §4.1 |
| **Add** | A pre-registered M1 go/no-go; the probe battery; clean-subset reporting; feature standardisation for taps; a CSLS control; budget and time estimates in the plan | §1.4, §3, §6 |
| **Drop/defer** | The Inkling 2 Mini frozen reader (keep the API readout in M0) | Not runnable on the target GPUs as specified; no precedent |
| **Drop/defer** | Matryoshka 128-d and int8 option caches → only if catalogs exceed about 1M options | Memory is not binding |
| **Drop/defer** | The Jev API reference | Published 72.6 suffices; protects the budget |
| **Drop/defer** | Calibration as a rung | It is an orthogonal procedure |

---

## Changes I would make before locking

1. **Rewrite [ADR1] "State and question towers"** as one causal state⊕question pass with state-KV reuse. Fuse standardised h_S and h_{S⊕Q} with an interaction term. Keep separate towers as an ablation.
2. **Make question-conditioned option caching an explicit option-tower ablation,** with the question hash in the cache key.
3. **Specify the boolean and ordinal heads** concretely: LM-row initialisation, no option text, and routing for label-word choice questions.
4. **Restrict coverage-gate training** to none-pairs built from hard-label choice items. Evaluate the gate only on open-set protocols, and keep p_⊥ out of closed-set suites.
5. **Make C causal by default,** with a contract for sharing state KV between the fast and slow paths. Demote `prefix_bidir` to an H5 ablation.
6. **Correct the Context claims in [ADR1]:**
   - Jeva vs InternDecision;
   - #13's unseen-label rows;
   - "levels off around 4B";
   - NDM's mask;
   - "all arms fit one GPU".
7. **Remove** the Inkling 2 Mini reader and Matryoshka from M1, and the Jev API from M0.
8. **Expand M0:**
   - add Jeva-4B, Jeff-Qwen-4B, Qwen3-Embedding and the probes (with an unseen split);
   - reproduce 59.3;
   - reproduce 41.4 through our own encoder (after fixing `inference.py` truncation) and add the parity test;
   - pin S1A;
   - add the W2 quality suite.
9. **Replace the ladder** with the baseline-anchored factorial plus drop-ones from §3. Treat calibration as orthogonal, start the LoRA pilot in parallel, and use a task-cluster bootstrap.
10. **Pre-register the M1 go/no-go** against the calibrated same-backbone readout and a W2 latency ratio.
11. **In [ADR2]:**
    - fit calibration leave-source-out;
    - specify the shingling;
    - hash label sets;
    - report a clean subset;
    - flag contaminated references.
12. **Write into [PLAN]:**
    - per-GPU latency targets;
    - the FP8 W8A8 path;
    - an M1 budget of about $200–400;
    - about 3 months to an M1 decision.

**Sources checked this session:**
- [CLM #13](https://github.com/Contrastive-LM/CLM/issues/13)
- S1-Arena: [README](https://github.com/TianyuCodings/S1-Arena), [schema.py](https://github.com/TianyuCodings/S1-Arena/blob/main/s1arena/schema.py), [docs/models.md](https://github.com/TianyuCodings/S1-Arena/blob/main/docs/models.md)
- [Jeva-4B card](https://huggingface.co/LIQIIIII/Jeva-4B)
- [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- Datasheets: [L4](https://www.nvidia.com/en-us/data-center/l4/), [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/), [L40S](https://www.nvidia.com/en-eu/data-center/l40s.md)
- [Modal pricing](https://modal.com/pricing)
- [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)
- [Block-FP8 on Ada write-up](https://levelup.gitconnected.com/blockwise-fp8-on-ada-gpus-why-your-l4-falls-back-to-marlin-measured-e74a0dfa3024) (could not be fetched; unverified)
- Papers: [Poly-encoders](https://arxiv.org/abs/1905.01969), [Qwen3 Embedding](https://arxiv.org/abs/2506.05176), [LIMIT](https://arxiv.org/abs/2508.21038), [Massive activations](https://arxiv.org/abs/2402.17762), [Smooth ECE](https://arxiv.org/abs/2309.12236)
- Not re-checked this session: [CSLS](https://arxiv.org/abs/1710.04087)
