# Response to the independent reviews

Before locking the architecture, on 2026-11-22, we commissioned three independent reviews of the survey, ADR 0001, ADR 0002 and plan v2. Reviewers received the documents, not the author's conclusions. All three are published verbatim:

| Review | Scope | Headline |
|---|---|---|
| [01 · Technical review](reviews/01-technical-review.md) | Survey, ADRs and plan, checked against all 31 system notes and the raw notes | 27 issues (3 critical, 20 major), plus 12 items of missing content |
| [02 · Fact-check](reviews/02-fact-check.md) | 40 survey claims checked against primary sources (papers, code, issues, model cards) | 27 verified, 7 minor mismatches, 6 context mismatches, **no wrong digits** |
| [03 · Architecture review](reviews/03-architecture-review.md) | Devil's-advocate stress test of ADR 0001 ("build B, designed for C"), ADR 0002 and plan v2 | "Modify and re-sequence, do not replace": 12 changes before locking |

This page tracks every finding. **Outcome:** the architecture was locked on 2026-11-22 as the re-sequenced design in [ADR 0003](../adr/0003-architecture-resequenced.md): the typed-decision model on a forked state first, and the cached-option path behind a pre-registered go/no-go.

**Status values**
- **Fixed** names the commit.
- **Partly** says what remains.
- **Open** is scheduled for survey v1.1.
- **Resolved / Adopted** marks items settled when the architecture was locked on 2026-11-22 ([ADR 0003](../adr/0003-architecture-resequenced.md), [ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md), [plan v3](plan.md)).
- **Declined** gives a reason.

## 02 · Fact-check: claims that were not fully verified

| # | Finding | Status |
|---|---|---|
| 7 | The ">65 %, <100 ms" frontier region is not empty: four open-LLM answer-token readouts sit there | **Fixed** in 68c1019. Table 11 now has all 22 rows; abstract conclusion 1 and §9.2 are rewritten; Fig. 2 and the README frontier are rebuilt (70dfdf6) |
| 8 | CLM is the fastest only among System-1-native rows; its own backbone's readout is faster and more accurate | **Fixed** in 68c1019 (abstract, §1.1) |
| 10 | Jeva's AUROC comes from §4.2 / Fig. 6b, not Tables 1–2 | **Fixed** in 68c1019 |
| 14 | Gemma-4-31B with Jeff's chat prompt fits temperatures of 5.96 / 7.24 / 7.24, not 2.7–6.0 (Table 19) | **Fixed** in 23bf3ac (§5.1) |
| 16 | Jeff reports its order-invariance cost as within noise, and option sorting as free | **Fixed** in 68c1019 (FM1, H4, Table 10 row 6) |
| 17 | Jeff is bidirectional for every model except Gemma-4-31B | **Fixed** in 68c1019 (mask table) |
| 20 | Kev-27B decontamination used exact match, 8-gram Jaccard > 0.2 or containment ≥ 0.5 | **Fixed** in 68c1019 |
| 21 | Kev's buried-state gains are synthetic; real documents moved only ±1 pp | **Fixed** in 68c1019 (FM6, Table 10 row 10) |
| 22 | Laya's option budget includes the question text | **Fixed** in 23bf3ac (§3.4) |
| 23 | Laya issue #741 was a single-seed third-party run; the three-seed PR #1012 found a tie | **Fixed** in 68c1019 (§4.3, Table 10 row 14) |
| 25 | Laya's evidence-position cells have n = 20 (±0.22) | **Fixed** in 68c1019 (FM6) |
| 26 | Appendix A's θ = 4.6423 belongs to a different head | **Fixed** in 23bf3ac (Appendix A) |
| 28 | CLM truncation: on real vLLM 0.30.0, issue #29 measures −0.1169 (the −0.0006 is from an MLX emulation) | **Fixed** in 23bf3ac (FM6, Table 10 row 9) |
| 29 | Strands' question-ignoring persists from v7 to v19 (92–95 %) | **Fixed** in 68c1019 (FM7) |
| 30 | StartLux throughput figures are in App. D.4, not §6 | **Fixed** in 68c1019 (mask table) |
| 32 | Jev's fitted temperatures of 0.66 and 0.21 both come from Suite A | **Fixed** in 68c1019 |
| 33 | The yes/no flip range covers four checkpoints; Kev-0.8B flips 14.3 | **Fixed** in 68c1019 (FM2, Table 10 row 7) |
| 36 | Reading-right-acting-wrong spans 73–78 % for System One models; 85 % is GPT-6.1 | **Fixed.** Survey in 68c1019 (FM8); README tile regenerated with the same commit as this entry |
| 37 | Reading Right's +16.3 needs the oracle sub-goal; JEV's own sub-goals give +6.4 | **Fixed** in 68c1019 (Table 10 row 16) |
| 38 | NDM's adequacy head is a trade-off: recall 0.66 vs 0.25, but false alarms 0.29 vs 0.025 and AUROC 0.76 vs 0.79 | **Fixed** in 68c1019 (§3.2, §4.4, Table 10 row 5) |
| 39 | Risk-budget rates are over 250 / 209 / 153 settings, and the baseline is Qwen3-Embedding-8B | **Fixed** in 68c1019 (§5.5) |
| 40 | AnyJev uses 7.3 of 18 on tuned data and 10.6 of 18 on disjoint data (Table 3) | **Fixed** in 68c1019 (§3.6) |

## 01 · Technical review

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Critical | B's question-blind state tower is not what NDM or Jeva do (NDM's question attends to the state) | **Resolved.** Survey fixed in 23bf3ac. The locked design ([ADR 0003](../adr/0003-architecture-resequenced.md)) reads the state through exact state forking; question-blind towers survive only as an M2 ablation |
| 2 | Critical | The abstract asserts causes for CLM's gap; the scale cap explains Brier, not accuracy; the truncation and training-data confounds are omitted | **Fixed** in the survey (68c1019, 23bf3ac): candidate causes, not established ones; truncation and training-data confounds; the scale cap moved to Brier. M0 arms (CLM with full context vs the tail cut) are **open** for M0 |
| 3 | Critical | Table 10 mostly cites evidence that a flaw exists, not that a fix works | **Partly.** Rows 5, 6, 7, 10, 11, 14 and 16 were revised in 68c1019, and rows 2, 4 and 9 in 23bf3ac. Restructuring (flaw measured / fix measured / status / citations) is open |
| 4 | Major | Reproduction and calibration claims lack provenance; the low-confidence bin is anomalous | **Partly.** The tie bug is fixed in PR #41; the wording ("within 0.4 pp", different sample) and the in-distribution and tie caveats are in 23bf3ac. Split provenance is open; randomised gold position and a cross-dataset fit are M0 |
| 5 | Major | P3 misstates the IIA trade-off | **Fixed** in 3516adf |
| 6 | Major | Four equations are wrong or overstated (null formula, Bongard REINFORCE identity, AnyJev "exact", soft-CE "exact") | **Fixed.** Bongard identity removed in 3516adf; null formula, AnyJev and soft-CE in 2e5b364 |
| 7 | Major | Four equations are unsourced or incomplete (InfoNCE mask, RLCD std, vegaml E_k, Jeva attention terms) | **Fixed.** RLCD std in 2e5b364; InfoNCE mask, vegaml E_k and Jeva attention terms in 46660c4 |
| 8 | Major | Taxonomy families overlap and several systems are misplaced | **Open.** Planned: two axes (interaction/cache pattern × readout) plus attributes |
| 9 | Major | Table 1's caching columns are wrong for F1, F2, F3 and F5 | **Fixed** in 7d6f736 |
| 10 | Major | Calibration is presented as free, but the sources record trade-offs | **Partly.** Table 10 row 4 and §5.6 state the costs (23bf3ac); the §4.2 objective wording and the Jeff locator are fixed in 3516adf; a dedicated §5 paragraph is open |
| 11 | Major | K-aware calibration is misattributed; the ADRs adopt fitting rules that the sources contradict | **Fixed.** H3 in 3516adf, Table 5 in fd4b1fb; leave-source-out fitting and $K$-conditioning as an experiment in [ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md) |
| 12 | Major | The "RL earns its keep" examples are CE or closed-form proper-score losses | **Fixed** in 3516adf (conclusion 3, §4.3) |
| 13 | Major | The adaptation and scale claims do not follow (Qwen3.5 keeps rising after 4B; head-only arm within noise) | **Fixed** in 3516adf (Table 4 caption, §4.5, H6). **Declined in part:** the review's "frozen 9B 82.2 vs LoRA-adapted Jeff-Qwen-4B 79.3" compares accuracy on Jeff's seven held-out sets (App. F) with OpenDecisions Capability (Table 3), which are different benchmarks |
| 14 | Major | Strong baselines and benchmark-validity caveats are missing | **Partly.** The [start-here guide](README.md) adds baselines and a benchmark card; they are part of M0 in [plan v3](plan.md) |
| 15 | Major | W2 and the latency frontier are argued one-sidedly | **Fixed.** The "empty region" in 68c1019; W2 counter-evidence and measured latency anchors in Table 12 in 7d0ad66 |
| 16 | Major | Metric conventions (ECE bins, Brier scale) are unspecified | **Fixed** in e899364 (§2.2 conventions). [plan v3](plan.md) reports both estimators |
| 17 | Major | The single-pass trajectory shortcut breaks under CLM's layout | **Fixed** in 3516adf (§6.2); [plan v3](plan.md) replaces the shortcut with exact state forking plus a parity test |
| 18 | Major | Failure-mode provenance is overstated; FM6, FM11 and FM13 are mismeasured | **Mostly fixed.** FM2 in 68c1019; provenance, FM6, FM11 and FM13 in c39e048. A per-mode provenance column and FM1's protocol column are open |
| 19 | Major | ADR 0001 has unestablished hardware feasibility (Inkling 2 Mini; `prefix_bidir` on DeltaNet layers) | **Resolved** in [ADR 0003](../adr/0003-architecture-resequenced.md): causal throughout; Inkling through Tinker plus an off-target H100 reader arm (the owner's decision); per-GPU measurements in M1.4 |
| 20 | Major | ADR 0001 lacks a training specification and acceptance criteria | **Resolved:** pre-registered criteria in [ADR 0003](../adr/0003-architecture-resequenced.md); per-stage data and arms in [plan v3](plan.md) |
| 21 | Major | ADR 0002 has gaps: releasable-weights licensing, closed-model provenance, registry, dedup unit | **Resolved** in [ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md) |
| 22 | Major | The plan's success criteria are vague and it bundles factors | **Resolved** in [plan v3](plan.md): baseline-anchored factorial with drop-ones, numeric criteria, pre-registered go/no-go |
| 23 | Major | Four figures are unused; three have defects | **Fixed.** Frontier rebuilt in 68c1019; all figures embedded with corrected captions in c399b9f |
| 24 | Minor | Internal inconsistencies | **Partly.** Table 11 has all rows (68c1019); the rest is open |
| 25 | Minor | Smaller faithfulness errors | **Partly.** AnyJev's "7 of 18" was fixed (68c1019); the rest is open |
| 26 | Minor | Notation collisions | **Partly.** The [start-here guide](README.md) fixes a notation table (cd6449c); aligning the survey is open |
| 27 | Nit | Naming, undefined terms, small wording issues | **Open** |
| §3 | Missing | The 12 missing pieces for a one-stop guide (reading paths, decision guide, worked example, glossary, disambiguation, baselines, benchmark card, evidence grades, contradictions register, history, hardware primer) | **Fixed** in cd6449c: the [start-here guide](README.md) |

## 03 · Architecture review: changes before locking

| # | Recommended change | Status |
|---|---|---|
| 1 | Rewrite "state and question towers" as one causal state⊕question pass with state-KV reuse; keep the towers as an ablation | **Adopted** in [ADR 0003](../adr/0003-architecture-resequenced.md), as exact state forking: the question reads the cached state |
| 2 | Make question-conditioned option caching an explicit ablation | **Adopted** as an M2 ablation ([plan v3](plan.md)) |
| 3 | Specify the boolean and ordinal heads: LM-row initialisation, no option text, routing for label-word choices | **Adopted** in [ADR 0003](../adr/0003-architecture-resequenced.md) |
| 4 | Train the coverage gate only on none-pairs from hard-label choice items; evaluate it only on open-set protocols | **Adopted** in [ADR 0003](../adr/0003-architecture-resequenced.md) |
| 5 | Make C causal by default and share state KV with the fast path; `prefix_bidir` becomes an H5 ablation | **Adopted:** M1 is causal throughout and forks the state ([ADR 0003](../adr/0003-architecture-resequenced.md)) |
| 6 | Correct ADR 0001's Context: Jeva vs InternDecision; #13's unseen rows; "levels off around 4B"; NDM's mask; "all arms fit one GPU" | **Adopted:** [ADR 0003](../adr/0003-architecture-resequenced.md)'s Context records each correction |
| 7 | Remove the Inkling 2 Mini reader and Matryoshka from M1, and the Jev API from M0 (Inkling stays as an API readout and teacher) | **Adopted with one change by the owner:** Matryoshka and the Jev API are dropped; Inkling stays through Tinker *and* as an off-target H100 frozen-reader arm ([ADR 0003](../adr/0003-architecture-resequenced.md)) |
| 8 | Expand M0:<br>• add Jeva-4B, Jeff-Qwen-4B, Qwen3-Embedding and probes with an unseen split<br>• reproduce 59.3, and 41.4 through our own encoder<br>• add a parity test<br>• pin S1-Arena<br>• add a W2 quality suite | **Adopted** in [plan v3](plan.md) (M0.1–M0.5); the encoder truncation fix is in PR #41 |
| 9 | Replace the ladder with a baseline-anchored factorial plus drop-one ablations; calibration is orthogonal; start the LoRA pilot in parallel | **Adopted** in [plan v3](plan.md) |
| 10 | Pre-register the M1 go/no-go against the calibrated same-backbone readout and a W2 latency ratio | **Adopted** in [ADR 0003](../adr/0003-architecture-resequenced.md) |
| 11 | ADR 0002: leave-source-out calibration fits, specified shingling, hashed label sets, a clean subset, flagged contaminated references | **Adopted** in [ADR 0004](../adr/0004-data-calibration-and-decontamination-amendments.md) |
| 12 | Plan: per-GPU latency targets, an FP8 W8A8 path, an M1 budget of about $200–400 and about 3 months to an M1 decision | **Adopted** in [plan v3](plan.md); spend is approved at each milestone's start |

## Code defects found during review

| Defect | Found by | Status |
|---|---|---|
| The inference encoder truncated long inputs from the wrong end. CLM and S1-Arena keep the *last* 2,048 tokens | Architecture review, §4.1 | **Fixed** in PR #41 (`configure_tokenizer`, with a regression test) |
| ECE counted exact ties between the gold option and a negative as hits | Architecture review, §4.3 item 11; technical review #4(d) independently flagged the symptom (anomalous lowest-confidence bin) | **Fixed** in PR #41 (strict `gold > max(negatives)`, with a regression test). The calibration numbers in the survey will be re-measured in M0 |
| The README's findings tile showed our own rounded numbers (51.7 / 69.2 / 68.6) as the published reference; the CLM README reports 52.1 / 69.2 / 68.5 | Found while applying fact-check corrections | **Fixed** in 8abc4b2 ("reproduced within 0.4 pp") |

## Library sweep (2026-11-22): gaps in the survey

A [web and literature sweep](library-sweep.md) run alongside the reviews found work the survey does not yet cover. Its papers are now indexed (family `unclassified` until read); the survey text still has to absorb them.

| Gap | Status |
|---|---|
| A competing survey, "The Hitchhiker's Guide to System 1 Models" (2610.33775): cite it and state how ours differs | **Open** |
| Missing systems: Fastino GLiDE (2611.26509), Decision 1.0 (2611.18629) and vLLM-SR Decision 3.0, Baseten Mammoth-1 and Elephant-1, Perplexity Decider v1.3, Gemini Decision 1, Microsoft-Decision-1.1, Liquid d1.2, firelex/jeff, Valen, Karpo, Piccolo | **Open** |
| Artificial Analysis Decision Index v2/v2.1, now quoted by vendors, is missing from the benchmark tables | **Open** |
| vLLM, SGLang and llama.cpp ship System One endpoints; the serving section does not mention them | **Open** |
| Direct evidence on per-option vs joint scoring (Joint or Per-Option?, 2610.30970; MetaEncoder, 2610.11316) and a frozen-LM bi-encoder (2611.23832) bears on H1 and Candidate A | **Being read** before the architecture lock |
| A missing failure mode: option *names* overriding option definitions (2609.26758, 2610.02586) | **Open** |
