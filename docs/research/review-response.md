# Response to the independent reviews

Before locking the architecture, on 2026-11-22, we commissioned three independent reviews of the survey, ADR 0001, ADR 0002 and plan v2. Reviewers received the documents, not the author's conclusions. All three are published verbatim:

| Review | Scope | Headline |
|---|---|---|
| [01 · Technical review](reviews/01-technical-review.md) | Survey, ADRs and plan, checked against all 31 system notes and the raw notes | 27 issues (3 critical, 20 major), plus 12 items of missing content |
| [02 · Fact-check](reviews/02-fact-check.md) | 40 survey claims checked against primary sources (papers, code, issues, model cards) | 27 verified, 7 minor mismatches, 6 context mismatches, **no wrong digits** |
| [03 · Architecture review](reviews/03-architecture-review.md) | Devil's-advocate stress test of ADR 0001 ("build B, designed for C"), ADR 0002 and plan v2 | "Modify and re-sequence, do not replace": 12 changes before locking |

This page tracks every finding.

**Status values**
- **Fixed** names the commit.
- **Partly** says what remains.
- **Open** is scheduled for survey v1.1.
- **At lock** changes an ADR or the plan and waits for the architecture decision.
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
| 1 | Critical | B's question-blind state tower is not what NDM or Jeva do (NDM's question attends to the state) | **At lock.** The recommended rewrite is one causal state⊕question pass with state-KV reuse, keeping towers as an ablation. Survey H1, Candidate A and Table 10 row 2 were **fixed** in 23bf3ac |
| 2 | Critical | The abstract asserts causes for CLM's gap; the scale cap explains Brier, not accuracy; the truncation and training-data confounds are omitted | **Fixed** in the survey (68c1019, 23bf3ac): candidate causes, not established ones; truncation and training-data confounds; the scale cap moved to Brier. M0 arms (full context vs tail cut) are **At lock** |
| 3 | Critical | Table 10 mostly cites evidence that a flaw exists, not that a fix works | **Partly.** Rows 5, 6, 7, 10, 11, 14 and 16 were revised in 68c1019, and rows 2, 4 and 9 in 23bf3ac. Restructuring (flaw measured / fix measured / status / citations) is open |
| 4 | Major | Reproduction and calibration claims lack provenance; the low-confidence bin is anomalous | **Partly.** The tie bug is fixed in PR #41; the wording ("within 0.4 pp", different sample) and the in-distribution and tie caveats are in 23bf3ac. Split provenance is open; randomised gold position and a cross-dataset fit are M0 |
| 5 | Major | P3 misstates the IIA trade-off | **Fixed** in 3516adf |
| 6 | Major | Four equations are wrong or overstated (null formula, Bongard REINFORCE identity, AnyJev "exact", soft-CE "exact") | **Fixed.** Bongard identity removed in 3516adf; null formula, AnyJev and soft-CE in 2e5b364 |
| 7 | Major | Four equations are unsourced or incomplete (InfoNCE mask, RLCD std, vegaml E_k, Jeva attention terms) | **Fixed.** RLCD std in 2e5b364; InfoNCE mask, vegaml E_k and Jeva attention terms in 46660c4 |
| 8 | Major | Taxonomy families overlap and several systems are misplaced | **Open.** Planned: two axes (interaction/cache pattern × readout) plus attributes |
| 9 | Major | Table 1's caching columns are wrong for F1, F2, F3 and F5 | **Fixed** in 7d6f736 |
| 10 | Major | Calibration is presented as free, but the sources record trade-offs | **Partly.** Table 10 row 4 and §5.6 state the costs (23bf3ac); the §4.2 objective wording and the Jeff locator are fixed in 3516adf; a dedicated §5 paragraph is open |
| 11 | Major | K-aware calibration is misattributed; the ADRs adopt fitting rules that the sources contradict | **Partly.** H3 was fixed in 3516adf; Table 5 is open. **At lock** for ADR 0001 and ADR 0002 |
| 12 | Major | The "RL earns its keep" examples are CE or closed-form proper-score losses | **Fixed** in 3516adf (conclusion 3, §4.3) |
| 13 | Major | The adaptation and scale claims do not follow (Qwen3.5 keeps rising after 4B; head-only arm within noise) | **Fixed** in 3516adf (Table 4 caption, §4.5, H6). **Declined in part:** the review's "frozen 9B 82.2 vs LoRA-adapted Jeff-Qwen-4B 79.3" compares accuracy on Jeff's seven held-out sets (App. F) with OpenDecisions Capability (Table 3), which are different benchmarks |
| 14 | Major | Strong baselines and benchmark-validity caveats are missing | **Partly.** The [start-here guide](README.md) adds baselines and a benchmark card; adding them to the M0 protocol is **At lock** |
| 15 | Major | W2 and the latency frontier are argued one-sidedly | **Fixed.** The "empty region" in 68c1019; W2 counter-evidence and measured latency anchors in Table 12 in 7d0ad66 |
| 16 | Major | Metric conventions (ECE bins, Brier scale) are unspecified | **Fixed** in e899364 (§2.2 conventions). The protocol estimator is confirmed **At lock** |
| 17 | Major | The single-pass trajectory shortcut breaks under CLM's layout | **Fixed** in 3516adf (§6.2). **At lock** (plan M1.1) |
| 18 | Major | Failure-mode provenance is overstated; FM6, FM11 and FM13 are mismeasured | **Mostly fixed.** FM2 in 68c1019; provenance, FM6, FM11 and FM13 in c39e048. A per-mode provenance column and FM1's protocol column are open |
| 19 | Major | ADR 0001 has unestablished hardware feasibility (Inkling 2 Mini; `prefix_bidir` on DeltaNet layers) | **At lock.** The recommendation is to drop the Inkling 2 Mini reader, make C causal and run a feasibility spike |
| 20 | Major | ADR 0001 lacks a training specification and acceptance criteria | **At lock** |
| 21 | Major | ADR 0002 has gaps: releasable-weights licensing, closed-model provenance, registry, dedup unit | **At lock** |
| 22 | Major | The plan's success criteria are vague and it bundles factors | **At lock** (pre-registered go/no-go, factorial ladder) |
| 23 | Major | Four figures are unused; three have defects | **Fixed.** Frontier rebuilt in 68c1019; all figures embedded with corrected captions in c399b9f |
| 24 | Minor | Internal inconsistencies | **Partly.** Table 11 has all rows (68c1019); the rest is open |
| 25 | Minor | Smaller faithfulness errors | **Partly.** AnyJev's "7 of 18" was fixed (68c1019); the rest is open |
| 26 | Minor | Notation collisions | **Partly.** The [start-here guide](README.md) fixes a notation table; aligning the survey is open |
| 27 | Nit | Naming, undefined terms, small wording issues | **Open** |
| §3 | Missing | The 12 missing pieces for a one-stop guide (reading paths, decision guide, worked example, glossary, disambiguation, baselines, benchmark card, evidence grades, contradictions register, history, hardware primer) | **In progress:** [start-here guide](README.md) |

## 03 · Architecture review: changes before locking

| # | Recommended change | Status |
|---|---|---|
| 1 | Rewrite "state and question towers" as one causal state⊕question pass with state-KV reuse; keep the towers as an ablation | **At lock** (recommended) |
| 2 | Make question-conditioned option caching an explicit ablation | **At lock** (recommended) |
| 3 | Specify the boolean and ordinal heads: LM-row initialisation, no option text, routing for label-word choices | **At lock** (recommended) |
| 4 | Train the coverage gate only on none-pairs from hard-label choice items; evaluate it only on open-set protocols | **At lock** (recommended) |
| 5 | Make C causal by default and share state KV with the fast path; `prefix_bidir` becomes an H5 ablation | **At lock** (recommended) |
| 6 | Correct ADR 0001's Context: Jeva vs InternDecision; #13's unseen rows; "levels off around 4B"; NDM's mask; "all arms fit one GPU" | **At lock** |
| 7 | Remove the Inkling 2 Mini reader and Matryoshka from M1, and the Jev API from M0 (Inkling stays as an API readout and teacher) | **At lock** (recommended) |
| 8 | Expand M0:<br>• add Jeva-4B, Jeff-Qwen-4B, Qwen3-Embedding and probes with an unseen split<br>• reproduce 59.3, and 41.4 through our own encoder<br>• add a parity test<br>• pin S1-Arena<br>• add a W2 quality suite | **At lock** (recommended). The encoder truncation fix is in PR #41 |
| 9 | Replace the ladder with a baseline-anchored factorial plus drop-one ablations; calibration is orthogonal; start the LoRA pilot in parallel | **At lock** (recommended) |
| 10 | Pre-register the M1 go/no-go against the calibrated same-backbone readout and a W2 latency ratio | **At lock** (recommended) |
| 11 | ADR 0002: leave-source-out calibration fits, specified shingling, hashed label sets, a clean subset, flagged contaminated references | **At lock** (recommended) |
| 12 | Plan: per-GPU latency targets, an FP8 W8A8 path, an M1 budget of about $200–400 and about 3 months to an M1 decision | **At lock** (needs budget approval) |

## Code defects found during review

| Defect | Found by | Status |
|---|---|---|
| The inference encoder truncated long inputs from the wrong end. CLM and S1-Arena keep the *last* 2,048 tokens | Architecture review, §4.1 | **Fixed** in PR #41 (`configure_tokenizer`, with a regression test) |
| ECE counted exact ties between the gold option and a negative as hits | Architecture review, §4.3 item 11; technical review #4(d) independently flagged the symptom (anomalous lowest-confidence bin) | **Fixed** in PR #41 (strict `gold > max(negatives)`, with a regression test). The calibration numbers in the survey will be re-measured in M0 |
