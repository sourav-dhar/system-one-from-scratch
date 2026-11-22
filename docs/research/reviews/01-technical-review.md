> **Technical review of survey v1.0, ADRs 0001-0002 and plan v2.** Independent review commissioned on 2026-11-22 before locking the architecture. Reviewers received the documents, not the author's conclusions. Published verbatim; how each finding was addressed is tracked in [`../review-response.md`](../review-response.md).

# Review: System One survey, ADRs 0001–0002 and plan v2

**Key to references** (no files were modified, no git commands run):
- `L<n>` = C:\system-one-from-scratch\docs\research\survey.md
- `ADR1:<n>` = C:\system-one-from-scratch\docs\adr\0001-architecture-late-interaction-two-path.md
- `ADR2:<n>` = C:\system-one-from-scratch\docs\adr\0002-data-licensing-and-decontamination.md
- `plan:<n>` = C:\system-one-from-scratch\docs\research\plan.md
- `<system>:<n>` = C:\system-one-from-scratch\docs\research\systems\<system>.md
- `rawNN:<n>` = C:\system-one-from-scratch\docs\research\notes\raw\NN-*.md
- Data: C:\system-one-from-scratch\results\benchmark\heads.json
- Script: C:\system-one-from-scratch\scripts\research_figures.py
- Figures: C:\system-one-from-scratch\docs\research\figures\

**How I checked.** I read the survey, the ADRs, the plan, the script, heads.json and all five figures in full. I also read the CLM, Jeva, NDM, Jeff and Small Heads notes, and raw notes 01, 03, 07 and 08. Three parallel read-only passes checked every remaining claim against the other 26 system notes and raw notes 00, 02, 04, 05 and 06. I re-opened every quote that a critical or major issue rests on; a few minor citations come only from those passes.

## 1. Overall assessment

**Strengths.**
- **Sourcing** is unusually careful for a field ten weeks old:
  - All 69 citation keys resolve.
  - Inferences are mostly marked and unread sources are declared.
  - Appendix A logs what was re-verified.
- **Numbers.** The large majority of individual numbers match their notes. Every number in Table 6 and the §4.5 truncation figures matches heads.json.
- **Framing.** The W1/W2 framing, the cost-and-cacheability lens, the failure-mode catalogue and the project's own CLM measurements are real contributions.

**Main weakness.** The synthesis layer (abstract conclusions, the fix matrix, the hypotheses, the ADRs) is more confident and more one-directional than the notes it sits on. Caveats the notes state explicitly were dropped on the way up:
- Jeva's comparison is confounded.
- NDM's AUROC result is mixed.
- Bongard's stage 3 is cross-entropy (CE), not RL.
- Temperature scaling has side effects.
- Kev's held-out refit failed, and its long-context fix did not transfer.
- Several benchmarks have validity problems.
- Fine-tuned classifiers beat Jev.

Almost every dropped caveat cuts against the chosen design. The consequences:
- An abstract causal claim that includes a non-cause and omits two confounds.
- A "who fixed whom" table whose evidence column mostly measures flaws, not fixes.
- A taxonomy whose families overlap and that misplaces vegaml, NDM and Bongard.
- Hypotheses with no decision thresholds.
- An accepted ADR whose central "separate state and question towers" is not what either cited system (Jeva, NDM) does, with backbone arms not shown to fit an L4, A10 or L40S.

Most of this can be fixed by propagating what the notes already say.

## 2. Issues (by severity)

### Critical

**1. ADR B's question-blind state tower is not what the cited evidence did.**
- **Where:** ADR1:33 "Encode the state and the question **separately** and fuse them late. This removes CLM's suffix collapse." The same idea appears in L871 (H1), L897 (Candidate A) and L763 (Table 10 row 2, "Separate h_S and h_Q, fused late (NDM)").
- **Problem:** Neither cited system encodes the question without the state.
  - Jeva's request tower encodes type, instructions and state together.
  - NDM pools h_S and h_Q separately, but its question block attends to the state, which is its ancestor in the tree.
  - A question-blind state vector removes the state⊗question interaction that L68 says W1 accuracy depends on ("reading the state *in light of the question*").
  - So Jeva's 78.30 says nothing about B's towers, and "removes CLM's suffix collapse" is untested.
- **Evidence:**
  - raw07:212 "z_0 = P_x E_θ(T_x(t,ι,x))"
  - raw03:108 "the request template holds decision type, question and state"
  - raw07:446 "a token sees only its own block and its ancestors"
  - ndm:42
  - jeva:250 warns against predicting frozen-backbone results from Jeva.
- **Fix:**
  - Compute h_Q as a continuation over the cached state KV, as NDM does: pool h_S at the end of the state, pool h_Q after question tokens that attend to the state, and fuse [h_S; h_Q]. This keeps state reuse and charges only question tokens per question.
  - Add a ladder rung comparing (a) question-blind towers, (b) a state-aware h_Q and (c) Jeva's joint (t, ι, x) template.
  - Reword H1, §11 A and Table 10 row 2 to match.

**2. Abstract conclusion 1 claims causes that the evidence does not establish.**
- **Where:** L34 "The gap traces to specific, fixable causes: … an uncalibrated scale saturated at its cap …". ADR1:15 repeats it: "The causes are specific".
- **Problem:**
  - (a) A rescaled softmax cannot change the argmax on choice or noul items. The cap therefore explains Brier (0.749 vs 0.715, clm:143), not 41.4 vs 59.3. Score items would be affected only if graded on a rounded expectation, and no note says they are.
  - (b) No experiment on S1-Arena isolates any of these causes. The paper gives "no explanation of why CLM fails" (raw03:21). The listed causes come from CLM issues on BoolQ, Banking77 and Emotion. §10 H1 correctly treats the same claim as untested.
  - (c) Two confounds are omitted:
    - S1-Arena gave CLM only its last 2,048 tokens, while the 59.3 readout saw full inputs (raw03:18 "No other model's input is shortened").
    - CLM never trained on typed or rubric data (raw03:314, marked inference). This matches its per-domain pattern: Pro −31.0, Batch −26.5, Social −24.7, but Route only −4.4 (clm:285).
- **Fix:**
  - Rewrite the conclusion as "consistent with" and list candidate causes, including training data and truncation.
  - Move the scale to a Brier finding and point to H1.
  - Add M0 arms that run CLM with full context and with tail truncation, and report the share of S1-Arena items longer than 2,048 tokens.

**3. Table 10 ("who fixed whom") mostly cites evidence of the flaw, not of the fix.**
- **Where:** §8.2 (L758-780), the table behind contribution 5 (L81), which ADR1 relies on. No cell carries a citation. Several fixes are undemonstrated or contradicted.
- **Row 1.**
  - "Typed 78.30 vs 43.05" compares an adapted 4B trained on an unnamed curriculum of about 1.1M items with a frozen, zero-shot 8B. §3.2 itself says the contributions "cannot be separated".
  - InternDecision-4B, a joint model, scores 78.70 on the same Typed test (jeva:154). That test is probably among its training sources.
- **Row 2.**
  - "0.958 vs 0.452" measures the flaw; no fix result exists.
  - sbert2s1 puts the question on the option side (raw07:483), so its option cache is keyed by (question, option) and needs K new embeddings per question.
  - "Yes" and "one extra short embedding per question" describe NDM only.
- **Row 3.**
  - "0.000 → 1.059" splices two measurements: 0.000 is for short labels, and the trailing-period fix goes 0.256 → 1.059 (L720, L1001).
  - Bongard's contextual readout is a joint path, so "Yes" is wrong for it.
- **Row 4.** Our 0.065 → 0.013 is one global scalar fitted in-distribution, not a "per-type temperature, fit on held-out data".
- **Row 7.** The fix is marked "(inference)" in raw07:750. Kev-0.8B, also a fine-tuned checkpoint, flips 14.3 per 100 (raw07:612).
- **Row 8.** Clef has neither code rows nor a pointer; it uses a 4-layer joint-schema head (clef:15-19). Its 94.2 is an internal run.
- **Row 10.** "+17–20 pp" omits that "buried-synthetic-state training did not transfer to real documents (PLAN #11)" (kev:88).
- **Row 11.** The flaw is listed as fixed, while its own evidence cell says the fixes lost.
- **Row 12.** 2.3% → 56.6% comes from a λ = 0 vs 0.004 control on a loss weight, not from centring (bongard:105, :68).
- **Row 14.** #741 has no seeds; "'soft-CE is better' is not established" (laya:193). RLCD had the best ECE.
- **Row 15.**
  - "Kev v1" appears in no note.
  - 0.137 → 0.067 is a refit on the target workload.
  - Kev's registered held-out-dataset refit "failed its adoption rule" (kev:111).
- **Row 17.**
  - SanSi's 72.0 trails Qwen3.5-4B (73.8) and Kev-4B (74.3) on SanSi's own suite (sansi:88-89).
  - Beveridge's recursion adds +0.0 (raw07:517).
  - Beveridge is not cited.
- **Fix:**
  - Split the evidence into "flaw measured" and "fix measured (same system? controlled?)".
  - Add a status column: demonstrated, partial, proposed or failed.
  - Cite every cell.
  - Re-derive rows 1–4, 7, 8, 10–12, 15 and 17.

### Major

**4. Our own reproduction and calibration claims overstate the result and lack provenance.**
- **Where:** L360 "reproduces the published heads exactly: 51.73 / 69.16 / 68.63% on n = 191,161"; L936 "to four digits"; L38 "0.065 … reduced to 0.013 by one fitted scalar"; L566.
- **Problem:**
  - (a) The README reports 52.1 / 69.2 / 68.5 on "about 100K held-out questions" (clm:109). The CLM note itself says the reproduction is "within 0.4 points" (clm:229).
  - (b) How the 191,161 items relate to the authors' held-out split is never stated. If they are not the authors' held-out documents, the heads were trained on some of them.
  - (c) The temperature is fitted on chunk 0028 and evaluated on chunk 0029 of the same distribution. That is the held-out-items fit §5.3 says does not transfer, and the abstract omits the caveat.
  - (d) The lowest-confidence reliability bin is anomalous in all three heads. heads.json gives mid n = 409, confidence 0.104, accuracy 0.467; pre 443/0.105/0.424; post 409/0.105/0.447. Near-uniform predictions that are right 42–47% of the time suggest a tie-break artifact (gold at a fixed index with first-index argmax) or duplicate candidates. The same harness is meant to validate every reference model (§12).
- **Fix:**
  - Say "within 0.4 pp, different n".
  - Document the split by doc_id.
  - Randomize the gold position, break ties randomly, and report the tie count.
  - Add a cross-dataset fit before headlining 0.013.
  - State the estimator (15 equal-width top-label bins) and the fitting objective.

**5. P3 states the IIA trade-off wrongly and contradicts conclusion 2.**
- **Where:** L125 "IIA buys order invariance and cacheability."
- **Problem:**
  - Independent scoring is sufficient for both properties but not necessary. Jeva is set-dependent yet exactly equivariant, and it caches option K/V (Prop. 3; L35).
  - What IIA actually saves is the per-request set block.
  - Prop. 1 is an existence result only (raw07:292).
- **Fix:** rewrite the sentence and add the existence caveat.

**6. Four equations are wrong or overstated.**
- **(a) L129, the null formula.**
  - It holds only if all K options share the score s. It is exact in general if s is replaced by s̄ = τ·log((1/K)Σᵢ e^{sᵢ/τ}). The symbol s also collides with the state.
  - "A coverage gate … does not have this bias" needs a caveat. Prop. 2 covers only options added below the current maximum (raw07:239). The expected maximum of K distractor scores still rises with K, which is a different K-drift.
- **(b) L425, the Bongard justification.**
  - "Expected REINFORCE gradient is half the negative Brier gradient" is false in general.
  - Take a ~ q = softmax(z), y ~ t and reward r = 1[a = y]. Then ∇_z E[r] = J·t, while −½∇_z E‖q − e_y‖² = J·(t − q), with J = diag(q) − qqᵀ.
  - They differ by J·q = ∇½‖q‖², which is zero only when q is uniform on its support.
  - Even if the identity held, it would justify a Brier loss, not the CE the survey describes.
  - Quote the paper's exact statement and conditions, or drop the identity.
- **(c) L317, AnyJev.**
  - "Exact under an additive position-bias model": cyclic rotations of one base order cancel only absolute-position effects, and shuffles change each option's neighbours.
  - The residual flips (0.138, and 11.6 under Runoff's protocol, raw07:630) show the model fails in practice.
  - The label prior enters with exponent 0.75 (anyjev:14).
- **(d) L418, "the explanation … is exact".**
  - The policy-gradient term estimates the gradient of a Gaussian-smoothed reward that also includes spherical (weight 0.75) and RPS terms.
  - Soft-CE matches only the log-score part, and only as σ → 0.

**7. Four equations are unsourced or incomplete.**
- **(a) L352, InfoNCE.**
  - The README objective has no same-document mask (raw07:85).
  - The only mask is finetune.py's, over rows sharing (task_id, step_idx) (raw07:104).
  - The survey nonetheless attributes D(i) to pre- and mid-training.
- **(b) L406, RLCD advantage.** The code divides by the batch std of the centred reward, std(r − r̄) (laya:115-117), not by std_batch(r).
- **(c) L309, vegaml.**
  - E_k is never defined.
  - The source defines E_k = ‖z_T − c_k‖²/(2σ_k²) − log a_k, with z_T the settled particle (vegaml:89).
  - P_k is therefore set-dependent.
- **(d) L225, Jeva.** The survey omits the attention weights a_i^h (cosine attention with τ_ℓ initialised at 0.15) and α = τσ(γ)/√L (raw07:216-218).
- **(e) Table 1.** It drops the √256 scalings and Kev's biases that §3.3 and §3.5 include.

**8. The taxonomy's families overlap, and several placements contradict the notes.**
- **vegaml.** Figure 1 (L160-163) puts F7/vegaml under "Options scored independently (IIA)". But "Options are read **jointly** with the state and question in one sequence" (vegaml:38), and its wells interact through z_T (see 7c).
- **NDM.** It is labelled F2, yet it lacks the cross-request option caching that defines F2 (L236; ndm:53). Table 2 hedges with "F2/F4".
- **SemIf and AnyJev.** They appear both in F3 (L256) and in F8 (L313-317).
- **Unclassified or misfiled systems.**
  - AnyS1 and NDM have set interaction but sit outside F2.
  - Beveridge and this-that are not classified at all, and Table 2 has no F8 row.
  - GLiNER2.5-Decide is listed as a Laya derivative (L284-287), but it belongs to the GLiNER2 line (gliner-decide:3,7).
- **L13, "each defined by its scoring equation".** F3, F6 and F8 share the letter readout.
- **Arrows.** No paper made the F2→F3 move; Kev predates Jeva.
- **Fix:** use two orthogonal axes plus attributes.
  - Axis A, interaction/cache pattern: independent cached options, late interaction over cached options, per-request tree/prefix, schema-first prefix, fully joint.
  - Axis B, readout: cosine, pointer/bilinear, LM-head letters/codes, marker MLP, energy.
  - Attributes: frozen, LoRA, full fine-tune or training-free; null mechanism; order-invariance mechanism.

**9. Table 1's caching columns are wrong.**
- **F1 state reuse "per request".** CLM needs a state pass per question (clm:74), mitigated only by vLLM prefix caching.
- **F2 "per request".** Jeva never caches the request side (raw07:227).
- **F5 "yes (encoder output)".**
  - From stage 3 the encoder reads the question instructions, so "A hit requires identical II" (bongard:43).
  - Also, "adding other questions to a request changes answers" (bongard:151). That is a coherence failure missing from §7.
- **F3 "Option reuse: no", and L276.** this-that's schema-first prefix is reused across states (this-that:30).
- **Fix:** correct these cells and add schema-first reuse as a W2-relevant pattern.

**10. Calibration is presented as free, but the notes record trade-offs.**
- **Where:**
  - L576 "at no accuracy cost". This holds trivially, since rescaling cannot move the argmax.
  - Table 10 row 4 lists the cost as "Negligible".
  - L388 "Adding a Brier term or label smoothing is the cheapest calibration lever".
- **Evidence:**
  - **Jeff (zero-shot Gemma-31B):** choice calibration 58.6 → 84.0, but score competence 64.5 → 54.1 and exact-probability items 76.3 → 52.5 (jeff:74).
  - **Phocinae:** ECE 0.2519 → 0.0168, but Brier 0.0207 → 0.1218 (phocinae:19).
  - **Strands:** "Calibrating score changes its answer" (strands-decider:75).
  - **Kev:** calibration losses "produced no candidate" (kev:118).
  - **AnyS1:** the survey's own table shows CE+Brier 0.148 vs CE 0.146.
  - **Jeff loss ablation:** "after temperature scaling" is uncertain (jeff:79), and "[@jeff, App. G]" (L394) points to the order-invariance appendix (jeff:117).
- **Fix:**
  - Add a paragraph on the costs of calibration.
  - Rewrite L388: label smoothing helped in AnyS1, and Brier helped in SanSi but not in AnyS1 or Kev.
  - Fix the citation.

**11. The evidence for K-aware calibration is misattributed, and the ADRs adopt contradicted fitting rules.**
- **Where:** Table 5 (L528, L531); H3 (L878) "Laya, NDM and decider use K"; ADR1:40; ADR2:29 "DecisionBank by group_id, Nemotron by doc_id".
- **Evidence:**
  - NDM uses a "per-primitive temperature in [0.25, 10]" with no K (ndm:67).
  - Laya's fine-tunes get only type-level temperatures (laya:202, marked inference).
  - StartLux uses "One temperature per question type" and fuses its two passes as p = 0.25p⁽⁰⁾ + 0.75p⁽¹⁾. There is no separate post-reasoning temperature (startlux:53, :23).
  - Bongard does use per-K temperatures (bongard:90) but is missing from the table.
  - Kev found per-(type, K) temperatures worse out of distribution, and its held-out-dataset refit failed (kev:111).
  - ADR2's group_id and doc_id splits hold out items of the training sources, which §5.3 calls a failure mode.
- **Fix:**
  - Correct Table 5 and H3.
  - In ADR1, make K-conditioning a ladder rung rather than a decision.
  - In ADR2, fit on whole held-out source datasets and evaluate on different ones.

**12. §4.3's "RL earns its keep" examples are not RL, or have closed-form targets.**
- **Where:** L424-426.
- **Evidence:**
  - **Bongard stage 3** uses direct CE that "replaces REINFORCE" (bongard:79, :161).
    - Its targets are rollout frequencies or come from exact oracles.
    - 50.6 → 64.8 is a whole-stage effect, not an RL ablation.
  - **decider's RL stage** has four terms. One is a "Belief log-score against exact laws" (decider:62): a direct proper-score loss on a closed-form target. The notes do not credit PPO with the drop.
- **Omitted counter-evidence:**
  - "v1 (with CISPO) beats v2 on JevBench public and hard" (jeeves:71).
  - "RL hurts long computations: −5.9" (startlux:75).
  - sbert2s1's PG-LOO gains +2.5.
- **Fix:** recast Bongard and decider as evidence for direct proper-score losses. Soften conclusion 3 (L36) to "no controlled evidence that RL beats direct proper-score losses for single-step decisions".

**13. The adaptation and scale claims do not follow from the numbers.**
- **Where:**
  - L467 "levels off around 4B".
  - H6 (L881) "Small Heads (4B ≈ 8B …) and Jeff's level-off".
  - ADR1:25 "beats a better head on a frozen backbone … levels off after".
  - L454, Table 4's "controlled ablation".
  - L471 "LoRA on an instruct checkpoint … the head is not the bottleneck".
- **Evidence:**
  - **Scale curve:**
    - Qwen3.5 zero-shot scores 76.8 at 4B, 82.2 at 9B and 86.5 at 27B: +9.7 after 4B (jeff:114).
    - Jeff's own data has frozen zero-shot 9B at 82.2 above the LoRA'd Jeff-Qwen-4B at 79.3 (jeff:90). That directly contradicts H6.
    - Small Heads LoRA-adapts every size (small-heads:41), so "4B ≈ 8B" compares adapted with adapted.
  - **Head-only arm:** Jeff's head-only arm uses the same head, not a better one. Its 78.9 ± 2.5 vs 79.6 is within noise.
  - **Table 4's rows are not controlled:**
    - Kev-27B's "full FT" release is a 0.85·SFT + 0.15·v1 blend on new data (kev:82).
    - Kev's same-data full fine-tune, which "gained nothing and raised ECE by about 0.045" (raw03:182), is omitted.
    - The decider row compares a zero-parameter stock readout with a different, trained backbone.
  - **Instruct default:** base, instruct and thinking checkpoints tie within 0.25 (raw07:403), and AnyS1 and Kev both train -Base checkpoints.
  - **Head matters:** a linear slot scores 0.6619 vs a pointer's 0.7250, and "half the gain is having option representations at all" (strands-decider:56-61).
- **Fix:**
  - Rewrite these claims.
  - Test H6 at matched data and state the counterexample.
  - Drop "controlled" from Table 4's caption.

**14. Strong baselines and benchmark-validity caveats are omitted, and the protocol inherits the gap.**
- **Fine-tuned DeBERTa beats Jev.** "Fine-tuned DeBERTa wins nearly everything: D1 0.793, CLINC 0.953, emotion 0.938, vs Jev 0.732 / 0.913 / 0.504" (raw07:611).
  - The notes recommend logistic-regression and DeBERTa baselines (raw07:779).
  - Neither appears in §12 or plan M0.5.
- **TypedBench validity.** "DistilBERT scores 1.000 (states are templated); 273/500 phishing states duplicated" (raw07:597). The survey still uses it for FM4, FM5, L517, L558 and L965.
- **JevAdvBench labels.** "68% of the labels are Jev's own stable answers; adaptive search was run only on Jev" (raw07:686). FM9 compares models without this caveat.
- **Jeff's OpenDecisions score.** Jeff's 89.0 has "recipe settings … selected on OpenDecisions", and its teacher is also an OpenDecisions panel model (jeff:128). L272 and L331 report it without this.
- **DeepSWE.** Train and eval share all 113 tasks (raw04:64), and 31/38 vs 27/38 is a 4-task difference (L71).
- **S1-Arena.**
  - It cut Laya to 512 tokens (raw03:18).
  - Its released data (92,125 questions, 16 domains) differs from the paper's 78,362 and 13 (raw04:84).
  - Jev, Vega and MS-Decision-1 give no answer on 13.2–13.7% of single steps (raw07:573).
- **Fix:**
  - Add a "Baselines to beat" section and add those baselines to §12 and M0.5.
  - Add a validity column to Table 9.
  - Caveat FM9, L272 and L71.

**15. The W2 case and the latency frontier are one-sided.**
- **Where:**
  - L71.
  - L818 "No published system occupies it".
  - L819 "Cached designs (F1/F2) are the only route to single-digit milliseconds under reuse".
  - Table 12, B "~1 ms revisit / ~30 ms new state".
- **CLM is weak on W2 too:**
  - /v1/rank over 29,852 commands scores 0/20 vs BM25's 8/20, with hubness on 9/20 (clm:275-277).
  - Zero-shot, Jev matches or beats CLM on BFCL (99.2 vs 95.2) and WikiRacing (30/30 vs 26/30).
  - The T-Rex parity "comes mostly from the safety harness": agreement 0.658 vs 0.987, shield interventions 4,883 vs 28 (clm:184-189, :298).
- **Single-digit ms outside F1/F2:** NDM at 0.8–4.9 ms per extra question (L799), Laya at 4.6 ms (L829), Beveridge at 4.9 ms on an H200 (raw07:520).
- **The "empty region" is unmeasured, not empty:**
  - Jeff-Large runs at p50 44 ms on an H200 (jeff:121).
  - TypedBench measured Kev-4B at p50 208 ms (raw07:595).
  - Hosted rows include network time.
- **Jeva's measured latency** is 46.6 and 75.1 ms on an H200, and the request side is never cached (raw07:227, :282-285). That does not support Table 12's "~1 ms revisit".
- **Fix:**
  - Add this counter-evidence.
  - Replace "empty" with "unmeasured" and drop "only route".
  - Rebuild Table 12 from measured latencies.
  - State the hardware for every latency.

**16. Metric definitions are unspecified, so the tables mix conventions.**
- S1-Arena uses Brier on a 0–2 scale and top-label ECE with 10 equal-mass bins (raw07:551). Our heads.json uses 15 equal-width bins.
- Laya #741's Brier of 0.0501 at 80.5% accuracy implies a per-option mean (inference).
- Jeva reports a binary Noul Brier, and Table 6 uses the 11-way sum.
- TypedBench reports ECE/null, and L558's "value of calibration" is never defined.
- **Fix:** define the ECE estimator and Brier normalisation in §2.2, tag each table with its convention, and fix one estimator, with CIs, in §12.

**17. The single-pass trajectory shortcut (§6.2) breaks under CLM's layout, and plan M1.1 relies on it.**
- **Where:** L615 "one forward pass over a trajectory yields every step's state embedding"; plan:32.
- **Problem:**
  - The shortcut holds only if each step's text is an exact prefix of the trajectory and pooling happens at the end of that prefix.
  - CLM appends "\n\n{instructions}" after the state (clm:60), wraps the state in a chat template, and serves only the tail 2,048 tokens.
- **Fix:** put nothing after the state, pool at a sentinel token, match the truncation, and validate against per-step encodings on a sample.

**18. The failure-mode catalogue overstates its provenance, and some items are mismeasured.**
- **Provenance.** L15 says the modes were quantified by "independent evaluations", but FM6, FM11 and FM12 are self-reports.
- **FM6 (L690) may measure truncation.** The `laya` checkpoint has a 512-token context (laya:23, :88), so evidence at the end of 7,000 filler tokens would be cut off. The checkpoint used is not recorded.
- **FM2.**
  - "Fine-tuned checkpoints flip 4.2–6.5" omits Kev-0.8B's 14.3 (raw07:612).
  - L292's "polarity flips at confidence 1.0" is actually a constant negative answer given with label words (laya:312), and the evidence is Laya-only.
- **FM13.** vegaml's "unbound" detector over-rejects normal inputs; it does not accept out-of-scope ones.
- **FM11.** It compares Kev-4B without reasoning against Jeeves v2 (9B) with reasoning (jeeves:46, :53).
- **P5 (L133).** It omits that Jev's 0.064 beats Qwen3.8-27B's 0.293 and 0.122 (raw07:662).
- **FM1.**
  - Its table mixes protocols that the notes call "not comparable with each other" (laya:256).
  - Von's 49.5% → 0 is a version-to-version change (von:39).
- **Fix:**
  - Add a provenance column.
  - Re-check FM6.
  - Correct FM2, FM11 and FM13.
  - Add a protocol column to FM1.

**19. ADR 0001's hardware feasibility is unestablished, and arm C's mask conflicts with the hybrid backbone.**
- **Where:**
  - ADR1:49 "All arms fit one mid-range GPU".
  - ADR1:56, the Inkling arm.
  - ADR1:45, which puts C's slow path on the *same* backbone under prefix_bidir.
- **Inkling 2 Mini:**
  - The notes say only "H100/H200 verified; vLLM needs the inkling-2-mini docker image" (inkling:138), and "on Hopper, Inkling's NVFP4 falls back to BF16 dequantization" (inkling:163).
  - 26.4B parameters in BF16 is about 53 GB.
  - Native FP4/MX formats are Blackwell-only: Ada (L4, L40S) has FP8, Ampere (A10) has neither. This is general knowledge, not from the notes.
  - Layers 6–18 run 3×, so hidden-state hooks fire three times (inkling:34).
  - "Frontier" overclaims: the note says "not a guaranteed upgrade over Qwen3-8B" (inkling:213).
- **prefix_bidir on the hybrid backbone:**
  - L269 itself says block masks cannot be honoured on Gated DeltaNet layers.
  - Jeff's bidirectional mode flips only the full-attention layers (jeff:23).
  - prefix_bidir has no ablation (strands-decider:121).
  - Small Heads found bidirectional attention costs 3.5–6.0 points in-distribution and 5.4–11.8 out of distribution (raw07:405). H5 omits this.
- **Fix:**
  - Run a feasibility spike on an L4 and an L40S before committing.
  - Drop the A10 or its FP8 step.
  - Specify how prefix_bidir maps onto DeltaNet layers, or use a non-hybrid backbone for C.

**20. ADR 0001 lacks a training specification and decision rules.**
- **Missing pieces:**
  - The loss for each stage (InfoNCE? soft-CE? coverage BCE? Noul/Score auxiliaries?).
  - The data for each stage.
  - Acceptance targets, such as what share of the 41.4 → 59.3 gap B must close at a matched backbone.
  - The trigger for starting milestone 2.
- **Missing alternatives:**
  - Replicate Jeva as published.
  - An NDM-style tree-mask encoder.
  - A schema-first joint decoder.
  - Classifier baselines.
- **Context errors.** Context 1 restates the causes as fact (issue 2), and Context 4's "better head" is the same head (issue 13).
- **Fix:** add Training and Acceptance-criteria sections.

**21. ADR 0002 has gaps on licensing, closed-model provenance and the registry.**
- **Release vs. licences.** ADR2:19 permits non-commercial and unlicensed data for research runs, but plan M3 ships a public v0.2.0. No note says publishing weights trained on these is allowed:
  - NVIDIA data: "solely for the purpose of internal training" (raw04:17-18).
  - DecisionBank: "Do not use commercially" (raw04:58).
  - kev-suites: no licence (raw04:59).
  - OpenDecisions: "do not use for training" (raw04:85).
  - The ungated mirror's licence tag looks inconsistent with NVIDIA's terms (raw04:19).
- **Closed-model provenance.** ADR2:30 says "never depend on closed-model terms", but:
  - The CLM negatives replayed in M1.2 were generated by Gemini 2.5 Flash-Lite (raw04:21-23).
  - The teachers behind typed-decisions and DecisionBank's 60,219 LLM-written items are unnamed (raw04:60; laya:172).
- **Registry gaps.** It omits JevAdvBench (used in the robustness battery), the kev-suites test splits, and the MMLU-Pro, CSQA, ARC, HellaSwag and BFCL v4 overlaps (raw04:67-72).
- **Dedup spec.**
  - The "13-gram overlap" rule (ADR2:27) cannot fire on short items.
  - The matching unit is undefined; option-only matches such as yes/no would flag everything.
  - MinHash parameters are missing.
  - Decision 4's "source-level" rule is illustrated with item-level examples.
- **Fix:**
  - Allow-list what may go into releasable checkpoints.
  - Mine hard negatives instead of replaying Gemini outputs (raw04:128).
  - Define the dedup unit and tokenizer.

**22. The plan's success criteria are vague, and it breaks its own one-factor rule.**
- **Success criteria:**
  - M0.4's "within noise" (plan:25) is undefined for a deterministic model. It must also pin the S1-Arena version (issue 14).
  - M1 has no budget and no exit criteria.
  - The hypotheses have no thresholds: H1 "most", H5 "most of", H7 "little loss", H8 "nearly lossless", H9 "dominates … cost-adjusted accuracy".
  - The evidence for H9 already cuts against it: System One Auto scores 82.25 against Vega's 87.88.
- **One-factor rule:**
  - Rungs 3, 5 and 7 each bundle several factors, against plan:12.
  - A cumulative ladder measures order-dependent marginal effects; add leave-one-out runs from full B.
  - Rung 4 has no hypothesis.
- **Battery and hardware:**
  - M0.2 omits negation pairs, framing ranges and adversarial subsets, which §12 and H10 list.
  - The FP8 validation in M1.4 cannot run on an A10.
- **Fix:** set numeric acceptance criteria per step and per hypothesis, and split the bundled rungs.

**23. Four of the five figures are never used, and three have defects.**
- **Unused.** Only Figure 2 is referenced, by path only (L822). clm_calibration, clm_reliability, clm_truncation and order_flips are neither embedded nor cited.
- **clm_reliability.png.** Half the image is blank: the long footnote (research_figures.py:182-189) widens the tight bounding box.
- **s1arena_frontier.png:**
  - It shows 7 of the 9 System-1 rows without saying so (raw08:9).
  - It draws the 59.3 readout as a line with no latency.
  - It omits the closed-LLM ceiling, Opus 5.5 at 81.1 / 0.232 (raw07:568).
  - It does not mark the 512- and 2,048-token input cuts.
- **Captions and titles:**
  - order_flips says "compare within rows", but each row holds one value; it should say "do not compare across rows".
  - The clm_calibration title should say the split is in-distribution.
  - clm_truncation does not say whether truncated vectors were renormalised.

### Minor

**24. Internal inconsistencies:**
- **CLM ECE:** 0.065 (L38, Tables 6 and 10) vs 0.064 (L514); these are chunk 0029 vs the full test.
- **CLM θ:** 4.6132 (L199) vs 4.6423 (L1006); name the checkpoint for each.
- **Kev-4B hard-v1 ECE:** 0.084 (L526) vs 0.137 (L536); the notes cannot reconcile them.
- **Jeva parameters:** "about 9.4M" (Table 10) is the block alone, while 13.37M includes the projections.
- **Kev stage 4:** "Skills and devtools" (L479) vs "hard/devtools" (L589).
- **System count:** Table 11 has 7 rows, while L71 says "nine".
- **Dead cross-references:** "§7.1/§7.7/§7.14" (L210, L276, L471, L820) do not exist; they mean FM1/FM7/FM14.
- **Kev-4B memory:** 14.3 GB vs the "8–14 GB" row (L858).
- **Kev sizes:** App. A says "0.5B–27B", but kev:3 lists 0.8B/4B/9B/27B.
- **"± 2.1" at L385** is in no note.
- **vega-U range:** 0.765–0.811 vs "0.738–0.811 … four readers" (vegaml:18).

**25. Smaller faithfulness errors:**
- **L11 counts:** "forty implementations / two dozen evaluation papers" vs "about 40 … the prior study missed, plus 20 new arXiv papers", which mixes model and evaluation papers (raw02:1). L62's count is not in the notes.
- **L50, "three types":** Jev 1.14 adds `box` (raw00:70).
- **L336, pplx:**
  - "Full SFT" is documented for v1 only (pplx-decider:26-28).
  - The DI version is unstated: 61.56 here vs 62.25 elsewhere.
- **L263, Decision 2.0:** one row per question by default; tree packing is opt-in (decision2:31-32).
- **L265, Kev:** isolated options see "state, instruction and itself" (kev:22); the −5.8 pp comes from the archived PLAN (kev:93).
- **L267:**
  - Jeff's bidirectional mode also runs on Gemma-4-12B (jeff:23).
  - pplx loses 3.6 on WinoGrande when served causally (pplx-decider:44). That is not "small".
- **L305, SanSi:** 72.0 − 58.4 = 13.6, not 13.5 (sansi:110).
- **L317, AnyJev:** "7 of 18" is an H100 serving measurement, not Table 1 (anyjev:61).
- **L333, StartLux:** no note names Qwen as the backbone (startlux:12).
- **L338, Laya:** 0.766 has "no committed result file" and exceeds the 0.735 teacher ceiling (laya:234-235).
- **L408, Laya:** a blog post is cited as [@laya-repo], and it describes the unshipped base run (laya:132).
- **L426, StartLux:** RLCD² rewards post-reasoning improvement over the direct answer (startlux:44).
- **L469, soups:** this framing is the notes' inference, and Kev lists checkpoint averaging and WiSE-FT (−4.4 pp) as negatives (kev:96).
- **L517, Jev:** an MMLU ECE has no over/under direction, so it cannot settle the calibration dispute.
- **L593, Strands v2:** trained with "no teacher, no KL" (strands-decider:66).
- **L607, negatives:** tagged "nvidia-open-data-license" (raw04:56).
- **L733:** "rules out pure IIA" is an inference, not a source finding (raw00:84).
- **L736:** no note links Strands to Kev.
- **L748:** NDM's link to CLM is the notes' own inference (ndm:53, :146).
- **L798, Bongard:** the speed-up is on a 1B-1B model and needs identical instructions (bongard:43).

**26. Notation collisions.** These need a notation table:
- s is used for the state, for Jeva's scores, for CLM's state embedding, and for the score in P4.
- x is the state in Jeva.
- T is both a temperature and a loop count.
- τ is both a temperature and a risk threshold (L560).
- λ is both CLM's scale and Bongard's loss weight.
- E is both the encoder and the energy.
- z is both the logits and Jeva's state vector.

### Nits

**27.**
- **Naming:** the model is "CLM-v0.1-8B" (raw08:5).
- **Abstract:** its failure-mode list is not in FM order.
- **L834:** the HF post's lower binary-quantization figure is *without* rescoring.
- **L840:** "≈20%" is 21% if layers are 1-indexed and 17% if 0-indexed.
- **L558:** define "−1,220" as cost(argmax) − cost(Bayes) = 1,080 − 2,300 (raw03:326).
- **Undefined terms:** "Nox", "Tinker" and the name "Noul".
- **Table 2:** its caption promises DI versions, but pplx's is missing.
- **Jeff's tournament mode:** it "was not evaluated systematically" (jeff:21).
- **OpenDecisions licence:** the wording is "do not use for training".
- **JevBench:** "300 public" comes from Parallel; the cited repo holds 231 public items (raw04:86).
- **vegaml refit:** "≥10 examples required, 30–100 recommended".

## 3. Missing content for a one-stop guide (in priority order)

1. **A start-here page.**
   - Reading paths for newcomers, builders and researchers.
   - A one-page decision guide that maps workload, K, reuse, latency budget and null needs to a family and readout.
2. **A worked example.**
   - One `/v1/systemone` request with a choice, a noul and a score question (JSON in and out).
   - Trace it through CLM, Jeva and a letter readout with toy numbers.
   - Steps: logits → softmax → temperature → coverage gate → score expectation.
3. **A notation table and a full glossary.** The current glossary has 7 entries. Missing terms include:
   - W1/W2, IIA, equivariance, coverage vs adequacy.
   - InfoNCE, RLCD, PG-LOO, CISPO, RPS.
   - ECE estimators, ECE/null, Brier normalisations, Learn-then-Test.
   - Matryoshka, Gated DeltaNet, radix cache, FP8/NVFP4/MXFP8.
   - State sensitivity, U1D0, Capability (I, C), DI versions, transfer-v4/breadth/hard-v1.
4. **A name-disambiguation box.**
   - Jev, Kev, Jeva, Jeff, Jeeves, JEV-27B, JevBench, JevAdvBench, Jev Decision Index.
   - Vega-27B vs vegaml vs vega-U.
   - Laya vs Laya 0.5.
5. **Diagrams.**
   - The five attention masks: causal rows, block/branch, option isolation, prefix_bidir, prefix tree.
   - Jeva's block and its caches.
   - ADR B/C data flow.
   - A dated lineage graph.
   - Embed the four unused figures.
6. **A "baselines to beat" section, with numbers:**
   - same-backbone zero-shot readout (59.3 on S1-Arena);
   - logistic regression or PCA-Ridge on frozen embeddings;
   - fine-tuned DeBERTa-v3-large;
   - BM25 for large catalogs.
7. **A benchmark reference card.** For each benchmark: version, size, scoring rule (and whether it is calibration-sensitive, as with OpenDecisions Capability and JevBench's 0.2–0.8 rule), licence, contamination, validity issues, and which systems were evaluated.
8. **Evidence grading on every table row:** self-reported, independently evaluated, or reproduced here. Also list weights and licences, for example StartLux CC BY-NC, Liquid's LFM licence, and Gemma terms for Bongard.
9. **An evaluation recipe.**
   - Exact protocols for flips, yes/no swaps, K-sweeps, null-by-gold-removal, the ECE estimator, latency reporting (hardware, cache state, p50/p95), seeds and CIs.
   - Map it to the evidence audit's checklist.
10. **A hardware primer for L4/A10/L40S.**
    - Memory arithmetic: CLM's 512-d fp16 cache is about 1 KB per option, while Jeva at d = 768 is about 4.5 KB per option including K/V.
    - Data types supported by each GPU generation.
11. **A register of open contradictions:**
    - pplx DI 61.56 vs 62.25;
    - Decision Index 110,201 vs 132,422 requests;
    - the S1-Arena paper vs its release;
    - Kev hard-v1 ECE 0.084 vs 0.137;
    - the missing condition on Von's band rule.
12. **A short dated history of the field since 15 September 2026.**
