# Kev

> **Family:** decoder-readout · **Team:** Jared Palmer · **Released:** repo created 2026-09-17; Kev 1.0 shipped 2026-10-01 per the README (the earlier landscape note dates the Kev-4B card to 2026-09-24) · **License:** Apache-2.0 · **Code/weights:** complete code (training, evaluation, serving, MLX, CUDA graphs, Modal); weights for 0.8B / 4B / 9B (LoRA adapters plus head on Qwen3.5-*-Base) and 27B v2 (full weights on post-trained Qwen3.8-27B); suites frozen in the repo or in `jaredpalmer/kev-suites`; the 27B training corpus is private (manifest only) · **Sources read:** README, `kev/model.py`, `kev/api.py`, `kev/train.py`, `kev/data.py`, `kev/shared_prefix.py`, `kev/calibrate.py`, `kev/cuda_graphs.py`, `kev/fused_qwen35.py`, current `PLAN.md` and the archived `research-archive-2026-09-24` `PLAN.md`, the kev-4b and kev-27b model cards. Only lightly read: `serve.py` batching and the archived PLAN.md history beyond the lines quoted. Not available: the 27B training corpus.

## Summary

Kev is an open, Apache-2.0 "Jev-like" decision-model family that speaks TypeSafe's `/v1/systemone` wire format; its architecture follows Archer Hume's third-party reverse-engineering of Jev ([repo](https://github.com/jaredpalmer/kev)) [@kev-repo] [@jev-hume]. State, question and options are read in one causal sequence, and each option is scored by a 256-dimensional pointer head that matches the option's closing token against the question's final `<decide>` token. The idea that matters most is **question branching off a shared state**: every question restarts its position ids right after the state and sees only the state and itself (via a block-causal mask, or, on Gated DeltaNet hybrids, as a separate row continuing from the cached state), so the state is encoded once and reused across questions and requests. The repository also documents a long list of negative results, notably that learning rate was the main lever and that an in-distribution temperature does not transfer to hard workloads.

## Architecture

**Backbones.**
- Kev 0.8B / 4B / 9B: LoRA adapters plus a pointer head on Qwen3.5-*-Base ([README](https://github.com/jaredpalmer/kev/blob/main/README.md) "Models") [@kev-repo]. Kev-4B's base has 32 layers, 24 of them Gated DeltaNet, hidden size 2560; the LoRA (33.8M parameters) is merged at load ([kev-4b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- Kev-27B v2: full weights on post-trained Qwen3.8-27B ([kev-27b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)).

**Delimiters** ([kev/model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)).
- Five reused Qwen special tokens: `<|fim_prefix|>` = state, `<|fim_middle|>` = question, `<|box_start|>` / `<|box_end|>` = option open / close, `<|fim_suffix|>` = decide. No new embedding rows are added.
- `user_tokens()` rewrites `<|x|>` in user text to `<¦x¦>`, so caller text cannot forge a boundary.

**Sequence layout** (`encode`). One record is
`[<state> S]` followed, for each question, by `[<q> instr <opt> o₁ </opt> … <opt> o_K </opt> <decide>]`.
- **Position-id trick:** every question branch restarts positions at `len(S)`, so all questions sit at identical positions after the state.
- **Option isolation (optional flag):** all option spans share the same position ids; each span sees only the state, the instruction and itself; `<decide>` is placed at `p0 + len(instr) + longest`.

**Attention mask** (`branch_mask_batch`). Token $i$ attends to token $j$ iff $j \le i$ and ($\mathrm{seg}[j] = 0$ or $\mathrm{seg}[j] = \mathrm{seg}[i]$), where segment 0 is the shared state *(inference from the layout)*. With option isolation, option tokens see only their own span, and `<decide>` sees everything in its question. On non-DeltaNet (attention-only) models this mask isolates questions inside one packed sequence ([repo](https://github.com/jaredpalmer/kev)).

**Gated DeltaNet backbones.** DeltaNet layers cannot honour a block mask. On hybrid Qwen3.5 (4B: 24 of 32 layers DeltaNet), each question therefore runs as **one causal row = state + its branch** (`rows_of`). The row form equals the packed form to fp32 noise (`test_rows_match_packed`). Serving copies the conv and recurrent states of the cached prefix per chunk (`_rows_hidden`), and `rows_per_pass` bounds memory: Kev-4B on MLX with a 4.8k-token state and 64 questions uses 24.7 GB in one pass vs 8.9 GB row by row ([kev/model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py)).

**Readout** (`PointerHead`, $d_p = 256$). With $q, k$ = `Linear(d, 256)` with bias and no normalisation ([model.py](https://raw.githubusercontent.com/jaredpalmer/kev/main/kev/model.py)):

$$z_k = \frac{(W_k\,h_{\langle/\mathrm{opt}\rangle_k} + b_k)\cdot(W_q\,h_{\langle\mathrm{decide}\rangle} + b_q)}{\sqrt{256}}, \qquad p = \mathrm{softmax}(z/T) \text{ over the } K \text{ options.}$$

The temperature is applied only at inference/eval. The head runs in fp32. Because `<decide>` is the last token of its branch, it sees every option. The 27B head has 2.6M parameters (27B card).

**Question types** ([kev/api.py](https://github.com/jaredpalmer/kev/blob/main/kev/api.py)). Noul becomes two options `[no, yes]`; Score levels become options and the reported score is the expected level; $K \le 255$.

**LoRA** (`DecisionModel.__init__`). $r$ = `lora`, $\alpha = 2r$, dropout 0.05, task `FEATURE_EXTRACTION`. Targets "all" = q/k/v/o and gate/up/down plus the DeltaNet `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`. Shipped: r16 / α32, 33.8M parameters on 4B ([kev-4b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).

**Context.** Training defaults: `MAX_STATE` 384, `MAX_BRANCH` 1024, `MAX_PACKED` 2048. Serving: up to 65,536 state tokens plus 8,192 per question (model.py constants). Validated context is 8,192 tokens below 27B (the 16k lower bound is −3.4 pp at 4B) and 65,536 for 27B after training on 32k states (model cards, as summarised in the decoder-family dossier; [repo](https://github.com/jaredpalmer/kev)). The landscape note phrases this as "served up to 65,536 tokens of state; validated at 8k".

| Property | Value |
|---|---|
| Passes per request | Attention-only backbones: one packed pass (state + all question branches). DeltaNet hybrids: one state pass, then one row per question continuing from the cached state (batched up to `rows_per_pass`). All shipped Kev sizes sit on hybrids (Qwen3.5; Qwen3.8-27B has 48 DeltaNet layers per the pplx-decider notes), so the row form is the one served *(inference)* |
| State prefix reusable | Yes (prefix cache; rows continue from the cached state) |
| Option embeddings reusable | No; options are encoded inside each question branch, conditioned on state and instruction *(inference)* |
| Max K | 255 |
| Context | Served: 65,536 state tokens + 8,192 per question. Validated: 8,192 (0.8B–9B), 65,536 (27B) |

## Training

**Loss** (`question_loss` in [kev/train.py](https://github.com/jaredpalmer/kev/blob/main/kev/train.py)).
- Cross-entropy over options, or soft cross-entropy when a soft target $t$ exists: $\mathcal{L}_{\mathrm{CE}} = -\sum_i t_i \log p_i$.
- At most one modifier: label smoothing, Brier, or focal.
- Optional ordinal CDF term: $\mathcal{L}_{\mathrm{ord}} = w_{\mathrm{ord}} \cdot \mathrm{mean}\big((\mathrm{cumsum}\,p - F_{\mathrm{obs}})^2\big)$.
- Optional anchor KL to the frozen base model's zero-shot distribution, and optional symmetric permutation KL between two option orders.
- Optimiser: AdamW, OneCycle (10% warm-up), gradient clip 1.0, separate head learning rate.

**Augmentation** ([kev/data.py](https://github.com/jaredpalmer/kev/blob/main/kev/data.py) `augment`, `none_pair`).
- Choice options are always permuted.
- `p_none` = 0.1: remove the true option and add a "none of the above" variant as gold (13 phrasings).
- `p_none_distract` = 0.12: add a none option as a wrong answer.
- `p_distract` = 0.15: add an absurd distractor (e.g. "The colour purple").
- **None-pair minimal pairs:** the same state and question rendered twice with identical option order, once with the true option present (none is wrong) and once with it removed (none is right). A quarter of Choice records also produce a none-pair (`p_none_pair` 0.25) ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).
- Soft-target questions are only permuted.

**Kev-4B four-stage curriculum** (card "Training procedure").

| Stage | Data | Settings |
|---|---|---|
| 1 | decision-v7: 12,576 records (10,000 from ten public datasets at 1k each; 896 policy minimal pairs; 1,680 from 60 random rule trees) | 2 epochs, LR 5e-5, effective batch 8; a quarter of Choice records also produce a none-pair |
| 2 | Dates and missing evidence: 1,425 records | 1 epoch at 2e-5, replay 2,000 |
| 3 | CFPB documents: 5,219 records (states ≤7,552 tokens per the methods note) | replay 2,000 |
| 4 | hard-v1 (6,000) + devtools-v1 (5,320) | replay 4,000, 1,915 steps |

- Compute: 56 min on an H100 for the base stage; 1.4 h on an H200 for skills.
- "No output of Jev ... was used" (card).

**Kev-27B v2** ([27B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)).
- Full-weight fine-tune of Qwen3.8-27B on 145,840 records (337,130 questions), states ≤32,768 tokens.
- 8×H200 with FSDP2; LR 2e-6 (backbone) and 1e-4 (head); 128 records per step; 1,140 steps; 129 GPU-hours.
- Labels come from code or from open models (GLM-5.3, DeepSeek-V4-Pro, Inkling, Mistral Large 3, gpt-oss-120b, MiMo-V2.6-Pro).
- Decontamination by 8-gram Jaccard > 0.2 removed 6,388 records.
- The released checkpoint is **weight-averaged 0.85·SFT + 0.15·v1 (merged LoRA)**.

**Shared-prefix training** ([kev/shared_prefix.py](https://github.com/jaredpalmer/kev/blob/main/kev/shared_prefix.py)). The state runs once with gradients flowing through the DeltaNet conv and recurrent state; this matches the row form to about 1e-5 in fp32.

**What the research log found** (current [PLAN.md](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md) and the [archived PLAN.md](https://github.com/jaredpalmer/kev/blob/research-archive-2026-09-24/PLAN.md) "Overnight synthesis").
- **The lever was learning rate:** 2e-4 → 5e-5 gave +4.7 pp [+0.4, +9.6] by reducing drift. LR 2e-4 for 2 epochs at 4B "collapsed (dev 0.58)".
- **Buried evidence:** placing the state among 1–4k tokens of unrelated records cost 22–43 pp; training on such states recovered +17 to +20 pp. However, buried-synthetic-state training did not transfer to real documents (PLAN #11).
- **Diminishing data returns:** the first 6k hard-v1 records gave +26 pp; the next 12k gave +5 pp.
- **Soft targets:** ambiguity soft targets beat hard labels (Brier −0.011, accuracy +1.1).
- **Data vs method:** full fine-tuning on Kev's own data gained nothing and raised ECE by about 0.045; broad data gave +9.1 pp. Separately, "more public data" is listed as a negative (raises dev, flattens transfer). The notes report both without reconciling them.
- **None-pairs:** none_present 0.85 → 0.93 at 4B.
- **Option isolation:** exact flip rate 0.0 at 0.6B with "no accuracy gain", but "option isolation at low lr 0.729 (−5.8 pp, significant)" at 4B. (The methods note, which read only the current PLAN.md, lists option isolation as negative "with no deltas given"; the delta comes from the archived PLAN.md.)
- **No gain from:** special embeddings, head_dim 1024, perm_kl, ord_w, 3 epochs, rank or target ablations.
- **Question-side LoRA** cost −3.4 to −6.0 pp at 4B/9B.
- **Also negative:** DeltaNet-frozen LoRA; the post-trained 9B base; Qwen3.6-35B-A3B (+1.2 pp, worse calibration, 8× memory); checkpoint averaging (row 4.5); 9B → 0.8B self-distillation (row 4.11); WiSE-FT (α 0.5: −4.4 pp); anchoring to base answers ("not a lever").
- **Fine-tuning erodes base capability:** MMLU −11 pp at 8B; PAWS −14 pp at 0.6B. The skill is lost in the representation, not at the readout: date arithmetic is 0.53 via the LM head and 0.53 via the pointer.

## Calibration

**Shipped method.** One global temperature per checkpoint, fit by NLL: 4B T = 2.41, 9B 2.19, 0.8B 2.35, 27B 1.32 ([README](https://github.com/jaredpalmer/kev/blob/main/README.md) "Kev 1.0"). The 4B temperature was fitted in-distribution on 1,264 decision-v7 dev rows. Its ECE ([kev-4b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)):

| Panel | ECE |
|---|---|
| breadth-v1 test | 0.029 |
| transfer-v4 locked test | 0.017 |
| hard-v1 test | 0.084 |
| documents-v1 test | 0.101 |

**Refit and tooling.**
- A registered refit on held-out datasets gave T = 2.30 and failed its adoption rule (card "Calibration"). The methods note separately reports that fitting on held-out *datasets* gave breadth ECE 0.0085, and that fitting T on held-out items of the *training* sources is a declared failure mode ([PLAN.md](https://raw.githubusercontent.com/jaredpalmer/kev/main/PLAN.md)).
- `kev.calibrate` reports raw, shipped, workload and out-of-fold arms for deployer refits ([kev/calibrate.py](https://github.com/jaredpalmer/kev/blob/main/kev/calibrate.py)).

**Documented calibration failures** ([PLAN.md](https://github.com/jaredpalmer/kev/blob/main/PLAN.md) "What we have learned" #5, #13).
- An in-distribution T does not transfer to hard workloads. Served hard-v1 ECE is 0.137 / 0.073 / 0.047 for 4B / 9B / 27B, vs 0.067 / 0.034 / 0.041 after refit.
- **Per-(type, K) temperatures made calibration worse:** out-of-distribution Score items have unseen K, and grouped scaling scrambles the cross-group confidence ranking (archived PLAN.md row 2a).
- A logistic reliability head made AURC worse: 4B 0.0583 → 0.0709 (row 4.10).
- Calibration losses (label smoothing, CE + Brier, focal) produced no candidate (round 3).
- Kev-27B v2 is overconfident on CUAD long contracts: ECE 0.053 vs 0.007 for v1 (27B card).

**Third-party calibration findings.**
- Rafe & Das: stored temperatures break at large K; Kev-9B's ECE goes from 0.051 at T = 1 to 0.377 as shipped, at 150 options ([2610.00346](https://arxiv.org/html/2610.00346)) [@rafedas].
- TypedBench: Kev-9B's "value of calibration" (cost(argmax) − cost(Bayes)) is −3,180, i.e. Bayes decisions from its probabilities cost more than argmax ([2610.11392](https://arxiv.org/html/2610.11392)) [@typedbench].
- Calibration & Risk-Budgeted Decisions: "Kev-ft" (9,988 records, 2 epochs) certifies 6/6 datasets under Learn-then-Test ([2610.19811](https://arxiv.org/html/2610.19811)) [@riskbudget].

## Results

**Kev-4B (author-reported, [kev-4b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).**

| Suite | Kev-4B | Jev |
|---|---|---|
| transfer-v4 locked test (acc / Brier) | 0.838 / 0.224 | – |
| transfer-v4 dev ("new sources") | 0.817 | 0.857 |
| breadth-v1 test (14 held-out sets) | 0.690 | 0.757 |
| Breadth index | 38.0 | 54.0 |
| MMLU-Pro | 0.565 | 0.840 |
| Unanswerable items answered at p ≥ 0.9 | 0.00 | 0.09 |
| `deadline` date arithmetic | 0.65 | 0.95 |

(The transfer-v4 dev row is from the landscape note citing the same card. The landscape note's "0.648 vs 0.817 on new sources" for Kev-0.8B vs Kev-4B appears to refer to the same dev split *(inference)*.)

**Kev-27B (author-reported, [27B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)).**

| Suite | Kev-27B v2 | Kev-27B v1 | Jev | AutoJev |
|---|---|---|---|---|
| Breadth index | 52.3 | 50.2 | 54.0 | 50.0 |
| Locked transfer-v4 (acc / Brier) | 0.8887 / 0.154 | – | – | – |
| CUAD long-contract ECE | 0.053 | 0.007 | – | – |

- README dev/test table: Kev-27B "New sources" 0.851 / 0.889 vs Jev 0.857 (dev) ([README](https://github.com/jaredpalmer/kev/blob/main/README.md)).
- Size: Kev-0.8B lags Kev-4B badly on new sources, 0.648 vs 0.817 ([repo](https://github.com/jaredpalmer/kev)).
- Confident errors: Kev-9B's confident-error rate on new sources fell from 8.2% to 2.4% (Jev 3.7%) ([README](https://github.com/jaredpalmer/kev)).

**Third-party evaluations that include Kev.**

| Evaluation | Setup | Kev | Comparators | Source |
|---|---|---|---|---|
| Jeva Table 1 | Typed Decisions test | Kev-4B 66.00 | Jeva-4B 78.30; InternDecision-4B 78.70; CLM-8B 43.05 | [Jeva](https://arxiv.org/html/2610.18773) [@jeva] |
| Jeva Table 1 | BANKING77, K = 16 | Kev-4B 82.47 | Jeva-4B 91.62; CLM-8B 18.15 | [Jeva](https://arxiv.org/html/2610.18773) [@jeva] |
| Jeva null detection | BANKING77 K = 16, 3,080 requests (588 answer-absent), AUROC | 93.61 | Jeva 98.66; InternDecision-4B 84.27; JevK5 79.44 | [Jeva](https://arxiv.org/html/2610.18773) [@jeva] |
| SanSi Table 2 | SanSi suite | Kev-4B (SanSi authors' data) 74.3 | SanSi 72.0; SanSi-2.6B 75.8; Qwen3.5-4B 73.8 | [SanSi](https://arxiv.org/html/2610.07730) [@sansi] |
| SanSi App. Table 14 | released Kev-4B, overall / sources no SanSi model trained on | 68.2 / 72.0 | Jev 78.9 / 83.5 | [SanSi](https://arxiv.org/html/2610.07730) [@sansi] |
| AnyS1 Table 4 | Any-DP Dev | Kev-4B 57.8 | AnyS1 83.8; standard pointer 82.7; Jev 78.4 | [AnyS1](https://arxiv.org/html/2611.23751) [@anys1] |
| TypedBench | (row as given in the notes) | Kev-9B 0.565; value of calibration −3,180 | Jev priority accuracy 0.874 [0.708, 0.946] | [TypedBench](https://arxiv.org/html/2610.11392) [@typedbench] |
| Rafe & Das | Yes/no swap, flips per 100 | Kev-0.8B 14.3; Kev-9B in the 4.2–6.5 group | Jev 50.5; Laya 91.3; Gemma 17.2 | [2610.00346](https://arxiv.org/html/2610.00346) [@rafedas] |
| Rafe & Das | Out-of-scope acceptance at a held-out threshold | Kev-9B 0.015 | Jev 0.310 | [2610.00346](https://arxiv.org/html/2610.00346) [@rafedas] |
| Rafe & Das Table 10 | Cascade first stage | Kev-0.8B escalates 99–100% of decisions | this-that → Jev on CLINC 0.927 at 0.43 of Jev's cost | [2610.00346](https://arxiv.org/html/2610.00346) [@rafedas] |
| ProbabilityBench | Exact-probability test bed | Kev appears only in the price table | – | [2610.17016](https://arxiv.org/html/2610.17016) [@probabilitybench] |
| Jeeves README | "Test overall" (Jeeves' suite; reasoning when confidence < 0.9 for Jeeves) | Kev-9B 0.822 | Jeeves v2 0.905; Jev 0.857 | [Jeeves](https://github.com/PostHog/jeeves) [@jeeves] |

The landscape note also records SanSi reporting Jev 11.5 pp above released Kev-4B "per search summary, not verified in full text"; the later full read (72.0 vs 83.5 on the unseen-source subset) is consistent with that gap.

## Serving and efficiency

**Kev-4B** ([card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)): H100, bf16, fused fla Triton kernels plus CUDA graphs.

| Workload | New state | Cached state |
|---|---|---|
| 6 questions, short state (H100) | 18.1 ms | 12.9 ms |
| 5 questions, 2,200-token state (H100) | 89.4 ms | 22.5 ms |
| L40S | 41.5 ms | 27.7 ms |

- Throughput 100.8 req/s at 64 clients; 14.3 GB resident.
- Served path vs fp32 eval path: max $|\Delta p|$ 0.017, no changed answers.
- The landscape note states Kev-4B fits on a 24 GB GPU ([repo](https://github.com/jaredpalmer/kev)).

**Kev-27B on H200** ([27B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) "Serving parity"): a 64k-token state takes 9.4 s new / 733 ms cached; 65.5 GB resident; 36.4 req/s at 64 clients.

**Kernels and batching.**
- [cuda_graphs.py](https://github.com/jaredpalmer/kev/blob/main/kev/cuda_graphs.py) batches the state passes and row passes of all waiting requests with a fixed-layout state bank.
- [fused_qwen35.py](https://github.com/jaredpalmer/kev/blob/main/kev/fused_qwen35.py) fuses the DeltaNet q/k/v/z/b/a GEMMs, the conv, the gated delta rule and RMSNorm.
- Memory: row-by-row execution on MLX cuts Kev-4B memory from 24.7 GB to 8.9 GB for a 4.8k-token state with 64 questions.
- Deployment: `/v1/systemone` server and a one-command Modal deploy ([repo](https://github.com/jaredpalmer/kev)). Kev cites Hydragen, DeFT and FIRST as related serving/readout work [@hydragen] [@deft] [@first].

**Third-party latency.** TypedBench reports Kev 0.8B / 4B / 9B p50 of 62 / 208 / 325 ms ([2610.11392](https://arxiv.org/html/2610.11392)) [@typedbench]; the notes do not record TypedBench's hardware or protocol, so these are not comparable to the card's numbers.

## Flaws and failure modes

**(a) Flaws of Kev itself**
- **Option order can change an answer** (4B card, Limitations). The decoder-family cross-table groups Kev's and Strands' pointer shifts as "small but not zero" (~0.016 distribution shift; the 0.016 figure is from Strands' architecture.md). The exact fix, option isolation, cost −5.8 pp at 4B (archived PLAN.md).
- **Knowledge gap vs Jev:** MMLU-Pro 0.565 vs 0.840; `deadline` date arithmetic 0.65 vs 0.95 (4B card). The research log attributes this to fine-tuning eroding base capability (MMLU −11 pp at 8B; PAWS −14 pp at 0.6B) in the representation, not the readout ([PLAN.md](https://github.com/jaredpalmer/kev/blob/main/PLAN.md) #13).
- **Long context validated only to 8,192 tokens below 27B**; the 16k lower bound is −3.4 pp at 4B. Buried evidence costs 22–43 pp, and synthetic buried-state training did not transfer to real documents (PLAN #11).
- **In-distribution temperature does not transfer:** served hard-v1 ECE 0.137 (4B) vs 0.067 after refit; documents-v1 ECE 0.101.
- **Stored temperatures fail at large K:** ECE 0.051 → 0.377 at 150 options for Kev-9B ([2610.00346](https://arxiv.org/html/2610.00346)) [@rafedas].
- **27B v2 is overconfident on CUAD** long contracts (ECE 0.053 vs v1 0.007) (27B card); the notes record no cause.
- **Small sizes are weak first stages:** Kev-0.8B escalates 99–100% of decisions in a cascade ([2610.00346](https://arxiv.org/html/2610.00346)), and lags Kev-4B 0.648 vs 0.817 on new sources.
- **Training fragility:** LR 2e-4 for 2 epochs at 4B collapsed (dev 0.58).
- **Negative value of calibration** for Kev-9B in TypedBench (−3,180) [@typedbench].

**(b) Flaws of other systems that Kev claims to fix**
- **Cross-question interference:** question isolation means one question cannot influence another (README "Flaws and fixes").
- **Confident answers on unanswerable items (Jev):** Kev 0.00 vs Jev 0.09 at p ≥ 0.9 (4B card); confident errors on new sources Kev-9B 2.4% vs Jev 3.7% (README).
- **Independent corroboration:** Rafe & Das find Kev-9B accepts 0.015 of out-of-scope requests vs Jev's 0.310, and Kev-9B flips 4.2–6.5 per 100 on yes/no swap vs Jev 50.5 ([2610.00346](https://arxiv.org/html/2610.00346)) [@rafedas].
- **Training-data provenance:** Kev explicitly used no Jev outputs (4B card); the landscape note reads this as a signal that distilling from Jev is a sensitive point *(inference, as in the note)*.

## Implications for our design

All bullets are *(inference)*, grounded in the evidence above and in the notes' own "what it means for us" sections.

**Adopt**
- **Minimal-pair "none" augmentation** (none_present 0.85 → 0.93 at 4B) and Kev-style absurd distractors; these are data-side changes that do not touch an option cache.
- **Buried-evidence training states** for long inputs, while validating on *real* documents, since synthetic burial did not transfer.
- **Staged curriculum with replay** and a **low learning rate** (LR was Kev's lever: +4.7 pp).
- **Temperature fit on held-out datasets, not held-out items**, plus a deployer refit tool with out-of-fold reporting (`kev.calibrate`).
- **A shared-state branch layout** (position-id restart + block mask, or per-question rows continuing a cached state on DeltaNet) for any joint path we add; the 4B card shows 89.4 → 22.5 ms with a cached 2,200-token state.
- **Kev-4B as the open, small, `/v1/systemone`-compatible baseline** (fits a 24 GB GPU per the landscape note).

**Avoid**
- **Per-(type, K) temperatures and reliability heads without out-of-distribution validation** (both were Kev negatives). This is in tension with the methods note's recommendation to fit $T(\text{type}, \log K)$; Rafe & Das's K = 150 failure argues that *some* K dependence is needed, but it must be validated on unseen datasets and unseen K.
- **Exact option isolation as a default** unless a 2–6 pp cost is acceptable.
- **Assuming more same-distribution data transfers** (dev up, transfer flat).
- **Note on caching:** Kev's pointer head is a joint, non-cacheable path for options (option representations depend on the state), so it cannot replace a cached-option fast path; the methods note names a Jeva-style late-interaction design as the cacheable analog [@jeva].

## References

- [@kev-repo] https://github.com/jaredpalmer/kev — README, PLAN.md, model cards ([kev-4b](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md), [kev-27b](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)), [model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py), [api.py](https://github.com/jaredpalmer/kev/blob/main/kev/api.py), [train.py](https://github.com/jaredpalmer/kev/blob/main/kev/train.py), [data.py](https://github.com/jaredpalmer/kev/blob/main/kev/data.py), [shared_prefix.py](https://github.com/jaredpalmer/kev/blob/main/kev/shared_prefix.py), [calibrate.py](https://github.com/jaredpalmer/kev/blob/main/kev/calibrate.py), [cuda_graphs.py](https://github.com/jaredpalmer/kev/blob/main/kev/cuda_graphs.py), [fused_qwen35.py](https://github.com/jaredpalmer/kev/blob/main/kev/fused_qwen35.py), [archived PLAN.md](https://github.com/jaredpalmer/kev/blob/research-archive-2026-09-24/PLAN.md)
- [@kev-suites] https://huggingface.co/datasets/jaredpalmer/kev-suites
- [@jev-hume] https://archerhume.com/posts/jevs-architecture-unmasked/
- [@hydragen], [@deft], [@first] (works cited by Kev)
- [@jeva] https://arxiv.org/html/2610.18773
- [@sansi] https://arxiv.org/html/2610.07730
- [@anys1] https://arxiv.org/html/2611.23751
- [@typedbench] https://arxiv.org/html/2610.11392
- [@rafedas] https://arxiv.org/html/2610.00346
- [@riskbudget] https://arxiv.org/html/2610.19811
- [@probabilitybench] https://arxiv.org/html/2610.17016
- [@jeeves] https://github.com/PostHog/jeeves
