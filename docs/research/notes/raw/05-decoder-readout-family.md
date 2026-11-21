I covered all 14 systems in the decoder-readout family from primary sources: the repo code and docs, the arXiv HTML, Hugging Face cards and remote code, and vendor pages. The things I could not read are listed in §4.

# Decoder-based System-One readouts: primary-source dossier (read 2026-11-21)

Conventions:
- "Card" means a model card.
- "(inference)" marks my reasoning, not a sourced fact.
- K is the number of options and T is the temperature.
- Numbers are copied exactly from the cited file or table.

---

## 0. Taxonomy at a glance

| System | Readout | Mask | Layout and passes | State KV reusable across questions? |
|---|---|---|---|---|
| **Kev** | New pointer head: q-proj(⟨decide⟩) · k-proj(⟨/opt⟩ₖ)/√256 | Block-causal packed (attention-only backbones); one causal row per question (DeltaNet); optional option isolation | `[<state> S][<q> instr <opt>o</opt>… <decide>]` per question | Yes (prefix cache; rows continue from the cached state) |
| **Jeff-1.1** | LM-head letter rows A–P plus a zero-init affine correction | Bidirectional over the prompt in full-attention layers (Gemma-31B stays causal) | One full prompt per question, 1 pass (≤17 passes if K>16) | No; `use_cache=False` |
| **Strands v1** | Pointer head (~1M params) | Causal | One prompt; state KV broadcast to question suffixes | Yes |
| **Strands v2** | 255-code token readout (LM-head-initialised) | `prefix_bidir`: the state is bidirectional within itself and never sees the question; each question is bidirectional within itself | State once, question suffixes | Yes, by construction |
| **Jeeves** | Kev-style pointer, read after an optional reasoning trace | Causal | Reason, then re-ask the question, then ⟨decide⟩ | Not stated |
| **decider (Mapika)** | LM-head letter rows (A…Z, two-letter codes up to 255) at one answer slot per question | Causal | Many slots in one prompt, or one row per question; optional schema-first cache | Optional |
| **StartLux** | LM-head letters A–Z | Instant: bidirectional in full-attention layers; AutoThink: causal | One pass per question; groups of ≤25 options with the top 3 advancing | Prevented by the bidirectional mask (paper §6) |
| **AnyJev L0** | Label-token logits averaged over K cyclic rotations, divided by a label prior | Causal (a "set" mask was tried and failed) | K prefills (adaptive about 7 of 18) | State prefix yes; question block not exact |
| **NanoJev** | Scalar per candidate "leaf", plus a set-attention residual | Causal | One path per candidate (the state is repeated) | Optional tree sharing |
| **Decision 2.0** | Endpoint pointer: bilinear plus MLP | Causal | One row per question; opt-in shared context | Opt-in, with a near-tie fallback |
| **pplx-decider v1.1** | 255-row readout mapped to vocabulary ids | Noncausal in the 16 full-attention layers | One pass | No (radix cache disabled) |
| **Liquid d1-3B** | Max over token forms per option at the answer slot | Causal LFM2 | One pass; packed states | Not stated |

---

## 1. Per-system dossiers

### 1.1 Kev (Jared Palmer)

**Identity**
- Repo created 2026-09-17; Apache-2.0; 10,076 stars ([repo](https://github.com/jaredpalmer/kev), via gh API).
- Kev 1.0 shipped 2026-10-01 in four sizes:
  - 0.8B, 4B and 9B are LoRA adapters plus a head on Qwen3.5-*-Base.
  - 27B v2 is full weights on post-trained Qwen3.8-27B ([README](https://github.com/jaredpalmer/kev/blob/main/README.md) "Models").
- The code is complete: training, evaluation, serving, MLX, CUDA graphs and Modal. Suites are frozen in the repo or in `jaredpalmer/kev-suites`.
- The 27B training corpus is private; only its manifest is public ([kev-27b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) "Training data").
- The architecture follows Archer Hume's "Jev's Architecture Unmasked" (README).

**Architecture** ([kev/model.py](https://github.com/jaredpalmer/kev/blob/main/kev/model.py))

*Delimiters.*
- Five reused Qwen special tokens serve as delimiters: `<|fim_prefix|>`=state, `<|fim_middle|>`=q, `<|box_start|>`/`<|box_end|>`=opt open/close, `<|fim_suffix|>`=decide. No new embedding rows are added.
- `user_tokens()` rewrites `<|x|>` in user text to `<¦x¦>`, so caller text can never forge a boundary.

*Encoding (`encode`).* The record is `[<state> S]`, then for each question `[<q> instr <opt> o₁ </opt> … <decide>]`.
- **Position-id trick:** every question branch restarts positions at `len(S)`, so all questions sit at identical positions after the state.
- **Option isolation (flag):** every option span shares the same position ids. Each span sees only state, instruction and itself. `<decide>` sits at `p0+len(instr)+longest`.

*Masks.* `branch_mask_batch`:
- A token attends to j iff j≤i and (seg[j]=0 or seg[j]=seg[i]).
- With option isolation, option tokens see only their own span. `<decide>` sees everything in its question.

*DeltaNet backbones.* Gated DeltaNet layers cannot honour a block mask. On hybrid Qwen3.5 (4B: 24 of 32 layers DeltaNet), each question therefore runs as **one causal row = state + its branch** (`rows_of`). The row form equals the packed form to fp32 noise (`test_rows_match_packed`).
- Serving copies the conv and recurrent states of the cached prefix per chunk (`_rows_hidden`).
- `rows_per_pass` bounds memory. Kev-4B on MLX with a 4.8k-token state and 64 questions: 24.7 GB in one pass vs 8.9 GB row by row.

*Readout* (`PointerHead`, dp=256):
- `z = (W_k h_opt) · (W_q h_decide) / √256`; at inference `z/T`; softmax over K.
- The head runs in fp32. 27B head: 2.6M params (27B card).

*Question types* ([kev/api.py](https://github.com/jaredpalmer/kev/blob/main/kev/api.py)):
- Noul becomes 2 options `[no, yes]`.
- Score levels become options; the reported score is the expected level.
- K ≤ 255.

*LoRA* (`DecisionModel.__init__`):
- r=lora, α=2r, dropout 0.05, task FEATURE_EXTRACTION.
- Targets "all" = q/k/v/o and gate/up/down, plus DeltaNet `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`.
- Shipped: r16/α32, 33.8M params on 4B ([kev-4b card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)).

*Context.* Training: MAX_STATE 384, MAX_BRANCH 1024, MAX_PACKED 2048 by default. Serving: up to 65,536 state tokens plus 8,192 per question (model.py constants).

**Training**

*Loss* (`question_loss` in [kev/train.py](https://github.com/jaredpalmer/kev/blob/main/kev/train.py)):
- Cross-entropy over the options, or soft cross-entropy −Σ tᵢ log pᵢ when a soft target exists.
- At most one optional modifier: label smoothing, Brier, or focal.
- Optional ordinal CDF term `ord_w·mean((cumsum p − observed CDF)²)`.
- Optional anchor KL to the frozen base's zero-shot distribution, and optional symmetric permutation KL between two option orders.
- Optimizer: AdamW, OneCycle (10% warmup), grad clip 1.0, separate head LR.

*Augmentation* ([kev/data.py](https://github.com/jaredpalmer/kev/blob/main/kev/data.py) `augment`, `none_pair`):
- Always permute Choice options.
- p_none=0.1: remove the true option and add a "none of the above" variant as gold (13 phrasings).
- p_none_distract=0.12: add a none option as a wrong answer.
- p_distract=0.15: add an absurd distractor (for example "The colour purple").
- **None-pair minimal pairs:** the same state and question rendered twice with identical option order, once with the true option present (none is wrong) and once with it removed (none is right).
- Soft-target questions are only permuted.

*Kev-4B 4-stage curriculum* (card "Training procedure"):
1. decision-v7: 12,576 records for 2 epochs. 10,000 come from ten public datasets at 1k each, 896 are policy minimal pairs, and 1,680 come from 60 random rule trees. LR 5e-5, effective batch 8, a quarter of Choice records also produce a none-pair.
2. Dates and missing evidence: 1,425 records, 1 epoch at 2e-5, replay 2,000.
3. CFPB documents: 5,219 records, replay 2,000.
4. hard-v1 (6,000) plus devtools-v1 (5,320): replay 4,000, 1,915 steps.
- Compute: 56 min on an H100 for the base stage; 1.4 h on an H200 for skills.
- "No output of Jev ... was used."

*Kev-27B v2* (card):
- Full-weight FT of Qwen3.8-27B on 145,840 records (337,130 questions), states ≤32,768 tokens.
- 8×H200 with FSDP2. LR 2e-6 for the backbone and 1e-4 for the head. 128 records per step, 1,140 steps, 129 GPU-hours.
- Labels come from code or from open models (GLM-5.3, DeepSeek-V4-Pro, Inkling, Mistral Large 3, gpt-oss-120b, MiMo-V2.6-Pro).
- Decontamination by 8-gram Jaccard >0.2 removed 6,388 records.
- The release is **weight-averaged 0.85·SFT + 0.15·v1 (merged LoRA)**.

*Shared-prefix training* ([kev/shared_prefix.py](https://github.com/jaredpalmer/kev/blob/main/kev/shared_prefix.py)):
- The state runs once with gradients flowing through the DeltaNet conv and recurrent state.
- It matches the row form to about 1e-5 in fp32.

**Calibration**
- One global temperature, fit by NLL: 4B T=2.41, 9B 2.19, 0.8B 2.35, 27B 1.32 (README "Kev 1.0").
- The 4B temperature was fitted in-distribution on 1,264 decision-v7 dev rows. Its ECE:

| Panel | ECE |
|---|---|
| breadth-v1 test | 0.029 |
| transfer-v4 locked test | 0.017 |
| hard-v1 test | 0.084 |
| documents-v1 test | 0.101 |

- A registered refit on held-out datasets gave T=2.30 and failed its adoption rule (card "Calibration").
- `kev.calibrate` reports raw, shipped, workload and out-of-fold arms for deployer refits ([kev/calibrate.py](https://github.com/jaredpalmer/kev/blob/main/kev/calibrate.py)).
- Documented failure modes ([PLAN.md](https://github.com/jaredpalmer/kev/blob/main/PLAN.md) "What we have learned" #5, #13):
  - An in-distribution T does not transfer to hard workloads. Served hard-v1 ECE is 0.137 / 0.073 / 0.047 for 4B / 9B / 27B, vs 0.067 / 0.034 / 0.041 after refit.
  - **Per-(type, K) temperatures made calibration worse.** Out-of-distribution Score items have unseen K, and grouped scaling scrambles the cross-group confidence ranking ([archived PLAN.md](https://github.com/jaredpalmer/kev/blob/research-archive-2026-09-24/PLAN.md) row 2a).
  - A logistic reliability head made AURC worse: 4B 0.0583→0.0709 (row 4.10).
  - Calibration losses (label smoothing, CE+Brier, focal) gave no candidate (round 3).

**Results**

*Kev-4B* (card):

| Suite | Kev-4B | Jev |
|---|---|---|
| transfer-v4 locked test (acc / Brier) | 0.838 / 0.224 | – |
| breadth-v1 test (14 held-out sets) | 0.690 | 0.757 |
| Breadth index | 38.0 | 54.0 |
| MMLU-Pro | 0.565 | 0.840 |
| Unanswerable items answered at p≥0.9 | 0.00 | 0.09 |
| `deadline` date arithmetic | 0.65 | 0.95 |

*Kev-27B v2* (card):

| Suite | Kev-27B v2 | Kev-27B v1 | Jev | AutoJev |
|---|---|---|---|---|
| Breadth index | 52.3 | 50.2 | 54.0 | 50.0 |
| Locked transfer-v4 (acc / Brier) | 0.8887 / 0.154 | – | – | – |
| CUAD long-contract ECE | 0.053 | 0.007 | – | – |

- The README dev/test table puts Kev-27B "New sources" at 0.851/0.889 vs Jev 0.857 (dev).

**Serving** (4B card)
- H100, bf16, fused fla Triton kernels plus CUDA graphs:
  - 6 questions, short state: 18.1 ms new / 12.9 ms cached.
  - 5 questions, 2,200-token state: 89.4 / 22.5 ms.
  - 100.8 req/s at 64 clients; 14.3 GB resident.
- L40S: 41.5 / 27.7 ms.
- Served vs fp32 eval path: max |Δp| 0.017, no changed answers.
- 27B on H200: a 64k state takes 9.4 s new / 733 ms cached; 65.5 GB resident; 36.4 req/s at 64 clients ([27B card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) "Serving parity").
- [cuda_graphs.py](https://github.com/jaredpalmer/kev/blob/main/kev/cuda_graphs.py) batches the state passes and row passes of all waiting requests with a fixed-layout state bank.
- [fused_qwen35.py](https://github.com/jaredpalmer/kev/blob/main/kev/fused_qwen35.py) fuses the DeltaNet q/k/v/z/b/a GEMMs, the conv, the gated delta rule and RMSNorm.

**Negative results** (archived PLAN.md "Overnight synthesis" plus current PLAN.md #13)
- **Option isolation** gives an exact flip rate of 0.0 at 0.6B with "no accuracy gain", but "option isolation at low lr 0.729 (−5.8 pp, significant)" at 4B.
- No gain from: special embeddings, head_dim (1024), perm_kl, ord_w, 3 epochs, rank or target ablations.
- **Question-side LoRA** cost −3.4 to −6.0 pp at 4B/9B.
- The DeltaNet-frozen LoRA, the post-trained 9B base and Qwen3.6-35B-A3B (+1.2 pp, worse calibration, 8× memory) did not help.
- Also negative: checkpoint averaging (4.5), 9B→0.8B self-distillation (4.11), WiSE-FT (α 0.5: −4.4 pp), anchoring to base answers (not a lever), more public data (raises dev, flattens transfer).
- LR 2e-4 on 2 epochs at 4B "collapsed (dev 0.58)".
- **The lever was LR**: 2e-4→5e-5 gave +4.7 pp [+0.4, +9.6] by reducing drift.
- **Fine-tuning erodes base capability**: MMLU −11 pp at 8B; PAWS −14 pp at 0.6B.
- The skill is lost in the representation, not at the readout: date arithmetic 0.53 via the LM head and 0.53 via the pointer.

**Flaws and fixes**
- Flaws:
  - "Option order can change an answer" (4B card, Limitations).
  - The knowledge gap.
  - Long-context validated only to 8,192 tokens below 27B (16k lower bound −3.4 pp at 4B).
  - 27B v2 is overconfident on CUAD.
- Claimed fixes: question isolation (one question cannot influence another), and abstention on unanswerable items (0.00 vs Jev 0.09).

### 1.2 Jeff-1.1 / "Releasing the Inner Jev" ([arXiv 2610.17205 v2](https://arxiv.org/html/2610.17205), 19 Nov 2026)

**Identity**
- Geuter, Cheng and Alvarez-Melis (Harvard, Kempner Institute, Microsoft Research).
- Code: [j-geuter/Jeff](https://github.com/j-geuter/Jeff), Apache-2.0, created 2026-10-08, with training and serving code.
- Configs exist for Large (Gemma-4-31B-it), Medium (Gemma-4-12B), Qwen-27B (Qwen3.6-27B) and Qwen-4B (Qwen3.5-4B).

**Readout** ([src/jeff/head.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/head.py))
- `z = c·tanh(W₀h/c) + A·(h−μ)/σ + b`, where:
  - W₀ is the LM-head rows for the letters A–P;
  - A and b start at zero, so training begins exactly at the instruct model's letter readout;
  - μ and σ come from 2,048 training rows;
  - c = 30 (Gemma soft-cap); Qwen has none.
- K ≤ 16 slots. For K>16 (up to 256), `decide_wide` runs groups of ≤16 plus a final round of the group winners (≤17 passes). The paper says "we have not evaluated this mode systematically" (§6).

**Prompt** ([prompting.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py))
- SemIf's byte-identical prompt: a system message ("Apply the supplied criterion… Respond with only its uppercase letter") plus a JSON object `{evidence, criterion, options:[{letter,description}]}`.
- Chat template with thinking off. Inputs over 8,192 tokens are rejected, or filtered by a keyword heuristic in `longinput.py`, which was added because "JevBench… counts a refused item as wrong".

**Mask** ([bidir.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/bidir.py))
- "full" mode sets `is_causal=False` on the full-attention layers only: Gemma's global layers, or Qwen's 1-in-4 attention layers. Sliding-window and DeltaNet layers stay causal.
- Jeff-1.1-Large config: `"bidirectional": null`. Qwen-4B: `"full"`.
- Ablation: "bidirectional helps little, and even slightly hurts the 31B", −0.2 ± 1.1 Capability (§4).

**Training** ([train.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/train.py), [configs](https://github.com/j-geuter/Jeff/tree/main/configs/train), paper Table 7)
- LoRA r32/α64, dropout 0. Targets q/k/v/o/gate/up/down, plus `in_proj_qkv, in_proj_z, out_proj` on Qwen.
- LR 2e-5 for LoRA and 1e-4 for the head (Qwen-4B: 1e-4 for both). WSD schedule: 250 warmup steps, 1,500 steps total (800 for 4B).
- 65,536 padded tokens per step (8,192 tokens × grad_accum 2 × 4 GPUs). Max 8,192 tokens.
- Options are reshuffled each use; score levels are not shuffled.
- Loss is soft cross-entropy only (`soft_ce`). "No RL, reward models, or auxiliary losses."
- The release is a three-seed **weight soup**.

**DecisionBank-260k** (Table 2): 261,561 items.

| Component | Items | Target |
|---|---|---|
| decider public and custom questions (79 tasks, ≤2,094 per task) | 163,732 | 50/50 label and teacher |
| Human-vote sets | 9,168 | 50/50 votes and teacher |
| LLM-written | 60,219 | Teacher only |
| Programmatic | 28,442 | Exact |

- Teacher: Qwen3.8-27B with reasoning, mean of 2 samples.

**Calibration** (§4, App. H)
- Per-type T fitted by NLL on held-out families, clamped to ≥1:

| Model | choice | noul | score |
|---|---|---|---|
| Large | 1.49 | 1.57 | 1.11 |
| Qwen-27B | 1.42 | 1.23 | 1.01 |
| Medium | 1.42 | 1.35 | 1.06 |
| Qwen-4B | 1.29 | 1.23 | 1.17 |

- **Snapping rule:** if 0.2 < P(yes) < 0.8, the answer moves to 0.199 or 0.801. The code says why: "JevBench counts a yes/no answer with 0.2<P(yes)<0.8 as wrong" ([model.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/model.py) `JeffModel` docstring).
- Temperature scaling side effects (Gemma-31B zero-shot): choice calibration 58.6→84.0, but score competence 64.5→54.1, and exact-probability items 76.3→52.5.

**Results**

*Table 3 (OpenDecisions Capability = (I+C)/2):*

| System | Capability | JevBench-Open |
|---|---|---|
| Jeff-1.1-Large | **89.0** (I 83.5 / C 94.5) | 93.1% |
| Gemma-4-31B zero-shot | 79.6 | – |
| Jev 1.13 | 80.1 | 86.6% |
| Jeff-Qwen-27B | 86.8 | – |
| Jeff-Medium | 85.0 | – |
| Jeff-Qwen-4B | 79.3 | – |
| StartLux-27B Instant | 88.5 | – |
| pplx-decider-v1.3 | 85.1 | – |

*Table 5 ablations (Jeff 1.0 recipe, Gemma-31B, reference 84.4):*

| Variant | Capability |
|---|---|
| Head only, frozen backbone | 78.9 |
| LoRA, head frozen at init | 84.9 |
| Gemma *base* + LoRA | 70.3 |
| Public data only | 82.5 |
| Hard labels | 84.1 |
| Rank 16 / 64 / 128 | 84.7 / 84.5 / 84.9 |

*Other results:*
- Size scaling, zero-shot (App. F): Qwen3.5 0.8B/2B/4B/9B/27B = 48.9 / 57.6 / 76.8 / 82.2 / 86.5, "level off after about 4B".
- Knowledge: Jev 92.9 vs Large 89.5.
- +R reasoning when unsure: 92.5 at 16.5% reasoning, 94.1 at 50%.

**Order-invariant encodings** (App. G)
- The 4B model flips about 6% of choice and yes/no answers when the order is reversed.
- Order-invariant variants score 0.6–2.4 below the letter format: "the encoding does not help".
- Calibration losses (frozen 4B head): no loss lowers ECE by >0.01 vs cross-entropy; focal −2.1 ± 1.7.

**Serving** (App. J, Table 25)
- One H200, compiled: p50 Large 61→44 ms, Medium 35→21 ms, Qwen-4B 48→30 ms.
- Jev API p50 0.189 s.
- No prefix cache (`use_cache=False` in `hidden`).

**Flaws** (§6)
- 78% English; recipe settings were selected on OpenDecisions; the teacher is also an OpenDecisions panel model; 16 slots vs Jev's 256.
- **JevBench-specific noul snapping** distorts calibration (inference).

### 1.3 Strands Decider (Strands Labs / AWS)

**Identity**
- [Repo](https://github.com/strands-labs/strands-decider) Apache-2.0, created 2026-09-29.
- [Blog](https://strandsagents.com/blog/introducing-strands-decider) 2026-10-01 (Brooker, Chambers, Nonato de Paula).
- Released: v1 (Qwen3.5-2B plus 4 Gemma-4 sizes, 2026-10-09) and v2 (5 Gemma-4 sizes, 2611) ([CHANGELOG](https://github.com/strands-labs/strands-decider/blob/main/CHANGELOG.md)).
- Code completeness:
  - The training code is present.
  - The v1 data builders for 4 of 9 files are marked "NOT REPRODUCIBLE".
  - The v2 data mix of **17.48M rows** "is not in this repository" ([docs/training.md](https://github.com/strands-labs/strands-decider/blob/main/docs/training.md)).

**v1 pointer** ([docs/architecture.md](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md))
- Prompt shape: `<state>…</state><question type=…>…<options>1. … </options></question><answer>`.
- `logit_k = ⟨q(h_answer), k(h_option_k)⟩/√dim`, with dim 256 (~1M params), random init, head LR 1e-3.
- Torso: Qwen3.5-2B-Base, 18 of 24 layers DeltaNet. LoRA r16/α32 on attention, MLP and DeltaNet projections.

**v1 loss** (v19 recipe)
- Cross-entropy with 10% ordinal smoothing on Score.
- KL to the frozen torso's option-number readout (w 0.3); the v1 release anchors to released v19 instead.
- KL to a teacher (w 1.0) only where the teacher beats the student.
- "Option order is re-shuffled on every example, every epoch"; scores are only reversed (p 0.5).

**v1 data** ([config](https://github.com/strands-labs/strands-decider/blob/main/configs/v1/strands-decider-2B-qwen3.5-v1-2610.yaml))
- 246,678 rows. The teacher is gemma-4-31B-it, used where it agrees with gold.
- Three-seed soup, stored as one rank-48 adapter with stacked heads.

**Pointer vs slot head** ([research/history.md](https://github.com/strands-labs/strands-decider/blob/main/research/history.md))

| Head | Params | Score |
|---|---|---|
| Linear slot | 53k | 0.6619 |
| Slot + mean(option states) | 2.12M | 0.7056 |
| **Pointer, dim 16** | **70k** | **0.7200** |
| Pointer, dim 256 | 1.05M | 0.7250 |

- "Roughly half the gain is having option representations at all, half is addressing them per option."

**v2 token readout**
- One linear map with 255 rows (codes A…IV, without GZ, each one Gemma token), initialised from the LM-head rows, with the Gemma soft-cap, masked past K. One global T.
- **`prefix_bidir` mask:** "A state token sees all of the state, in both directions, and never the question. A question token sees the state and all of its own question." Hence "the torso encodes the state once, and each question reads that encoding."
- 31B v2 recipe ([config](https://github.com/strands-labs/strands-decider/blob/main/configs/v2/strands-decider-31B-gemma4-v2-2611.yaml)):
  - LoRA r64/α128, LR 1e-4 for adapters and readout, 2,680 rows per step, 1 epoch.
  - max_length 16,384; no teacher, no KL, no ordinal smoothing.
  - Three-run average, then one temperature.

**Calibration** ([evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md) "Calibration")
- v19 per-primitive T fitted against ECE: noul 0.91, choice 0.77, score 0.96 (below 1, i.e. **underconfident**).
- Held-out ECE 0.054 overall, but **noul 0.162**.
- "Calibrating score changes its answer."
- "No single temperature per primitive serves both" short classification and long documents. v19 is underconfident on multi-step documents: right 0.83 of the time at a stated 0.64.

**Results**
- v1 2B soup: JevBench 180/231, Brier 0.280, ECE 0.072, HotpotQA (never trained) 0.832, held-out short tasks 0.653 (results.md "v1").
- v2 ([docs/models.md](https://github.com/strands-labs/strands-decider/blob/main/docs/models.md)):

| Model | DI 0.3 | JevBench | OpenDecisions | DecisionPivot | H100 median latency |
|---|---|---|---|---|---|
| 31B | 71.5 | 93.1 | 92.2 | 84.3 | 99 ms (1 question) / 614 ms (30 questions) |
| E2B | 57.1 | – | 77.7 | – | 62 ms |

**Serving**
- Shared-prefix KV broadcast: 1 vs 6 questions on a ~640-token state, 219.4 vs 217.4 ms ([designs/inference.md](https://github.com/strands-labs/strands-decider/blob/main/designs/inference.md)).
- The hybrid cache fork copies the DeltaNet states "following decider-2b's `shared_prefix.py`".
- RTX 3090: v19 p50 115 ms.
- Deploy targets: Lambda, SageMaker, MLX and vLLM.

**Flaws**
- **Question-ignoring:** v7 "gives the same answer to a changed question 94–99% of the time". Fixes v11a/b both lost on JevBench. v11b "answers yes on 60 of 74 yes/no tasks" (history.md v11).
- Pointer order sensitivity: reversing options moves the distribution ~0.016 (architecture.md).
- v1 card: "score and noul transfer poorly".

### 1.4 Jeeves (PostHog)

**Identity**
- [Repo](https://github.com/PostHog/jeeves) MIT, created 2026-09-29, by Nicholas Waltz. "Inspired by Kev".
- 9B weights on [HF](https://huggingface.co/PostHog/jeeves). Full training code plus data prep. The SFT reasoning chains are "not published yet".

**Architecture** (README "How it works")
- Kev's delimiter tokens and pointer: query at `<decide>`, key at `</opt>`.
- Qwen3.5-9B, LoRA r16 on all projections.
- Prompt: `<state> … <q> instr <opt>…</opt> … <think>` → reasoning → `</think>` → **the question and options repeated** → `<decide>`.
- Ablations in the README: plain-text tags instead of special tokens, or not repeating the question after reasoning, "worsened performance".

**Training**
- SFT: 1 epoch, 16,867 steps on 4 GPUs, 453,531 questions from ~175 public sources.
- About 15% of examples carry a self-generated reasoning chain, kept only if it closes within 1,024 tokens, does not loop and reaches the right answer.
- v1 added **CISPO** ([trainer.py](https://github.com/PostHog/jeeves/blob/master/trainer.py) `train_cispo`):
  - group size G=8, 8 prompts per step;
  - reward = p_correct (the pointer head's probability of gold after reasoning) × (1 − length penalty, hinge 2048→1024 tokens, cap 0.1);
  - advantage = reward − group mean;
  - IS weight `exp(logp−old).clamp(max=1+0.2)` detached;
  - plus rollout cross-entropy (0.5) and anchor cross-entropy (0.5).
- "CISPO did not improve v2."

**Calibration** ([calibrate.py](https://github.com/PostHog/jeeves/blob/master/calibrate.py))
- Two temperatures: 1.52 direct and **5.42 after reasoning**. "Every training chain reaches the right answer, so answers after reasoning come out overconfident."

**Results** (README table; reasoning when confidence <0.9)

| Suite | Jeeves v2 | Jev | Kev-9B |
|---|---|---|---|
| Test overall | 0.905 | 0.857 | 0.822 |
| JevBench public | 0.896 (v1 0.913) | 0.866 | – |
| JevBench hard | 0.793 (v1 0.820) | 0.730 | – |
| MMLU-Pro | 0.770 | 0.840 | – |

**Serving**
- No thinking: 22 ms p50. `nothink_threshold` 0.9: 1.2 s p50 / 8.2 s p90 (H100, FP8).
- **Diffusion drafter** (Orthrus-style) for Gated DeltaNet: block 4 gives 176 vs 109 chain tokens/s.

**Flaws**
- v2 "is overconfident without reasoning on some long-policy and multi-hop items, so it skips reasoning on them."

### 1.5 decider (Mapika)

**Identity**
- [Repo](https://github.com/Mapika/decider) Apache-2.0, created 2026-09-16, by Mark Marosi.
- Models: 0.8B / 2B / 4B (Qwen3.5-Base), 35B-A3B, and Gemma-4 12B and 31B, plus stock-weight "decider-chat" configs.
- Full SFT builders for ~95 datasets. The RL code lives in a separate repo.

**Readout** ([decider/model.py](https://github.com/Mapika/decider/blob/main/decider/model.py))
- **No new head.** `logits = lm_head.weight[letters] · h_slot`, soft-capped and masked past K. Letters run A–J, then K–Z, then two-letter tokens up to 255.
- **Multiple answer slots in one prompt**, one per question. "Every question can also be scored in its own row."
- Two layouts, trained 50/50:
  - state-first (the default);
  - schema-first, whose question-and-options prefix is cached per schema. It "trades accuracy for speed" (README "How it works").

**Training** ([train.py](https://github.com/Mapika/decider/blob/main/decider/train.py))
- **Full-parameter** fine-tune, LR 1e-5, cosine schedule, AdamW β(0.9, 0.95), cross-entropy plus optional Brier or label smoothing.
- One epoch = 1.47M examples / 455M tokens, 5.3 h on a GH200 (README "Train your own").
- Mixture ([decider/data/mixture.py](https://github.com/Mapika/decider/blob/main/decider/data/mixture.py)):

| Component | Count |
|---|---|
| wide (full label sets ≤255) | 8,000 per task |
| padded with unrelated labels | 25,000 |
| described options (Qwen3.5-27B-written descriptions) | 40,000 |
| JSON states | 32,000 + 20,000 indexed |
| single-question | 30,000 |
| isolated yes/no per Score level | 26,000 |
| programmatic rules with state and rule twins | 90,000 |
| contrastive pairs | ×3 |
| replay | 200,000 |

- Plus "none" augmentation.
- Teacher data comes from a local Qwen3.5-27B ([teacher_labels.py](https://github.com/Mapika/decider/blob/main/decider/data/teacher_labels.py)), filtered because the "checker shares the bias being fixed".

**RL stage v8→v10** ([docs/RL.md](https://github.com/Mapika/decider/blob/main/docs/RL.md))
- PPO actor (0.1) on live MiniWoB++ and games.
- **Belief log-score** against exact laws (0.2), reading options in two orders.
- **Rendering-consistency KL** across reversed order and schema-first (1.0).
- Retention KL to v8 (0.7), with a hard budget.
- LR 1e-6. Belief excess 0.473→0.219 nats.

**Calibration**
- Per-type temperatures. `temperature_by_options` T(n) = max(min, a + b·ln n); decider-chat-gemma4-31b uses a=10.124, b=−1.633.
- **decider-12b v2 sets Noul T=0.05 "chosen for JevBench"**:

| Noul T | AUROC | ECE |
|---|---|---|
| 0.05 | 0.878 | 0.151 |
| 1.0 | 0.934 | 0.038 |

- At T=0.05, "most answers round to exactly 0 or 1" (README "Temperatures").

**Results**
- Decision Index 0.2.1: decider-chat-gemma4-31b (stock weights plus the readout) 57.33, #2 of 70.
- The trained 35B / 4B / 2B score 47.11 / 40.70 / 28.97: "the base model sets most of the score."
- decider-31b v2: DI 0.3 public 63.90.

**Flaws** (README "Limits")
- Hard-tier ECE 0.31 (2B v10).
- "Rules written into the question are not followed at this size."
- Long JSON arrays: 0.51 at 64 records vs 0.70 with one.
- GGUF multi-row packing moves probabilities by up to 0.16 at Q4_K_M.

### 1.6 StartLux-Decision ([arXiv 2610.17309 v2](https://arxiv.org/html/2610.17309), 20 Oct 2026)

**Identity**
- Zhuang, Su, Pan, Wong, Chen and Kuo (StartLux; HKU; SUSTech).
- [Repo](https://github.com/StartLuxLabs/StartLux-Decision) Apache-2.0, 1,780 stars. **Inference only:** vLLM plugin, llama.cpp bidirectional patches, MLX.
- Weights 0.8B–27B, Instant and AutoThink.

**Architecture** (paper §3)
- Hybrid linear-plus-full-attention backbone. At 4B and 9B, 8 of 32 layers are made bidirectional.
- Letter logits A–Z through the LM head; Eq. 1 is a temperature softmax.
- One prompt per question. Groups of ≤25 with the top 3 advancing.
- Instant is bidirectional; AutoThink is causal.

**AutoThink**
- If max p₀ ≥ 0.9 (paper), return p₀. Otherwise reason (≤4,096 tokens) and fuse `p = 0.25·p₀ + 0.75·p₁`; α was picked by held-out Brier (§5.2).
- The repo's serving default gate is **0.6** ([README](https://github.com/StartLuxLabs/StartLux-Decision)).

**RLCD²** (Eq. 3–6)
- Reward `r = S(p₁,t) − S(p₀,t) − λ·max(0,|τ|−B)`, with λ=1e-4 and B=3,072.
- S is a decision-weighted proper score: `s(q,z) = zW(q) − C(q)`, w = ½Uniform + ½Beta(8,2).
- GRPO-style: group-centred advantages clipped to [−5, 5], not std-normalised, plus a KL penalty.
- Trained only on items with max p₀ < 0.9.
- G, ε, β and LR are **not reported**.

**SFT**
- Stage 1: full SFT, soft cross-entropy, ~2.7M examples (0.8–9B) or ~0.8M (27B), with hash-based option permutation.
- Stage 2: adapters on ~1.7M examples, then a 3–4-seed average.
- Stage 3: a reasoning adapter on 4k–21k examples.
- Teachers are unnamed.

**Calibration**
- Per-type temperature; "all fitted temperatures exceed 1" (overconfident).
- DecisionPivot calibration error, AutoThink / Instant: 4B 2.4 / 7.3; 27B 5.2 / 5.1.

**Results**

*Table 4, DI 0.3 public:*

| System | DI 0.3 |
|---|---|
| 27B AutoThink | 73.37 |
| 27B Instant | 72.83 |
| Perplexity Decider v1.1 | 62.25 |
| Jev | 57.96 |
| d1-3B | 48.99 |
| Decision 2.0 Sol | 30.05 |

- DecisionPivot pair accuracy: 27B Instant 75.4, AutoThink 86.9, **Jev 31.0**.
- Bidirectional attention: up to +2.4 DI points at 4 of 5 scales (Fig. 4).

**Serving** (§6)
- H200 / vLLM Instant: 27B 51.9 ms median; 4B 27.4 ms.
- "Prefix caching changed 2.6% of readouts by more than 0.05" (App. D.3).
- On a 3,200-token document with 32 questions, 27B throughput goes 2.8→27 decisions/s when document representations are reused. "Bidirectional attention prevents that reuse."

**Flaws and claims**
- 151-option intent: 86.7% vs 95.7% at 26 options.
- RL hurts long computations: −5.9 pts on 2,048–4,095-token traces.
- Claims Jev has 13.8 pp calibration error on DecisionPivot ("systematic overconfidence").
- [docs/none_probability.md](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/none_probability.md): an untrained, appended "None of the above" letter gives AUROC 0.920 for no-correct-answer detection; threshold 0.2 flags 66.3% at 2.2% false alarms.

### 1.7 AnyJev / Tacit (Nokia et al., [arXiv 2610.00831](https://arxiv.org/html/2610.00831), 30 Sep 2026)

**Identity**
- Zhang, Yang, Shi et al. (Nokia, Tencent Hunyuan, PolyU).
- [Repo](https://github.com/nokia-applied-research/AnyJev) Apache-2.0.

**L0 (training-free) readout**
- Restrict the next-token distribution to the K label tokens and renormalise.
- **Rotation averaging:** mean of the log-probabilities over the K cyclic rotations, then softmax. This is exact under an additive position-bias model.
- **Label prior:** p ∝ p̃ / q̂^α, with α=0.75 and q̂ a running mean over unlabelled inputs.
- Adaptive stopping on the log-odds gap τ, certified at ≤1% disagreement (Clopper–Pearson).
- The tried "set" mask scored 0.042 vs 0.633, against a chance level of 0.056 (§8).

**Results**

*Table 1 (K=20, means over 11 models):*

| Metric | Raw | + rotations | + prior |
|---|---|---|---|
| banking20 accuracy | 0.665 | 0.737 | 0.752 |
| banking20 flip rate | 0.328 | 0.138 | – |
| Injection (K=2) flip rate | 0.180 | 0.000 | – |
| ECE after temperature | – | – | 0.088 |

- JevBench pooled debiasing: +0.006 [−0.004, +0.018], i.e. **no gain at K≈3.6**.
- Serving (H100, 18 options): 16.7 decisions/s with all rotations → 37.2/s adaptive (7.3 rotations).

**Tacit** ([tacit.py](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/tacit.py), [HF Tacit-9B](https://huggingface.co/morriszjm/Tacit-9B))
- Self-distillation: the base model writes its own decision problems, answers them with reasoning, and is trained to give those answers in one pass. "No human labels, external datasets or other models."
- Tacit-9B vs base Qwen3.5-9B: JevBench 0.823 vs 0.805; bev 0.725 vs 0.680.
- Adaptive escalation (τ=0.5, cap 20%): JevBench 0.887.
- Tacitus-31B: DI 0.4.1 Full 66.47.
- The objective, LoRA vs full fine-tuning, and data sizes are not disclosed. **The training code is not in the repo.**

### 1.8 NanoJev (Tianyu Chen, S1-Arena author)

**Identity**
- [Repo](https://github.com/TianyuCodings/NanoJev) MIT, created 2026-09-17. Qwen3-0.6B, full fine-tune.
- Focus: games (ViZDoom, Maze, Snake).

**Architecture** ([scripts/train_toy_decisions.py](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/train_toy_decisions.py) `DecisionModel`)
- Each candidate is its own "leaf" path: state + instruction + candidate. The path "repeats prefixes; tree sharing is a separately verified optimization".
- Score: `z = w·LN(h_last)`, plus for Choice a **set-attention residual**: MultiheadAttention(128, 4 heads) over the candidates with a log K feature, zero-initialised output.
- Boolean logits are [0, z].
- Permutation-equivariant by construction (inference).
- The contract doc confirms there is no cross-question attention ([docs/TYPESAFE_CONTRACT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md)).

**RLCD experiment** ([docs/RLCD_EXPERIMENT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md))
- An unbiased sampled-Brier policy gradient, with E[R] = ‖q‖² − ‖p−q‖² over M samples.
- It matched but did not beat direct cross-entropy or Brier: test L2 0.1184 vs CE 0.1242 vs Brier 0.1384, one seed.
- Correctness-only REINFORCE was bad: 0.0171 vs 0.0026 on CPU.

**Results**
- ViZDoom Basic 128/128 vs Jev 56/128 (README).
- AnyJev's BEV table gives NanoJev 32.99%.

### 1.9 SemIf-OpenJev (Theo Lee)

**Identity**
- [Repo](https://github.com/TheoLeeCJ/SemIf-OpenJev) MIT, 4,970 stars. **No training.**
- Frozen Qwen3.5-4B letter logits for 2–16 options ([docs/METHOD.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/METHOD.md)).
- Its prompt became Jeff's prompt.

**Results** ([docs/RESULTS.md](https://github.com/TheoLeeCJ/SemIf-OpenJev/blob/main/docs/RESULTS.md))

| Workload | Direct letter logits | Qwen3-Reranker-4B | Jev |
|---|---|---|---|
| Authored (balanced accuracy) | 0.813 | 0.625 | – |
| TypeSafe subset agreement | 0.845 | – | 0.883 |

- **Option reversal caused 10 flips of 36** for direct logits.
- RTX 3090 with parallel suffixes after one state prefill: 20.03 decisions/s vs 2.33 fresh. Argmax drift was 6 of 777.

### 1.10 vLLM-SR Decision 2.0 and System One Auto

**Identity**
- Six Apache-2.0 models, Kai-0.6B to Vega-27B (2 Oct 2026) ([docs](https://vllm-sr.ai/docs/model-runtime/supported-models/decision-2/overview/)).
- The head is **now readable in HF remote code** (`decision2/_vendor/dev2model/decision_model.py` in [Kai-0.6B](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/main)).

**Readout**
- Prompt: `Context:{state} Task type … Question … Options: <option>{json}</option>… Select the single option… Decision:`.
- `CandidateHead`:
  - s_k = (K·LN(h_endpoint_k)) · (Q·LN(h_Decision)) / √256 + w·GELU(M_c·LN(h_k) + M_q·LN(h_q));
  - fp32 head; causal mask; one row per question.
- Kai config: "base-random-head", a **uniform soup of 6 arms**.
- Calibration:
  - per-type temperatures in [0.05, 20] (`calibration.py`);
  - plus a per-level **score bias** for 5-level scores, fitted with class-balanced NLL (`score_bias.json`).

**Serving**
- Shared context (`shared_ctx.py`): the longest common prefix is cut before the first option. "Tree" mode packs prefix and suffixes into one row with two attention kernels merged by log-sum-exp; DeltaNet continues from the prefix state.
- Near-ties (margin < τ) fall back to the exact path.
- "Up to 3.4× faster for 128 questions on Vega-27B." Fused Triton kernels are 11–48% faster with no changed decision. ROCm merged rows changed decisions (docs).

**Results**
- Vega: JevArena 74.0, DI 56.5, 71.4 ms per single question ([card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B)).
- Training data and loss are undisclosed. The code shows only a `teacher_probs` replay field.

**System One Auto** ([ModelSystem.One](https://modelsystem.one/news/vllm-sr-decision-2-family/))
- Kai first, escalating to Vega when the gate fails. The gate rule is not stated.
- JevBench public: Kai 65.80%, Auto 82.25%, Vega 87.88%.
- The "54.11%" saving is estimated, not billed.

### 1.11 pplx-decider v1.1 (Perplexity; AutoJev lineage)

**Identity**
- Qwen3.8-27B. Readout `[255, 5120]` mapped to vocabulary token ids.
- Inference code at `source/src/autojev/model.py` ([HF card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)).
- v1 = AutoJev-27B, full-weight SFT on 73k synthetic rows (Kev PLAN.md "Against Jev").

**Recipe and results**
- v1.1: 1 epoch on 626,033 rows (530,103 from tasksource); T=1.0087 (v1 2.2076). Decision Index 61.56 vs v1 56.4 ([SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Perplexity/PPLX-Decider-v1.1-27B.md)).
- **Causal mask removed** in the 16 full-attention layers; the 48 DeltaNet layers stay causal.
- Serving it causally changed the top answer on **101 of 2,167** items. WinoGrande 88.87% vs 92.42%.

**Serving cost**
- SGLang disables the radix cache and chunked prefill: "nothing is reused across requests".
- GB300: 53 ms for a short question; four questions, 4,540 tokens: 183 ms.

### 1.12 Liquid d1

**Identity**
- Open d1-3B and d1-omni-600M (2026-10-07) ([blog](https://www.liquid.ai/blog/d1-open)).
- d1-3B = LFM2.5-VL-3B averaged with LFM2.5-2.6B.

**Readout and training**
- Readout ([prompt.py](https://huggingface.co/LiquidAI/d1-3B/blob/main/prompt.py)): single-token codes A–Z, then 00…; each option scores the **max over token forms** (for example "A" and " A"); Noul uses yes/no forms; Score uses digits (2–10 levels).
- Training: seed and mixture merges. The blog says "long inputs, shuffled answer options, and fixing data shortcuts mattered more" than advanced techniques. The objective is not stated.

**Results**
- DI v0.2.1 public: 48.57.
- RTX 4090: 8 ms per question; 102 ms for a 3.4k-token state.

### 1.13 Microsoft-Decision-1 and the OpenAI Decisions API (hosted)

**Microsoft-Decision-1** ([blog](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), 2026-10-09)
- Post-trained Qwen3.5-9B. The readout is not described.
- Claims:
  - flip rate 1.3% averaged across 8 perturbation types, with 0 under option paraphrase, reversal or shuffle;
  - p50 about 35× faster than GPT-6 Sol;
  - $0.042 per M input tokens.
- The training method is undisclosed.

**OpenAI Decisions API** ([guide](https://developers.openai.com/api/docs/guides/decisions))
- `gpt-6-luna`; predicate, choice and score question types; 1–400 questions per request.
- An implicit cache breakpoint sits at the end of the shared `input`, with a ≥1,024-token prefix; **question content is not part of the reusable prefix**.
- $0.10 per M uncached input tokens. No calibration claims.
- "Choice value changes can alter results even when descriptions stay the same."

---

## 2. Cross-system evidence on known failure modes

| Failure | Evidence |
|---|---|
| **Order sensitivity** | Causal readouts are not invariant: Jeff-4B flips ~6% on reversal (App. G); SemIf 10/36 flips; AnyJev raw K=20 flips 0.33. Kev and Strands pointers are "small but not zero" (~0.016 distribution shift). Exact fixes cost accuracy: Kev option isolation −5.8 pp at 4B; Jeff order-invariant encodings −0.6 to −2.4. Partial fixes: AnyJev rotation averaging (flip 0.33→0.14, costs K passes); decider's rendering-consistency KL and optional reversed second reading; NanoJev's set head (exact). |
| **Yes/no** | Strands noul ECE 0.162 even after fitting. Leaderboard-driven *distortion*: Jeff snaps P(yes) out of [0.2, 0.8]; decider-12b uses noul T=0.05 (ECE 0.151 vs 0.038). Both are attributed to JevBench scoring. Strands v11b drifted to "yes" on 60 of 74. |
| **Large K** | Letter readouts cap at 16 (Jeff) or 26 (StartLux) and use tournaments: StartLux 151-option 86.7% vs 95.7%, elimination 0.2–0.7%. Pointer and 255-code readouts (Kev, Strands, decider, Decision 2.0, pplx) do one pass up to 255. AnyJev cost scales with K. |
| **Overconfidence** | Instruct Gemma is overconfident (T 2.7–6.0) while pretrained Gemma is calibrated (Jeff App. I). StartLux: all T>1. Post-reasoning answers are overconfident (Jeeves T 5.42; StartLux post-reasoning error 8.1 vs default 3.7). LoRA small models can be *under*confident (Strands T<1). |
| **Long context** | Kev's validated context is 8,192 below 27B, and 65,536 for 27B after training on 32k states. Jeff and pplx are trained to 8,192. Strands v2 is trained to 16,384. Kev's buried-synthetic-state training did not transfer to real documents (PLAN #11). |
| **Question-ignoring** | Strands v7 gave the same answer to a changed question 94–99% of the time. Cause: one fixed question per task in the data. |
| **Knowledge erosion** | Kev: −11 pp MMLU at 8B, date arithmetic lost in the representation; a lower LR is the fix. Jeff: a base model plus LoRA scored 70.3 vs instruct zero-shot 79.6. |
| **Batch and runtime nondeterminism** | Decision 2.0 on ROCm; StartLux prefix caching changed 2.6% of readouts by >0.05; decider GGUF packing up to 0.16; Decision 2.0 needs a near-tie fallback. |

---

## 3. What a from-scratch builder should adopt or avoid (inference, grounded in the above)

**Adopt**
1. **Start from an instruct checkpoint and initialise the readout from its letter/LM-head rows** (Jeff's z = W₀h + A·std(h) + b; Strands v2 and pplx 255-code rows). Zero-shot readout is already strong: Gemma-31B 79.6 vs Jev 80.1. Training then starts at a good point.
   - If you want *no-ceiling, runtime-defined options*, a pointer on option end-tokens is cheap and strong. Strands' 70k-param pointer beat a 2.1M slot head; Decision 2.0 adds an MLP term.
2. **Shuffle options every epoch** (every system). Add **none-of-the-above minimal pairs** (Kev: none_present 0.85→0.93 at 4B). Keep **minimal pairs and contrastive twins together** (Kev, decider, Strands, Liquid).
3. **Train toward soft targets where they exist**, from a reasoning teacher mixed 50/50 with labels (Jeff; Kev PLAN #10). Use a teacher only where it beats the student (Strands v12/v14).
4. **Use a low LR, or LoRA plus weight-soup or blend toward a less-drifted parent:**
   - Kev: LR is the lever; 27B v2 blends 0.85·SFT with 0.15·v1.
   - Jeff, Strands, Decision 2.0 and StartLux soup 3–6 seeds.
5. **Use a cacheable state prefix.** Choose between:
   - causal attention plus per-question rows continuing the state cache (Kev, Strands v1, Decision 2.0; DeltaNet needs conv and recurrent state forking); or
   - Strands v2's **`prefix_bidir`**: state bidirectional only within itself, which keeps caching.
   - Full-prompt bidirectional attention (pplx, StartLux, Jeff) gives small gains (StartLux ≤+2.4; Jeff −0.2 ± 1.1; pplx causal-serving loses 3.6 on WinoGrande) and kills reuse.
6. **Calibrate with one temperature per type, fit on held-out *datasets*, not held-out items** (Kev #5). Ship a workload-refit tool (`kev.calibrate`), and report ECE out-of-fold.
7. **Make "none fits" a first-class signal**, either trained (Kev) or as an appended letter (StartLux AUROC 0.92).
8. **Probe what metrics hide**: question sensitivity (Strands), the order-flip rate, the 8-perturbation flip protocol (Microsoft), and the validated context-length rule (Kev).

**Avoid**
1. Calibration tricks that game the leaderboard: noul snapping, T=0.05.
2. Per-(type, K) temperatures and reliability heads without out-of-distribution validation (both Kev negatives).
3. Exact order isolation as a default unless you can absorb ~2–6 pp (Kev, Jeff).
4. Training on one fixed question per task (Strands v11).
5. Assuming more same-distribution data helps transfer (Kev: dev up, transfer flat or down).
6. Base checkpoints for a letter readout (Jeff Table 5).
7. Reasoning escalation without its own post-reasoning temperature (Jeeves, StartLux).

---

## 4. Not read or not found

**Papers**
- Jeff paper: the first fetch window cut off at App. D.1; App. G's encoding *method* text and Tables 6/9/11/12 in full were not obtained (Table 12 lowest-family values were).
- StartLux RLCD² hyperparameters (G, ε, β, LR, batch) and the SFT/LoRA hyperparameters are absent from the paper text I read. Its training code is not public.

**Code or data not published**
- Tacit's self-distillation objective, data size and code.
- Decision 2.0 training data and loss, and the Auto gate rule.
- Strands v2's 17.48M-row data mix.
- Kev-27B's training corpus (manifest only).
- Strands v1's four "NOT REPRODUCIBLE" input files.
- Jeeves' reasoning chains.

**Hosted systems**
- Microsoft-Decision-1: weights, readout and the full list of 8 perturbation types.
- OpenAI internals and calibration.
- Liquid d1: training objective and license text.

**Lightly read**
- Kev `serve.py` batching and the archived PLAN.md history beyond the lines quoted.
- decider `docs/RESULTS.md` and `HISTORY.md`.
- vLLM-SR blog posts beyond the docs page.
