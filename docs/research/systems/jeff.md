# Jeff-1.1 ("Releasing the Inner Jev")

> **Family:** decoder-readout · **Team:** Geuter, Cheng and Alvarez-Melis (Harvard, Kempner Institute, Microsoft Research) · **Released:** arXiv 2610.17205; the catalog dates it 2026-10-12, and the decoder-family dossier read v2 of 19 Nov 2026; code repo created 2026-10-08 · **License:** code Apache-2.0; paper under the arXiv non-exclusive licence (link only) · **Code/weights:** training and serving code at [j-geuter/Jeff](https://github.com/j-geuter/Jeff); weights in an HF collection ([collection](https://huggingface.co/collections/jgeuter/jeff-11-6ada6708f3ed49beaf0f97ae)); data released as [DecisionBank](https://huggingface.co/datasets/jgeuter/DecisionBank) and [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions) · **Sources read:** arXiv HTML v2 (the first fetch window cut off at App. D.1; App. G's encoding *method* text and Tables 6/9/11/12 in full were not obtained, only Table 12's lowest-family values), `src/jeff/head.py`, `prompting.py`, `bidir.py`, `train.py`, `model.py`, `longinput.py`, `configs/train`. A guessed HF path for Jeff-1.1-31B returned 401; the collection link is the correct one.

## Summary

Jeff-1.1 shows that an instruction-tuned LLM already contains a strong System One model: reading the option-letter logits at the last prompt position of Gemma-4-31B-it gives 79.6 OpenDecisions Capability zero-shot, against 80.1 for Jev ([arXiv](https://arxiv.org/html/2610.17205)) [@jeff]. The recipe then adds a zero-initialised affine correction to the LM-head letter rows, LoRA on the backbone, soft-target cross-entropy on the 261,561-item DecisionBank, per-type temperatures with T ≥ 1, and a three-seed weight soup, reaching 89.0 Capability (+8.9 over Jev). The idea that matters most is **initialising the readout exactly at the instruct model's letter readout and adapting the backbone**: a trained head on a frozen backbone scored *below* zero-shot (78.9 vs 79.6), and a base (non-instruct) checkpoint with the same recipe scored 70.3. The model reads one full prompt per question with no prefix cache, and the readout has 16 slots.

## Architecture

**Backbones and configs** ([repo](https://github.com/j-geuter/Jeff)) [@jeff-repo]. Large = Gemma-4-31B-it; Medium = Gemma-4-12B; Qwen-27B = Qwen3.6-27B; Qwen-4B = Qwen3.5-4B. The chat template is used with thinking disabled ([arXiv](https://arxiv.org/html/2610.17205)).

**Prompt** ([prompting.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py)). SemIf's byte-identical prompt [@semif]: a system message ("Apply the supplied criterion… Respond with only its uppercase letter") plus a JSON object `{evidence, criterion, options: [{letter, description}]}`. Options are labelled A–P.

**Readout** ([src/jeff/head.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/head.py)), read at the last prompt position:

$$z = c\,\tanh\!\left(\frac{W_0 h}{c}\right) + A\,\frac{h-\mu}{\sigma} + b,$$

where $W_0$ are the LM-head rows for the letters A–P; $A$ and $b$ are initialised at zero, so training starts exactly at the instruct model's letter readout; $\mu$ and $\sigma$ are computed from 2,048 training rows; $c = 30$ is Gemma's soft-cap (Qwen has none). A softmax runs over the first $K$ of the 16 letter rows. The methods note describes the correction as "A·std(h)+b".

**More than 16 options.** `decide_wide` splits up to 256 options into groups of ≤16 plus a final round among the group winners (≤17 passes). The paper says "we have not evaluated this mode systematically" (§6).

**Attention mask** ([bidir.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/bidir.py)). The "full" mode sets `is_causal=False` on the full-attention layers only (Gemma's global layers, or Qwen's 1-in-4 attention layers); sliding-window and DeltaNet layers stay causal. Jeff-1.1-Large's config has `"bidirectional": null` (causal); Qwen-4B uses `"full"`. The methods note says bidirectional attention is used for Gemma-4-12B, Qwen3.6-27B and Qwen3.5-4B but not Gemma-4-31B, consistent with the configs. The paper's ablation: "bidirectional helps little, and even slightly hurts the 31B", −0.2 ± 1.1 Capability (§4).

**Long inputs.** Inputs over 8,192 tokens are rejected, or filtered by a keyword heuristic in `longinput.py`, which was added because "JevBench… counts a refused item as wrong".

**No caching.** `use_cache=False` in `hidden`; there is no prefix cache ([arXiv](https://arxiv.org/html/2610.17205) App. J).

| Property | Value |
|---|---|
| Passes per request | One full prompt per question, 1 pass (≤17 passes when K > 16 via `decide_wide`) |
| State prefix reusable | No (`use_cache=False`; most configs are bidirectional in full-attention layers) |
| Option embeddings reusable | No (options are rendered inside the joint prompt) |
| Max K | 16 native; up to 256 via the unevaluated tournament mode |
| Context | 8,192 tokens (longer inputs rejected or keyword-filtered) |

## Training

**Recipe** ([train.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/train.py), [configs](https://github.com/j-geuter/Jeff/tree/main/configs/train), paper Table 7).
- LoRA r32 / α64, dropout 0, on q/k/v/o and gate/up/down, plus `in_proj_qkv, in_proj_z, out_proj` on Qwen [@lora].
- LR 2e-5 (LoRA) and 1e-4 (head); Qwen-4B uses 1e-4 for both. WSD schedule: 250 warm-up steps, 1,500 steps total (800 for 4B).
- 65,536 padded tokens per step (8,192 tokens × grad_accum 2 × 4 GPUs); max 8,192 tokens.
- Options are reshuffled on every use; score levels are not shuffled.
- **Objective:** soft cross-entropy only (`soft_ce`), $\mathcal{L} = -\sum_{k=1}^{K} t_k \log \mathrm{softmax}(z)_k$. "No RL, reward models, or auxiliary losses."
- The release is a three-seed **weight soup**; its gain is not isolated (methods note).

**DecisionBank-260k** (Table 2; 261,561 items) [@decisionbank].

| Component | Items | Target |
|---|---|---|
| decider public and custom questions (79 tasks, ≤2,094 per task) | 163,732 | 50/50 dataset label and teacher |
| Human-vote sets | 9,168 | 50/50 vote distribution and teacher |
| LLM-written | 60,219 | Teacher only |
| Programmatic | 28,442 | Exact |

- Teacher: Qwen3.8-27B with reasoning, mean of 2 samples.
- The largest component comes from the decider (Mapika) question builders [@decider].

## Calibration

**Method** (§4, App. H). One temperature per question type, fitted by NLL on held-out task *families* and clamped to T ≥ 1:

| Model | choice | noul | score |
|---|---|---|---|
| Large | 1.49 | 1.57 | 1.11 |
| Qwen-27B | 1.42 | 1.23 | 1.01 |
| Medium | 1.42 | 1.35 | 1.06 |
| Qwen-4B | 1.29 | 1.23 | 1.17 |

**Snapping rule.** If $0.2 < P(\text{yes}) < 0.8$, the answer is moved to 0.199 or 0.801. The code states why: "JevBench counts a yes/no answer with 0.2<P(yes)<0.8 as wrong" ([model.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/model.py) `JeffModel` docstring).

**Effects of temperature scaling** (Gemma-31B zero-shot).
- Temperature scaling alone took zero-shot Gemma-4-31B from 72.9 to 79.6 Capability (methods note).
- Side effects: choice calibration 58.6 → 84.0, but score competence 64.5 → 54.1 and exact-probability items 76.3 → 52.5 (decoder-family dossier).
- Instruct Gemma is overconfident (T 2.7–6.0) while pretrained Gemma is calibrated (App. I).

**Calibration losses.**
- The methods note: Brier, focal, soft-ECE and MMCE — none lowers ECE by more than 0.01 after temperature scaling; an expected-utility loss gives +0.9 ± 0.9 accuracy and its ECE gain disappears after temperature scaling.
- The decoder-family dossier: on a frozen 4B head, no loss lowers ECE by > 0.01 vs cross-entropy; focal −2.1 ± 1.7. (The two notes describe the comparison baseline differently — "after temperature scaling" vs "vs cross-entropy" — and may refer to different experiments.)

## Results

**Table 3: OpenDecisions Capability = (I + C)/2** [@opendecisions].

| System | Capability | JevBench-Open |
|---|---|---|
| Jeff-1.1-Large | **89.0** (I 83.5 / C 94.5) | 93.1% |
| Jeff-Qwen-27B | 86.8 | – |
| Jeff-Medium | 85.0 | – |
| Jeff-Qwen-4B | 79.3 | – |
| Gemma-4-31B zero-shot | 79.6 | – |
| Jev 1.13 | 80.1 | 86.6% |
| StartLux-27B Instant | 88.5 | – |
| pplx-decider-v1.3 | 85.1 | – |

LoRA SFT adds 7.8–15.7 points over zero-shot (landscape note).

**Table 5: ablations** (Jeff 1.0 recipe, Gemma-4-31B, reference 84.4).

| Variant | Capability | Δ vs reference |
|---|---|---|
| Zero-shot | 79.6 | −4.8 |
| Head only, frozen backbone | 78.9 | −5.5 |
| LoRA, head frozen at init | 84.9 | +0.6 |
| Gemma *base* + LoRA (not instruction-tuned) | 70.3 | −14.1 |
| Public data only / no LM-labeled data | 82.5 | −1.9 |
| Hard labels instead of soft | 84.1 | −0.2 |
| LoRA rank 16 / 64 / 128 | 84.7 / 84.5 / 84.9 | flat |
| Bidirectional attention | −0.2 ± 1.1 | – |

(The zero-shot row comes from the methods note's version of this table and the bidirectional row from §4, not Table 5. The dossier labels the 82.5 row "Public data only"; the methods note labels it "No LM-labeled data".)

**Other results.**
- Zero-shot size scaling (Qwen3.5, App. F): 0.8B / 2B / 4B / 9B / 27B = 48.9 / 57.6 / 76.8 / 82.2 / 86.5, which "level off after about 4B".
- Knowledge: Jev 92.9 vs Large 89.5.
- "+R" (reasoning when unsure): 92.5 at 16.5% reasoning, 94.1 at 50%.
- Order-invariant encodings (App. G): the 4B model flips about 6% of choice and yes/no answers when option order is reversed; order-invariant (option-parallel) variants score 0.6–2.4 below the letter format — "the encoding does not help".

## Serving and efficiency

- One H200, compiled (App. J, Table 25): p50 Large 61 → 44 ms, Medium 35 → 21 ms, Qwen-4B 48 → 30 ms. Jev API p50 is 0.189 s in the same table.
- The landscape note instead gives "about 58 ms per decision on H200 (31B)"; this does not match either of App. J's Large figures (61 / 44 ms), and the note does not say which table it came from.
- No prefix cache (`use_cache=False`); every question re-encodes the full prompt.

## Flaws and failure modes

**(a) Flaws of Jeff itself**
- **Self-declared (§6):** 78% English; recipe settings were selected on OpenDecisions; the teacher (Qwen3.8-27B) is also an OpenDecisions panel model; 16 slots vs Jev's 256.
- **Leaderboard-specific calibration distortion:** the noul snapping rule exists to avoid a JevBench scoring rule and distorts calibrated probabilities *(inference, as in the dossier)*. The 8,192-token keyword filter similarly exists because JevBench counts refusals as wrong.
- **Temperature scaling trades off sub-skills:** score competence 64.5 → 54.1 and exact-probability items 76.3 → 52.5 on Gemma-31B zero-shot.
- **Order sensitivity:** ~6% flips on reversal at 4B; the exact order-invariant encodings cost 0.6–2.4 points.
- **Large K is unvalidated:** the ≤17-pass tournament for K > 16 was not evaluated systematically.
- **No reuse:** `use_cache=False` and bidirectional full-attention layers in most configs mean no state reuse across questions *(inference for the bidirectional part; StartLux states that bidirectional attention prevents reuse)*.
- **Knowledge gap:** 89.5 vs Jev 92.9.
- **Base checkpoints fail:** base + LoRA 70.3 vs instruct zero-shot 79.6.

**(b) Flaws of other systems Jeff claims to address**
- **Closed decision models (Jev):** an open recipe on open weights reaches 89.0 vs Jev 1.13's 80.1 Capability and 93.1% vs 86.6% on JevBench-Open.
- **Head-only training on frozen features** (the closest analog to a CLM-style recipe, per the methods note) underperforms zero-shot letter readout: 78.9 vs 79.6. The landscape note adds that S1-Arena shows CLM-v0.1-8B losing 17.8 pp to its own backbone's answer-token readout ([S1-Arena](https://arxiv.org/html/2610.21244)) [@s1arena].

## Implications for our design

All bullets are *(inference)*, grounded in the evidence above and in the methods and decoder-family notes.

**Adopt**
- **A same-backbone letter-logit readout as both a baseline and a slow path** (the methods note costs it at about 1 day zero-shot, about 1 week with LoRA). It is joint encoding and cannot reuse cached option embeddings, so it complements rather than replaces a cached-option fast path.
- **Initialise any readout from the instruct model's LM-head letter rows with a zero-init correction**, so training starts at a strong point; avoid base checkpoints for a letter readout.
- **Adapt the backbone (LoRA)** rather than training a head on frozen features: head-only was −5.5 vs the reference and below zero-shot.
- **Per-type temperatures, T ≥ 1, fit on held-out task families**; temperature scaling alone gave +6.7 Capability to zero-shot Gemma.
- **DecisionBank soft targets** (label/teacher 50/50, exact targets for programmatic items) as ready-made training data; soft vs hard labels was −0.2 here, but LM-labeled data was worth 1.9.

**Avoid**
- **Noul snapping and other benchmark-specific post-processing**.
- **Expecting gains from bidirectional attention** (−0.2 ± 1.1), from LoRA rank (flat 16–128), or from exotic calibration losses (≤0.01 ECE after temperature scaling).
- **Treating a 16-slot readout as general:** any K > 16 requires tournaments that were never evaluated.

## References

- [@jeff] https://arxiv.org/html/2610.17205 (also https://arxiv.org/abs/2610.17205)
- [@jeff-repo] https://github.com/j-geuter/Jeff — [head.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/head.py), [prompting.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/prompting.py), [bidir.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/bidir.py), [train.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/train.py), [model.py](https://github.com/j-geuter/Jeff/blob/main/src/jeff/model.py), [configs](https://github.com/j-geuter/Jeff/tree/main/configs/train)
- HF collection: https://huggingface.co/collections/jgeuter/jeff-11-6ada6708f3ed49beaf0f97ae
- [@decisionbank] https://huggingface.co/datasets/jgeuter/DecisionBank (viewer: https://huggingface.co/datasets/jgeuter/DecisionBank/viewer/DecisionBank-260k)
- [@opendecisions] https://huggingface.co/datasets/jgeuter/OpenDecisions
- [@semif] https://github.com/TheoLeeCJ/SemIf-OpenJev
- [@decider] https://github.com/Mapika/decider
- [@lora]
- [@s1arena] https://arxiv.org/html/2610.21244
