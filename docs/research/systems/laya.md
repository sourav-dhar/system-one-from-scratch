# Laya

> **Family:** encoder (non-autoregressive, one `[MASK]` per option) · **Team:** Convai Innovations (repo owner NandhaKishorM; the preprint's Crossref record lists Nandakishor M and Anjali M) · **Released:** GitHub repo created 2026-09-18; HF weights around 2026-09-18 per secondary sources; Laya 0.5 (date not recorded) makes the vegaml engine the default · **License:** Apache-2.0 · **Code/weights:** [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) [@laya-repo]; [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) [@laya-card] (`laya`, `laya-multilingual`, `laya-typed-decisions`) · **Sources read:** code (`laya/common.py`, `train.py`, `calibrate.py`, `agent.py`, `serve.py`), `BENCHMARKS.md`, `research/README.md`, issues #131/#156/#185/#394/#741/#963, HF card, the author's dev.to post, and third-party evaluations (S1-Arena, TypedBench, Rafe & Das, Runoff, sbert2s1, Intern-Decision, Decision Index, JevBench). **Not read:** the "From Laya to Vega" preprint [@laya-paper] returned HTTP 403. Only its Crossref metadata and abstract were read, and nothing below is attributed to it beyond those. **Not shipped:** the base-checkpoint training corpus and the base RLCD run.

## Summary

Laya is the most-starred open System One implementation: 35,758 GitHub stars per the GitHub API in the encoder dossier, and 35,727 in a catalog snapshot dated 2026-11-20 ([repo](https://github.com/NandhaKishorM/laya); [@laya-repo]). It fully fine-tunes a ModernBERT-large (421M) or mmBERT-base (322M) encoder. It places a `[MASK]` token before each option, refines the sequence with a 2-layer Transformer head, and scores each option from the hidden state at its `[MASK]`.

The training objective is called "RLCD": a policy gradient on Gaussian-perturbed logits with proper-scoring-rule rewards. In the released code the policy-gradient term is *added to* a soft cross-entropy. The repository's own three-arm ablation ([issue #741](https://github.com/NandhaKishorM/laya/issues/741)) finds that the RL term adds nothing measurable. Calibration is a per-type scalar temperature, clamped to $[0.5, 5]$. The clamp was added after a shipped temperature of 0.1006 reported coin flips at confidence 1.0 ([#394](https://github.com/NandhaKishorM/laya/issues/394)).

Results split sharply. Zero-shot, the base checkpoint is near chance on typed-decisions (0.362, below the 0.461 majority baseline). Fine-tuned, it reports 0.766 against Jev 1.13's 0.727, a number with "no committed result file behind it yet" ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)). Independent evaluations place it near the bottom of the field (S1-Arena 44.5%, rank 27/29).

Since Laya 0.5, `laya.Agent()` routes to the frozen-reader vegaml engine by default, and the classic encoder needs `vegaml=False` ([HF card](https://huggingface.co/convaiinnovations/laya)). The vegaml engine is described in [vegaml.md](vegaml.md).

**Lineage.** The author's [dev.to post](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me) says a 2025 predecessor used frozen embeddings plus a PPO value network (arXiv 2503.23303, 2510.01237).

## Architecture

### Checkpoints

| Checkpoint | Backbone | Parameters | Context |
|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (up to 8k) |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 |

The ModernBERT-large backbone has 28 layers, $d=1024$ and 16 heads, per the dev.to post. It is fully fine-tuned.

### Input layout

The layout comes from `laya/common.py::build_sequence` / `build_head` [@laya-repo]:

```
[CLS] "{type} question: {ins}" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
```

- **Option budget.** Each option is capped at 48 tokens. If the option block exceeds `head_max_len` (192 for English, 256 for multilingual), every option is cut to $\max\big(4, \lfloor(\texttt{head\_max\_len}-16)/K\rfloor\big)$ tokens.
  - *(inference)* At $K=77$ this formula gives the 4-token floor for both budgets. That is consistent with the notes' "about 3–4 tokens each".
- **State budget.** The state gets whatever remains: $\texttt{max\_len} - |\text{head}| - 1$ tokens. It is right-truncated, or left-truncated for conversation lists.
- **Sanitising.** Literal `[MASK]` strings in user text are replaced with a space.
- **Noul options** are hard-coded as `false: …` / `true: …`.
- **Order.** The question and options come *before* the state.

### Scoring model (`class DecisionModel`)

Let $x$ be the token sequence and $t$ the question type. The code has three types, which index a type embedding and a temperature buffer. Let $m_1,\dots,m_K$ be the positions of the $K$ `[MASK]` tokens. The modules, in the order the notes list them, are:

$$h_i \leftarrow \mathrm{Enc}_\theta(x)_i + e_{\text{type}}(t) \quad \text{for every position } i$$

$$\tilde H = \mathrm{TransformerEncoder}_{2}(H), \qquad d_{\text{model}}=d,\; n_{\text{head}}=d/64,\; d_{\text{ff}}=4d,\; \text{dropout } 0.1,\; \texttt{norm\_first=True}$$

$$z_k = W_2\,\mathrm{GELU}\big(W_1\,\mathrm{LN}(\tilde h_{m_k}) + b_1\big) + b_2, \qquad W_1 \in \mathbb{R}^{d\times d},\; W_2 \in \mathbb{R}^{1\times d}$$

$$p = \mathrm{softmax}\big(\hat z / T_t\big), \qquad \hat z_k = \begin{cases} z_k & \text{valid option} \\ -10^4 & \text{padding} \end{cases}$$

The notes say the logits are masked to $-10^4$ and softmaxed per question. They also say a `temperature` buffer holds one value per type. *(inference)* Dividing by $T_t$ at inference follows from the calibrator, which fits $\mathrm{softmax}(z/T)$ (see [Calibration](#calibration)).

**Act/escalate head.** This is a separate MLP over the CLS state and summary statistics of the question's output:

$$a = W_4\,\mathrm{GELU}\big(W_3\,[\,h_{\text{CLS}};\ \pi_{(1)};\ \pi_{(1)}-\pi_{(2)};\ H(p)/\log K;\ K/255\,]\big), \qquad W_3 \in \mathbb{R}^{256\times(d+4)},\; W_4 \in \mathbb{R}^{n_{\text{act}}\times 256}$$

Here $\pi_{(1)}, \pi_{(2)}$ are the code's "top1" and "top2". The notes do not say whether these are probabilities or logits. In [#185](https://github.com/NandhaKishorM/laya/issues/185) this head is dead (see [Flaws](#flaws-and-failure-modes)).

**Two confidence fields** (`agent.py::_decode_answers`):
- `confidence` for choice and score is $1 - H(p)/\log K$. It is not calibrated.
- `answer_confidence` is $\max_k p_k$.

### Passes per request and caching

- **One row per question.** Each question is its own row. The state is tokenized once and reused across rows (`agent.py::_encode_state`), and the batch runs as one tensor.
  - So a request with $N$ questions costs one batched forward pass over $N$ rows of at most `max_len` tokens.
  - The landscape notes paraphrase the HF card as "all questions are answered in one forward pass". *(inference)* That is consistent with one batched call. Compute still grows linearly with $N$: on a T4, 1/5/10/50 questions take 39.5/84.5/158.6/771 ms (English) ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) "Speed").
- **No caching across questions** *(inference)*. The encoder is bidirectional and the state sits after the question-specific head, so no prefix is shared between questions.

### Order-blind "parallel layout" (shipped disabled)

`parallel_layout` / `parallel_option_masks` give every option the same start position and block cross-option attention.
- The docstring says "sequential [is what] every published checkpoint was trained on".
- The mode needs transformers ≥5 (`uses_parallel_layout`).
- Laya cites Parallel Context Windows [@pcw].

### Properties

| Property | Value |
|---|---|
| Backbone | ModernBERT-large 421M (`laya`, `laya-typed-decisions`); mmBERT-base 322M (`laya-multilingual`); fully fine-tuned |
| Attention | Bidirectional encoder |
| Context | 512 (`laya`); 1024, up to 8k (`laya-multilingual`); 1024 (`laya-typed-decisions`) |
| Option encoding | `[MASK]` before each option, in sequence; ≤48 tokens per option; option block ≤192/256 tokens |
| Head | 2-layer pre-norm Transformer, then LN–MLP scorer at each `[MASK]`; type embedding; act/escalate MLP |
| Question types | 3 (one temperature each) |
| Passes per request | One encoder row per question, batched |
| Cacheable across questions or requests | No *(inference)* |
| Order invariance | No. The parallel layout exists but no checkpoint was trained with it |
| Output | Per-question softmax; `confidence` $=1-H/\log K$; `answer_confidence` $=\max p$; act probability |
| Calibration | Per-type scalar $T$, clamped to $[0.5, 5]$ |

## Training

### RLCD objective as implemented

Source: `laya/common.py::proper_reward` and `laya/train.py::rlcd_loss` [@laya-repo]. The derivation is written from the code because the preprint could not be read.

Take a question with $K$ options, logits $z\in\mathbb{R}^K$, a target distribution $t\in\Delta^{K-1}$ (one-hot or soft) and a fixed noise scale $\sigma$. Let $\mathrm{sg}$ denote stop-gradient.

$$\varepsilon \sim \mathcal{N}(0, \sigma^2 I_K),\quad \bar\varepsilon = \varepsilon - \tfrac{1}{K}\textstyle\sum_k \varepsilon_k,\quad \tilde z = \mathrm{sg}(z) + \bar\varepsilon,\quad q = \mathrm{softmax}(\tilde z),\quad S \text{ samples (default } 4) \tag{1}$$

$$r(q,t) = \sum_k t_k \max(\log q_k, -9.21) \;+\; w_{\text{sph}}\,\frac{t^\top q}{\lVert q\rVert_2} \;-\; w_{\text{rps}}\,\mathbb{1}[\text{score}]\,\frac{1}{K-1}\sum_k \big(F_{q,k} - F_{t,k}\big)^2 \tag{2}$$

In (2), $F_{q,k}$ and $F_{t,k}$ are the cumulative distributions of $q$ and $t$ at option $k$.
- The three terms are a clipped log score, a spherical score, and a ranked probability score (RPS) that applies only to `score` questions.
- `train.py` defaults to $w_{\text{sph}}=0.75$ and $w_{\text{rps}}=1.0$. `proper_reward` itself defaults to $w_{\text{sph}}=0.5$.
- *(inference)* $-9.21 \approx \ln 10^{-4}$.

$$A^{(s)} = \frac{r^{(s)} - \frac{1}{S}\sum_{s'} r^{(s')}}{\operatorname{std}_{\text{batch}}\big(r - \bar r\big)} \tag{3}$$

In (3), the baseline is the mean over the $S$ samples of the same question. The result is divided by **one** standard deviation taken over the whole batch (`adv.std()`).

$$\log \pi(\tilde z \mid z) = -\frac{\lVert \tilde z - z\rVert_2^2}{2\sigma^2} \tag{4}$$

$$\mathcal{L} = -\operatorname{mean}_{b,s}\big(A^{(s)} \log\pi(\tilde z^{(s)}\mid z)\big) \;+\; \mathrm{softCE}(z, t), \qquad \mathrm{softCE}(z,t) = -\sum_k t_k \log \mathrm{softmax}(z)_k \tag{5}$$

- Setting `loss="soft-ce"` drops the policy-gradient term.
- *(inference)* Because $\tilde z$ is built from $\mathrm{sg}(z)$, the policy-gradient term reaches $z$ only through the $z$ in (4).

$$\sigma_e = 0.4 + (0.1 - 0.4)\,\frac{e}{E-1} \quad \text{(linear per epoch from 0.4 to 0.1, \texttt{sigma\_at})} \tag{6}$$

(7) **TD($\lambda$) is defined but unused.** `td_lambda_targets` has $\lambda=1$: a Monte-Carlo terminal target $G$ backed up over a trajectory's prefixes, for noul. It is defined and unit-tested, but **`train.py` never calls it**. Code search hits only `common.py`, `__init__.py`, docs and tests.

### Discrepancy: author's description vs released loop

The dev.to post describes the **base** training run, which is not shipped. `train.py` is the released (fine-tune) loop. The two cannot be reconciled from the sources.

| Setting | dev.to (base run) | Released `train.py` |
|---|---|---|
| Loss | "pure policy gradient (zero supervised cross-entropy loss)" | Policy gradient + soft-CE |
| Samples per question | $G=8$ | $S=4$ |
| $\sigma$ schedule | 1.0 → 0.3 | 0.4 → 0.1 |
| $w_{\text{sph}}$ | 0.5 | 0.75 (`proper_reward` default 0.5) |
| Act cost matrix | +1 / −3 / escalate −0.5 | not recorded in the notes |
| TD($\lambda$) for multi-turn | The HF card (per the landscape notes) describes TD($\lambda$) over prefixes | Defined, never called |

The HF card, as summarised in the landscape and methods notes, also describes REINFORCE with a group-mean baseline ("GRPO-style") and does not give $\sigma$.

### Hyperparameters

These are the `TrainConfig` defaults; the code says they "train the way the published notebook does".

| Setting | Value |
|---|---|
| Epochs | 4 |
| Batch | micro 8 × accumulation 8 = 64 effective |
| Learning rate | encoder 2.5e-5, head 1e-4, minimum 1e-6 |
| Optimizer / schedule | AdamW, weight decay 0.01, cosine |
| Gradient clip | 1.0 |
| Precision | fp16 autocast on CUDA |
| Calibration holdout | min(400, 10%), seed 20260922 |
| Option shuffling | `shuffle_options=()`, i.e. **off by default** |

### Data

- **Base model.** The dev.to post says "100% human-labeled public datasets" in six families:
  - support/intent;
  - NLI/fact-checking;
  - safety;
  - guardrails;
  - rubric ratings;
  - multi-turn sales/support.

  The augmentations are option shuffle, paraphrase, JSON vs raw state, and distractor questions. **Sizes are not given.** The data-inventory notes found no named dataset: a Kaggle notebook builds the data and the teacher is unnamed. The sources do not reconcile "human-labeled" with a teacher.
- **Fine-tune.** The fine-tune uses `LocalLLaMA/typed-decisions` [@typed-decisions], with 6,000 decisions in the train split ([#741](https://github.com/NandhaKishorM/laya/issues/741)).
  - Its `gold` field holds "a teacher's probabilities per question" ([docs/finetune.md](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)).
  - The dataset has 1,600 cases (400 test cases, 2,000 test decisions). Each case is one of 4 workflows × 5 typed questions. It is Apache-2.0 and synthetic, built from a teacher.

### Issue #741: the RL term adds nothing measurable

[Issue #741](https://github.com/NandhaKishorM/laya/issues/741) ran three arms on an A100, on typed-decisions, with 1,200 test decisions:

| Arm | Accuracy | ECE | Brier |
|---|---|---|---|
| soft-CE | **80.50%** | 11.82% | **0.0501** |
| RLCD (PG + soft-CE) | 79.08% | **11.18%** | 0.0543 |
| Direct proper-score gradient | 79.92% | 12.04% | 0.0516 |

Several sources agree with the result:
- **The issue's explanation:** soft-CE already *is* the expected log score, and the policy-gradient term is a high-variance estimate of a gradient that can be computed exactly.
- **`train.py`'s docstring** concedes the point ("#741 measured no gain").
- **The Crossref abstract** says "a supervised baseline matches the RL objective on single-step questions" ([Crossref](https://api.crossref.org/works/10.20944/preprints202611.0781.v1)).
- **sbert2s1** reports that Laya's RLCD estimator is 3.6–15× too large in scale (Table 6), and that a leave-one-out variant (PG-LOO) fixes this. After temperature scaling, every objective it tests sits at 3.6–4.4 ECE (Table 4) ([arXiv 2610.02486](https://arxiv.org/html/2610.02486); [@sbert2s1]).

**Gaps.**
- The notes do not say which 1,200 of the 2,000 test decisions #741 used.
- No seeds or confidence intervals are recorded. *(inference)* The 1.4-point accuracy gap is therefore suggestive, not conclusive. The conclusion "no gain" is robust, but "soft-CE is better" is not established.

## Calibration

**Fitter** (`laya/calibrate.py::_fit_unclamped_temperature`). One scalar $T$ per group, fit by LBFGS on $\log T$ to minimise the soft NLL of $\mathrm{softmax}(z/T)$:

$$T^\star_g = \arg\min_{T>0} \; -\sum_{i\in g}\sum_k t_{ik}\,\log \mathrm{softmax}(z_i/T)_k, \qquad T_g = \mathrm{clip}(T^\star_g, 0.5, 5)$$

- **Buckets.** One per type, plus `temp_bucket` = type × $K\in\{2, 3\text{–}5, 6\text{–}10, 11+\}$. sbert2s1 uses the same buckets, also fit by L-BFGS on $\log T$.
- **Floors.** `MIN_BUCKET_N = 2000` and `MIN_TYPE_N = 10`. *(inference)* With `calib_max = 400`, fine-tunes only ever get type-level temperatures.
- **Clamp** `common.py::clamp_temperature`, $[0.5, 5]$. It was added after the shipped `choice:11+` temperature of 0.1006, a roughly 10× sharpener, published coin flips as 1.0 confidence ([#394](https://github.com/NandhaKishorM/laya/issues/394)).
- **Abstention.** `fit_abstention_thresholds` uses a selective-risk cut: target error 0.10, bucket floor 100.
- **Before vs after refit** ([BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) "Calibration"):

| Checkpoint | ECE as shipped | ECE after refit |
|---|---|---|
| `laya` | 0.466 | 0.081 |
| `laya-multilingual` (shipped with no temperatures) | 0.314 | 0.106 |

**Conflicting attributions.**
- The methods notes paraphrase the HF card as adding split-conformal sets at $\alpha=0.10$, with abstention that keeps error among answered items at or below 10%.
- The landscape notes attribute conformal sets to the 0.5 vegaml engine. vegaml does ship split-conformal thresholds (see [vegaml.md](vegaml.md)).
- The notes on the classic `calibrate.py` record only the selective-risk abstention fitter. Whether the classic encoder path has a conformal fitter is **not resolved** by the notes.

**Third-party evidence.**
- TypedBench fits $T = 5.04$ for Laya; a fitted $T>1$ means overconfident ([arXiv 2610.11392](https://arxiv.org/html/2610.11392); [@typedbench]).
- **Laya's claim against Jev.** Laya claims a post-refit ECE of 0.081 vs 0.246 for Jev. Laya's own typed-decisions table, however, shows Jev with *better* ECE: 0.144 vs 0.213 for `laya-typed-decisions`.
- *(inference)* The value 0.246 also appears, in the Intern-Decision README, as *Laya's own* JevBench-Hard ECE. The notes do not let us tell whether this is a coincidence or a transcription error.

## Results

### Typed-decisions (400 cases / 2,000 decisions; [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md))

| Model | Accuracy | Brier | ECE |
|---|---|---|---|
| `laya-typed-decisions` (fine-tuned) | 0.766 | 0.061 | 0.213 |
| `laya` base | 0.362 | 0.315 | — |
| `laya-multilingual` base | 0.352 | — | — |
| Jev 1.13 (published) | 0.727 | 0.148 | 0.144 |

- Reference points: majority class 0.461, random 0.318, teacher ceiling 0.735.
- The 0.766 has "no committed result file behind it yet".
- *(inference)* It also exceeds the 0.735 teacher ceiling.

### Public sets (N = 400)

| Dataset | `laya` | `laya-typed-decisions` | Jev |
|---|---|---|---|
| AG News | 0.950 | 0.953 | 0.910 |
| DAIR Emotion | 0.595 | 0.600 | 0.480 |
| banking77 | 0.425 | 0.492 | 0.870 |

### Themes and languages

- **Themes.**
  - In the training mix: spam 0.993, phishing 0.980.
  - Held out: jailbreak 0.708, toxicity 0.530, routing 0.639. `laya-multilingual` scores 0.123 on routing.
- **MASSIVE** (51 languages, 20 options): macro accuracy is 0.227 for English `laya` and 0.401 for multilingual.
- **Khmer** on English `laya`: 0.000 accuracy at 0.952 confidence.
- **Long documents.** With 7,000 filler tokens, accuracy is 0.85 when the evidence is at position 0 and 0.40 at position 1.0 ([research/README.md](https://github.com/NandhaKishorM/laya/blob/main/research/README.md)).

### Option-order sensitivity, by source

The figures come from different protocols and are not comparable with each other.

| Source | Setting | Flip rate |
|---|---|---|
| [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) | massive_intent.en, $K=20$ | 0.150 (English) / 0.230 (multilingual); Jev 0.13 |
| Phocinae technical report ([phocinae.md](phocinae.md)) | in-domain / out of domain | 3.7% / 19.4% |
| sbert2s1 Table 12 [@sbert2s1] | Laya fine-tuned; generic "C" `[MASK]` head | 9.8%; 15.7% |
| Runoff Table 3 [@runoff] | Laya native | 30.6 |
| Cross-cutting table in the flaw-paper notes | range | 30.6–66.8% (the notes give no primary source for the upper bound) |

### Third-party evaluations

| Evaluation | Laya result | Reference points |
|---|---|---|
| S1-Arena, non-agentic ([arXiv 2610.21244](https://arxiv.org/html/2610.21244) Table 1, App. B; [@s1arena]) | 44.5% accuracy, Brier 0.660, p50 136 ms, rank 27/29. "The Laya library cuts an input to 512 tokens"; no other model's input is shortened except CLM's last-2,048 tail | Vega-27B (vLLM-SR) 73.0, Jev 72.6, Kai-0.6B 50.3, CLM 41.4 |
| TypedBench [@typedbench] | Priority 0.343 (prior 0.338); $T=5.04$; collapses before 100 options; ignores negation (0.10–0.28); upper-casing −0.305; keeps ordinal order on 1 of 4 ladders at 5 and 10 levels | — |
| Rafe & Das ([arXiv 2610.00346](https://arxiv.org/html/2610.00346); [@rafedas]) | Yes/no swap: 91.3 flips per 100; accuracy 0.852 at 20 options → 0.020 at 50; as a cascade first stage it escalates 99–100% of decisions | Jev 50.5 flips; fine-tuned decision models ≤6.5 |
| Intern-Decision's seven suites ([README](https://github.com/InternLM/Intern-Decision); [@interndecision-repo]) | Average 57.77; WildJailBreak 14.84; JevBench-Hard Brier 0.804 / ECE 0.246 | Intern-Decision-4B 90.02; Jev 88.74 |
| Decision Index 0.2.1, Cloudflare's run ([clef-flash card](https://huggingface.co/Cloudflare/clef-flash)) | BANKING77 14.3, CLINC150 3.2, MMLU 30.7; median 5.8 ms, p95 222.5 ms | BANKING77: Clef 94.2, Jev 79.7 |
| JevBench v1.4 ([Von card](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md)) | I 36.1, C 63.7, composite 30.3, sealed ECE 0.172 | Von: 34.5 / 75.7 / 27.5 / 0.107 |
| JevBench v1.5.4 ([BenchLM](https://benchlm.ai/md/decision-models.md)) | Composite about 0 | — |
| fast-decisions ([GLiNER2.5-Decide card](https://huggingface.co/fastino/GLiNER2.5-Decide)) | "Laya Router" 46.6% (checkpoint not specified) | GLiNER2.5-Decide 60.2, JevK5 57.6 |
| sbert2s1 [@sbert2s1] | A Laya-style `[MASK]` cross head ("C") scores 32.8 held-out vs 23.4 for a bi-encoder ("B"); on seen tasks B is within 0.6 of C | — |

- The two JevBench versions disagree badly: composite 30.3 on v1.4 vs about 0 on v1.5.4. The landscape notes warn that leaderboards sharing the "JevBench" name use different protocols.
- A pentest-harness paper evaluates Jev and Laya ([arXiv 2609.28940](https://arxiv.org/abs/2609.28940); [@pentest]). The notes record only its title.

## Serving and efficiency

- **`laya-serve`** exposes `POST /v1/systemone`, `/v1/systemone/batch` and `/v1/route`. Auth is optional through `LAYA_API_KEY`, and `LAYA_MAX_BATCH_TOKENS` defaults to 131072 ([serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)).
- **SDKs and ports.**
  - Shipped: ONNX, C++, .NET, Rust and TypeScript SDKs.
  - Community ports: MLX, Core ML, laya.cpp, llama.cpp and Unsloth.
- **Hosted access.** Laya was free on the Vercel AI Gateway through October 31 ([Vercel changelog](https://vercel.com/changelog/laya-decision-model-now-available-on-ai-gateway-free-through-october-31)).

| Setting | Latency or throughput |
|---|---|
| T4, English, 1/5/10/50 questions | 39.5 / 84.5 / 158.6 / 771 ms. The landscape notes quote 32.8–39.5 ms for one question |
| TileLang fast path, RTX 4070 Ti S, 1 question | 17.7 → 4.6 ms (`laya`); 14.1 → 2.8 ms (multilingual) |
| CPU, 4-core EPYC | 580 ms per question (English); 193 ms (multilingual) |
| H100 NVL, TensorRT FP16 (independent capacity study) | 105 decisions/s at p99 ≤ 50 ms; 175 decisions/s at p99 ≤ 130 ms; all 74 backend rows matched the FP32 parity answers |
| S1-Arena p50 | 136 ms |
| Decision Index (Cloudflare's run) | median 5.8 ms, p95 222.5 ms |

*(inference)* On a T4, each extra question costs about 15 ms: (771 − 39.5)/49. This is the price of the one-row-per-question layout. A state-first causal layout could reuse the state's KV for all questions.

## Flaws and failure modes

### (a) Laya's own flaws, with evidence

Numbers already tabulated under [Results](#results) are only referenced here.

| # | Flaw | Evidence |
|---|---|---|
| 1 | Near chance zero-shot | typed-decisions 0.362 < 0.461 majority ([HF card](https://huggingface.co/convaiinnovations/laya) "Honest Limits") |
| 2 | Option-count collapse | banking77 exactly 0.425 on both base checkpoints (77 options share the 192/256-token head budget); right-vs-wrong confidence gap 0.009 at $K=20$ ([#394](https://github.com/NandhaKishorM/laya/issues/394)); 0.852 → 0.020 from 20 to 50 options (Rafe & Das); collapse before 100 options (TypedBench) |
| 3 | Yes/no label flips | With `true`/`false` or `yes`/`no` label words, noul returns the negative label in both polarities at confidence 1.0 ([#156](https://github.com/NandhaKishorM/laya/issues/156)); 91.3 flips per 100 (Rafe & Das); sbert2s1 finds no encoder reliably above chance on label-word options |
| 4 | Slot bias | `laya-multilingual` never picks the first-listed score option (0/290, English); noul AUROC 0.355 ([#131](https://github.com/NandhaKishorM/laya/issues/131)) |
| 5 | Dead act head | Head activations reach about 300× the encoder's scale; act logits [4499, −3677]; `act_probability` always 1.0 ([#185](https://github.com/NandhaKishorM/laya/issues/185)) |
| 6 | Collapse to the prior on small fine-tunes | At 72 updates the within-row logit range falls to 0.01, and 6 of 15 seeds never leave the $\ln K$ plateau ([#963](https://github.com/NandhaKishorM/laya/issues/963)) |
| 7 | Over-confident as shipped | ECE 0.466; $T=0.1006$ on `choice:11+` ([#394](https://github.com/NandhaKishorM/laya/issues/394)); TypedBench $T=5.04$; Khmer 0.000 accuracy at 0.952 confidence |
| 8 | Long-document position sensitivity | 0.85 at position 0 vs 0.40 at position 1.0 with 7,000 filler tokens |
| 9 | Order sensitivity | Only the sequential layout is trained; `shuffle_options` is off by default; flip rates 0.150 to 30.6 depending on protocol |
| 10 | Training objective | Policy gradient adds no gain (#741); estimator 3.6–15× too large in scale (sbert2s1); TD($\lambda$) advertised but never called; base run and data not released |
| 11 | Negation and wording | TypedBench: ignores negation (0.10–0.28); upper-casing −0.305 |
| 12 | Short context in third-party runs | S1-Arena cut inputs to 512 tokens |
| 13 | Unbacked headline | 0.766 has no committed result file and exceeds the stated 0.735 teacher ceiling |

### (b) What Laya claims to fix in other systems

- **Against Jev:**
  - latency, 6–8× faster;
  - weights that users can adapt;
  - post-refit ECE, 0.081 vs 0.246.
- **Caveats.**
  - Laya's own typed-decisions table shows Jev with better ECE: 0.144 vs 0.213.
  - The 0.246 coincides with Laya's own JevBench-Hard ECE as reported by Intern-Decision.
  - Jev beats Laya on banking77 (0.870 vs 0.425/0.492), and in every third-party comparison in the notes that includes both: S1-Arena, Intern-Decision's suites, and Decision Index BANKING77.

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **Plain soft-CE + Brier as the proper-score loss.** Skip noisy-logit policy gradient. #741, the Crossref abstract and sbert2s1 all point the same way.
- **Temperature scaling with guard rails:**
  - per type, and per $K$ only when the bucket has enough data (Laya's floors are a reasonable template);
  - a clamp on $T$;
  - ECE reported before and after the refit.
- **A single calibrated confidence field.** Expose $\max p$ (or Jev's normalised margin), not $1-H/\log K$.
- **Noul hygiene.** Use neutral option keys and same-state negated twins. Laya's #156 and Rafe & Das's 91.3 flips show label-word options failing in encoders.
- **Normalised inputs to auxiliary heads.** The act head died from a roughly 300× scale mismatch (#185). Any coverage or abstain head we add over raw scores needs LayerNorm or standardisation.
- **Option shuffling by default during training.**
- **Laya as a baseline.** Use it as a cheap encoder baseline and an "after fine-tuning" comparison. It speaks `/v1/systemone`, so it is a config entry in a TypeSafe-compatible harness.

**Avoid**
- **A fixed per-option token budget inside a shared sequence.** This causes the banking77/K=77 collapse. Our bi-encoder encodes each option separately and so avoids it by construction.
- **A layout that puts the state after question-specific tokens.** It forbids KV or embedding reuse across questions.
- **Small encoder backbones for reading-heavy tasks.** Zero-shot near chance and a 44.5% S1-Arena score argue against them.
- **Trusting shipped temperatures.** Refit them on held-out data from *other* datasets.

## References

- Laya code, issues and benchmarks: [github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) [@laya-repo]
  - [BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
  - [research/README.md](https://github.com/NandhaKishorM/laya/blob/main/research/README.md)
  - [docs/finetune.md](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md)
  - [serve.py](https://github.com/NandhaKishorM/laya/blob/main/laya/serve.py)
  - Issues [#131](https://github.com/NandhaKishorM/laya/issues/131), [#156](https://github.com/NandhaKishorM/laya/issues/156), [#185](https://github.com/NandhaKishorM/laya/issues/185), [#394](https://github.com/NandhaKishorM/laya/issues/394), [#741](https://github.com/NandhaKishorM/laya/issues/741), [#963](https://github.com/NandhaKishorM/laya/issues/963)
- Laya model card: [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) [@laya-card]
- "From Laya to Vega" preprint, **not retrievable (403)**: [doi.org/10.20944/preprints202611.0781.v1](https://doi.org/10.20944/preprints202611.0781.v1) [@laya-paper]. Crossref metadata and abstract: [api.crossref.org](https://api.crossref.org/works/10.20944/preprints202611.0781.v1)
- Author's post: [dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)
- vegaml (Laya 0.5 default engine): [github.com/NandhaKishorM/vegaml](https://github.com/NandhaKishorM/vegaml) [@vegaml]
- typed-decisions dataset: [HF](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) [@typed-decisions]
- Parallel Context Windows (cited by Laya): [@pcw]
- S1-Arena: [arXiv 2610.21244](https://arxiv.org/html/2610.21244) [@s1arena]
- TypedBench: [arXiv 2610.11392](https://arxiv.org/html/2610.11392) [@typedbench]
- Rafe & Das: [arXiv 2610.00346](https://arxiv.org/html/2610.00346) [@rafedas]
- Runoff: [arXiv 2610.18233](https://arxiv.org/html/2610.18233) [@runoff]
- sbert2s1: [arXiv 2610.02486](https://arxiv.org/html/2610.02486) [@sbert2s1]
- Pentest harnesses (Jev and Laya): [arXiv 2609.28940](https://arxiv.org/abs/2609.28940) [@pentest]
- Intern-Decision README: [github.com/InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision) [@interndecision-repo]
- Clef-flash card (Decision Index 0.2.1 run): [huggingface.co/Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
- Von model card (JevBench v1.4): [MODEL_CARD.md](https://github.com/wfzyx/von/blob/main/hf/MODEL_CARD.md) [@von]
- GLiNER2.5-Decide card (fast-decisions): [HF](https://huggingface.co/fastino/GLiNER2.5-Decide) [@gliner-decide]
- BenchLM JevBench list: [benchlm.ai](https://benchlm.ai/md/decision-models.md)
- Vercel AI Gateway changelog: [vercel.com](https://vercel.com/changelog/laya-decision-model-now-available-on-ai-gateway-free-through-october-31)
