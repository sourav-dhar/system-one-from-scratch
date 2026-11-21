# vegaml (Vega engine, vega-U)

> **Family:** frozen-reader (a small decision engine over intermediate hidden states of a frozen causal LM) · **Team:** Nandakishor M (author of Laya; GitHub NandhaKishorM, HF nandakishorm) · **Released:** repo created 2026-10-09; since Laya 0.5, `laya.Agent()` routes here by default · **License:** Apache-2.0 · **Code/weights:** [github.com/NandhaKishorM/vegaml](https://github.com/NandhaKishorM/vegaml) [@vegaml] (315 stars); default model [nandakishorm/vega-U](https://huggingface.co/nandakishorm/vega-U) · **Sources read:** `vegaml/_common.py`, `ttt.py`, `learn/learner.py`, `notebooks/vega_finetune.py`, README, [docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md), [docs/calibration.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/calibration.md), the vega-U HF card and its `vega_config.json`. **Not available:** the engine's pre-training code, which the README says ships "only in the training tree". **Not read:** the "From Laya to Vega" preprint [@laya-paper] (HTTP 403); only its Crossref metadata and abstract were read, and nothing here is attributed to it. **Not found:** any independent evaluation. `ttt.py` is listed as read, but the notes record nothing from it.

**Name collision.** The "Vega" in S1-Arena, the vLLM-SR cascade and the methods notes is vLLM Semantic Router's **Decision-2.0-Vega-27B**, a different, Qwen3.8-27B-based system. Its results include S1-Arena 73.0% / Brier 0.343 / 228 ms, agentic loop success 26.5%, and "System-1 vs backbone +2.9". None of those numbers belong to vegaml.

## Summary

vegaml replaces Laya's fine-tuned encoder with two parts:
- a **frozen causal LM ("reader")**, run only up to its deepest tapped layer. Its LM head never runs.
- a **14.3M-parameter, attention-free "engine"**. The engine turns pooled span features into a 64-d particle, rolls it for up to 12 damped steps in a potential with one Gaussian well per option, and reads probabilities from the energies of the wells.

The released default, **vega-U**, maps several readers into one shared 1024-d feature space through per-reader linear "ports", so one engine serves many backbones. A new reader can be attached without labels.

Calibration has three parts: a per-decision "thermal" temperature computed from the settling dynamics, split-conformal thresholds, and abstain and "unbound" thresholds.

The self-reported results are:
- held-out accuracy 0.738–0.811 and zero-shot accuracy 0.504–0.629 across four readers, with ECE 0.007–0.021;
- a per-customer "Learner" that rises from 84.7% to 96.0% after 120 confirmed answers.

These numbers have three caveats:
- the "held-out" sets are in-distribution splits of the 708,657-item training mixture;
- the thermal temperature improves held-out NLL by about 0.002 over a scalar;
- the authors report that "on familiar labels… a plain classifier on the same features matches or beats the engine".

No independent evaluation exists in the notes.

## Architecture

### Prompt (`build_prompt`, state first by default)

```
Observation: {state} Measurement ({type}): {instructions} Possible outcomes: * {opt1} * {opt2} Outcome:
```

- Options are capped at 32 tokens.
- When the state is shorter than `query_first_max_state = 4096` tokens, the question is *also* placed before the state ([README "How it works"](https://github.com/NandhaKishorM/vegaml)).
- Options are read **jointly** with the state and question in one sequence. They are not encoded independently.

### Reader (`Perception`)

The reader is a frozen causal LM, run only up to its deepest tapped layer.
- **Tapped layers.** For the 0.8B reader (`n_layers` 24), `feature_layers` is [13, 19] and `option_layers` is [19].
- **Pooled features.** At each feature layer $\ell$ the reader pools:
  - the mean over the state span, $\bar h^{\ell}_{\text{state}}$;
  - the mean over the question span, $\bar h^{\ell}_{q}$;
  - the final token, $h^{\ell}_{\text{last}}$.

  At each option layer it pools the mean over each option span, $\bar h^{\ell}_{o_k}$.
- **Long states.** 16 chunk means (`memory_chunks`) feed a `MemoryReader`.

### vega-U port

For reader $r$ and tapped layer $\ell$, the port first standardizes per reader and then applies a linear map into a shared 1024-d space ([docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md)):

$$\phi^{\ell}_r(h) = P^{\ell}_r\,\mathrm{standardize}_r(h) \in \mathbb{R}^{1024}$$

### Engine (`class VegaEngine`)

The engine has latent dimension $D=64$, `world_dim` 256 and `proj_hidden` 2048. Softmax-weighted scalar mixes combine the tapped layers: $x_\bullet = \sum_\ell \mathrm{softmax}(\beta)_\ell\,\phi^{\ell}(\cdot)$ for each pooled span $\bullet$.

**Initial state and wells**

$$s = \mathrm{MLP}(x_{\text{state}}),\qquad s' = \mathrm{WorldDynamics}(s)\ \ (\text{``foresight''})$$

$$\text{probe} = \mathrm{MLP}(x_q) + e_{\text{type}}(t),\qquad \text{impulse} = \mathrm{MLP}(x_{\text{last}}),\qquad (z_0, p_0) = \mathrm{MLP}\big([s;\ s';\ \text{probe};\ \text{impulse}]\big)$$

$$\big(c_k,\ \log a_k,\ \sigma_k\big) = \mathrm{MLP}\big([x_{o_k};\ \text{probe}]\big),\qquad \log a_k \in [-4, 4],\qquad \sigma_k = 0.3 + \mathrm{softplus}(\cdot)$$

For `score` questions, the well centres $c_k$ are placed on a learned one-dimensional rail.

**Potential**

$$U(z) = \tfrac{1}{2}\kappa\lVert z\rVert^2 - \sum_{k=1}^{K} a_k \exp\!\Big(-\frac{\lVert z - c_k\rVert^2}{2\sigma_k^2}\Big),\qquad \kappa = \exp(\texttt{log\_kappa}),\ \kappa_{\text{init}} = 0.05$$

**Dynamics.** There are $N = 12$ steps with $\Delta t_{\max} = 0.5$, governed by a selective-SSM "thermostat". At step $n$:

$$u_n = \mathrm{ssm\_in}\big([z_n;\ p_n;\ \nabla U(z_n);\ \lVert\nabla U(z_n)\rVert]\big) \to (u_{dt}, u_{fr}, u_b),\qquad \Delta t_n = \Delta t_{\max}\,\sigma(u_{dt}),\qquad \gamma_n = \mathrm{softplus}(u_{fr})$$

$$h_{n+1} = e^{-A\Delta t_n} h_n + \Delta t_n u_b,\qquad f_n = 0.1\,\tanh\big(\mathrm{ssm\_out}(h_{n+1})\big),\qquad g_n = \frac{\nabla U(z_n)}{1 + \lVert\nabla U(z_n)\rVert/20}$$

$$p_{n+1} = p_n e^{-\gamma_n\Delta t_n} - \Delta t_n g_n + \Delta t_n f_n,\qquad z_{n+1} = z_n + \Delta t_n\, p_{n+1}\quad\text{(symplectic Euler)}$$

- The particle is "settled" when its kinetic energy is below 0.01 and $\lVert\nabla U\rVert < 0.1$.
- A `steps_used` count enters the calibrator. *(inference)* The roll-out can therefore stop before $N$ steps.

**Readout**

$$E_k = \frac{\lVert z_T - c_k\rVert^2}{2\sigma_k^2} - \log a_k,\qquad P_k = \mathrm{softmax}\big(-E/\tau\big)_k$$

Here $\tau$ is the per-decision temperature described under [Calibration](#calibration).

**Size.** The engine has 14.3M parameters and no attention (README); on disk it is about 55 MB.

### Passes per request and caching

- **One reader pass per question.** "Three questions about one state mean three prompts and three passes."
  - The exception is states of at least 4,096 tokens. There the state prefix is encoded once and each question continues from a copy of it (README "73,728-token context").
  - *(inference)* Below 4,096 tokens the question is duplicated *before* the state, so the state is no longer a shared prefix. That explains why prefix reuse applies only to long states.
- **Cacheable** *(inference)*:
  - the reader's KV for long-state prefixes;
  - the per-question engine output.
- **Not cacheable** *(inference)*: option features. They are pooled from a joint prompt, so they depend on the state and question, and cannot be reused across requests.

### Properties

| Property | Value |
|---|---|
| Reader | Frozen causal LM (vega-U card: Qwen3.5-0.8B, gemma-3-1b-it, Qwen3.5-4B, Qwen3.8-27B), truncated at the deepest tapped layer; no LM head |
| Features | Span means (state, question, each option) and the last token at 2 intermediate layers (0.8B: layers 13 and 19 of 24); 16 chunk means for long states |
| Port | Per-reader standardize, then a linear map per layer into 1024-d |
| Engine | 14.3M parameters, no attention; $D=64$; Gaussian-well potential; 12-step damped symplectic dynamics with an SSM thermostat |
| Readout | $P=\mathrm{softmax}(-E/\tau)$ over well energies |
| Context | Up to 73,728 tokens |
| Passes per request | One reader pass per question; a shared state prefix only for states of 4,096 tokens or more |
| Cacheable | Long-state prefix KV; per-question engine output *(inference)* |
| Option encoding | Joint with state and question; ≤32 tokens per option |
| Calibration | Per-decision thermal $\tau$ (θ per type), split-conformal thresholds, abstain and unbound thresholds |
| Adaptation | Rank-32 adapters (about 0.84M parameters); per-customer Learner |

## Training

**Engine pre-training is described only in prose, and its code is not released** (README "What is new here"):
- The engine is trained "on the settling behaviour, not on next-token likelihood".
- It uses "a counterfactual objective [that] pairs items that share an answer space".
- A one-step world-dynamics block is "trained to imagine the next state".
- Two rank-32 low-rank adapters sit on 7 engine projections, with per-question sigmoid gates.
- The loss weights, optimiser, schedule and data composition are not disclosed.

**vega-U** was trained jointly on Qwen3.5-0.8B and Gemma-3-1B ([vega-U card](https://huggingface.co/nandakishorm/vega-U)):
- 708,657 uid-matched items;
- records of up to 73,728 tokens;
- a cross-reader invariance loss.

The notes do not name the source datasets.

**Attaching a new reader** (`vegaml.attach`) needs no labels ([docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md)):
1. The anchor reader writes the targets.
2. A ridge port is fit per layer.
3. The port is refined for a few hundred steps by distillation, with an anchor MSE term.

This "cold" attachment is how the Qwen3.5-4B and Qwen3.8-27B rows in [Results](#results) were produced.

**Released adapter recipe** ([notebooks/vega_finetune.py](https://github.com/NandhaKishorM/vegaml/blob/main/notebooks/vega_finetune.py)):
- **Loss:** cross-entropy on cached features plus a gate BCE. Positives come from the task; negatives come from `ag_news`.
- **Optimiser:** AdamW with OneCycle, learning rate 3e-4.
- **Size and length:** rank 32, 3 epochs, about 0.84M parameters.
- **Evidence:** it was run **only on a 96-row smoke test**, reaching 0.313 → 0.542 on 48 items. The notebook calls this "not a result".

**Learner** (`learn/learner.py`, 14 templates). Small heads over frozen features. The methods notes say they are refit every 50 confirmed answers.

## Calibration

**Per-decision "thermal" temperature.** The configuration stores one $\theta=(b_t, w_t)$ per question type (`vega_config.json` `calibration.theta`):

$$\log\tau = b_t + w_t^{\top}\Big[\log(1+\mathrm{KE}_T),\ \log(1+d_{\text{near}}),\ \tfrac{\texttt{steps\_used}}{\texttt{n\_steps}},\ \log K\Big]$$

Here $\mathrm{KE}_T$ is the residual kinetic energy and $d_{\text{near}}$ the distance to the nearest well. The notes do not specify the distance normalisation.

**The gain over a scalar temperature is tiny** (`heldout_calib_nll`):

| Temperature model | Held-out calibration NLL |
|---|---|
| Scalar | 0.48719 |
| Thermal | 0.48476 |

**Split-conformal thresholds** [@conformal]:

| Type | $\alpha = 0.10$ | $\alpha = 0.05$ |
|---|---|---|
| choice | 0.744 | 0.803 |
| score | 0.837 | 0.924 |
| noul | 0.571 | 0.686 |

- **Abstain threshold:** risk 0.1 → 0.5965.
- **Unbound threshold:** 1.377.
- The notes do not say which statistic each threshold is applied to.

**User refit** ([docs/calibration.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/calibration.md)). Fit a per-question temperature, then a conformal threshold. At least 10 examples are required; 30–100 are recommended.

## Results

All numbers below are self-reported, from the vega-U card and the README.

| Reader | How attached | Held-out acc | Zero-shot acc | ECE | AURC | Acc @ 50% coverage |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | cold | 0.811 | 0.629 | 0.021 | 0.053 | 0.976 |
| Qwen3.5-4B | cold | 0.791 | 0.568 | 0.017 | 0.063 | 0.965 |
| Qwen3.5-0.8B | joint | 0.765 | 0.534 | 0.010 | 0.073 | 0.956 |
| gemma-3-1b-it | joint | 0.738 | 0.504 | 0.007 | 0.093 | 0.927 |

- **Test sizes.** The 27B row was scored on $n=21{,}008$; the others on $n=50{,}676$. *(inference)* The 27B row is therefore not strictly comparable with the others.
- **Definitions.** "Held-out" means in-distribution splits of the 708k-item training mixture. The notes do not define the "zero-shot" protocol.
- **Ordinal questions.** The `score` type sits at about 0.61 for every reader.
- **Learner** (14 templates): 84.7% on day one → 96.0% after 120 confirmed answers, versus Jev 89.5% → 92.6%.
- **4B reader against Jev:**
  - banking77: 87.5% with ECE 0.028, versus Jev at 79.4% as shipped;
  - typed-decisions: 73.7% with ECE 0.031, versus Jev at 74.7% / 0.047.
- **Reported losses:**
  - As a ToxicChat prompt-injection gate, Jev catches 79.1% at 0.3% refused, versus 73.0% at 0.7% for vega.
  - "On familiar labels… a plain classifier on the same features matches or beats the engine."

No third-party benchmark (S1-Arena, TypedBench, Rafe & Das, Decision Index, JevBench) reports vegaml in the notes.

## Serving and efficiency

- **Compute saved.** The reader runs only to its deepest tapped layer, and the LM head never runs. For the 0.8B reader the top tapped layer is 19 of 24. *(inference)* The layers above it are skipped entirely, so prefill cost falls roughly in proportion.
- **Engine cost.** The engine is 14.3M parameters with no attention, about 55 MB. *(inference)* It costs little next to the reader pass.
- **Long context.** The context reaches 73,728 tokens. States of at least 4,096 tokens share one prefix encoding across questions.
- **Integration.** vegaml is the default backend of `laya.Agent()` since Laya 0.5 ([Laya HF card](https://huggingface.co/convaiinnovations/laya)). The classic Laya encoder needs `vegaml=False` (see [laya.md](laya.md)).
- **Gaps.** The notes record **no latency or throughput numbers** for vegaml.
- **Inkling.** No vegaml port exists for any Inkling model. On Inkling 2 Mini, `find_layers` forward hooks would fire three times on the looped core layers (see [inkling.md](inkling.md)).

## Flaws and failure modes

### (a) vegaml's own flaws

| Flaw | Evidence |
|---|---|
| OOD ("unbound") detector fires on in-domain data | `unbound` "fired on 71% of normal banking77 rows" |
| Held-out sets are in-distribution | "Held-out" numbers are splits of the 708k-item training mixture |
| Per-decision temperature barely helps | 0.48719 → 0.48476 held-out NLL (about 0.002) |
| Multi-question requests are not amortised | One reader pass per question when the state is shorter than 4,096 tokens |
| The engine is no better than a linear head on familiar labels | Authors: "a plain classifier on the same features matches or beats the engine" |
| Weak on ordinal questions | `score` at about 0.61 for every reader |
| Loses to Jev on a safety gate | ToxicChat: 73.0% at 0.7% refused vs Jev 79.1% at 0.3% |
| Training is not reproducible | Engine pre-training code is unreleased; the data sources of the 708,657 items are not named in the notes |
| Fine-tuning recipe is unvalidated | Only a 96-row smoke test, "not a result" |
| No independent evaluation | None found in the notes |
| No component ablations | The notes record no comparison of the well dynamics against a plain MLP head beyond the authors' own remark |

### (b) Flaws of Laya it claims to fix

| Laya flaw | vegaml's claim | Assessment *(inference)* |
|---|---|---|
| Near-chance zero-shot (0.362 on typed-decisions) | Zero-shot 0.534–0.629 | Measured on the vega-U mixture split, not on typed-decisions. On typed-decisions the 4B reader reports 73.7%, against Laya's fine-tuned 0.766, which has no result file |
| 512-token context | 73,728 tokens | Plausible by design. No long-context accuracy curve is recorded |
| Uncalibrated gating | Conformal sets plus abstain | Thresholds are shipped, but `unbound` fires on 71% of in-domain banking77 rows |
| Needs fine-tuning | Learner: 84.7% → 96.0% after 120 answers | Self-reported, on 14 templates |

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **Intermediate-layer taps as a cheap ablation.** Pooling at about 55–80% depth both cuts prefill and may improve features. For background, "Layer by Layer" reports intermediate layers often beat the last ([@layerbylayer]). For our bi-encoder, this is a pooling change and leaves the option cache intact.
- **Split-conformal sets and selective-risk abstention on top of calibrated probabilities.** The methods notes estimate about 2 days. Pair them with Learn-then-Test thresholds [@ltt]. In the risk-budget study, naive and empirical thresholds exceeded the budget in 42.0% and 51.7% of settings, against 7.8% for LTT ([arXiv 2610.19811](https://arxiv.org/html/2610.19811); [@riskbudget]).
- **$\log K$ and margin features in the temperature model, but measured first against a per-type scalar.** vegaml's own data show only about 0.002 NLL of gain.
- **Per-customer few-shot heads over cached frozen features** (the Learner pattern). This fits a frozen backbone well.
- **Label-free backbone porting** (a ridge port, then distillation to an anchor) as a recipe for swapping readers without relabelling. It applies only if we keep a fixed head across backbones.
- **A linear-probe baseline on the same features.** Make it mandatory. vegaml concedes that a plain classifier matches the engine on familiar labels. CLM issue [#13](https://github.com/Contrastive-LM/CLM/issues/13) shows logistic regression on frozen Qwen3-8B embeddings beating the fine-tuned head on Banking77 (0.900 vs 0.795) [@clm-repo].

**Avoid**
- Calling in-distribution splits of the training mixture "held-out".
- Shipping an OOD or "unbound" detector without measuring its false-positive rate on in-domain data.
- A question-first duplication that breaks state-prefix sharing for short states.
- Reading options jointly with the state, if cross-request option caching matters.
- Assuming a frozen reader plus a small head is enough. Jeff-1.1's head-only frozen-backbone arm scored *below* zero-shot readout (78.9 vs 79.6), while the full LoRA recipe reached 84.4 [@jeff]. vegaml publishes no frozen-vs-adapted comparison on its own engine.

## References

- vegaml code and docs: [github.com/NandhaKishorM/vegaml](https://github.com/NandhaKishorM/vegaml) [@vegaml]
  - [docs/vega-u.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/vega-u.md)
  - [docs/calibration.md](https://github.com/NandhaKishorM/vegaml/blob/main/docs/calibration.md)
  - [notebooks/vega_finetune.py](https://github.com/NandhaKishorM/vegaml/blob/main/notebooks/vega_finetune.py)
- vega-U model card and `vega_config.json`: [huggingface.co/nandakishorm/vega-U](https://huggingface.co/nandakishorm/vega-U)
- Laya model card (0.5 default routing): [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) [@laya-card]; code [@laya-repo]
- "From Laya to Vega" preprint, **not retrievable (403)**: [doi.org/10.20944/preprints202611.0781.v1](https://doi.org/10.20944/preprints202611.0781.v1) [@laya-paper]; Crossref: [api.crossref.org](https://api.crossref.org/works/10.20944/preprints202611.0781.v1)
- Background:
  - conformal prediction [@conformal]
  - Learn-then-Test [@ltt]
  - risk-budgeted decisions [@riskbudget], [arXiv 2610.19811](https://arxiv.org/html/2610.19811)
  - intermediate-layer embeddings [@layerbylayer]
  - Jeff-1.1 frozen-backbone ablation [@jeff], [arXiv 2610.17205](https://arxiv.org/html/2610.17205)
- Name-collision reference, vLLM-SR Decision-2.0-Vega-27B: [HF card](https://huggingface.co/vllm-sr/Decision-2.0-Vega-27B); S1-Arena [@s1arena]
