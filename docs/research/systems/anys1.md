# AnyS1

> **Family:** decoder readout (joint sequence; pointer head plus an option-interaction self-attention layer) · **Team:** USTC / NTU · **Released:** arXiv [2611.23751](https://arxiv.org/html/2611.23751) (submission date not recorded in the notes we read) · **License:** not disclosed / not read · **Code/weights:** project page only; no code, weights or data link found · **Sources read:** the arXiv HTML paper (Tables 1–4). **Gaps:** no code or data release, so the head's exact implementation and the Any-DP corpus cannot be inspected.

## Summary

AnyS1 is a 4B System-1 model for the decision points of agent workflows. Its target is to answer, cheaply, the decisions an agent would otherwise send to a System-2 LLM.

- **Encoding:** state, question and options are read in one Qwen3.5-4B-Base sequence with LoRA.
- **Head:** a pointer projection, plus **one position-free bidirectional self-attention layer over the decision and option-end vectors**, which lets options interact before scoring.
- **Training data:** Any-DP, 25,986 consensus states mined from 1,072 `SKILL.md` skills.
- **Loss:** label-smoothed cross-entropy.
- **Deployment:** a confidence gate delegates a decision to System-1 only when $\max p\ge\tau$.

Results:

- The option-interaction head adds 1.1 Dev points over a standard pointer head (83.8 vs 82.7).
- Label smoothing gives the best calibration under shift (ECE 0.094).
- Episode cost savings are 75–91% when the gate is used.

## Architecture

**Backbone and sequence.** Qwen3.5-4B-Base with LoRA r16, over one sequence:

$$\langle\text{state},\ \text{question},\ \langle\mathrm{opt}\rangle o_1\langle/\mathrm{opt}\rangle,\ \dots,\ \langle\mathrm{opt}\rangle o_K\langle/\mathrm{opt}\rangle,\ \langle\mathrm{dec}\rangle\rangle .$$

**Head.**

- A pointer projection with $d_p=256$ maps the $\langle\mathrm{dec}\rangle$ and $\langle/\mathrm{opt}\rangle$ hidden states into a scoring space.
- On top sits **one position-free, 4-head, bidirectional self-attention layer** over the $\langle\mathrm{dec}\rangle$ and $\langle/\mathrm{opt}\rangle$ vectors, with zero-initialised residuals. The head therefore starts as a plain pointer head.

Because the layer has no positions, it is permutation-equivariant over the option vectors. The backbone sequence is still ordered, so the model as a whole is not order-invariant by construction *(inference)*.

**Gate.** System-1 answers when $\max_i p_i\ge\tau$ and otherwise delegates to System-2. In "cold" mode, $\tau=0.918$ is fit on 153 examples.

**Passes per request** *(inference)*: one joint pass per question over state + question + options, followed by the small head. Options are read in context, so no option representation is request-independent or cacheable across requests.

| Property | Value |
|---|---|
| Backbone | Qwen3.5-4B-Base, LoRA r16 |
| Encoding | joint sequence with `⟨opt⟩…⟨/opt⟩` spans and a final `⟨dec⟩` |
| Head | pointer projection ($d_p=256$) + one position-free 4-head bidirectional self-attention layer over `⟨dec⟩`/`⟨/opt⟩` vectors; zero-init residuals |
| Option interaction | yes, in the head (and via the backbone's context) |
| Loss | label-smoothed CE, ε = 0.1 |
| Gate | delegate to System-1 if $\max p\ge\tau$; cold-mode τ = 0.918 |
| Cached | nothing request-independent *(inference)* |
| Latency | 21.8–31.7 ms per decision (H100) |

## Training

- **Data:** Any-DP, 25,986 consensus states mined from 1,072 `SKILL.md` skills (Table 1). Teacher agreement is 89.2%.
- **Loss:** label-smoothed cross-entropy with $\varepsilon=0.1$, i.e. targets $(1-\varepsilon)\,y+\varepsilon/K$ in the standard formulation *(rendering)*.
- **Schedule:** 1 epoch at learning rate 1e-4.

## Calibration

**Training objective vs calibration (Table 3c):**

| Objective | ECE | Gated agreement |
|---|---|---|
| Label smoothing (ε = 0.1) | 0.094 | 97.7% |
| Hard-label CE | 0.146 | 92.4% |
| CE + Brier | 0.148 | 93.7% |

- Label smoothing beats both hard-label CE and CE + Brier on ECE and on agreement after gating.
- This is the opposite of SanSi's finding, where adding Brier helped [@sansi]. The two experiments differ in setting, so the results need not conflict *(inference)*.
- The gate threshold τ = 0.918 is fit on 153 examples.

## Results

**Head comparison (Table 4, Dev):**

| Model | Dev |
|---|---|
| AnyS1, option-interaction head | 83.8 |
| AnyS1, standard pointer head | 82.7 |
| Jev | 78.4 |
| Kev-4B | 57.8 |

**Training-state source (Table 3b):** LLM-written synthetic states score 65.0, against 83.8 for real container states. This comparison is confounded by corpus size (84,481 vs 25,986).

**Agent episodes (Table 2):**

- The gate yields 75–91% episode cost savings.
- The gate-free "All-S1" setting collapses ALFWorld to 40.3, against 95.5 with the gate.

No independent evaluation of AnyS1 appears in the sources we read.

## Serving and efficiency

- 21.8–31.7 ms per decision on an H100.
- The cost saving comes from the gate, which routes most decisions away from the System-2 LLM: 75–91% per episode.
- Without the gate, accuracy collapses (ALFWorld 40.3 vs 95.5). The saving therefore depends on a well-calibrated threshold.

## Flaws and failure modes

### (a) AnyS1's own flaws, with evidence

- **Small head gain.** The option-interaction layer adds only +1.1 Dev, and the paper reports no variance.
- **Synthetic vs real states is confounded** by a 3.25× difference in corpus size (84,481 vs 25,986).
- **A small calibration set for the gate** (153 examples for τ = 0.918), and **one training epoch**.
- **The gate is load-bearing.** Without it, ALFWorld falls to 40.3.
- **No code, weights or data link**; only a project page.
- **No cross-request option caching**, because options are encoded jointly with the state *(inference)*.

### (b) Flaws of other systems it claims to fix

- **Independent option scoring in pointer heads.** The option-interaction layer gives 83.8 vs 82.7.
- **Overconfident hard-label training under distribution shift.** Label smoothing gives ECE 0.094 vs 0.146.
- **Paying System-2 cost for every agent decision.** The confidence gate cuts episode cost by 75–91%.
- **Unrealistic synthetic training states.** Real container states score 83.8 vs 65.0, albeit confounded.

## Implications for our design

*(All items are inference unless cited.)*

**Adopt**

- **Label smoothing (ε = 0.1)** as a free calibration-under-shift baseline. Compare it against CE + Brier on our own held-out datasets, since AnyS1 and SanSi disagree.
- **Real tool-output states** in training data, in preference to LLM-written states.
- **A confidence gate tuned for agreement with System-2**, which fits a cached bi-encoder fast path in front of a slower readout. The threshold should be fit with a risk-controlling procedure such as Learn-then-Test [@ltt] rather than on 153 examples.

**Adapt** the option-interaction idea to cached options. AnyS1's layer needs the joint sequence, but the same position-free self-attention over option vectors could run over cached, request-independent option embeddings (an NDM-style set head [@ndm]) or through the state (Jeva [@jeva]).

## References

- [@anys1] AnyS1, arXiv 2611.23751: https://arxiv.org/html/2611.23751
- [@sansi] SanSi, arXiv 2610.07730: https://arxiv.org/html/2610.07730
- [@ndm] Narrow Decision Models, arXiv 2610.25553: https://arxiv.org/html/2610.25553
- [@jeva] Jeva, arXiv 2610.18773: https://arxiv.org/html/2610.18773
- [@ltt] Learn-then-Test, arXiv 2110.01052
