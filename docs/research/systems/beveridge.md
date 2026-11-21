# Beveridge

> **Family:** encoder (ModernBERT-large reader plus engineered numeric features, an exact rule engine and a small recurrent decision network) · **Team:** not recorded in our notes (repository owner `sathiskumarjothi0-oss`) · **Released:** arXiv [2611.18655](https://arxiv.org/html/2611.18655) (submission date not recorded in the notes we read) · **License:** not disclosed / not read · **Code/weights:** [bev-decisions-pypi](https://github.com/sathiskumarjothi0-oss/bev-decisions-pypi) · **Sources read:** the arXiv HTML paper (Tables 2, 4, 6 and 7; §10). **Gaps:** the code was not read; the composition of the training data and of Beveridge-Bench is not recorded in our notes.

## Summary

Beveridge augments a text encoder with hand-built numeric reasoning. The components are:

- **Large-C**, a 398.3M-parameter ModernBERT-large that reads questions and options *before* the document (up to 4,096 tokens);
- 22 typed value features for every number or date;
- a parameter-free rule engine that emits exact facts (comparisons, sums, counts, windows);
- a 3.5M-parameter decision network with evidence attention, a recurrent refinement step and an exact yes/no path.

On the authors' own Beveridge-Bench it beats Jev, 93.0 vs 87.9, and automates 76% of decisions at 95% accuracy, against Jev's 59%. It is weak on general text tasks: SST-5 33.2, Case holdings 42.0, Banking77 63.6. The ablations show that the gains come from the exact facts (−17.4 without them) and the operator path (−11.5), while the three-step recursion contributes nothing (+0.0 with one step).

## Architecture

**Reader.** Large-C (398.3M) is ModernBERT-large. Questions and options are placed **before** the document, which can be up to 4,096 tokens long. One likely effect is that truncation cuts the document rather than the question, which is the opposite of CLM's failure in its issues #6 and #29 *(inference)*.

**Engineered features.**

- 22 typed value features for each number or date in the input.
- A **parameter-free rule engine** that emits exact facts: `cmp`, `sum`, `count`, `window`.

**Decision network (3.5M parameters).**

- Evidence attention.
- A shared recurrent layer applied 3 times, with halting.
- Option logits from an MLP over $z$, $o$ and their element-wise product. We read $z$ as the refined decision state and $o$ as the option vector *(inference)*:

$$\ell_o=\mathrm{MLP}\big([\,z;\ o;\ z\odot o\,]\big).$$

- An exact yes/no path for Boolean questions.

**Passes per request** *(inference)*: one joint encoder pass over questions + options + document, one rule-engine evaluation, and one decision-network evaluation (up to 3 recurrent steps). The options are encoded jointly with the document, so no request-independent option representation exists.

| Property | Value |
|---|---|
| Reader | Large-C: ModernBERT-large, 398.3M; questions and options before the document; up to 4,096 tokens |
| Features | 22 typed value features per number or date; parameter-free exact-fact engine (cmp, sum, count, window) |
| Decision network | 3.5M: evidence attention; shared recurrent layer ×3 with halting; $\mathrm{MLP}([z;o;z\odot o])$; exact yes/no path |
| Calibration | per-type temperature from a grid; Large-C uses 1.5 / 1.2 / 1.0 |
| Order sensitivity | 3.4% of answers change under option shuffling |
| Latency | 4.9 ms (H200) |
| Cached | not described |

## Training

Our notes do not record the training data, losses or schedule beyond the components above.

## Calibration

- Per-type temperatures are chosen from a grid.
- Large-C uses 1.5 / 1.2 / 1.0. The mapping of these values to question types is not recorded in our notes.
- No ECE figures are recorded in our notes.

## Results

**Main results (Table 2):**

| Benchmark | Beveridge | Jev |
|---|---|---|
| Beveridge-Bench | 93.0 | 87.9 |
| SST-5 | 33.2 | n.r. |
| Case holdings | 42.0 | n.r. |
| Banking77 | 63.6 | 79.2 |

Bongard's same-item comparison reports Jev at 79.8 on Banking77 [@bongard]. The two papers use different protocols, so the 79.2 and 79.8 figures for Jev need not conflict *(inference)*.

**Ablations (Table 4):**

| Removed / changed | Δ |
|---|---|
| Exact facts | −17.4 |
| Operator path | −11.5 |
| One recursive step instead of three | **+0.0** |

**Robustness and automation.**

- Shuffling the options changes 3.4% of answers (§10).
- At 95% accuracy, Beveridge automates 76% of decisions, against Jev's 59% (Table 6).

**Latency (Table 7):** 4.9 ms on an H200.

No independent evaluation of Beveridge appears in the sources we read.

## Serving and efficiency

- 4.9 ms per decision on an H200, with a 398.3M reader and a 3.5M decision network.
- The rule engine is parameter-free.
- The recurrent layer can be cut to one step at no accuracy cost (+0.0), which would save two refinement steps *(inference)*.

## Flaws and failure modes

### (a) Beveridge's own flaws, with evidence

- **Narrow gains.** The headline win is on the authors' own suite; on general text tasks it is weak (SST-5 33.2, Case holdings 42.0, Banking77 63.6 vs Jev 79.2).
- **Dependence on hand-engineered features.** Removing the exact facts costs 17.4 points and removing the operator path costs 11.5. The model's advantage is in domain-engineered numeric processing.
- **Unnecessary recursion.** One step matches three (+0.0), so the recurrent halting machinery adds complexity without accuracy.
- **Not order-invariant.** Shuffling options changes 3.4% of answers.

### (b) Flaws of other systems it claims to fix

- **Weak numeric and date reasoning in single-pass decision models.** Exact facts are worth +17.4, and Beveridge-Bench is 93.0 vs Jev 87.9.
- **Low automation rate at a fixed accuracy target.** Beveridge automates 76% vs Jev's 59% at 95% accuracy.
- **Latency.** 4.9 ms per decision.

The cross-cutting analysis lists Beveridge's exact-fact path as the fix for multi-step numeric reasoning that single passes miss (for example, this-that's 0.560 on multi-step arithmetic [@thisthat]).

## Implications for our design

*(All items are inference unless cited.)*

- **Limited relevance to a general bi-encoder.** The gains come from domain-engineered numeric features evaluated on the authors' own suite.
- **Worth considering:** an optional **exact-fact feature path** for numeric or date-heavy states. It is compatible with cached option embeddings, because it touches only the state side.
- **Do not adopt** recurrent refinement; the paper's own ablation shows it adds nothing.
- **Measure order sensitivity** whenever options are read jointly. A bi-encoder avoids the 3.4% shuffle sensitivity by construction.

## References

- [@beveridge] Beveridge, arXiv 2611.18655: https://arxiv.org/html/2611.18655
- Code: https://github.com/sathiskumarjothi0-oss/bev-decisions-pypi
- [@bongard] Bongard (Jev Banking77 comparison), arXiv 2609.39111: https://arxiv.org/html/2609.39111
- [@thisthat] this-that-model-1.0, arXiv 2609.23886: https://arxiv.org/html/2609.23886
