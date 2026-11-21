# Intern-Decision

> **Family:** decoder readout (a causal LM that reads letter logits at JSON placeholders; the paper calls it "autoregressive masked language modeling"). It is a causal model, not an encoder · **Team:** InternLM (Shanghai AI Lab); paper by Tong Zhu, Zhe Xu, Zefeng He et al. · **Released:** arXiv 2026-10-15 · **License:** code Apache-2.0; paper CC-BY-SA 4.0 · **Code/weights:** [github.com/InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision) [@interndecision-repo] (812 stars), with XTuner training configs and temperature calibration. Weights are 0.8B / 2B / 4B on Qwen3.5. The encoder dossier notes that the training data and weights are not in the repo; the catalog lists the three weight sizes · **Sources read:** the repo, including [configs/training/qwen35.py](https://github.com/InternLM/Intern-Decision/blob/main/configs/training/qwen35.py) and `src/eval/fit.py`, and the paper's HTML ([arXiv 2610.22131](https://arxiv.org/html/2610.22131); [@interndecision]) up to Appendix C. **Not read:** the appendices after C.

## Summary

Intern-Decision turns a Qwen3.5 model into a multimodal multi-field decision model.
- **Input.** The assistant turn is pre-filled with a JSON skeleton containing one placeholder per decision field.
- **Readout.** At each field, the LM head's logits at the position before the placeholder are restricted to the legal option letters and divided by a temperature.
- **Passes.** All fields, up to 16, are read in **one causal pass**.
- **Training.** A small mixture: 17,880 samples / 23,500 decisions, of which 5,500 have images. It runs for 2 epochs (156 updates) with full-vocabulary cross-entropy.
- **Calibration.** One scalar temperature per checkpoint.
- **Results.** The 4B model averages 90.02 over its seven suites, against Jev's 88.74.

Two caveats limit that comparison:
- Four of the seven suites (AG News, ToolACE, Typed Decisions and WildJailbreak) also appear in its training data.
- Options are never shuffled, and Jeva measures 22.67% permutation flips for InternDecision-4B.

## Architecture

**Layout.** The assistant turn holds a JSON skeleton, for example `{"team": "<decision>", …}`. Options are letters drawn from 62 symbols. Each request has up to 16 fields, all read in one causal pass.

**Readout.** With $p_j$ the position of field $j$'s placeholder, $h$ the backbone hidden states, $C_j$ the legal candidates of field $j$ and $T$ a temperature:

$$z_{p_j-1} = \mathrm{LMHead}\big(h_{p_j-1}\big) \tag{Eq. 1}$$

$$\mathcal{L} = -\sum_j \log \mathrm{softmax}\big(z_{p_j-1}\big)_{y_j}\quad\text{(full-vocabulary cross-entropy at the shifted positions)} \tag{Eq. 2}$$

$$P_j(c\mid x; T) = \frac{\exp\big(z_{p_j-1,c}/T\big)}{\sum_{c'\in C_j}\exp\big(z_{p_j-1,c'}/T\big)},\qquad c\in C_j \tag{Eq. 3}$$

The paper describes Eq. 2 in words as "full-vocabulary CE at the shifted positions". The summed form above is our rendering.

**Field isolation.** "Earlier placeholders are visible to later fields… but their gold or predicted answers are not fed back." *(inference)* Field $j$'s representation can therefore depend on earlier fields' question and option text, so fields are not isolated from each other.

**Order.** There is no option-order randomization: "Field and option order remain unchanged."

**Multimodal input.** Image inputs pass through the Qwen3.5 vision tower, which is frozen in training.

**Passes and caching** *(inference, as noted in the encoder dossier)*. The state comes first and the schema after it. This causal layout, shared with Clef, is the one that allows prefix-KV reuse of the state across requests that share it.

| Property | Value |
|---|---|
| Backbone | Qwen3.5, at 0.8B / 2B / 4B; vision tower frozen |
| Attention | Causal |
| Option encoding | Letter symbols (62) inside a JSON skeleton |
| Head | The pretrained LM head; softmax over the legal letters (Eq. 3) |
| Fields per request | Up to 16 |
| Passes per request | One causal pass for all fields |
| Cacheable | State-prefix KV *(inference)* |
| Order invariance | No; order fixed during training |
| Calibration | One scalar $T$ per checkpoint |

## Training

From Tables 9 and 10 of the paper and [configs/training/qwen35.py](https://github.com/InternLM/Intern-Decision/blob/main/configs/training/qwen35.py):

| Setting | Value |
|---|---|
| Data | 17,880 samples / 23,500 decisions; 5,500 with images |
| Synthetic text | From a deepseek-v4.1-flash teacher |
| Public data | Includes AG News, ToolACE, Typed Decisions and WildJailbreak |
| Schedule | 2 epochs / 156 updates |
| Learning rate | 2e-6 |
| Batch | 16 packed 8k sequences |
| Hardware | 4 GPUs |
| Frozen parts | Vision tower |

- *(inference)* The four public datasets listed are four of the seven evaluation suites, so those suites are in-distribution.
- The notes do not state whether the language model is fully fine-tuned or adapted with LoRA.

## Calibration

**Fitting.** One scalar $T$ per checkpoint, fit by NLL bisection over $[0.01, 100]$ (`src/eval/fit.py`):

$$T^\star = \arg\min_{T\in[0.01,\,100]} \; -\sum_{i}\log P_{j_i}\big(y_i\mid x_i; T\big)$$

| Checkpoint | $T$ |
|---|---|
| 0.8B | 2.748 |
| 2B | 2.101 |
| 4B | 1.992 |

**Interpretation** *(inference)*:
- All three temperatures are above 1, so the raw readouts are overconfident.
- The correction shrinks with model size.
- A single $T$ ignores the type- and $K$-dependence that Laya, sbert2s1, Bongard and Small Heads model.

**Reported calibration:**
- JevBench-Hard: Brier 0.347, ECE 0.065.
- A known-distribution pilot: 0.550 / 0.089, against Jev's 0.595 / 0.130.

## Results

| Evaluation | Intern-Decision | Comparison | Source |
|---|---|---|---|
| Seven-suite average (4B) | 90.02 | Jev 88.74; Laya 57.77 | [README](https://github.com/InternLM/Intern-Decision), paper |
| JevBench-Hard (Brier / ECE) | 0.347 / 0.065 | Laya 0.804 / 0.246 | same |
| Known-distribution pilot | 0.550 / 0.089 | Jev 0.595 / 0.130 | paper |
| Latency | 44.16 ms on an RTX 4090 (289 tokens, 3 fields) | — | paper |
| Jeva Table 1, Typed Decisions (4B) | 78.70 | Jeva-4B 78.30; Kev-4B 66.00; CLM-8B 43.05 | [arXiv 2610.18773](https://arxiv.org/html/2610.18773) [@jeva] |
| Jeva Table 2, choice flips under permutation (4B) | 22.67% | Jeva 0.00%; Jev-1.13 17.00%; JevK5 26.17% | [@jeva] |
| Jeva null detection, BANKING77 K=16, AUROC (4B) | 84.27 | Jeva 98.66; Kev 93.61; JevK5 79.44 | [@jeva] |
| Jeva ToolRet | fits only 65/84 requests at K=32 | Jeva 86.90 at K=32 | [@jeva] |
| Decision Index Full (sub-1B model; reported in Von's notes) | 11.8 | — | [Von](https://github.com/wfzyx/von) [@von] |

The paper itself admits three limitations:
- no matched-data ablations;
- no comparison with a classification head;
- latency reported without a stated protocol.

## Serving and efficiency

- **Speed:** 44.16 ms on an RTX 4090 for 289 tokens and 3 fields, with the measurement protocol unstated.
- **Pass count:** all of a request's fields share one causal pass.
- *(inference)* The state-first layout permits prefix-KV caching in standard LLM servers.
- **Capacity:** the letter alphabet (62 symbols) and the 16-field cap bound the decision size per request. On Jeva's ToolRet, only 65 of 84 requests fit at $K=32$.

## Flaws and failure modes

### (a) Intern-Decision's own flaws

| Flaw | Evidence |
|---|---|
| Order sensitivity | "Field and option order remain unchanged" in training; 22.67% choice flips under permutation (Jeva Table 2) |
| In-distribution evaluation | AG News, ToolACE, Typed Decisions and WildJailbreak are both training data and evaluation suites |
| Weak null detection | AUROC 84.27, against Jeva 98.66 and Kev 93.61 |
| Large-option-set capacity | Fits only 65/84 ToolRet requests at $K=32$ |
| Fields are not isolated | Earlier placeholders are visible to later fields *(inference: answers may depend on co-asked questions)* |
| Coarse calibration | One scalar $T$ per checkpoint |
| Small sub-1B quality | Decision Index Full 11.8, per Von's notes |
| Self-admitted gaps | No matched-data ablations, no classification-head comparison, latency protocol unstated |

### (b) Flaws of other systems it claims to fix

- **Efficiency.** The paper's framing ("Efficient Multimodal Decision Making") is one causal pass for up to 16 fields, plus image inputs. This addresses the per-question cost of row-per-question designs such as Laya, and of text-only systems.
- **Calibration.** It claims better calibration than Jev on JevBench-Hard and on the known-distribution pilot (0.550 / 0.089 vs 0.595 / 0.130).
- **Assessment** *(inference)*. The 90.02 vs 88.74 average over Jev is confounded by four in-distribution suites. The permutation flips (22.67%) are worse than Jev's (17.00%).

## Implications for our design

All items in this section are *(inference)*.

**Adopt**
- **The pretrained LM head restricted to legal letters** (Eq. 3) as the zero-training, same-backbone readout baseline for our Qwen3-8B. It is the same family of readout that scored 59.3 vs CLM's 41.4 on S1-Arena.
- **A state-first, schema-last layout** for any joint slow path, so that state KV can be cached.
- **NLL temperature fitting, done per type and as a function of $K$** on held-out datasets.

**Avoid**
- Training without option shuffling. Measure flip rates under permutation for every model we report.
- Letting later fields attend to earlier fields' question text, if answers must not depend on what else is asked.
- Evaluating on suites whose sources appear in training.

## References

- Code and README: [github.com/InternLM/Intern-Decision](https://github.com/InternLM/Intern-Decision) [@interndecision-repo]
  - [configs/training/qwen35.py](https://github.com/InternLM/Intern-Decision/blob/main/configs/training/qwen35.py)
- Paper: "Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling", [arXiv 2610.22131](https://arxiv.org/html/2610.22131) [@interndecision]
- Jeva (third-party comparisons): [arXiv 2610.18773](https://arxiv.org/html/2610.18773) [@jeva]
- Von notes (Decision Index Full): [github.com/wfzyx/von](https://github.com/wfzyx/von) [@von]
- Laya (comparison rows): [@laya-repo]
- Clef (same state-first causal layout): [@clef]
- S1-Arena (letter-readout baseline evidence): [@s1arena]
