# SanSi

> **Family:** looped decoder (weight-shared layers applied repeatedly) with a typed letter readout after every loop · **Team:** Gan, Lee, Kang (University of Minnesota) · **Released:** arXiv [2610.07730](https://arxiv.org/html/2610.07730) (submission date not recorded in the notes we read) · **License:** code Apache-2.0; weight release not recorded · **Code/weights:** [minnesotanlp/Sansi](https://github.com/minnesotanlp/Sansi) [@sansi-repo] · **Sources read:** arXiv HTML paper through App. D; code files [`sansi/model.py`](https://github.com/minnesotanlp/Sansi/blob/main/sansi/model.py) (`CoupledReadout`) and [`sansi/loss.py`](https://github.com/minnesotanlp/Sansi/blob/main/sansi/loss.py). **Gaps:** appendices E–K were truncated, including the stopping-rule Table 22 and the calibration Table 28; the training sources are not named; no built-data release was found.

## Summary

SanSi reads a typed decision after **every** loop of a looped language model, Ouro-1.4B (24 shared layers run for T = 8 loops). This makes compute adjustable at test time. The readout is the frozen LM-head letter rows plus a zero-initialised rank-16 correction and a per-loop temperature. Every loop is trained with cross-entropy plus Brier.

Results:

- Overall accuracy is 72.0 with ECE 0.093. Averaging the probabilities of all 8 loops cuts ECE to 0.044 at no accuracy cost.
- Most of the gain arrives by loop 3.
- The gains depend on looped pre-training: adding loops to a non-looped model (SmolLM2) hurts it.
- On sources that none of the authors' models trained on, Jev leads SanSi-2.6B, 83.5 vs 71.9.

## Architecture

**Backbone.** Ouro-1.4B, or 2.6B for SanSi-2.6B. The same 24 layers are applied repeatedly without re-injecting the input:

$$h_t=N\big(F_\theta(h_{t-1})\big),\qquad t=1,\dots,T,\quad T=8 .$$

**Prompt.** `<state>\n\nQuestion:…\nOptions: (A)…\nAnswer:`. State, question and lettered options form one joint sequence.

**Readout after every loop** (`CoupledReadout`). With $z_t$ the final-position hidden state after loop $t$,

$$\ell_t=\frac{W z_t+U_tV_t z_t}{\tau_t},\qquad \tau_t=\exp(a_t),$$

- $W$ holds the frozen LM-head rows for the option letters.
- $U_tV_t$ is a rank-16 correction. $U_t$ is zero-initialised, so the readout starts as the plain letter readout.

**Trainable parameters.** LoRA r64 (α 128) with 60.6M parameters, plus 0.27M of readout parameters. The method notes describe the base LM as frozen under the LoRA.

**Passes per request** *(inference)*:

- One joint forward pass that traverses the 24-layer stack T times. Eight loops cost 7.7× one pass.
- An exit gate can stop early.
- Every question needs its own prompt, because options are lettered in-context.
- Serving many questions from one cached state prefix would require storing K/V for every loop.

| Property | Value |
|---|---|
| Backbone | Ouro-1.4B (24 shared layers) or 2.6B, looped T = 8 times |
| Encoding | joint prompt (state, question, lettered options) |
| Readout | frozen LM-head letter rows + rank-16 per-loop correction, per-loop temperature $\exp(a_t)$ |
| Trainable | LoRA r64 / α 128 (60.6M) + readouts (0.27M) |
| Compute per decision | up to 8 loops (7.7× one pass); exit gate optional |
| Cached | nothing request-independent; prefix caching would need per-loop K/V |
| Permutation behaviour | not invariant by construction; options are shuffled during training *(inference)* |

## Training

**Loss** ([loss.py](https://github.com/minnesotanlp/Sansi/blob/main/sansi/loss.py)). Every loop is supervised:

$$\mathcal L=\frac1T\sum_{t=1}^{T}\Big[\mathrm{CE}(p_t,y)+\mathrm{Brier}(p_t,y)\Big].$$

Targets are one-hot, uniform for unanswerable items, or the crowd distribution where one exists.

**Hyperparameters (App. B).**

- AdamW with β = (0.9, 0.95).
- Learning rate 1e-4 for the LoRA and 1e-3 for the readouts.
- 200 warm-up steps, then cosine decay to 10%.
- 1,000 steps × 16 items.
- Options are shuffled on every draw.

**Data.**

- Training: 12,800 items from 20 sources.
- Test: 10,027 items from 59 sources.
- The data inventory also records a 2,471-item dev split, built by `data.build_main`, and describes the main set as coming from 59 sources overall.
- The sources are not named.
- The depth tasks are synthetic: liar chains and object swaps.

## Calibration

- The Brier term is the calibration mechanism. Without it, ECE at loop 8 is 0.104 instead of 0.093 (Table 3).
- **Averaging the probabilities of all 8 loops cuts ECE to 0.044** at no accuracy cost.
- The calibration appendix (Table 28) was truncated in our read.

## Results

**Main comparison (Table 2):**

| Model | Accuracy | ECE |
|---|---|---|
| SanSi (8 loops) | 72.0 | 0.093 |
| SmolLM2-1.7B | 58.4 | n.r. |
| Qwen3.5-4B | 73.8 | n.r. |
| Kev-4B (trained on the authors' data) | 74.3 | n.r. |
| SanSi-2.6B | 75.8 | n.r. |

**Jev and Kev on the SanSi suite.**

- The Jev API reaches 78.9 overall.
- On sources none of the authors' models trained on, Jev scores 83.5 vs SanSi-2.6B's 71.9.
- The released Kev-4B scores 68.2 overall and 72.0 on that subset (App. Table 14). So Jev's lead over released Kev-4B on the subset is 11.5 points, matching a figure in the landscape notes.

**Accuracy by loop:**

| Loop | Accuracy / gain |
|---|---|
| 1 | 58.4 |
| 2 | +8.5 |
| 3 | +3.5 |
| 4 | +1.2 |
| 5–8 (together) | +0.4 |

The increments sum to the 8-loop figure of 72.0 *(inference: arithmetic)*.

The method notes describe the headline gain as "+13.5 pp over a same-shape non-looped model (72.0 vs 58.4)". The listed values differ by 13.6, and our notes give 58.4 both as loop-1 accuracy and as the SmolLM2-1.7B row. The two baselines may or may not be the same quantity; our notes do not settle it.

**Ablations (Table 3 and text):**

| Change | Effect |
|---|---|
| Train only the last loop | loop-1 accuracy drops to 35.6 |
| Remove the Brier term | ECE at loop 8: 0.104 vs 0.093 |
| Average the 8 loops' probabilities | ECE 0.044, no accuracy cost |
| RL instead of SFT | no gain (71.7) |
| Add loops to a model not pre-trained to loop (SmolLM2) | drops to 33.5 |

**Where the gains concentrate:** multi-step reasoning (+15.1), long documents (+16.7) and knowledge (+17.6).

**Other results.**

- *Stopping:* Ouro's exit gate reaches the same 72.0 with 44% fewer loops.
- *Depth tasks:* liar chains hold to depth 3 at 1 loop and to depth 11 at 4 loops.
- *As a GRPO reward model:* correct answers rise from 37.0% to 52.1%.

No independent evaluation of SanSi appears in the sources we read.

## Serving and efficiency

- Eight loops cost 7.7× a single pass. The exit gate recovers 44% of loops at equal accuracy.
- No latency figures are recorded in our notes.
- Because the readout is joint and lettered, there is no request-independent option representation to cache.
- Prefix caching for many questions would need K/V stored per loop, which multiplies cache memory by the number of loops *(inference)*.

## Flaws and failure modes

### (a) SanSi's own flaws, with evidence

- **Requires looped pre-training.** Adding loops to SmolLM2 drops it to 33.5, so the method does not transfer to ordinary decoders.
- **Compute.** Up to 7.7× a single pass per decision, and loops 5–8 add only +0.4.
- **Generalisation gap.** On unseen sources, SanSi-2.6B scores 71.9 vs Jev's 83.5.
- **Unnamed sources** and truncated appendices (stopping rule, calibration tables).
- **Not order-invariant by construction**, because options are lettered in a joint prompt. Training-time shuffling is the only defence *(inference)*.

### (b) Flaws of other systems it claims to fix

- **Fixed compute per decision in single-pass System One models.** SanSi makes depth adjustable (1–8 loops) and reports +13.5 pp over the non-looped baseline, with the largest gains on multi-step reasoning, long documents and knowledge.
- **Single-readout overconfidence.** Per-loop CE + Brier, plus averaging across loops, gives ECE 0.044.
- **Limited reasoning depth.** Liar chains reach depth 11 at 4 loops vs depth 3 at 1 loop.

## Implications for our design

*(All items are inference unless cited.)*

**Adopt**

- A per-readout CE + Brier loss. Removing Brier raised ECE from 0.093 to 0.104. Cross-cutting evidence agrees: AnyS1's label smoothing [@anys1] and NDM's CE + Brier [@ndm] point the same way.
- Averaging probabilities across several readout depths. For a frozen Qwen3-8B, this translates into heads on hidden states from several layers (for example $L/2$, $3L/4$, $L$), with a loss on each exit and averaged predictions. Hidden states are already extracted, so the cost is small.

**Avoid**

- Looping itself. It is **not compatible** with a frozen single-pass encoder, and it does not help non-looped backbones.
- Its joint lettered prompt, which gives up option caching.

## References

- [@sansi] SanSi, arXiv 2610.07730: https://arxiv.org/html/2610.07730
- [@sansi-repo] https://github.com/minnesotanlp/Sansi — [`model.py`](https://github.com/minnesotanlp/Sansi/blob/main/sansi/model.py), [`loss.py`](https://github.com/minnesotanlp/Sansi/blob/main/sansi/loss.py)
- [@anys1] AnyS1, arXiv 2611.23751: https://arxiv.org/html/2611.23751
- [@ndm] Narrow Decision Models, arXiv 2610.25553: https://arxiv.org/html/2610.25553
- [@layerbylayer] intermediate-layer embeddings (background for multi-exit readouts), arXiv 2502.02013
