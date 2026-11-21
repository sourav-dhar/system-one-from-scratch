# AnyJev / Tacit

> **Family:** decoder-readout · **Team:** Zhang, Yang, Shi et al. (Nokia Applied Research, Tencent Hunyuan, PolyU) · **Released:** arXiv 2610.00831, 30 Sep 2026 · **License:** code Apache-2.0; paper CC-BY 4.0 · **Code/weights:** [repo](https://github.com/nokia-applied-research/AnyJev) (1,629 stars per the catalog) with the training-free L0 method; Tacit self-distilled weights on HF (e.g. [Tacit-9B](https://huggingface.co/morriszjm/Tacit-9B); the catalog lists Tacit 1.7B to 9B) · **Sources read:** arXiv HTML (Table 1, §8), [anyjev/tacit.py](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/tacit.py), the Tacit-9B card. Not disclosed: Tacit's self-distillation objective, LoRA vs full fine-tuning, data sizes and training code.

## Summary

AnyJev turns any instruction-following LLM into a decision model without training ([arXiv](https://arxiv.org/html/2610.00831)) [@anyjev] [@anyjev-repo]. Its L0 readout restricts the next-token distribution to the K label tokens, then debiases it twice: **rotation averaging** (average log-probabilities over all K cyclic rotations of the option list, exact under an additive position-bias model) and a **label prior** estimated from unlabelled inputs. The idea that matters most is that position bias can be cancelled by construction at the cost of up to K prefills, which adaptive stopping cuts to about 7 of 18. Tacit adds a self-distillation stage in which a base model writes its own decision problems and learns to answer them in one pass.

## Architecture

**L0 (training-free) readout.**
1. Restrict the next-token distribution to the $K$ label tokens and renormalise: $\tilde p_k = p(\ell_k \mid x) / \sum_{j \le K} p(\ell_j \mid x)$.
2. **Rotation averaging:** for each of the $K$ cyclic rotations $r$ of the option list, compute $\log \tilde p^{(r)}_k$ for option $k$; take $\bar s_k = \frac{1}{K}\sum_{r} \log \tilde p^{(r)}_k$ and apply a softmax. This is exact under an additive position-bias model.
3. **Label prior:** $p_k \propto \tilde p_k / \hat q_k^{\alpha}$, with $\alpha = 0.75$ and $\hat q$ a running mean over unlabelled inputs.
4. **Adaptive stopping** on the log-odds gap $\tau$, certified at ≤1% disagreement (Clopper–Pearson).

**Mask.** Causal. A "set" mask was tried and failed: 0.042 vs 0.633, against a chance level of 0.056 (§8).

| Property | Value |
|---|---|
| Passes per request | K prefills for full rotation averaging; adaptive stopping uses about 7.3 of 18 at K = 18 |
| State prefix reusable | Yes for the state prefix; the question block is not exact across rotations (decoder-family taxonomy) |
| Option embeddings reusable | No |
| Max K | Not stated in the notes; cost scales with K |
| Context | Not stated in the notes |

## Training

- **L0:** none.
- **Tacit** ([tacit.py](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/tacit.py), [HF Tacit-9B](https://huggingface.co/morriszjm/Tacit-9B)): self-distillation. The base model writes its own decision problems, answers them with reasoning, and is trained to give those answers in one pass. "No human labels, external datasets or other models."
- The objective, LoRA vs full fine-tuning, and data sizes are not disclosed. The decoder-family dossier states that **the training code is not in the repo**; the catalog instead describes the repo as "training-free method plus self-distillation scripts". The two notes disagree on whether self-distillation code is published.

## Calibration

- ECE after temperature scaling, with rotations and prior: 0.088 (Table 1, K = 20, mean over 11 models).
- The label prior is the method's main bias correction; no per-type temperature scheme is described in the notes.

## Results

**Table 1** (K = 20, means over 11 models).

| Metric | Raw | + rotations | + prior |
|---|---|---|---|
| banking20 accuracy | 0.665 | 0.737 | 0.752 |
| banking20 flip rate | 0.328 | 0.138 | – |
| Injection (K = 2) flip rate | 0.180 | 0.000 | – |
| ECE after temperature | – | – | 0.088 |

- JevBench pooled debiasing: +0.006 [−0.004, +0.018], i.e. **no gain at K ≈ 3.6**.
- **Tacit-9B** vs base Qwen3.5-9B: JevBench 0.823 vs 0.805; bev 0.725 vs 0.680.
- Tacit with adaptive escalation (τ = 0.5, cap 20%): JevBench 0.887.
- Tacitus-31B: DI 0.4.1 Full 66.47.
- AnyJev's BEV table gives NanoJev 32.99% [@nanojev].

**Third-party (Runoff, [2610.18233](https://arxiv.org/html/2610.18233))** [@runoff].
- Order flips: AnyJev averaging 11.6 vs Runoff 0.0 (Qwen3-8B, Table 2). This is of the same magnitude as AnyJev's own K = 20 mean of 0.138 after rotations *(inference: reading 11.6 as a percentage)*; the setups differ (one model vs an 11-model mean), and the notes do not give Runoff's K for this row.
- Speed: Runoff is 3–4× faster than AnyJev at K = 20, but slower at K ≤ 6 (Table 4).

## Serving and efficiency

- H100, 18 options: 16.7 decisions/s with all rotations → 37.2 decisions/s with adaptive stopping (7.3 rotations on average).
- Cost scales with K because each rotation is a separate prefill (decoder-family cross-table).

## Flaws and failure modes

**(a) Flaws of AnyJev itself**
- **Cost grows with K:** up to K prefills; adaptive stopping helps (7.3 of 18) but does not remove the scaling. Runoff is 3–4× faster at K = 20.
- **Not fully order-invariant:** flips fall from 0.328 to 0.138 at K = 20 but remain; Runoff measures 11.6 for AnyJev averaging on Qwen3-8B.
- **No gain at small K:** JevBench pooled debiasing +0.006 [−0.004, +0.018] at K ≈ 3.6.
- **Set-style masking fails:** 0.042 vs 0.633 (chance 0.056).
- **Tacit is underspecified:** objective, data sizes and fine-tuning mode are undisclosed.

**(b) Flaws of other systems it addresses**
- **Position bias in causal letter readouts:** raw K = 20 flip rate 0.328 → 0.138 with rotations; injection (K = 2) flips 0.180 → 0.000.
- **Label-frequency bias:** the label prior lifts banking20 accuracy from 0.737 to 0.752.

## Implications for our design

All bullets are *(inference)*.

- **Adopt (as a baseline):** rotation-averaged, prior-corrected letter readout on our backbone, as a training-free comparator for order robustness and accuracy.
- **Adopt:** adaptive stopping with a certified disagreement bound when any multi-pass ensemble is used.
- **Avoid:** relying on rotation averaging at production K; its cost scales with K, and a bi-encoder or set-equivariant head is order-invariant by construction at no extra passes.
- **Note:** debiasing showed no gain at K ≈ 3.6, so its value concentrates in large-K settings where its cost is highest.

## References

- [@anyjev] https://arxiv.org/html/2610.00831
- [@anyjev-repo] https://github.com/nokia-applied-research/AnyJev — [tacit.py](https://github.com/nokia-applied-research/AnyJev/blob/main/anyjev/tacit.py)
- Tacit-9B: https://huggingface.co/morriszjm/Tacit-9B
- [@runoff] https://arxiv.org/html/2610.18233
- [@nanojev] https://github.com/TianyuCodings/NanoJev
