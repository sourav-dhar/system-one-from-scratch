# NanoJev

> **Family:** decoder-readout · **Team:** Tianyu Chen (first author of System1-Arena) · **Released:** repo created 2026-09-17 (2,602 stars as of 2026-11-20 per the catalog) · **License:** MIT · **Code/weights:** end-to-end training pipeline; a 0.6B checkpoint (Qwen3-0.6B, full fine-tune) · **Sources read:** README, [scripts/train_toy_decisions.py](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/train_toy_decisions.py) (`DecisionModel`), [docs/TYPESAFE_CONTRACT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md), [docs/RLCD_EXPERIMENT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md).

## Summary

NanoJev is a small, fully fine-tuned Qwen3-0.6B decision model focused on games (ViZDoom, Maze, Snake), from the author of System1-Arena ([repo](https://github.com/TianyuCodings/NanoJev)) [@nanojev] [@s1arena]. Each candidate is scored on its own "leaf" path (state + instruction + candidate), and for Choice questions a small zero-initialised set-attention layer over the candidates adds cross-option interaction. The idea that matters most is this **leaf-plus-set-head** design: candidates are encoded independently, then compared by a permutation-equivariant set layer, which the decoder-family cross-table counts as an exact order-invariance fix. Its RLCD experiment is also a useful negative: an unbiased sampled-Brier policy gradient only matched direct cross-entropy or Brier training.

## Architecture

([scripts/train_toy_decisions.py](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/train_toy_decisions.py) `DecisionModel`)

- **Backbone:** Qwen3-0.6B, full fine-tune.
- **Leaf paths:** each candidate is its own path, state + instruction + candidate. The path "repeats prefixes; tree sharing is a separately verified optimization".
- **Score:** $z_k = w^\top \mathrm{LN}\big(h^{(k)}_{\text{last}}\big)$, the last hidden state of candidate $k$'s path.
- **Choice set head:** a **set-attention residual**, MultiheadAttention(128, 4 heads) over the candidates with a $\log K$ feature and a zero-initialised output, added to the per-candidate scores. (The notes do not give the exact composition of the residual with $z_k$.)
- **Boolean:** logits $[0, z]$.
- **Order invariance:** permutation-equivariant by construction *(inference, as in the dossier)*; the cross-system table lists "NanoJev's set head (exact)" among order-sensitivity fixes.
- **No cross-question attention**, confirmed by the contract doc ([TYPESAFE_CONTRACT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md)).
- **Mask:** causal.

| Property | Value |
|---|---|
| Passes per request | One path per candidate (the state is repeated per path) |
| State prefix reusable | Optional tree sharing, verified separately |
| Option embeddings reusable | No; each leaf encodes the candidate after the state *(inference)* |
| Max K | Not stated in the notes |
| Context | Not stated in the notes |

## Training

- Full fine-tune of Qwen3-0.6B; end-to-end pipeline in the repo.
- **RLCD experiment** ([docs/RLCD_EXPERIMENT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md)): an unbiased sampled-Brier policy gradient with $\mathbb{E}[R] = \lVert q\rVert^2 - \lVert p - q\rVert^2$ over $M$ samples.
  - Test L2: 0.1184 (RLCD) vs 0.1242 (cross-entropy) vs 0.1384 (Brier), one seed. The document's verdict is that RLCD "matched but did not beat" direct cross-entropy or Brier, although its L2 is numerically the lowest of the three; with a single seed the notes do not establish significance.
  - Correctness-only REINFORCE was bad: 0.0171 vs 0.0026 on CPU (the notes do not name the metric).

## Calibration

No calibration procedure or ECE values are recorded in the notes.

## Results

| Benchmark | NanoJev | Comparator | Source |
|---|---|---|---|
| ViZDoom Basic | 128/128 | Jev 56/128 | README |
| AnyJev BEV table | 32.99% | – | [AnyJev](https://arxiv.org/html/2610.00831) [@anyjev] |

## Serving and efficiency

- Per-candidate leaf paths repeat the state; "tree sharing" (sharing the state prefix across leaves) is a separately verified optimisation.
- No latency figures appear in the notes.

## Flaws and failure modes

**(a) Flaws of NanoJev itself**
- **Narrow focus and scale:** a 0.6B model aimed at games; on AnyJev's general BEV table it scores 32.99%.
- **Redundant computation without tree sharing:** the state is repeated for every candidate.
- **Single-seed RL evidence:** the RLCD comparison rests on one seed.

**(b) Flaws of other systems it addresses**
- **Order sensitivity in causal letter readouts:** the leaf-plus-set-head design is exact by construction (cross-system table).
- **Weak game control from general models:** ViZDoom Basic 128/128 vs Jev 56/128.

## Implications for our design

All bullets are *(inference)*.

- **Adopt:** a zero-initialised set-attention layer over independently encoded candidates, with a $\log K$ feature, as a cheap way to add option interaction while keeping order invariance. This is structurally close to a set head over cached option vectors.
- **Avoid:** defaulting to RL-style calibration objectives when direct proper-score losses (CE, Brier) reach a comparable result; NanoJev's single-seed RLCD comparison found only a match.
- **Note:** correctness-only rewards are clearly worse than proper-score rewards in NanoJev's test.

## References

- [@nanojev] https://github.com/TianyuCodings/NanoJev — [train_toy_decisions.py](https://github.com/TianyuCodings/NanoJev/blob/main/scripts/train_toy_decisions.py), [TYPESAFE_CONTRACT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md), [RLCD_EXPERIMENT.md](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md)
- [@s1arena] https://arxiv.org/html/2610.21244
- [@anyjev] https://arxiv.org/html/2610.00831
