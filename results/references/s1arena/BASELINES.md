# System1-Arena baselines

- **Data:** the pinned public static set (`ad05ed5`, 41,183 items).
- **Aggregation:** mean over 87 leaves.
- **Calibration:** temperatures per question type, fitted leave-one-domain-out, so no domain is calibrated on its own questions.
- **Source:** `scripts/run_bm25.py`. No GPU was used.

| Model | Accuracy | Brier, raw | Brier, calibrated | ECE (10 equal-mass), calibrated | Source |
|---|---|---|---|---|---|
| `bm25@k1=1.5,b=0.75` | 37.84 | 0.8209 | 0.6115 | 0.133 | this run |
| `Qwen/Qwen3-8B@label-think0` | 59.50 | 0.7139 | 0.4890 | 0.152 | ANCHORS.md |
| `Contrastive-LM/CLM-v0.1-8B` | 41.86 | 0.7411 | 0.6069 | 0.149 | ANCHORS.md |

## Accuracy by domain

| Domain | `bm25@k1=1.5,b=0.75` | `Qwen/Qwen3-8B@label-think0` | `Contrastive-LM/CLM-v0.1-8B` |
|---|---|---|---|
| basic_cls | 22.8 | 57.3 | 38.0 |
| batch | 46.6 | 68.8 | 41.8 |
| code | 42.1 | 55.0 | 40.1 |
| dialog | 40.6 | 57.9 | 41.2 |
| fore | 29.5 | 44.2 | 36.3 |
| judge | 47.6 | 59.2 | 45.4 |
| know | 38.9 | 66.1 | 48.2 |
| pro | 34.7 | 59.2 | 28.6 |
| ret | 43.4 | 64.5 | 48.3 |
| route | 39.2 | 53.7 | 50.0 |
| safety | 38.9 | 69.6 | 55.2 |
| social | 23.7 | 46.2 | 20.3 |
| struct | 44.4 | 63.3 | 41.5 |

## Accuracy by question type (pooled over questions)

| Type | `bm25@k1=1.5,b=0.75` | `Qwen/Qwen3-8B@label-think0` | `Contrastive-LM/CLM-v0.1-8B` |
|---|---|---|---|
| choice | 33.7 | 63.8 | 29.8 |
| score | 25.1 | 39.7 | 25.5 |
| boolean | 37.0 | 74.3 | 65.3 |

## Reading

- **BM25 is a lexical floor.** It can only help where option texts share rare words with the request, as in retrieval and routing.
- **Booleans.** S1-Arena's boolean questions carry no option text (42,869 questions here). BM25 answers them uniformly, and the tie goes to "true".
- **Its boolean accuracy is therefore the share of "true" gold labels** (37.0%), not a lexical result. On choice questions, where BM25 does match text, it scores 33.7%.
- **CLM against the floor.** BM25 is at least as accurate as CLM-v0.1-8B in 6 of 13 domains: batch, code, judge, pro, social, struct. That fits the documented failure where CLM ignores the state (survey FM7 / FM14): there, its cosine to the options carries little more than word overlap does.
- **Choice questions, pooled:** BM25 33.7% against CLM 29.8%. CLM's overall lead over the floor comes from booleans (65.3% against BM25's 37.0% 'always true'), not from reading options better.
- **What it means for the anchors.** A model that does not beat BM25 in a domain is not reading the options better than word overlap does.
