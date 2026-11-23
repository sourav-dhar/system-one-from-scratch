# System1-Arena anchors: our harness against the published leaderboard

- **Dataset:** `C-Tianyu/S1-Arena` at `ad05ed5`, the 13 static domains.
- **Reference:** the leaderboard snapshot of 2026-10-14 (harness `28e16b1`).
- **Accuracy:** the leaderboard's own aggregate, a mean over the 87 leaves of each leaf's mean over its tasks. The interval is a 95% paired bootstrap that resamples tasks within leaves and items within tasks.
- **Coverage:** the public dataset lacks 14 restricted sources (2,043 static requests in 9 leaves). The 8 complete domains therefore compare one to one, and the overall figure is computed on the public subset.
- **Acceptance (ADR 0003):** within 1.0 pp of the published accuracy.

| Model | Ours (95% CI) | Published | Δ overall | Δ over 8 complete domains | Brier ours / published | Within 1 pp |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-8B@label-think0` | 59.50 (58.01–61.11) | 59.25 | +0.25 pp | +0.05 pp | 0.7139 / 0.7153 | yes |
| `Contrastive-LM/CLM-v0.1-8B` | 41.86 (40.30–43.43) | 41.43 | +0.43 pp | +0.12 pp | 0.7411 / 0.7492 | yes |

## `Qwen/Qwen3-8B@label-think0`

- **How it was run:** answer-token readout, our transformers port of `lm:` (`bench/readout.py`).
- **Calibration:** NLL 4.283; ECE 0.356 (10 equal-mass bins) and 0.361 (15 equal-width bins).
- **Unanswered:** 0.06% of questions, scored as wrong.

| Domain | Ours | Published | Δ (pp) | Complete in the public set |
|---|---|---|---|---|
| basic_cls | 57.27 | 56.55 | +0.72 | no |
| know | 66.06 | 65.84 | +0.22 | yes |
| social | 46.24 | 46.84 | -0.60 | no |
| pro | 59.19 | 59.23 | -0.04 | no |
| ret | 64.54 | 64.54 | -0.00 | no |
| struct | 63.30 | 63.52 | -0.22 | yes |
| code | 54.96 | 54.75 | +0.21 | yes |
| fore | 44.18 | 39.39 | +4.79 | no |
| dialog | 57.91 | 57.67 | +0.24 | yes |
| batch | 68.80 | 68.67 | +0.13 | yes |
| judge | 59.21 | 59.30 | -0.09 | yes |
| safety | 69.65 | 69.53 | +0.12 | yes |
| route | 53.73 | 53.97 | -0.24 | yes |

## `Contrastive-LM/CLM-v0.1-8B`

- **How it was run:** CLM heads on our Qwen3-8B last-token encoder (`bench/clm.py`).
- **Calibration:** NLL 1.900; ECE 0.286 (10 equal-mass bins) and 0.281 (15 equal-width bins).
- **Unanswered:** 0.00% of questions, scored as wrong.

| Domain | Ours | Published | Δ (pp) | Complete in the public set |
|---|---|---|---|---|
| basic_cls | 37.98 | 35.64 | +2.34 | no |
| know | 48.20 | 48.33 | -0.13 | yes |
| social | 20.34 | 22.07 | -1.73 | no |
| pro | 28.60 | 28.17 | +0.43 | no |
| ret | 48.30 | 45.87 | +2.43 | no |
| struct | 41.53 | 40.65 | +0.88 | yes |
| code | 40.09 | 40.09 | -0.00 | yes |
| fore | 36.33 | 35.53 | +0.80 | no |
| dialog | 41.17 | 40.83 | +0.34 | yes |
| batch | 41.80 | 42.20 | -0.40 | yes |
| judge | 45.37 | 45.99 | -0.62 | yes |
| safety | 55.21 | 54.74 | +0.47 | yes |
| route | 50.04 | 49.58 | +0.46 | yes |
