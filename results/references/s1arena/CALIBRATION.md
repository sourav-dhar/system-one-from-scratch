# System1-Arena anchors, calibrated

- **Method:** post-hoc temperature scaling of each model's stored distributions ($p_T \propto p^{1/T}$), fitted **leave-one-domain-out**: each of the 13 domains is calibrated with temperatures fitted on the other 12 (ADR 0004).
- **What it changes:** temperature never changes the top option, so accuracy is untouched. Brier, NLL and ECE move.
- **Aggregation:** the leaderboard's mean over 87 leaves, on the pinned public set (`ad05ed5`).
- **Source:** `scripts/calibration_report.py`, with all numbers in `calibration.json`. No GPU was used.

| Model | Variant | Accuracy | Brier | NLL | ECE (10 equal-mass) | ECE (15 equal-width) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-8B@label-think0` | raw | 59.50 | 0.7139 | 4.283 | 0.356 | 0.361 |
| `Qwen/Qwen3-8B@label-think0` | one temperature | 59.50 | 0.5027 | 1.019 | 0.166 | 0.162 |
| `Qwen/Qwen3-8B@label-think0` | one per question type | 59.50 | 0.4890 | 0.988 | 0.152 | 0.148 |
| `Contrastive-LM/CLM-v0.1-8B` | raw | 41.86 | 0.7411 | 1.900 | 0.286 | 0.281 |
| `Contrastive-LM/CLM-v0.1-8B` | one temperature | 41.86 | 0.6096 | 1.273 | 0.151 | 0.125 |
| `Contrastive-LM/CLM-v0.1-8B` | one per question type | 41.86 | 0.6069 | 1.270 | 0.149 | 0.129 |

## Temperatures

- **`Qwen/Qwen3-8B@label-think0`.** One temperature: 8.85–9.55 across the 13 folds. Per type: boolean 11.64–13.26, choice 6.86–7.42, score 10.84–13.09. For reference, the in-distribution fit gives 9.07.
- **`Contrastive-LM/CLM-v0.1-8B`.** One temperature: 5.33–7.52 across the 13 folds. Per type: boolean 1.12–1.80, choice 6.55–9.07, score 24.03–48.17. For reference, the in-distribution fit gives 6.21.

## Reading

- **Both models are badly overconfident as they ship.** Their temperatures are well above 1.
  - The Qwen3-8B answer-token readout needs about T = 9; its raw NLL is 4.3 nats.
  - CLM needs about T = 6 on top of its capped scale of 100.
- **The calibrated Qwen3-8B readout is M1's bar** (plan v3: the calibrated zero-shot readout of the backbone).
  - With per-type temperatures it reaches Brier 0.489 at 59.5% accuracy.
  - On both accuracy and Brier it already beats the published, as-shipped numbers of three System One models: Decision 2.0 Kai-0.6B (50.3%, 0.537), Laya (44.5%, 0.660) and CLM-v0.1-8B (41.4%, 0.749). Theirs come from the leaderboard's full set; ours is on the public subset (41,183 of 43,226 requests).
- **One temperature cannot fix everything.** ECE stays near 0.15 after calibration, and per-domain miscalibration differs in sign and size (see the per-domain values in `calibration.json`). That is the room a trained, per-type calibrated M1 model has to win.
