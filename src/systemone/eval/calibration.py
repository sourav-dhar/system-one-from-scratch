"""Calibration: is a System One probability of 0.8 right 80% of the time?

The learned inverse temperature is trained for *retrieval among ~16k in-batch candidates*, but at
decision time the model chooses among a handful of options. Nothing forces the same scale to be
calibrated in both regimes, so we measure it (ECE, reliability bins) and fit a post-hoc scale on
a calibration split - one scalar, no retraining.
"""

from __future__ import annotations

import numpy as np
import torch


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    """Per-bin mean confidence, accuracy and count over equal-width confidence bins."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": float(m.sum()),
                "confidence": float(conf[m].mean()) if m.any() else float("nan"),
                "accuracy": float(correct[m].mean()) if m.any() else float("nan"),
            }
        )
    return bins


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over equal-width bins."""
    bins = reliability_bins(conf, correct, n_bins)
    n = sum(b["count"] for b in bins)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]))


def fit_scale(cos: np.ndarray, init: float = 20.0, steps: int = 200) -> float:
    """Inverse temperature minimizing the gold-option NLL on ``cos`` (``[N, 1+K]``, gold first).

    Optimizes ``log s`` with L-BFGS, so the result is always positive.
    """
    # padding is -inf; d(s*x)/ds = x would then give 0 * -inf = NaN, so use a finite stand-in
    x = torch.from_numpy(np.nan_to_num(np.asarray(cos, dtype=np.float64), neginf=-1e4))
    log_s = torch.tensor(float(np.log(init)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_s], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = -(log_s.exp() * x).log_softmax(1)[:, 0].mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_s.detach().exp())
