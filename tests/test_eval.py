import numpy as np
import pytest
import torch

from systemone.config import HeadConfig, ModelConfig
from systemone.data import PairSet
from systemone.eval import (
    EvalSuite,
    evaluate,
    expected_calibration_error,
    fit_scale,
    general_metrics,
    hard_metrics,
    reliability_bins,
)
from systemone.eval.metrics import summarize_hard
from systemone.model import DualEncoder


def identity_model(d: int = 16) -> DualEncoder:
    """Shared heads, so a state scores highest against an identical action vector."""
    torch.manual_seed(0)
    return DualEncoder(ModelConfig(head=HeadConfig(in_dim=d, width=64, out_dim=32), shared_heads=True))


def test_general_metrics_perfect_when_actions_equal_states() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 16)).astype(np.float32)
    m = general_metrics(identity_model(), PairSet(x, x.copy(), np.arange(2048)), block=1024)
    assert m["general/top1"] == 1.0
    assert m["general/top5"] == 1.0
    assert m["general/n"] == 2048


def test_general_metrics_needs_a_full_block() -> None:
    x = np.zeros((10, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="at least"):
        general_metrics(identity_model(), PairSet(x, x, np.arange(10)))


def test_hard_metrics_skip_rows_without_negatives_and_respect_padding() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 16)).astype(np.float32)
    negs = rng.standard_normal((6, 4, 16)).astype(np.float32)
    count = np.array([4, 0, 2, 1, 3, 4])
    negs[2, 2:] = x[2]  # padding identical to the gold answer must not count
    m = hard_metrics(identity_model(), PairSet(x, x.copy(), np.arange(6), negs, count))
    assert m["hard/n"] == 5
    assert m["hard/top1"] == 1.0
    assert m["hard/mrr"] == 1.0


def test_summarize_hard_known_values() -> None:
    cos = np.array([[0.9, 0.1, -np.inf], [0.2, 0.5, 0.1], [0.3, 0.3, -np.inf]])
    m = summarize_hard(cos, scale=10.0)
    assert m["hard/top1"] == pytest.approx(1 / 3)  # ties are misses: gold must be strictly higher
    assert m["hard/mrr"] == pytest.approx((1 + 1 / 2 + 1 / 2) / 3)
    p0 = np.exp(9) / (np.exp(9) + np.exp(1))
    assert m["hard/p_gold"] == pytest.approx(
        np.mean([p0, np.exp(2) / (np.exp(2) + np.exp(5) + np.exp(1)), 0.5])
    )


def test_ece_zero_when_calibrated_and_positive_when_overconfident() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 200_000)
    correct = rng.uniform(size=conf.size) < conf
    assert expected_calibration_error(conf, correct) < 0.01
    assert expected_calibration_error(np.full(1000, 0.95), np.arange(1000) < 600) == pytest.approx(0.35)
    bins = reliability_bins(conf, correct, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == conf.size


def test_fit_scale_recovers_the_generating_temperature() -> None:
    rng = np.random.default_rng(0)
    true_scale, n, k = 12.0, 40_000, 6
    cos = rng.uniform(-1, 1, (n, k))
    logits = true_scale * cos
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    gold = np.array([rng.choice(k, p=row) for row in p])
    cos[np.arange(n), 0], cos[np.arange(n), gold] = cos[np.arange(n), gold], cos[np.arange(n), 0].copy()
    cos[::3, -1] = -np.inf  # padding must not break the fit
    assert fit_scale(cos, init=50.0) == pytest.approx(true_scale, rel=0.1)


def test_evaluate_combines_suites_and_prefix_dims() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1024, 16)).astype(np.float32)
    negs = rng.standard_normal((1024, 3, 16)).astype(np.float32)
    suite = EvalSuite(
        general=PairSet(x, x, np.arange(1024)), hard=PairSet(x, x, np.arange(1024), negs, np.full(1024, 3))
    )
    m = evaluate(identity_model(), suite, dims=(8,))
    assert {"scale", "general/top1", "hard/top1", "hard@8/top1", "hard/ece"} <= set(m)
    assert "hard/top1" not in evaluate(identity_model(), suite, hard=False)
