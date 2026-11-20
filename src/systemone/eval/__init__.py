"""Benchmarks: in-batch retrieval, the hard-negative ("tricky Q&A") test, and calibration."""

from systemone.eval.calibration import expected_calibration_error, fit_scale, reliability_bins
from systemone.eval.metrics import EvalSuite, evaluate, general_metrics, hard_metrics, hard_scores

__all__ = [
    "EvalSuite",
    "evaluate",
    "expected_calibration_error",
    "fit_scale",
    "general_metrics",
    "hard_metrics",
    "hard_scores",
    "reliability_bins",
]
