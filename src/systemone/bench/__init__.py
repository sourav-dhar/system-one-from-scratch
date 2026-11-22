"""The benchmark harness: scoring, aggregation and statistics shared by reference models and ours."""

from systemone.bench.metrics import (
    METRICS,
    Scored,
    answer_distribution,
    ece_equal_mass,
    ece_equal_width,
    score,
    score_item,
    summarize,
)
from systemone.bench.stats import aggregate, paired_bootstrap

__all__ = [
    "METRICS",
    "Scored",
    "aggregate",
    "answer_distribution",
    "ece_equal_mass",
    "ece_equal_width",
    "paired_bootstrap",
    "score",
    "score_item",
    "summarize",
]
