"""Scoring, aggregation and the paired bootstrap.

``fixtures/s1arena_scoring.json`` holds random items and answers scored by S1-Arena's own
``metrics.score`` (harness 28e16b1). The answers include failed calls, questions left out, all-zero
replies, exact ties, ids that are not options, negative values and soft gold. Our scorer must match
upstream on every task to 1e-12.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from systemone.bench import (
    aggregate,
    answer_distribution,
    ece_equal_mass,
    ece_equal_width,
    paired_bootstrap,
    score,
    summarize,
)
from systemone.bench.metrics import Scored
from systemone.decisions import Answer, DecisionRequest, Gold, Item, ModelResponse, Question, s1arena

FIXTURE = Path(__file__).parent / "fixtures" / "s1arena_scoring.json"


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["items"] = [s1arena.row_to_item(r) for r in data["rows"]]
    data["responses"] = {k: ModelResponse.model_validate(v) for k, v in data["responses"].items()}
    return data


def one(question: Question, gold: Gold, probs: dict[str, float] | None, task: str = "d/l/t") -> Scored:
    item = Item(
        leaf=task.rsplit("/", 1)[0],
        task=task,
        request=DecisionRequest(state={}, questions={"q": question}),
        gold={"q": gold},
    )
    response = ModelResponse(answers={} if probs is None else {"q": Answer(probs=probs)})
    (scored,) = score([item], {item.id: response})
    return scored


ABC = Question(type="choice", instructions="?", criteria={"a": "A", "b": "B", "c": "C"})


# --- parity with the reference implementation -------------------------------------------------------


def test_matches_upstream_on_every_task(reference: dict[str, Any]) -> None:
    for task, expected in reference["expected"]["tasks"].items():
        mine = summarize(score([i for i in reference["items"] if i.task == task], reference["responses"]))
        assert mine["n_questions"] == expected["n_questions"]
        assert mine["unanswered"] == expected["unanswered"]
        for ours, theirs in (("accuracy", "accuracy"), ("brier", "brier"), ("ece_mass10", "ece")):
            assert mine[ours] == pytest.approx(expected[theirs], abs=1e-12), (task, ours)
        assert mine["accuracy_se"] == pytest.approx(expected["accuracy_se"], abs=1e-12, nan_ok=True)


def test_overall_is_a_mean_over_leaves(reference: dict[str, Any]) -> None:
    scored = score(reference["items"], reference["responses"])
    result = aggregate(scored)
    assert result["overall"]["accuracy"] == pytest.approx(
        reference["expected"]["overall_accuracy"], abs=1e-12
    )
    assert result["overall"]["accuracy"] != pytest.approx(result["pooled"]["accuracy"], abs=1e-6)
    assert result["overall"]["leaves"] == 3
    assert set(result["domains"]) == {"know", "route"}


# --- hand-computed cases ----------------------------------------------------------------------------


def test_a_confident_hit() -> None:
    s = one(ABC, Gold(label="b"), {"a": 0.2, "b": 0.5, "c": 0.3})
    assert (s.answered, s.hit, s.conf, s.p_gold, s.k) == (True, 1.0, 0.5, 0.5, 3)
    assert s.brier == pytest.approx(0.04 + 0.25 + 0.09)
    assert s.nll == pytest.approx(math.log(2))


def test_no_answer_is_wrong_and_uniform() -> None:
    for probs in (None, {"a": 0.0, "b": 0.0, "c": 0.0}, {"zzz": 1.0}):
        s = one(ABC, Gold(label="a"), probs)
        assert (s.answered, s.hit) == (False, 0.0)
        assert s.conf == pytest.approx(1 / 3)
        assert s.brier == pytest.approx((2 / 3) ** 2 + 2 * (1 / 3) ** 2)
        assert s.nll == pytest.approx(math.log(3))


def test_a_tie_goes_to_the_first_option() -> None:
    assert one(ABC, Gold(label="a"), {"a": 0.4, "b": 0.4, "c": 0.2}).hit == 1.0
    assert one(ABC, Gold(label="b"), {"a": 0.4, "b": 0.4, "c": 0.2}).hit == 0.0


def test_soft_gold() -> None:
    q = Question(type="score", instructions="?", criteria=["low", "mid", "high"])
    s = one(q, Gold(dist={"0": 0.0, "1": 0.25, "2": 0.75}), {"0": 0.1, "1": 0.3, "2": 0.6})
    assert s.hit == 1.0  # the mode of the gold distribution
    assert s.brier == pytest.approx(0.01 + 0.05**2 + 0.15**2)
    assert s.nll == pytest.approx(-(0.25 * math.log(0.3) + 0.75 * math.log(0.6)))


def test_unasked_items_are_skipped_and_none_is_a_failed_call() -> None:
    items = [
        Item(
            leaf="d/l",
            task="d/l/t",
            request=DecisionRequest(state={"i": i}, questions={"q": ABC}),
            gold={"q": Gold(label="a")},
        )
        for i in range(3)
    ]
    scored = score(
        items, {items[0].id: None, items[1].id: ModelResponse(answers={"q": Answer(probs={"a": 1.0})})}
    )
    assert [s.item_id for s in scored] == [items[0].id, items[1].id]
    assert [s.answered for s in scored] == [False, True]


def test_answers_with_an_error_still_count() -> None:
    response = ModelResponse(answers={"q": Answer(probs={"a": 0.9, "b": 0.1})}, error="rejected: q2 too long")
    probs, answered = answer_distribution(response, "q", ["a", "b"])
    assert answered
    assert probs == pytest.approx({"a": 0.9, "b": 0.1})


def test_ece_equal_mass_pools_ties() -> None:
    conf, hits = [0.9, 0.9, 0.6, 0.6], [1.0, 0.0, 1.0, 1.0]
    assert ece_equal_mass(conf, hits, bins=2) == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)
    assert ece_equal_mass(conf, [0.0, 1.0, 1.0, 1.0], bins=2) == ece_equal_mass(conf, hits, bins=2)
    assert math.isnan(ece_equal_mass([], []))


def test_ece_equal_width() -> None:
    assert ece_equal_width([0.95, 0.95, 0.55, 0.55], [1, 1, 0, 1]) == pytest.approx(0.5 * 0.05 + 0.5 * 0.05)
    assert math.isnan(ece_equal_width([], []))


# --- aggregation ------------------------------------------------------------------------------------


def records(task: str, hits: list[float], item_prefix: str = "") -> list[Scored]:
    leaf = task.rsplit("/", 1)[0]
    return [
        Scored(
            f"{item_prefix}{task}#{i}",
            task,
            leaf,
            "q",
            "choice",
            2,
            True,
            h,
            0.0 if h else 2.0,
            0.0,
            1.0,
            float(h),
        )
        for i, h in enumerate(hits)
    ]


def test_leaf_weighting() -> None:
    # leaf d/x has two tasks (1.0 and 0.0), leaf d/y one task (1.0) holding 10x the questions
    scored = records("d/x/a", [1.0]) + records("d/x/b", [0.0]) + records("d/y/c", [1.0] * 10)
    result = aggregate(scored)
    assert result["leaves"]["d/x"]["accuracy"] == 0.5
    assert result["overall"]["accuracy"] == pytest.approx(0.75)  # mean over leaves
    assert result["pooled"]["accuracy"] == pytest.approx(11 / 12)  # what an item mean would say
    assert result["domains"]["d"]["accuracy"] == pytest.approx(0.75)
    assert result["types"]["choice"]["n_questions"] == 12


# --- bootstrap --------------------------------------------------------------------------------------


def paired(hits_a: list[float], hits_b: list[float]) -> dict[str, list[Scored]]:
    tasks = ["d/x/a", "d/x/b", "d/y/c"]
    a = [s for t in tasks for s in records(t, hits_a)]
    b = [s for t in tasks for s in records(t, hits_b)]
    return {"A": a, "B": b}


@pytest.mark.parametrize("unit", ["task", "item"])
def test_bootstrap_of_identical_models(unit: str) -> None:
    rng = np.random.default_rng(1)
    hits = [float(x) for x in rng.integers(0, 2, 40)]
    out = paired_bootstrap(paired(hits, hits), b=200, unit=unit)  # type: ignore[arg-type]
    assert out["diff"]["A"]["B"]["accuracy"] == {"ci95": [0.0, 0.0], "p_better": 0.0}
    lo, hi = out["ci95"]["overall"]["A"]["accuracy"]
    assert lo <= np.mean(hits) <= hi


def test_bootstrap_of_a_dominating_model() -> None:
    out = paired_bootstrap(paired([1.0, 1.0, 0.0, 1.0] * 5, [0.0, 1.0, 0.0, 0.0] * 5), b=200)
    assert out["diff"]["A"]["B"]["accuracy"]["p_better"] == 1.0
    assert out["diff"]["A"]["B"]["brier"]["p_better"] == 1.0  # lower Brier is better
    assert out["diff"]["B"]["A"]["accuracy"]["ci95"][1] < 0
    assert set(out["ci95"]) == {"overall", "d"}


def test_bootstrap_is_deterministic_and_checks_its_input() -> None:
    data = paired([1.0, 0.0] * 10, [0.0, 1.0] * 10)
    assert paired_bootstrap(data, b=50, seed=3) == paired_bootstrap(data, b=50, seed=3)
    with pytest.raises(ValueError, match="unit"):
        paired_bootstrap(data, unit="leaf")  # type: ignore[arg-type]
    data["B"] = data["B"][:-1]
    with pytest.raises(ValueError, match="other questions"):
        paired_bootstrap(data)
