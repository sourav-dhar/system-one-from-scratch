"""The Jeva-Eval loader, on synthetic records in the documented format (the real data is gated)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from systemone.decisions import jeva_eval as je


def record(**kw: Any) -> dict[str, Any]:
    base = {
        "id": "r1",
        "state": "Please correct my invoice.",
        "question": "Which team should handle this?",
        "decision_type": "choice",
        "options": ["Billing", "Technical support", "Sales"],
        "allow_null": True,
        "target_probs": [0.0, 1.0, 0.0, 0.0],
    }
    return base | kw


def test_tasks_from_configs() -> None:
    assert je.task_of("banking77_test_K16") == ("jeva/banking77", "jeva/banking77/K16")
    assert je.task_of("toolret_nested_K2048") == ("jeva/toolret_nested", "jeva/toolret_nested/K2048")
    assert je.task_of("mind2web_cross_task_K16_present") == ("jeva/mind2web", "jeva/mind2web/cross_task_K16")
    assert je.task_of("typed_test") == ("jeva/typed", "jeva/typed/test")
    assert je.task_of("mldr_cache50") == ("jeva/mldr", "jeva/mldr/cache50")
    assert len(je.CONFIGS) == 22


def test_choice_with_the_answer_present_and_absent() -> None:
    present = je.row_to_item(record(), "banking77_test_K16")
    q = present.request.questions["answer"]
    assert q.criteria == {"o1": "Billing", "o2": "Technical support", "o3": "Sales"}
    assert present.gold["answer"].label == "o1"
    assert present.request.state == "Please correct my invoice."
    absent = je.row_to_item(record(target_probs=[1.0, 0.0, 0.0, 0.0]), "banking77_test_K16")
    assert absent.gold["answer"].absent
    partial = je.row_to_item(record(target_probs=[0.2, 0.6, 0.2, 0.0]), "banking77_test_K16")
    assert partial.gold["answer"].dist == pytest.approx({"o1": 0.75, "o2": 0.25, "o3": 0.0})
    assert partial.meta["null_mass"] == 0.2


def test_noul_score_and_retrieval_records() -> None:
    noul = je.row_to_item(
        record(
            decision_type="noul",
            options=["No", "Yes"],
            positive_index=1,
            allow_null=False,
            target_probs=[0.0, 0.3, 0.7],
        ),
        "typed_test",
    )
    q = noul.request.questions["answer"]
    assert q.type == "boolean"
    assert q.criteria == {"true": "Yes", "false": "No"}
    assert noul.gold["answer"].dist == pytest.approx({"true": 0.7, "false": 0.3})
    assert noul.gold["answer"].label == "true"

    score = je.row_to_item(
        record(
            decision_type="score",
            options=["low", "mid", "high"],
            values=[1, 3, 5],
            allow_null=False,
            target_probs=[0.0, 0.1, 0.2, 0.7],
        ),
        "typed_test",
    )
    assert score.request.questions["answer"].option_ids() == ["0", "1", "2"]
    assert score.gold["answer"].hard() == "2"
    assert score.meta["values"] == [1, 3, 5]

    big = 300  # beyond TypeSafe's 255-option wire limit: K-sweeps need the larger internal cap
    retrieval = je.row_to_item(
        record(
            options=[f"tool {i}" for i in range(big)],
            candidate_ids=[f"t{i}" for i in range(big)],
            qrels={"t7": 2, "t9": 1},
            eligible_grade=1,
            accuracy_rule="tied_top_set",
            allow_null=False,
            target_probs=[0.0] + [0.5 if i in (7, 9) else 0.0 for i in range(big)],
        ),
        "toolret_K512",
    )
    assert len(retrieval.request.questions["answer"].option_ids()) == big
    assert retrieval.meta["qrels"] == {"t7": 2, "t9": 1}
    assert retrieval.meta["accept"] == ["o8", "o10"]


def test_unknown_config() -> None:
    with pytest.raises(ValueError, match="unknown config"):
        je.load("banking77_K99")


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="Jeva-Eval is gated: needs an authorised HF_TOKEN")
def test_pinned_null_protocol() -> None:
    items = je.load("banking77_test_K16")
    assert len(items) == 3080
    assert sum(i.gold["answer"].absent for i in items) == 588
