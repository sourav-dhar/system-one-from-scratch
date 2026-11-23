"""The OpenDecisions loader: TypeSafe rows mapped onto the typed-decision schema."""

from __future__ import annotations

from typing import Any

import pytest

from systemone.decisions import opendecisions as od

ROWS: list[dict[str, Any]] = [
    {
        "id": "x-choice",
        "family": "llm_router",
        "state": "Turn these expense lines into JSON.",
        "question": {
            "type": "choice",
            "instructions": "Classify the task type.",
            "criteria": {"chitchat": "Small talk.", "extraction_json": "Pull details into JSON."},
        },
        "labels": ["chitchat", "extraction_json"],
        "expected": "extraction_json",
        "target_probs": None,
        "tier": "hard",
        "group": None,
        "label_source": "author+panel",
    },
    {
        "id": "x-noul",
        "family": "policy",
        "state": {"conversation": [{"role": "user", "content": "my SSN is ..."}]},
        "question": {
            "type": "noul",
            "instructions": "Must this stay on-premise?",
            "criteria": {"true": "Contains protected data.", "false": "Nothing protected."},
        },
        "labels": ["no", "yes"],
        "expected": "yes",
        "target_probs": {"yes": 0.75, "no": 0.25},
        "tier": "judge",
        "group": "g1",
        "label_source": "panel",
    },
    {
        "id": "x-score",
        "family": "difficulty",
        "state": "Say hi.",
        "question": {
            "type": "score",
            "instructions": "How hard?",
            "criteria": ["trivial", "routine", "expert"],
        },
        "labels": ["0", "1", "2"],
        "expected": 0,
        "target_probs": {"0": 0.8, "1": 0.2},
        "tier": "easy",
        "group": None,
        "label_source": "programmatic",
    },
]


def test_rows_map_onto_the_schema() -> None:
    choice, noul, score = (od.row_to_item(r) for r in ROWS)
    assert (choice.leaf, choice.task) == ("opendecisions/choice", "opendecisions/choice/llm_router")
    assert choice.request.state == "Turn these expense lines into JSON."
    assert choice.gold["answer"].label == "extraction_json"

    q = noul.request.questions["answer"]
    assert q.type == "boolean"
    assert q.option_texts() == ["Contains protected data.", "Nothing protected."]
    assert noul.gold["answer"].label == "true"
    assert noul.gold["answer"].dist == {"true": 0.75, "false": 0.25}
    assert noul.leaf == "opendecisions/boolean"

    assert score.gold["answer"].label == "0"
    assert score.gold["answer"].target(["0", "1", "2"]) == {"0": 0.8, "1": 0.2, "2": 0.0}
    assert score.meta["source_id"] == "x-score"
    assert score.meta["tier"] == "easy"
    assert all(i.meta["do_not_train"] for i in (choice, noul, score))


@pytest.mark.network
def test_pinned_release() -> None:
    items = od.load()
    assert len(items) == 2352
    assert sum(1 for i in items if i.gold["answer"].dist) == 129
    assert len({i.id for i in items}) == 2351  # one negation pair is duplicated in content
    assert {i.leaf for i in items} == {"opendecisions/choice", "opendecisions/boolean", "opendecisions/score"}
