"""The typed-decisions loader: JSON columns mapped onto items with soft gold."""

from __future__ import annotations

import json

import pytest

from systemone.decisions import typed_decisions as td

ROW = {
    "id": "invoice_processing_000007",
    "workflow": "invoice_processing",
    "split": "test",
    "state": json.dumps({"invoice": {"amount": 1200.0, "vendor": "Acme"}, "po_match": False}),
    "questions": json.dumps(
        {
            "approve": {
                "type": "noul",
                "instructions": "This invoice should be approved.",
                "criteria": {"true": "Approve it.", "false": "Hold it."},
            },
            "route": {
                "type": "choice",
                "instructions": "Who handles it?",
                "criteria": {"ap": "Accounts payable.", "fraud": "The fraud team."},
            },
            "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "medium", "high"]},
        }
    ),
    "gold": json.dumps(
        {
            "approve": {
                "label": "false",
                "probabilities": {"true": 0.300001, "false": 0.700001},
                "type": "noul",
            },
            "route": {"label": "ap", "probabilities": {"ap": 0.9, "fraud": 0.1}, "type": "choice"},
            "risk": {"label": "1", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}, "type": "score"},
        }
    ),
    "factors": json.dumps({"po_match": False}),
    "label_agreement": json.dumps({"approve": {"argmax_agree": True}}),
    "n_questions": 3,
}


def test_a_case_maps_onto_one_item() -> None:
    item = td.row_to_item(ROW)
    assert (item.leaf, item.task) == (
        "typed-decisions/invoice_processing",
        "typed-decisions/invoice_processing/test",
    )
    assert item.request.state == {"invoice": {"amount": 1200.0, "vendor": "Acme"}, "po_match": False}
    assert set(item.request.questions) == {"approve", "route", "risk"}
    assert item.request.questions["approve"].type == "boolean"
    approve = item.gold["approve"]
    assert approve.label == "false"
    assert approve.dist is not None
    assert sum(approve.dist.values()) == pytest.approx(1.0, abs=1e-12)  # 1.000002 renormalised
    assert item.gold["risk"].hard() == "1"
    assert item.meta["factors"] == {"po_match": False}
    assert item.meta["source_id"] == "invoice_processing_000007"


def test_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        td.load("dev")


@pytest.mark.network
def test_pinned_release() -> None:
    test = td.load("test")
    assert len(test) == 400
    assert sum(len(i.request.questions) for i in test) == 2000
    assert len({i.leaf for i in test}) == 4
    assert len(td.load("train")) == 1200
