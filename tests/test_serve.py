"""Contract tests for ``POST /v1/systemone``: the server against TypeSafe's documented wire format, and
the client round-tripping our requests through it.

The served model is a fake whose distributions are known, so every answer field can be checked by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from systemone.decisions import Answer, DecisionRequest, ModelResponse, Question  # noqa: E402
from systemone.serve import SystemOneClient  # noqa: E402
from systemone.serve.app import create_app  # noqa: E402
from systemone.serve.wire import confidence, from_wire_answer, to_wire_request  # noqa: E402


class Fake:
    """True with 0.8; choice: 0.6 on the last option, the rest spread evenly; score: the same over levels."""

    def __init__(self) -> None:
        self.seen: list[DecisionRequest] = []

    @property
    def model_id(self) -> str:
        return "toy/fake"

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]:
        self.seen.extend(requests)
        out = []
        for r in requests:
            answers = {}
            for name, q in r.questions.items():
                ids = q.option_ids()
                if q.type == "boolean":
                    probs = {"true": 0.8, "false": 0.2}
                else:
                    rest = 0.4 / (len(ids) - 1)
                    probs = {o: (0.6 if o == ids[-1] else rest) for o in ids}
                answers[name] = Answer(probs=probs)
            refused = "too long" in str(r.state)
            out.append(
                ModelResponse(error="rejected: too long")
                if refused
                else ModelResponse(answers=answers, usage={"input_tokens": 42})
            )
        return out


BODY: dict[str, Any] = {
    "model": "toy",
    "state": {"ticket": {"subject": "Refund?", "body": "I was charged twice."}},
    "questions": {
        "billing": {
            "type": "noul",
            "instructions": "Is this about billing?",
            "criteria": {"true": "money", "false": None},
        },
        "route": {
            "type": "choice",
            "instructions": {"task": "Route the ticket", "note": "one team"},
            "criteria": {"support": "General support", "billing": None, "legal": ["contracts", "disputes"]},
        },
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
        "only": {"type": "choice", "instructions": "Pick", "criteria": {"yes": "the only option"}},
    },
}


@pytest.fixture
def served() -> tuple[TestClient, Fake]:
    fake = Fake()
    return TestClient(create_app(fake, name="toy")), fake


def test_answers_follow_the_documented_shapes(served: tuple[TestClient, Fake]) -> None:
    http, fake = served
    reply = http.post("/v1/systemone", json=BODY)
    assert reply.status_code == 200
    body = reply.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "toy"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    a = body["answers"]
    assert a["billing"] == {"type": "noul", "noul": 0.8}
    assert a["route"]["type"] == "choice"
    assert a["route"]["choice"] == "legal"
    assert a["route"]["probabilities"] == pytest.approx({"support": 0.2, "billing": 0.2, "legal": 0.6})
    assert a["route"]["confidence"] == pytest.approx(0.6 - 0.2)
    assert a["urgency"]["score"] == pytest.approx(0.2 * 0 + 0.2 * 1 + 0.6 * 2)
    assert a["urgency"]["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert set(a["urgency"]) == {"type", "score", "confidence", "legend", "probabilities"}
    assert a["only"] == {"type": "choice", "choice": "yes", "confidence": 1.0, "probabilities": {"yes": 1.0}}
    # what the model saw: structured text rendered as prose, null descriptions as their names
    (request,) = fake.seen
    assert "only" not in request.questions  # a single option needs no model
    assert request.questions["route"].instructions == "task: Route the ticket\n\nnote: one team"
    assert request.questions["route"].criteria == {
        "support": "General support",
        "billing": "billing",
        "legal": "- contracts\n- disputes",
    }
    assert request.questions["billing"].criteria == {"true": "money"}


@pytest.mark.parametrize(
    "patch",
    [
        {"questions": {}},
        {"questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}},
        {"questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}}},
        {"questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
        {"questions": {"q": {"type": "noul", "criteria": {"maybe": "?"}}}},
        {"questions": {"q": {"type": "ranking"}}},
        {"state": 3},
    ],
)
def test_malformed_requests_get_422_detail_lists(
    served: tuple[TestClient, Fake], patch: dict[str, Any]
) -> None:
    http, _ = served
    reply = http.post("/v1/systemone", json={**BODY, **patch})
    assert reply.status_code == 422
    assert isinstance(reply.json()["detail"], list)


def test_refusals_unknown_models_and_images_get_400(served: tuple[TestClient, Fake]) -> None:
    http, _ = served
    assert http.post("/v1/systemone", json={**BODY, "model": "jev-latest"}).status_code == 400
    images = {**BODY, "assets": {"a1": {"type": "image", "format": "png", "data_base64": ""}}}
    assert http.post("/v1/systemone", json=images).status_code == 400
    refused = http.post("/v1/systemone", json={**BODY, "state": "too long"})
    assert refused.status_code == 400
    assert refused.json() == {"detail": "rejected: too long"}


def test_api_key_and_model_listing() -> None:
    http = TestClient(create_app(Fake(), name="toy", api_key="s3cret"))
    assert http.post("/v1/systemone", json=BODY).status_code == 401
    assert http.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert (
        http.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    )
    listing = http.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).json()
    assert listing["data"][0]["id"] == "toy"
    assert http.get("/health").json() == {"status": "ok", "model": "toy"}


def test_client_round_trip_reproduces_the_model(served: tuple[TestClient, Fake]) -> None:
    http, _ = served
    client = SystemOneClient("", model="toy", http=http, workers=1)
    requests = [
        DecisionRequest(
            state="plain text state",
            questions={
                "b": Question(type="boolean", instructions="True?", criteria={"true": "yes it is"}),
                "c": Question(type="choice", instructions="Which?", criteria={"x": "ex", "y": "why"}),
                "s": Question(type="score", instructions="How much?", criteria=["none", "some", "lots"]),
            },
        ),
        DecisionRequest(
            state={"note": "too long"},
            questions={"c": Question(type="choice", instructions="?", criteria={"x": "a", "y": "b"})},
        ),
    ]
    ok, refused = client.decide_many(requests)
    direct = Fake().decide_many(requests[:1])[0]
    for name in ("b", "c", "s"):
        assert ok.answers[name].probs == pytest.approx(direct.answers[name].probs)
    assert ok.error is None
    assert ok.usage == {"input_tokens": 42, "output_tokens": 0}
    assert refused.error is not None
    assert refused.error.startswith("rejected: HTTP 400")
    assert client.model_id == "systemone/toy"


def test_wire_helpers() -> None:
    assert confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    assert confidence([1.0]) == 1.0
    q = Question(type="score", instructions="?", criteria=["a", "b", "c"])
    assert from_wire_answer({"type": "score", "score": 1.4}, q) == {"0": 0.0, "1": 1.0, "2": 0.0}
    b = Question(type="boolean", instructions="?")
    assert from_wire_answer({"probability": 0.25}, b) == {"true": 0.25, "false": 0.75}
    assert from_wire_answer({}, b) is None
    wire = to_wire_request(DecisionRequest(state={}, questions={"b": b}), "m")
    assert wire == {"model": "m", "state": {}, "questions": {"b": {"type": "noul", "instructions": "?"}}}
