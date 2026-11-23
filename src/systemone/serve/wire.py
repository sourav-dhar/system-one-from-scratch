"""TypeSafe's ``POST /v1/systemone`` wire format, and conversion to and from typed decisions.

The contract is TypeSafe's (https://docs.typesafe.ai/api.md, read 2026-11-23).

**Request.**
* ``model`` (str), ``state`` (str | object | array) and ``questions`` ({name: question}, at least one).
* ``assets`` (images) is optional. This server refuses requests that carry it.

**Questions.**
* ``noul``: ``instructions``, plus optional ``criteria`` ``{"true": ..., "false": ...}``.
* ``choice``: ``instructions`` and ``criteria`` ``{name: description}``, with 1-255 options. A null
  description means the name alone carries the meaning.
* ``score``: ``instructions`` and ``criteria`` ``[level, ...]``, with 1-10 levels. A level's position is
  its score, from 0.
* ``box`` needs images; this server refuses it.
* Instructions and descriptions may be strings, objects, arrays or null. Here they are rendered as prose
  (:func:`systemone.decisions.text.to_text`).

**Response.** ``model``, ``answers`` ({name: answer}) and ``usage`` ``{input_tokens, output_tokens}``.
* ``noul``: ``{"type": "noul", "noul": P(true)}``.
* ``choice``: ``{"type": "choice", "choice", "confidence", "probabilities"}``.
* ``score``: ``{"type": "score", "score": E[level], "confidence", "legend", "probabilities"}``.

**Errors.** ``400 {"detail": ...}``, and ``422 {"detail": [{loc, msg, type}, ...]}`` for validation.

TypeSafe describes ``confidence`` as how concentrated the distribution is, without a formula. Here it
follows the open CLM implementation of this API: the top probability minus the mean of the others,
clipped to [0, 1].

A question with a single option or level has only one possible answer. The server answers it with
probability 1 and does not ask the model.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

from systemone.decisions.schema import (
    MAX_LEVELS,
    Answer,
    DecisionRequest,
    ModelResponse,
    Question,
)
from systemone.decisions.text import to_text

JSONValue = str | int | float | bool | dict[str, Any] | list[Any] | None
WIRE_MAX_OPTIONS = 255  # TypeSafe's documented limit (our internal schema allows more, for K-sweeps)


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONValue = None
    criteria: dict[str, JSONValue] | None = None

    @field_validator("criteria")
    @classmethod
    def _keys(cls, value: dict[str, JSONValue] | None) -> dict[str, JSONValue] | None:
        if value is not None and not set(value) <= {"true", "false"}:
            raise ValueError('noul criteria may only describe "true" and "false"')
        return value


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONValue = None
    criteria: dict[str, JSONValue] = Field(min_length=1, max_length=WIRE_MAX_OPTIONS)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONValue = None
    criteria: list[JSONValue] = Field(min_length=1, max_length=MAX_LEVELS)


WireQuestion = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, WireQuestion] = Field(min_length=1)
    assets: dict[str, Any] | None = None


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


WireAnswer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, WireAnswer]
    usage: Usage = Field(default_factory=Usage)


# --- wire -> typed decisions ---------------------------------------------------------------------------


def confidence(probs: list[float]) -> float:
    """The top probability minus the mean of the others, clipped to [0, 1]."""
    if len(probs) < 2:
        return 1.0
    top = max(range(len(probs)), key=probs.__getitem__)
    rest = [p for i, p in enumerate(probs) if i != top]
    return max(0.0, min(1.0, probs[top] - sum(rest) / len(rest)))


def to_question(q: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> Question | None:
    """Our question for a wire question; ``None`` for a single-option one, which has a fixed answer."""
    instructions = to_text(q.instructions).strip()
    if isinstance(q, NoulQuestion):
        described = {k: to_text(v) for k, v in (q.criteria or {}).items() if v not in (None, "")}
        return Question(type="boolean", instructions=instructions, criteria=described or None)
    if isinstance(q, ChoiceQuestion):
        if len(q.criteria) == 1:
            return None
        return Question(
            type="choice",
            instructions=instructions,
            criteria={k: to_text(v) if v not in (None, "") else k for k, v in q.criteria.items()},
        )
    if len(q.criteria) == 1:
        return None
    levels = [to_text(v) if v not in (None, "") else str(i) for i, v in enumerate(q.criteria)]
    return Question(type="score", instructions=instructions, criteria=levels)


def to_decision_request(body: SystemOneRequest) -> tuple[DecisionRequest | None, dict[str, dict[str, float]]]:
    """The model's request (``None`` if every question is trivial) and fixed answers for trivial ones."""
    questions: dict[str, Question] = {}
    fixed: dict[str, dict[str, float]] = {}
    for name, wire_q in body.questions.items():
        q = to_question(wire_q)
        if q is not None:
            questions[name] = q
        elif isinstance(wire_q, ChoiceQuestion):
            fixed[name] = {next(iter(wire_q.criteria)): 1.0}
        else:
            fixed[name] = {"0": 1.0}
    request = DecisionRequest(state=body.state, questions=questions) if questions else None
    return request, fixed


def to_wire_answer(
    wire_q: NoulQuestion | ChoiceQuestion | ScoreQuestion, probs: dict[str, float]
) -> WireAnswer:
    """A typed distribution as the wire answer for its question."""
    if isinstance(wire_q, NoulQuestion):
        return NoulAnswer(noul=float(probs["true"]))
    if isinstance(wire_q, ChoiceQuestion):
        ids = list(wire_q.criteria)
        values = [float(probs.get(o, 0.0)) for o in ids]
        best = ids[max(range(len(ids)), key=values.__getitem__)]
        return ChoiceAnswer(
            choice=best, confidence=confidence(values), probabilities=dict(zip(ids, values, strict=True))
        )
    ids = [str(i) for i in range(len(wire_q.criteria))]
    values = [float(probs.get(o, 0.0)) for o in ids]
    return ScoreAnswer(
        score=sum(i * p for i, p in enumerate(values)),
        confidence=confidence(values),
        legend=dict(zip(ids, wire_q.criteria, strict=True)),
        probabilities=dict(zip(ids, values, strict=True)),
    )


# --- typed decisions -> wire, and wire answers -> typed distributions (client side) ---------------------


def to_wire_request(request: DecisionRequest, model: str) -> dict[str, Any]:
    """Our request in TypeSafe's format."""
    questions: dict[str, dict[str, Any]] = {}
    for name, q in request.questions.items():
        if q.type == "boolean":
            entry: dict[str, Any] = {"type": "noul", "instructions": q.instructions}
            if q.criteria:
                entry["criteria"] = dict(q.criteria)  # type: ignore[arg-type]
        else:
            entry = {"type": q.type, "instructions": q.instructions, "criteria": q.criteria}
        questions[name] = entry
    return {"model": model, "state": request.state, "questions": questions}


def from_wire_answer(answer: dict[str, Any] | None, question: Question) -> dict[str, float] | None:
    """A wire answer as a distribution over the question's option ids; ``None`` if it carries none."""
    if not answer:
        return None
    options = question.option_ids()
    if question.type == "boolean":
        if "noul" in answer:
            p = float(answer["noul"])
        elif "probability" in answer:  # the AI-gateway variant of the same API
            p = float(answer["probability"])
        else:
            return None
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities")
    if probs:
        return {o: float(probs.get(o, 0.0)) for o in options}
    if question.type == "choice" and answer.get("choice") in options:
        return {o: float(o == answer["choice"]) for o in options}
    if question.type == "score" and "score" in answer:
        level = str(min(len(options) - 1, max(0, round(float(answer["score"])))))
        return {o: float(o == level) for o in options}
    return None


def to_model_response(body: dict[str, Any], request: DecisionRequest, latency_ms: float) -> ModelResponse:
    answers: dict[str, Answer] = {}
    missing = []
    for name, q in request.questions.items():
        probs = from_wire_answer((body.get("answers") or {}).get(name), q)
        if probs is None:
            missing.append(name)
        else:
            answers[name] = Answer(probs=probs)
    usage = body.get("usage") or {}
    return ModelResponse(
        answers=answers,
        latency_ms=latency_ms,
        usage={k: usage[k] for k in ("input_tokens", "output_tokens") if k in usage} or None,
        error=f"missing answers: {missing}" if missing else None,
    )
