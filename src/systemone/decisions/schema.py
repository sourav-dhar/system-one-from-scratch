"""Typed decisions: the request a model answers and the gold a suite scores it against.

A *decision request* is one state plus named questions. Each question has a type:

* ``choice`` picks one of 2-4,096 options (2-255 on the wire). ``criteria`` maps option id -> description.
* ``boolean`` asks whether a statement is true. Its option ids are ``"true"`` and ``"false"``.
  Criteria are optional; when given, they describe what each answer means
  (``{"true": ..., "false": ...}``). TypeSafe's API calls this type ``noul``; both spellings are
  accepted and stored as ``boolean``.
* ``score`` picks one of 2-10 ordered levels. ``criteria`` is the ordered list of level descriptions,
  and the option ids are ``"0"``, ``"1"``, and so on.

A model answers each question with a probability per option id. On open-set protocols it also gives
the probability that no listed option is right. Gold is a hard label, a soft distribution, or both.
On open-set protocols gold can instead be ``absent``: the right answer is not among the options.

The state may be an object, a string or an array, as in TypeSafe's ``POST /v1/systemone``.

The format is wire-compatible with the System1-Arena item schema, version 0.1
(https://github.com/TianyuCodings/S1-Arena, ``s1arena/schema.py``, Apache-2.0, from which the
validation rules and the id definition are adapted):
* an S1-Arena item parses unchanged (S1-Arena itself only uses object states and no boolean
  criteria, so this schema is a superset of it);
* :func:`content_id` reproduces S1-Arena's ids, so a loader can prove that no item differs from a pinned
  release.

Two fields extend the format for open-set protocols: ``Gold.absent`` and ``Answer.none``. The first
enters an id only when it is set, so the ids of closed-set items stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "0.1"
MAX_OPTIONS = 4096  # S1-Arena and TypeSafe's wire allow 255; K-sweeps (ToolRet) reach 2,048
MAX_LEVELS = 10
BOOLEAN_OPTIONS = ("true", "false")
TYPE_ALIASES = {"noul": "boolean"}  # TypeSafe's name for a yes/no question

QuestionType = Literal["choice", "boolean", "score"]


class Question(BaseModel):
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _alias(cls, value: Any) -> Any:
        return TYPE_ALIASES.get(value, value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check(self) -> Question:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_OPTIONS:
                raise ValueError(f"choice questions need 2..{MAX_OPTIONS} options given as a dict")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= MAX_LEVELS:
                raise ValueError(f"score questions need an ordered list of 2..{MAX_LEVELS} levels")
        elif self.criteria is not None and (
            not isinstance(self.criteria, dict) or not set(self.criteria) <= set(BOOLEAN_OPTIONS)
        ):
            raise ValueError('boolean criteria, if given, describe "true" and/or "false" in a dict')
        return self

    def option_ids(self) -> list[str]:
        """The ids a model must give probabilities for, in presentation order."""
        if self.type == "choice":
            assert isinstance(self.criteria, dict)
            return list(self.criteria)
        if self.type == "score":
            assert isinstance(self.criteria, list)
            return [str(i) for i in range(len(self.criteria))]
        return list(BOOLEAN_OPTIONS)

    def option_texts(self) -> list[str]:
        """The text of each option, aligned with :meth:`option_ids`.

        A boolean option without a description gives its id.
        """
        if self.type == "choice":
            assert isinstance(self.criteria, dict)
            return list(self.criteria.values())
        if self.type == "score":
            assert isinstance(self.criteria, list)
            return list(self.criteria)
        described = self.criteria if isinstance(self.criteria, dict) else {}
        return [described.get(o) or o for o in BOOLEAN_OPTIONS]


class DecisionRequest(BaseModel):
    """Everything the model is allowed to see."""

    state: dict[str, Any] | str | list[Any]
    questions: dict[str, Question]


class Gold(BaseModel):
    """A hard label, a soft label distribution, or both; on open-set protocols, possibly ``absent``."""

    label: str | None = None
    dist: dict[str, float] | None = None
    absent: bool = False  # extension: no listed option is right (gold-removed / open-set protocols)

    @model_validator(mode="after")
    def _check(self) -> Gold:
        if self.absent:
            if self.label is not None or self.dist is not None:
                raise ValueError("an absent gold has no label or distribution")
            return self
        if self.label is None and self.dist is None:
            raise ValueError("gold needs a label or a distribution")
        if self.dist is not None:
            total = sum(self.dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"gold distribution sums to {total}, not 1")
        return self

    def hard(self) -> str | None:
        """The single right option: the label, else the distribution's mode; ``None`` when absent."""
        if self.absent:
            return None
        if self.label is not None:
            return self.label
        assert self.dist is not None
        return max(self.dist, key=self.dist.__getitem__)

    def target(self, options: list[str]) -> dict[str, float]:
        """Target distribution for proper scoring: soft if available, else one-hot (all zero if absent)."""
        if self.dist is not None:
            return {o: float(self.dist.get(o, 0.0)) for o in options}
        return {o: float(o == self.label) for o in options}

    def wire(self) -> dict[str, Any]:
        """The S1-Arena form of this gold; the extension field appears only when it is used."""
        out: dict[str, Any] = {"label": self.label, "dist": self.dist}
        if self.absent:
            out["absent"] = True
        return out


class Item(BaseModel):
    """One benchmark request: what the model sees, the gold it never sees, and provenance."""

    schema_version: str = SCHEMA_VERSION
    id: str = ""
    leaf: str
    task: str
    request: DecisionRequest
    gold: dict[str, Gold]
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Item:
        for name, gold in self.gold.items():
            if name not in self.request.questions:
                raise ValueError(f"gold for unknown question {name!r}")
            question = self.request.questions[name]
            options = question.option_ids()
            if gold.absent and question.type != "choice":
                raise ValueError(f"only choice questions can have an absent gold ({name!r})")
            if gold.label is not None and gold.label not in options:
                raise ValueError(f"gold label {gold.label!r} is not an option of {name!r}")
            if gold.dist is not None and not set(gold.dist) <= set(options):
                raise ValueError(f"gold distribution of {name!r} has unknown options")
        if not self.id:
            self.id = content_id(self)
        return self

    @property
    def domain(self) -> str:
        return self.task.split("/", 1)[0]


def content_id(item: Item) -> str:
    """A stable id derived from content, so any change to an item changes its id.

    Byte-for-byte the S1-Arena definition: the task name with dots, a colon, and the first 12 hex
    digits of the SHA-256 of a JSON dump of task, request and gold. Key order is kept on purpose,
    because presenting the options in another order makes another item.
    """
    payload = json.dumps(
        {
            "task": item.task,
            "request": item.request.model_dump(),
            "gold": {name: gold.wire() for name, gold in item.gold.items()},
        },
        ensure_ascii=False,
    )
    return item.task.replace("/", ".") + ":" + hashlib.sha256(payload.encode()).hexdigest()[:12]


class Answer(BaseModel):
    """A model's answer to one question: a distribution over that question's option ids."""

    probs: dict[str, float]
    none: float | None = None  # extension: P(no listed option is right), for open-set protocols


class ModelResponse(BaseModel):
    """A model's reply to one request, as the runner stores it."""

    answers: dict[str, Answer] = Field(default_factory=dict)
    latency_ms: float | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    error: str | None = None
    raw_text: str | None = None  # the reply as written, kept for auditing (generative models)


def normalize(probs: dict[str, float], options: list[str]) -> dict[str, float]:
    """Restrict to known options, clip negatives and renormalize; uniform if nothing usable is left."""
    clean = {o: max(0.0, float(probs.get(o, 0.0) or 0.0)) for o in options}
    total = sum(clean.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: p / total for o, p in clean.items()}
