"""Baselines that need no GPU: BM25 lexical matching (Robertson & Zaragoza, 2009).

**Scoring.** The *query* is the state, rendered as prose (:func:`systemone.decisions.text.to_text`),
followed by the question's instructions. Each option's *document* is its text: a choice option's
description, or its id when the description is empty; a score level's text; a boolean's ``true`` /
``false`` description when the question gives one. Okapi BM25 scores every document against the query:

    score(q, d) = Σ_{t ∈ q} idf(t) · tf(t, d) · (k1 + 1)
                  / (tf(t, d) + k1 · (1 − b + b · |d| / avgdl))

with ``k1 = 1.5`` and ``b = 0.75`` (the usual defaults), and
``idf(t) = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))``.
Document frequencies and ``avgdl`` come from a *catalog*: every distinct option text of the suite.
Words that appear in many options therefore count for little.

**Probabilities.** The answer is softmax(score). BM25 scores are not probabilities, so this
distribution is uncalibrated by construction. Fit a temperature leave-source-out with
:mod:`systemone.bench.calibrate` before reading Brier or ECE.

**Without option text** (a boolean question with no descriptions, as in all of S1-Arena), BM25 has nothing
to match. It answers uniformly, and the tie goes to the first option ("true"). Such answers are flagged in
``usage``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from systemone.decisions.schema import Answer, DecisionRequest, Item, ModelResponse, Question
from systemone.decisions.text import to_text

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def option_texts(question: Question) -> list[str] | None:
    """The document for each option, or ``None`` when the options carry no text to match."""
    if question.type == "boolean":
        described = question.criteria if isinstance(question.criteria, dict) else {}
        if not described:
            return None
        return [str(described.get(o, "")) for o in question.option_ids()]
    return [t if t else o for o, t in zip(question.option_ids(), question.option_texts(), strict=True)]


class BM25Decider:
    """Okapi BM25 over each question's options, with document statistics from a catalog of option texts."""

    def __init__(self, catalog: Iterable[str], k1: float = 1.5, b: float = 0.75) -> None:
        docs = [tokenize(t) for t in dict.fromkeys(catalog)]
        self.k1, self.b = k1, b
        self.n_docs = max(1, len(docs))
        self.avgdl = sum(len(d) for d in docs) / self.n_docs if docs else 1.0
        self.df: Counter[str] = Counter(t for d in docs for t in set(d))

    @classmethod
    def from_items(cls, items: Sequence[Item], **kwargs: Any) -> BM25Decider:
        """A decider whose catalog is every option text of ``items``."""
        texts = [
            text
            for item in items
            for q in item.request.questions.values()
            for text in (option_texts(q) or [])
        ]
        return cls(texts, **kwargs)

    @property
    def model_id(self) -> str:
        return f"bm25@k1={self.k1:g},b={self.b:g}"

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query: Sequence[str], doc: Sequence[str]) -> float:
        tf = Counter(doc)
        norm = self.k1 * (1.0 - self.b + self.b * len(doc) / self.avgdl)
        return sum(self.idf(t) * tf[t] * (self.k1 + 1.0) / (tf[t] + norm) for t in set(query) if t in tf)

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]:
        out = []
        for request in requests:
            state = to_text(request.state)
            answers: dict[str, Answer] = {}
            no_text: list[str] = []
            for name, q in request.questions.items():
                ids = q.option_ids()
                docs = option_texts(q)
                if docs is None:
                    answers[name] = Answer(probs={o: 1.0 / len(ids) for o in ids})
                    no_text.append(name)
                    continue
                query = tokenize(f"{state}\n{q.instructions}")
                scores = [self.score(query, tokenize(d)) for d in docs]
                top = max(scores)
                exps = [math.exp(s - top) for s in scores]
                total = sum(exps)
                answers[name] = Answer(probs={o: e / total for o, e in zip(ids, exps, strict=True)})
            usage: dict[str, Any] = {"no_option_text": no_text} if no_text else {}
            out.append(ModelResponse(answers=answers, usage=usage or None))
        return out
