"""Scoring typed decisions: one record per question, then accuracy, Brier, NLL and two ECE estimators.

Every number comes only from each question's answer distribution, scored the way System1-Arena scores
it (harness 28e16b1, ``s1arena/metrics.py``), so reference models and ours are scored identically:

* **A question without an answer is wrong.** That covers a failed or refused call, a question the reply
  leaves out, and a reply with no probability on any listed option. For Brier, NLL and calibration such
  a question counts as the uniform distribution. Answers that come with an error still count.
* **The prediction is the most probable option.** A tie goes to the first option in presentation order,
  as upstream's ``max(options, key=...)`` does.
* **Brier** is the multi-class sum over options of (p - t)^2, in [0, 2]. The target t is the soft gold
  distribution when there is one, else one-hot.
* **NLL** is the cross-entropy -sum t log p, with p clipped at 1e-12. Upstream reports none.
* **ECE** is top-label: the confidence is the probability of the predicted option, and "correct" is a
  hit. We report two estimators, because published numbers use both:
  - 10 equal-mass bins with tied confidences pooled (S1-Arena's);
  - 15 equal-width bins (ours, and most papers').

Items missing from the response mapping were never asked, so they are not scored, as upstream does. A
``None`` response is a failed call.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from systemone.decisions.schema import Gold, Item, ModelResponse, Question, normalize
from systemone.eval.calibration import expected_calibration_error

EPS = 1e-12
METRICS = ("accuracy", "brier", "nll", "ece_mass10", "ece_width15")


@dataclass(frozen=True, slots=True)
class Scored:
    """One question of one item, scored for one model."""

    item_id: str
    task: str
    leaf: str
    question: str
    qtype: str
    k: int  # number of options
    answered: bool
    hit: float  # 1.0 if answered and the predicted option is the gold one
    brier: float
    nll: float
    conf: float  # probability of the predicted option; 1/k when unanswered
    p_gold: float  # probability on the gold option (nan when gold is absent)

    @property
    def domain(self) -> str:
        return self.task.split("/", 1)[0]


def answer_distribution(
    response: ModelResponse | None, name: str, options: list[str]
) -> tuple[dict[str, float], bool]:
    """The distribution a response gives for one question, and whether it gives one at all."""
    answer = response.answers.get(name) if response is not None else None
    probs = answer.probs if answer is not None else {}
    if not any(float(probs.get(o, 0.0) or 0.0) > 0 for o in options):
        return {o: 1.0 / len(options) for o in options}, False
    return normalize(probs, options), True


def score_question(
    item: Item, name: str, question: Question, gold: Gold, response: ModelResponse | None
) -> Scored:
    options = question.option_ids()
    probs, answered = answer_distribution(response, name, options)
    target = gold.target(options)
    truth = gold.hard()
    pred = max(options, key=probs.__getitem__)
    hit = float(answered and truth is not None and pred == truth)
    return Scored(
        item_id=item.id,
        task=item.task,
        leaf=item.leaf,
        question=name,
        qtype=question.type,
        k=len(options),
        answered=answered,
        hit=hit,
        brier=sum((probs[o] - target[o]) ** 2 for o in options),
        nll=-sum(t * math.log(max(probs[o], EPS)) for o, t in target.items() if t > 0),
        conf=probs[pred] if answered else 1.0 / len(options),
        p_gold=probs[truth] if truth is not None else float("nan"),
    )


def score_item(item: Item, response: ModelResponse | None) -> list[Scored]:
    """One record per gold-labelled question of the item."""
    return [
        score_question(item, name, item.request.questions[name], gold, response)
        for name, gold in item.gold.items()
    ]


def score(items: Iterable[Item], responses: Mapping[str, ModelResponse | None]) -> list[Scored]:
    """Score every item that has an entry in ``responses`` (keyed by item id)."""
    out: list[Scored] = []
    for item in items:
        if item.id in responses:
            out.extend(score_item(item, responses[item.id]))
    return out


def ece_equal_mass(conf: ArrayLike, correct: ArrayLike, bins: int = 10) -> float:
    """Top-label calibration error over equal-mass bins, with tied confidences pooled (S1-Arena's).

    Answers with the same confidence are interchangeable, so each takes the mean correctness of its
    group. The value then does not depend on how a sort orders ties.
    """
    c = np.asarray(conf, dtype=float)
    if c.size == 0:
        return float("nan")
    order = np.argsort(c, kind="stable")
    c = c[order]
    h = np.asarray(correct, dtype=float)[order]
    _, start, counts = np.unique(c, return_index=True, return_counts=True)
    h = np.repeat(np.add.reduceat(h, start) / counts, counts)
    total = len(c)
    return float(
        sum(
            len(s) / total * abs(c[s].mean() - h[s].mean())
            for s in np.array_split(np.arange(total), bins)
            if len(s)
        )
    )


def ece_equal_width(conf: ArrayLike, correct: ArrayLike, bins: int = 15) -> float:
    """Top-label calibration error over equal-width confidence bins."""
    c = np.asarray(conf, dtype=float)
    if c.size == 0:
        return float("nan")
    return expected_calibration_error(c, np.asarray(correct, dtype=float), bins)


def summarize(scored: Sequence[Scored]) -> dict[str, float]:
    """Pooled metrics over a set of scored questions (one task, one type, or anything else)."""
    n = len(scored)
    if n == 0:
        return {"n_questions": 0.0}
    hits = np.array([s.hit for s in scored])
    conf = [s.conf for s in scored]
    unanswered = sum(not s.answered for s in scored)
    return {
        "n_questions": float(n),
        "accuracy": float(hits.mean()),
        "accuracy_se": float(hits.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "brier": float(np.mean([s.brier for s in scored])),
        "nll": float(np.mean([s.nll for s in scored])),
        "ece_mass10": ece_equal_mass(conf, hits),
        "ece_width15": ece_equal_width(conf, hits),
        "unanswered": float(unanswered),
        "unanswered_rate": unanswered / n,
    }
