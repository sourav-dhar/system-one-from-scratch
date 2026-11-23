"""Post-hoc temperature scaling on stored answer distributions, fitted leave-source-out (ADR 0004).

**Rescaling.** A distribution over K options rescales as p_T ∝ p^(1/T), which is the same as dividing
the logits by T. Any model's stored answers can therefore be calibrated without running it again.
Temperature never changes the top option, so accuracy is unchanged; Brier, NLL and ECE move.

**Fitting.** T minimises the mean NLL of the gold (soft targets where given) over the *fitting* groups
only. NLL is convex in β = 1/T, so a bounded golden-section search finds the optimum.

**Leave-source-out.** For each held-out group (an S1-Arena domain, say), the temperatures are fitted on
every other group and applied to the held-out one. ADR 0004 replaced in-distribution fits because
those did not transfer: Kev documents the failure. The variants:
* ``global``: one temperature for every question;
* ``per_type``: one temperature per question type (choice / boolean / score).

Unanswered questions stay unanswered: they score as wrong and uniform, whatever the temperature.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from systemone.bench.metrics import answer_distribution
from systemone.decisions.schema import Answer, Item, ModelResponse

EPS = 1e-12
BETA_RANGE = (0.02, 20.0)  # 1/T between 1/50 and 20


@dataclass
class Segments:
    """Many questions' log-probabilities in one flat array, with targets aligned."""

    logp: np.ndarray  # flat log-probabilities
    target: np.ndarray  # flat targets, same layout
    starts: np.ndarray  # start offset of each question

    def nll(self, beta: float) -> float:
        z = beta * self.logp
        top = np.maximum.reduceat(z, self.starts)
        lse = (
            np.log(
                np.add.reduceat(
                    np.exp(z - np.repeat(top, np.diff(np.append(self.starts, len(z))))), self.starts
                )
            )
            + top
        )
        weighted = np.add.reduceat(self.target * z, self.starts)
        mass = np.add.reduceat(self.target, self.starts)
        return float(np.mean(mass * lse - weighted))


def segments(rows: Sequence[tuple[Sequence[float], Sequence[float]]]) -> Segments:
    """(probabilities, target) per question, as one :class:`Segments`."""
    logp: list[float] = []
    target: list[float] = []
    starts: list[int] = []
    offset = 0
    for probs, tgt in rows:
        starts.append(offset)
        logp.extend(math.log(max(p, EPS)) for p in probs)
        target.extend(tgt)
        offset += len(probs)
    return Segments(np.asarray(logp), np.asarray(target, dtype=float), np.asarray(starts, dtype=int))


def fit_beta(rows: Sequence[tuple[Sequence[float], Sequence[float]]], iterations: int = 80) -> float:
    """The β = 1/T that minimises mean NLL, by golden-section search over :data:`BETA_RANGE` (log scale)."""
    if not rows:
        return 1.0
    seg = segments(rows)
    lo, hi = math.log(BETA_RANGE[0]), math.log(BETA_RANGE[1])
    golden = (math.sqrt(5) - 1) / 2
    a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fa, fb = seg.nll(math.exp(a)), seg.nll(math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - golden * (hi - lo)
            fa = seg.nll(math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + golden * (hi - lo)
            fb = seg.nll(math.exp(b))
    return math.exp((lo + hi) / 2)


def rescale(probs: Mapping[str, float], beta: float) -> dict[str, float]:
    """p^β, renormalised (computed in log space)."""
    keys = list(probs)
    logp = np.array([math.log(max(float(probs[k]), EPS)) for k in keys]) * beta
    logp -= float(np.max(logp))
    p = np.exp(logp)
    p /= p.sum()
    return dict(zip(keys, p.tolist(), strict=True))


def _rows(
    items: Sequence[Item], responses: Mapping[str, ModelResponse | None]
) -> list[tuple[Item, str, str, list[float], list[float]]]:
    """(item, question, type, probabilities, target) for every answered, gold-labelled question."""
    out: list[tuple[Item, str, str, list[float], list[float]]] = []
    for item in items:
        response = responses.get(item.id)
        if response is None:
            continue
        for name, gold in item.gold.items():
            q = item.request.questions[name]
            options = q.option_ids()
            probs, answered = answer_distribution(response, name, options)
            if not answered or gold.absent:
                continue
            target = gold.target(options)
            out.append((item, name, q.type, [probs[o] for o in options], [target[o] for o in options]))
    return out


def leave_out_temperatures(
    items: Sequence[Item],
    responses: Mapping[str, ModelResponse | None],
    group_of: Callable[[Item], str],
    per_type: bool = False,
) -> dict[str, dict[str, float]]:
    """For each held-out group, the temperatures fitted on all other groups: {group: {key: T}}.

    ``key`` is ``"all"``, or the question type when ``per_type`` is set.
    """
    rows = _rows(items, responses)
    groups = sorted({group_of(r[0]) for r in rows})
    out: dict[str, dict[str, float]] = {}
    for held in groups:
        fit_rows = [r for r in rows if group_of(r[0]) != held]
        keys = sorted({r[2] for r in fit_rows}) if per_type else ["all"]
        out[held] = {}
        for key in keys:
            chosen = [(r[3], r[4]) for r in fit_rows if not per_type or r[2] == key]
            out[held][key] = 1.0 / fit_beta(chosen)
    return out


def apply_temperatures(
    items: Sequence[Item],
    responses: Mapping[str, ModelResponse | None],
    group_of: Callable[[Item], str],
    temperatures: Mapping[str, Mapping[str, float]],
) -> dict[str, ModelResponse | None]:
    """Responses with each answer rescaled by its group's (and type's) temperature."""
    out: dict[str, ModelResponse | None] = {}
    for item in items:
        response = responses.get(item.id)
        if item.id not in responses:
            continue
        if response is None:
            out[item.id] = None
            continue
        temps = temperatures.get(group_of(item), {})
        answers = {}
        for name, answer in response.answers.items():
            q = item.request.questions.get(name)
            t = temps.get(q.type if q is not None else "all", temps.get("all", 1.0))
            answers[name] = Answer(probs=rescale(answer.probs, 1.0 / t), none=answer.none)
        out[item.id] = response.model_copy(update={"answers": answers})
    return out


def in_distribution_temperature(
    items: Sequence[Item], responses: Mapping[str, ModelResponse | None]
) -> float:
    """One temperature fitted on the same questions it is scored on: an optimistic reference only."""
    return 1.0 / fit_beta([(r[3], r[4]) for r in _rows(items, responses)])


def by_type_counts(items: Sequence[Item]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        for name in item.gold:
            counts[item.request.questions[name].type] += 1
    return dict(counts)
