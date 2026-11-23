"""Temperature scaling on stored distributions, fitted leave-source-out."""

from __future__ import annotations

import math

import numpy as np
import pytest

from systemone.bench import calibrate as cal
from systemone.bench import score, summarize
from systemone.decisions import Answer, DecisionRequest, Gold, Item, ModelResponse, Question


def test_rescale_keeps_the_top_option() -> None:
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert cal.rescale(p, 1.0) == pytest.approx(p)
    sharp, flat = cal.rescale(p, 3.0), cal.rescale(p, 0.2)
    assert max(sharp, key=sharp.__getitem__) == max(flat, key=flat.__getitem__) == "a"
    assert sharp["a"] > p["a"] > flat["a"]
    assert sum(flat.values()) == pytest.approx(1.0)


def test_segment_nll_matches_a_direct_computation() -> None:
    rows = [([0.7, 0.2, 0.1], [1.0, 0.0, 0.0]), ([0.4, 0.6], [0.25, 0.75])]
    seg = cal.segments(rows)
    beta = 0.7

    def direct(probs: list[float], target: list[float]) -> float:
        z = [beta * math.log(p) for p in probs]
        lse = math.log(sum(math.exp(v) for v in z))
        return -sum(t * (v - lse) for t, v in zip(target, z, strict=True))

    assert seg.nll(beta) == pytest.approx(np.mean([direct(p, t) for p, t in rows]))


def overconfident_rows(n: int, k: int, scale: float, seed: int) -> list[tuple[list[float], list[float]]]:
    """Labels drawn from softmax(z); the model reports softmax(scale * z), so the right beta is 1/scale."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        z = rng.normal(0, 1.5, k)
        true = np.exp(z) / np.exp(z).sum()
        label = rng.choice(k, p=true)
        shown = np.exp(scale * z) / np.exp(scale * z).sum()
        rows.append((shown.tolist(), [float(i == label) for i in range(k)]))
    return rows


def test_fit_recovers_a_known_temperature() -> None:
    beta = cal.fit_beta(overconfident_rows(4000, 4, scale=2.5, seed=0))
    assert beta == pytest.approx(1 / 2.5, rel=0.1)


def item_with(group: str, i: int, probs: list[float], label: int) -> tuple[Item, ModelResponse]:
    ids = [f"o{j}" for j in range(len(probs))]
    q = Question(type="choice", instructions="?", criteria={o: o for o in ids})
    item = Item(
        leaf=f"{group}/l",
        task=f"{group}/l/t",
        request=DecisionRequest(state={"i": i}, questions={"q": q}),
        gold={"q": Gold(label=ids[label])},
    )
    return item, ModelResponse(answers={"q": Answer(probs=dict(zip(ids, probs, strict=True)))})


def test_leave_out_fits_never_see_the_held_out_group() -> None:
    items, responses = [], {}
    for group, scale, seed in (("a", 3.0, 1), ("b", 0.5, 2)):
        for i, (probs, target) in enumerate(overconfident_rows(600, 3, scale, seed)):
            item, response = item_with(group, i, probs, int(np.argmax(target)))
            items.append(item)
            responses[item.id] = response
    temps = cal.leave_out_temperatures(items, responses, lambda it: it.domain)
    # group a is calibrated with b's temperature and vice versa
    assert temps["a"]["all"] == pytest.approx(0.5, rel=0.25)
    assert temps["b"]["all"] == pytest.approx(3.0, rel=0.25)
    calibrated = cal.apply_temperatures(items, responses, lambda it: it.domain, temps)
    raw, fixed = summarize(score(items, responses)), summarize(score(items, calibrated))
    assert fixed["accuracy"] == raw["accuracy"]  # temperature never changes the top option


def test_per_type_keys_and_failed_calls() -> None:
    item, response = item_with("a", 0, [0.6, 0.3, 0.1], 0)
    other, other_response = item_with("b", 1, [0.2, 0.7, 0.1], 1)
    responses = {item.id: response, other.id: other_response}
    temps = cal.leave_out_temperatures([item, other], responses, lambda it: it.domain, per_type=True)
    assert set(temps["a"]) == {"choice"}
    out = cal.apply_temperatures(
        [item, other], {item.id: None, other.id: other_response}, lambda it: it.domain, temps
    )
    assert out[item.id] is None
    assert cal.by_type_counts([item, other]) == {"choice": 2}
