"""The robustness battery, checked against fake models whose biases are known.

* ``by_position`` puts 0.7 on whichever option is shown first, so it flips under reordering.
* ``by_id`` puts 0.7 on the alphabetically first option id, so it is invariant to order but not to renaming.
* ``by_text`` puts 0.7 on the longest description, so it is invariant to order, renaming and reversal.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from systemone.bench import probes
from systemone.decisions import Answer, DecisionRequest, Gold, Item, ModelResponse, Question

Policy = Callable[[Question], dict[str, float]]


def peaked(options: list[str], top: str) -> dict[str, float]:
    rest = 0.3 / (len(options) - 1)
    return {o: 0.7 if o == top else rest for o in options}


def by_position(q: Question) -> dict[str, float]:
    return peaked(q.option_ids(), q.option_ids()[0])


def by_id(q: Question) -> dict[str, float]:
    return peaked(q.option_ids(), sorted(q.option_ids())[0])


def by_text(q: Question) -> dict[str, float]:
    texts = dict(zip(q.option_ids(), q.option_texts(), strict=True))
    return peaked(q.option_ids(), max(texts, key=lambda o: (len(texts[o]), texts[o])))


def answer(items: list[Item], policy: Policy) -> dict[str, ModelResponse | None]:
    return {
        i.id: ModelResponse(answers={n: Answer(probs=policy(q)) for n, q in i.request.questions.items()})
        for i in items
    }


CHOICE = Question(
    type="choice", instructions="Pick.", criteria={"b": "bee", "a": "a long answer", "c": "sea"}
)
SCORE = Question(type="score", instructions="Rate.", criteria=["low", "middling", "high"])
BOOL = Question(type="boolean", instructions="Is it so?")


def item(**questions: Question) -> Item:
    gold = {
        n: Gold(label={"choice": "a", "score": "2", "boolean": "true"}[q.type]) for n, q in questions.items()
    }
    return Item(
        leaf="d/l", task="d/l/t", request=DecisionRequest(state={"x": 1}, questions=questions), gold=gold
    )


def run_probe(variants: list[probes.Variant], policy: Policy) -> dict[str, float]:
    responses = answer([v.item for v in variants] + [v.source for v in variants], policy)
    return probes.compare(variants, responses)


# --- order --------------------------------------------------------------------------------------------


def test_shuffle_keeps_ids_and_gold() -> None:
    src = item(c=CHOICE)
    v = probes.shuffle(src, random.Random(0))
    assert v is not None
    assert v.item.id != src.id
    assert set(v.item.request.questions["c"].option_ids()) == {"a", "b", "c"}
    assert v.item.request.questions["c"].option_ids() != CHOICE.option_ids()
    assert v.item.gold == src.gold
    assert v.item.meta["probe"] == "shuffle"
    assert v.item.meta["probe_of"] == src.id
    assert probes.shuffle(item(q=BOOL), random.Random(0)) is None


def test_order_probes_catch_position_bias_only() -> None:
    src = item(c=CHOICE)
    shuffled = [probes.shuffle(src, random.Random(seed)) for seed in range(8)]
    variants = [v for v in shuffled if v is not None]
    # a position-biased model flips exactly when another option lands first
    moved = [v.item.request.questions["c"].option_ids()[0] != "b" for v in variants]
    assert run_probe(variants, by_position)["flip_rate"] == pytest.approx(sum(moved) / len(moved))
    assert any(moved)
    assert run_probe(variants, by_id)["flip_rate"] == 0.0
    assert run_probe(variants, by_text)["flip_rate"] == 0.0


def test_reverse_mirrors_score_levels() -> None:
    src = item(s=SCORE)
    v = probes.reverse(src)
    assert v is not None
    assert v.item.request.questions["s"].criteria == ["high", "middling", "low"]
    assert v.item.gold["s"].label == "0"  # "high" was level 2
    assert v.to_source["s"] == {"0": "2", "1": "1", "2": "0"}
    result = run_probe([v], by_text)
    assert result["flip_rate"] == 0.0
    assert result["jsd"] == pytest.approx(0.0, abs=1e-12)
    assert result["acc_source"] == result["acc_variant"]


# --- identifiers -----------------------------------------------------------------------------------


def test_rename_catches_id_bias_only() -> None:
    src = item(c=CHOICE)
    v = probes.rename(src)
    assert v is not None
    assert v.item.request.questions["c"].criteria == {"o1": "bee", "o2": "a long answer", "o3": "sea"}
    assert v.item.gold["c"].label == "o2"
    assert run_probe([v], by_text)["flip_rate"] == 0.0
    assert run_probe([v], by_id)["flip_rate"] == 1.0  # "a" wins before, "o1" (= "b") after


def test_yes_no_swap() -> None:
    src = item(q=BOOL, c=CHOICE)
    pair = probes.yes_no_pair(src)
    assert pair is not None
    a, b = pair
    assert a.item.request.questions["q"].criteria == {"o1": "yes", "o2": "no"}
    assert b.item.request.questions["q"].criteria == {"o1": "no", "o2": "yes"}
    assert (a.item.gold["q"].label, b.item.gold["q"].label) == ("o1", "o2")
    assert a.item.request.questions["c"] == CHOICE  # other questions untouched
    responses = answer([a.item, b.item], by_position)
    assert probes.swap_flips([pair], responses) == {"n": 1.0, "flips_per_100": 100.0}
    responses = answer([a.item, b.item], by_text)  # "yes" is the longer word: always yes
    assert probes.swap_flips([pair], responses)["flips_per_100"] == 0.0
    assert probes.yes_no_pair(item(c=CHOICE)) is None


# --- sensitivity -------------------------------------------------------------------------------------


def test_question_and_state_sensitivity_variants() -> None:
    src = item(c=CHOICE)
    blank = probes.blank_question(src)
    assert blank.item.request.questions["c"].instructions == ""
    assert run_probe([blank], by_text)["jsd"] == 0.0  # this model ignores the question: sensitivity 0
    donor = Item(
        leaf="d/l",
        task="d/l/t",
        request=DecisionRequest(state={"x": 2}, questions={"c": CHOICE}),
        gold=src.gold,
    )
    other = probes.other_state(src, donor)
    assert other.item.request.state == {"x": 2}
    with pytest.raises(ValueError, match="another item"):
        probes.other_state(src, src)


def test_bury_splits_filler_at_whitespace() -> None:
    v = probes.bury(item(c=CHOICE), "alpha beta gamma delta", position=0.4)
    state = v.item.request.state
    assert isinstance(state, dict)
    assert state["input"] == {"x": 1}
    assert state["unrelated_before"] + " " + state["unrelated_after"] == "alpha beta gamma delta"
    assert not state["unrelated_before"].endswith("b")  # never mid-word
    with pytest.raises(ValueError, match="share"):
        probes.bury(item(c=CHOICE), "x", position=1.5)


# --- large K and null --------------------------------------------------------------------------------


def test_add_distractors() -> None:
    src = item(c=CHOICE)
    pool = [f"distractor {i}" for i in range(20)] + ["bee"]
    v = probes.add_distractors(src, pool, k=10, rng=random.Random(1))
    assert v is not None
    ids = v.item.request.questions["c"].option_ids()
    assert len(ids) == 10
    assert {"a", "b", "c"} <= set(ids)
    assert list(v.item.request.questions["c"].option_texts()).count("bee") == 1
    assert v.item.gold == src.gold
    with pytest.raises(ValueError, match="pool"):
        probes.add_distractors(src, ["only one"], k=10, rng=random.Random(1))
    assert probes.add_distractors(item(q=BOOL), pool, k=10, rng=random.Random(1)) is None


def test_remove_gold() -> None:
    src = item(c=CHOICE)
    v = probes.remove_gold(src)
    assert v is not None
    assert v.item.request.questions["c"].option_ids() == ["b", "c"]
    assert v.item.gold["c"].absent
    result = run_probe([v], by_text)
    assert result["variant_confidence"] == pytest.approx(
        0.7
    )  # confidently answering an unanswerable question
    assert math.isnan(result["acc_variant"])
    two = Question(type="choice", instructions="?", criteria={"a": "A", "b": "B"})
    assert probes.remove_gold(item(c=two)) is None


# --- metrics -----------------------------------------------------------------------------------------


def test_jsd_and_auroc() -> None:
    assert probes.jsd([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert probes.jsd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert probes.null_auroc([0.9, 0.8], [0.1, 0.8]) == pytest.approx((2 + 1 + 0.5) / 4)
    assert math.isnan(probes.null_auroc([], [0.1]))


def test_unanswered_pairs_are_counted_not_scored() -> None:
    src = item(c=CHOICE)
    v = probes.shuffle(src, random.Random(0))
    assert v is not None
    responses = answer([src], by_text) | {v.item.id: None}
    result = probes.compare([v], responses)
    assert (result["n"], result["unanswered"]) == (0.0, 1.0)
