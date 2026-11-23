"""The robustness battery: item transformations with a known effect on the answer, plus paired metrics.

Each probe turns an item into a *variant*. For every question, it records how the variant's option ids
map back to the original's. Running one model on both and comparing the mapped distributions measures
one property (survey §2, P2-P6; §7):

| Probe | Property | A well-behaved model |
|---|---|---|
| :func:`shuffle`, :func:`reverse` | P2 permutation equivariance | the same distribution: no flips |
| :func:`rename` | robustness to option ids | the same distribution with ids ``o1``... |
| :func:`yes_no_pair` | identifier bias | the same decision when yes and no trade ids |
| :func:`blank_question` | P6 question sensitivity | a changed answer (unchanged: question ignored) |
| :func:`other_state` | state sensitivity | a changed answer (unchanged: state ignored) |
| :func:`bury` | long-context robustness | the same answer inside unrelated text |
| :func:`add_distractors` | large-K robustness | the same answer among more options |
| :func:`remove_gold` | P4 null coverage | low confidence; P(none) rises if it has one |

Probes only build items. A model answers them through the usual runner, and :func:`compare`
pairs each variant with its source.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from systemone.bench.metrics import answer_distribution
from systemone.decisions.schema import MAX_OPTIONS, DecisionRequest, Gold, Item, ModelResponse, Question


@dataclass(frozen=True)
class Variant:
    """A transformed item, the item it came from, and per question the map variant id -> source id."""

    probe: str
    source: Item
    item: Item
    to_source: dict[str, dict[str, str]] = field(default_factory=dict)


def _variant(
    probe: str,
    source: Item,
    questions: Mapping[str, Question],
    gold: Mapping[str, Gold],
    to_source: dict[str, dict[str, str]],
    state: Any = None,
) -> Variant:
    item = Item(
        leaf=source.leaf,
        task=source.task,
        request=DecisionRequest(
            state=source.request.state if state is None else state, questions=dict(questions)
        ),
        gold=dict(gold),
        meta={**source.meta, "probe": probe, "probe_of": source.id},
    )
    for name, question in item.request.questions.items():
        to_source.setdefault(name, {o: o for o in question.option_ids()})
    return Variant(probe, source, item, to_source)


def _choice_items(question: Question) -> list[tuple[str, str]]:
    assert isinstance(question.criteria, dict)
    return list(question.criteria.items())


def _remap_gold(gold: Gold, to_variant: Mapping[str, str]) -> Gold:
    if gold.absent:
        return gold
    label = to_variant[gold.label] if gold.label is not None else None
    dist = {to_variant[k]: v for k, v in gold.dist.items()} if gold.dist is not None else None
    return Gold(label=label, dist=dist)


def shuffle(item: Item, rng: random.Random) -> Variant | None:
    """Choice options in a new random order (ids unchanged); ``None`` if the item has no choice question."""
    questions, changed = dict(item.request.questions), False
    for name, q in item.request.questions.items():
        if q.type == "choice":
            pairs = _choice_items(q)
            order = list(range(len(pairs)))
            while len(pairs) > 1 and order == list(range(len(pairs))):
                rng.shuffle(order)
            questions[name] = q.model_copy(update={"criteria": dict(pairs[i] for i in order)})
            changed = True
    return _variant("shuffle", item, questions, item.gold, {}) if changed else None


def reverse(item: Item) -> Variant | None:
    """Choice options in reverse order (ids unchanged); score levels mirrored (level i becomes K-1-i)."""
    questions, gold, to_source, changed = dict(item.request.questions), dict(item.gold), {}, False
    for name, q in item.request.questions.items():
        if q.type == "choice":
            questions[name] = q.model_copy(update={"criteria": dict(reversed(_choice_items(q)))})
            changed = True
        elif q.type == "score":
            assert isinstance(q.criteria, list)
            k = len(q.criteria)
            questions[name] = q.model_copy(update={"criteria": list(reversed(q.criteria))})
            to_source[name] = {str(j): str(k - 1 - j) for j in range(k)}
            if name in gold:
                gold[name] = _remap_gold(gold[name], {v: s for s, v in to_source[name].items()})
            changed = True
    return _variant("reverse", item, questions, gold, to_source) if changed else None


def rename(item: Item) -> Variant | None:
    """Choice option ids replaced by neutral ``o1``... in presentation order (descriptions kept)."""
    questions, gold, to_source, changed = dict(item.request.questions), dict(item.gold), {}, False
    for name, q in item.request.questions.items():
        if q.type == "choice":
            pairs = _choice_items(q)
            ids = {old: f"o{i + 1}" for i, (old, _) in enumerate(pairs)}
            questions[name] = q.model_copy(update={"criteria": {ids[o]: d for o, d in pairs}})
            to_source[name] = {new: old for old, new in ids.items()}
            if name in gold:
                gold[name] = _remap_gold(gold[name], ids)
            changed = True
    return _variant("rename", item, questions, gold, to_source) if changed else None


def yes_no_pair(item: Item) -> tuple[Variant, Variant] | None:
    """Each yes/no question as a two-option choice with neutral ids, once with ``o1`` = yes and once with
    ``o1`` = no. A model free of identifier bias gives the same decision in both."""
    if not any(q.type == "boolean" for q in item.request.questions.values()):
        return None
    out = []
    for probe, first in (("yes_no_swap_a", "yes"), ("yes_no_swap_b", "no")):
        questions, gold, to_source = dict(item.request.questions), dict(item.gold), {}
        for name, q in item.request.questions.items():
            if q.type != "boolean":
                continue
            ids = {"true": "o1", "false": "o2"} if first == "yes" else {"true": "o2", "false": "o1"}
            words = {"true": "yes", "false": "no"}
            criteria = {ids[t]: words[t] for t in sorted(ids, key=ids.__getitem__)}
            questions[name] = Question(type="choice", instructions=q.instructions, criteria=criteria)
            to_source[name] = {v: k for k, v in ids.items()}
            if name in gold:
                gold[name] = _remap_gold(gold[name], ids)
        out.append(_variant(probe, item, questions, gold, to_source))
    return out[0], out[1]


def blank_question(item: Item) -> Variant:
    """Every question's instructions removed: how much does the answer depend on the question?"""
    questions = {n: q.model_copy(update={"instructions": ""}) for n, q in item.request.questions.items()}
    return _variant("blank_question", item, questions, item.gold, {})


def other_state(item: Item, donor: Item) -> Variant:
    """The state of another item (same task, ideally) under this item's questions."""
    if donor.id == item.id:
        raise ValueError("the donor must be another item")
    return _variant("other_state", item, item.request.questions, item.gold, {}, state=donor.request.state)


def bury(item: Item, filler: str, position: float = 0.5) -> Variant:
    """The original input placed inside unrelated text.

    A ``position`` share of ``filler`` comes before the input and the rest after it, at a whitespace
    boundary. The state becomes ``{"unrelated_before", "input", "unrelated_after"}``.
    """
    if not 0.0 <= position <= 1.0:
        raise ValueError("position is a share of the filler, in [0, 1]")
    cut = int(len(filler) * position)
    while 0 < cut < len(filler) and not filler[cut].isspace():
        cut += 1
    state = {
        "unrelated_before": filler[:cut].strip(),
        "input": item.request.state,
        "unrelated_after": filler[cut:].strip(),
    }
    return _variant("bury", item, item.request.questions, item.gold, {}, state=state)


def add_distractors(item: Item, pool: Sequence[str], k: int, rng: random.Random) -> Variant | None:
    """Each choice question padded to ``k`` options with descriptions drawn from ``pool``, then shuffled.

    Distractors get ids ``d1``...; descriptions already present are never drawn. Returns ``None`` when
    there is no choice question to pad.
    """
    if not 2 <= k <= MAX_OPTIONS:
        raise ValueError(f"k must be in [2, {MAX_OPTIONS}]")
    questions, changed = dict(item.request.questions), False
    for name, q in item.request.questions.items():
        if q.type != "choice":
            continue
        pairs = _choice_items(q)
        need = k - len(pairs)
        if need <= 0:
            continue
        taken = {d for _, d in pairs}
        candidates = [d for d in dict.fromkeys(pool) if d not in taken]
        if len(candidates) < need:
            raise ValueError(f"the pool has {len(candidates)} unused descriptions; {need} are needed")
        extra = [(f"d{i + 1}", d) for i, d in enumerate(rng.sample(candidates, need))]
        merged = pairs + extra
        rng.shuffle(merged)
        questions[name] = q.model_copy(update={"criteria": dict(merged)})
        changed = True
    return _variant("distractors", item, questions, item.gold, {}) if changed else None


def remove_gold(item: Item) -> Variant | None:
    """The gold option removed from every choice question with a hard gold and 3 or more options."""
    questions, gold, changed = dict(item.request.questions), dict(item.gold), False
    for name, q in item.request.questions.items():
        g = item.gold.get(name)
        if q.type != "choice" or g is None or g.label is None or len(q.option_ids()) < 3:
            continue
        questions[name] = q.model_copy(
            update={"criteria": {o: d for o, d in _choice_items(q) if o != g.label}}
        )
        gold[name] = Gold(absent=True)
        changed = True
    return _variant("remove_gold", item, questions, gold, {}) if changed else None


# --- paired metrics ---------------------------------------------------------------------------------


def jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, in [0, 1]."""
    a, b = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = (a + b) / 2

    def kl(x: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / m[mask])))

    return (kl(a) + kl(b)) / 2


def _argmax(probs: Mapping[str, float], order: Sequence[str]) -> str:
    return max(order, key=lambda o: probs[o])


def compare(
    variants: Sequence[Variant],
    responses: Mapping[str, ModelResponse | None],
    source_responses: Mapping[str, ModelResponse | None] | None = None,
) -> dict[str, float]:
    """Paired metrics of variants against their sources, over questions both answered.

    The variant's distribution is mapped back onto the source's option ids. Options that only the
    variant has (distractors) or only the source has (a removed gold) are left out of the paired
    distributions.

    * ``flip_rate``: the share of questions whose top option changes.
    * ``jsd``: the mean Jensen-Shannon divergence between the paired distributions.
    * ``acc_source`` / ``acc_variant``: accuracies on the gold-labelled questions.
    * ``variant_confidence``: the mean top probability on the variant (for gold-removed variants, high
      values mean confidently wrong).
    """
    source_responses = source_responses if source_responses is not None else responses
    flips, divs, conf, acc_s, acc_v, unanswered = [], [], [], [], [], 0
    for v in variants:
        rs, rv = source_responses.get(v.source.id), responses.get(v.item.id)
        for name, vq in v.item.request.questions.items():
            sq = v.source.request.questions[name]
            v_ids, s_ids = vq.option_ids(), sq.option_ids()
            pv, ok_v = answer_distribution(rv, name, v_ids)
            ps, ok_s = answer_distribution(rs, name, s_ids)
            if not (ok_v and ok_s):
                unanswered += 1
                continue
            to_src = v.to_source.get(name, {})
            mapped = {to_src[o]: pv[o] for o in v_ids if to_src.get(o) in ps}
            shared = [o for o in s_ids if o in mapped]
            if len(shared) >= 2:
                a = [ps[o] for o in shared]
                b = [mapped[o] for o in shared]
                sa, sb = sum(a), sum(b)
                if sa > 0 and sb > 0:
                    a, b = [x / sa for x in a], [x / sb for x in b]
                    flips.append(float(shared[int(np.argmax(a))] != shared[int(np.argmax(b))]))
                    divs.append(jsd(a, b))
            conf.append(max(pv.values()))
            g_s, g_v = v.source.gold.get(name), v.item.gold.get(name)
            if g_s is not None and g_s.hard() is not None:
                acc_s.append(float(_argmax(ps, s_ids) == g_s.hard()))
            if g_v is not None and g_v.hard() is not None:
                acc_v.append(float(_argmax(pv, v_ids) == g_v.hard()))

    def mean(x: list[float]) -> float:
        return float(np.mean(x)) if x else math.nan

    return {
        "n": float(len(conf)),
        "unanswered": float(unanswered),
        "flip_rate": mean(flips),
        "jsd": mean(divs),
        "variant_confidence": mean(conf),
        "acc_source": mean(acc_s),
        "acc_variant": mean(acc_v),
    }


def swap_flips(
    pairs: Sequence[tuple[Variant, Variant]], responses: Mapping[str, ModelResponse | None]
) -> dict[str, float]:
    """Identifier bias: how often the yes/no decision changes when "yes" and "no" trade identifiers.

    Each pair comes from :func:`yes_no_pair`. A decision is "true" when P(yes) >= 0.5. The result is
    reported per 100 questions, as Rafe & Das report it.
    """
    flips: list[float] = []
    for a, b in pairs:
        for name in a.to_source:
            if a.source.request.questions[name].type != "boolean":
                continue
            yes = []
            for v in (a, b):
                ids = v.item.request.questions[name].option_ids()
                probs, ok = answer_distribution(responses.get(v.item.id), name, ids)
                if not ok:
                    break
                yes.append(sum(p for o, p in probs.items() if v.to_source[name][o] == "true"))
            if len(yes) == 2:
                flips.append(float((yes[0] >= 0.5) != (yes[1] >= 0.5)))
    return {"n": float(len(flips)), "flips_per_100": 100 * float(np.mean(flips)) if flips else math.nan}


def null_auroc(absent: Sequence[float], present: Sequence[float]) -> float:
    """AUROC of a null score: the probability that an answer-absent request outranks an answer-present one."""
    a, p = np.asarray(absent, dtype=float), np.asarray(present, dtype=float)
    if not len(a) or not len(p):
        return math.nan
    greater = (a[:, None] > p[None, :]).sum()
    ties = (a[:, None] == p[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(a) * len(p)))


PROBES: dict[str, Callable[..., Any]] = {
    "shuffle": shuffle,
    "reverse": reverse,
    "rename": rename,
    "yes_no_pair": yes_no_pair,
    "blank_question": blank_question,
    "other_state": other_state,
    "bury": bury,
    "add_distractors": add_distractors,
    "remove_gold": remove_gold,
}
