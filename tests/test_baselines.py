"""The BM25 baseline: Okapi scoring checked by hand, catalog statistics and the no-text fallback."""

from __future__ import annotations

import math

import pytest

from systemone.bench.baselines import BM25Decider, option_texts, tokenize
from systemone.decisions import DecisionRequest, Gold, Item, Question


def test_tokenize() -> None:
    assert tokenize("Reset my PASSWORD, please!") == ["reset", "my", "password", "please"]


def test_okapi_score_by_hand() -> None:
    bm = BM25Decider(["reset password", "billing refund", "password manager tips"], k1=1.5, b=0.75)
    # df(password) = 2 of 3 documents; avgdl = 7 / 3
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / (7 / 3))
    expected = idf * 1 * 2.5 / (1 + norm)
    assert bm.score(["password"], ["reset", "password"]) == pytest.approx(expected)
    assert bm.score(["unrelated"], ["reset", "password"]) == 0.0


def test_rare_shared_words_decide() -> None:
    q = Question(
        type="choice",
        instructions="Route the ticket.",
        criteria={"it": "password reset and account login", "fin": "billing refund and invoice"},
    )
    req = DecisionRequest(
        state={"ticket": "I was charged twice, please refund my invoice"}, questions={"q": q}
    )
    item = Item(leaf="d/l", task="d/l/t", request=req, gold={"q": Gold(label="fin")})
    bm = BM25Decider.from_items([item])
    (resp,) = bm.decide_many([req])
    probs = resp.answers["q"].probs
    assert probs["fin"] > probs["it"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert bm.model_id == "bm25@k1=1.5,b=0.75"


def test_booleans_without_text_are_uniform_and_flagged() -> None:
    q = Question(type="boolean", instructions="Is it spam?")
    assert option_texts(q) is None
    req = DecisionRequest(state={"text": "win money now"}, questions={"b": q})
    (resp,) = BM25Decider(["x"]).decide_many([req])
    assert resp.answers["b"].probs == {"true": 0.5, "false": 0.5}
    assert resp.usage == {"no_option_text": ["b"]}
    described = Question(
        type="boolean", instructions="?", criteria={"true": "spam offer money", "false": "normal mail"}
    )
    assert option_texts(described) == ["spam offer money", "normal mail"]


def test_empty_choice_descriptions_fall_back_to_ids() -> None:
    q = Question(type="choice", instructions="?", criteria={"refund": "", "login": "account access"})
    assert option_texts(q) == ["refund", "account access"]
