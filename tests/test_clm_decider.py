"""The CLM decider: text construction against the official client, and the scoring math.

``fixtures/clm_texts.json`` was written by the official ``clm.schema.build_pairs`` (contrastive-lm
0.1.0) on nested states, empty option descriptions, blank instructions, booleans and scores. Our port
must produce the same state and candidate texts, byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from systemone.bench.clm import CLMDecider, candidate_texts, state_text, tail_text, to_text
from systemone.config import HeadConfig, ModelConfig
from systemone.decisions import DecisionRequest, Question
from systemone.model import DualEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "clm_texts.json"


def test_texts_match_the_official_client() -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        request = DecisionRequest.model_validate(case["request"])
        for name, question in request.questions.items():
            expected = case["texts"][name]
            assert state_text(request.state, question.instructions) == expected["state"]
            official = dict(zip(expected["keys"], expected["candidates"], strict=True))
            ours = dict(zip(question.option_ids(), candidate_texts(question), strict=True))
            assert ours == official


def test_to_text_scalars() -> None:
    assert to_text(None) == ""
    assert to_text(True) == "true"
    assert to_text(2.5) == "2.5"
    assert to_text({"a": [], "b": {}}) == "a: \n\nb: "


class CharTokenizer:
    """One token per character."""

    def __call__(self, texts: list[str], add_special_tokens: bool = True) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(t) for t in texts]}

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


def test_tail_text_keeps_the_end() -> None:
    tok = CharTokenizer()
    assert tail_text(tok, tok.encode("abcdefghij"), 4) == "ghij"


def fake_embed(texts: list[str]) -> torch.Tensor:
    vecs = [torch.randn(8, generator=torch.Generator().manual_seed(sum(map(ord, t)) % 2**31)) for t in texts]
    return F.normalize(torch.stack(vecs), dim=-1)


def decider(maxlen: int = 2048) -> CLMDecider:
    torch.manual_seed(0)
    heads = DualEncoder(ModelConfig(head=HeadConfig(in_dim=8, width=16, out_dim=4, depth=3)))
    heads.logit_scale.data.fill_(2.0)
    return CLMDecider(heads, fake_embed, CharTokenizer(), maxlen=maxlen, name="toy/clm")


CHOICE = Question(type="choice", instructions="Which?", criteria={"a": "Alpha", "b": "Beta", "c": ""})


def test_probabilities_are_a_softmax_of_scaled_cosines() -> None:
    d = decider()
    request = DecisionRequest(state={"text": "hello"}, questions={"q": CHOICE})
    (response,) = d.decide_many([request])
    texts = [state_text(request.state, "Which?"), "Alpha", "Beta", "c"]
    with torch.no_grad():
        z = fake_embed(texts)
        s, o = d.heads.encode_state(z[:1])[0], d.heads.encode_action(z[1:])
        expected = F.softmax(d.heads.scale() * (o @ s), dim=0).tolist()
    assert list(response.answers["q"].probs) == ["a", "b", "c"]
    assert list(response.answers["q"].probs.values()) == pytest.approx(expected, rel=1e-5)
    assert response.error is None
    assert d.model_id == "toy/clm"


def test_long_texts_are_cut_and_empty_ones_refused() -> None:
    d = decider(maxlen=12)
    long_state = DecisionRequest(state={"text": "x" * 50}, questions={"q": CHOICE})
    empty = DecisionRequest(state={}, questions={"q": Question(type="boolean", instructions="")})
    first, second = d.decide_many([long_state, empty])
    assert first.usage is not None
    assert first.usage["cut_texts"] == 1
    assert set(first.answers["q"].probs) == {"a", "b", "c"}
    assert second.answers == {}
    assert second.error == "rejected: q: the state text is empty"
    assert d.model_id == "toy/clm@maxlen12-cut"


def test_boolean_answers_are_keyed_true_false() -> None:
    d = decider()
    request = DecisionRequest(
        state={"t": "s"}, questions={"b": Question(type="boolean", instructions="It holds.")}
    )
    probs: dict[str, Any] = d.decide_many([request])[0].answers["b"].probs
    assert list(probs) == ["true", "false"]
    assert sum(probs.values()) == pytest.approx(1.0)
