"""The answer-token readout and the resumable runner, on a fake tokenizer and a fake causal LM.

The fake LM's next-token distribution depends only on the last input token, so every label's
log-probability can be computed by hand. Slot reads and the prefix-tree walk (with cache cropping) are
checked against that.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from systemone.bench.readout import Readout, combine, labels_for, question_text, state_text
from systemone.bench.runner import read_responses, run, slug
from systemone.decisions import DecisionRequest, Gold, Item, ModelResponse, Question

V = 400
PIECES = ["yes", "Yes", "YES", "no", "No", "NO", "<|assistant|>"]


class FakeTokenizer:
    """Known pieces are one token each; every other character is its own token."""

    chat_template = "... enable_thinking ..."
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            piece = next((p for p in PIECES if text.startswith(p, i)), None)
            if piece is not None:
                ids.append(1 + PIECES.index(piece))
                i += len(piece)
            else:
                ids.append(10 + (ord(text[i]) % (V - 10)))
                i += 1
        return ids

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return f"<s>{messages[0]['content']}</s><u>{messages[1]['content']}</u><|assistant|>"


class FakeCache:
    def __init__(self) -> None:
        self.length = 0
        self.crops: list[int] = []

    def crop(self, n: int) -> None:
        self.crops.append(n)
        self.length = n


class FakeLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.randn(V, V, generator=torch.Generator().manual_seed(0)) * 3
        self.calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        past_key_values: FakeCache | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        self.calls += 1
        if past_key_values is not None:
            assert position_ids is not None
            assert int(position_ids[0, 0]) == past_key_values.length  # the step lands right after the cache
        cache = past_key_values if past_key_values is not None else FakeCache()
        cache.length += input_ids.shape[1]
        logits = self.table[input_ids[:, -1]][:, None, :]
        return SimpleNamespace(logits=logits, past_key_values=cache if use_cache else None)


def logp(prev: int) -> torch.Tensor:
    return FakeLM().table[prev].log_softmax(-1)


def readout(context: int = 4096, **kw: Any) -> Readout:
    return Readout(FakeTokenizer(), FakeLM().eval(), "fake/lm", context, "cpu", **kw)


def request(**questions: Question) -> DecisionRequest:
    return DecisionRequest(state={"text": "hello"}, questions=questions)


CHOICE = Question(type="choice", instructions="Pick.", criteria={"x": "ex", "y": "why", "z": "zed"})
BOOL = Question(type="boolean", instructions="Is it?")
MANY = Question(type="choice", instructions="Pick.", criteria={f"o{i}": f"opt {i}" for i in range(30)})


def test_prompt_pieces_match_upstream() -> None:
    assert labels_for(CHOICE) == ["A", "B", "C"]
    assert labels_for(BOOL) == ["yes", "no"]
    assert labels_for(MANY)[:2] == ["01", "02"]
    assert labels_for(Question(type="score", instructions="?", criteria=list("abcd"))) == ["0", "1", "2", "3"]
    assert question_text(CHOICE, ["A", "B", "C"]).splitlines() == [
        "Question: Pick.",
        "Options:",
        "A. ex",
        "B. why",
        "C. zed",
        "Reply with the label of one option.",
    ]
    assert question_text(BOOL, ["yes", "no"]) == "Question: Is it?\nReply yes or no."
    assert state_text(request(q=BOOL)) == 'Input:\n{\n "text": "hello"\n}'


def test_prepare_picks_slot_tree_or_refusal() -> None:
    r = readout()
    units = r.prepare([request(c=CHOICE, b=BOOL, m=MANY)])
    c, b, m = units
    assert c.variants is not None
    assert [len(v) for v in c.variants] == [1, 1, 1]
    assert b.variants == [[1, 2, 3], [4, 5, 6]]  # yes / Yes / YES and no / No / NO, summed
    assert m.variants is None
    assert m.paths is not None
    assert all(len(p) == 2 for p in m.paths)
    short = readout(context=100).prepare([request(c=CHOICE)])
    assert short[0].error is not None
    assert short[0].error.startswith("rejected: ")


def test_slot_scores_are_label_masses() -> None:
    r = readout()
    (resp,) = r.decide_many([request(b=BOOL, c=CHOICE)])
    unit_b, unit_c = r.prepare([request(b=BOOL, c=CHOICE)])
    lp = logp(unit_b.prompt_ids[-1])
    yes = sum(math.exp(lp[i]) for i in (1, 2, 3))
    no = sum(math.exp(lp[i]) for i in (4, 5, 6))
    assert resp.answers["b"].probs["true"] == pytest.approx(yes / (yes + no), rel=1e-5)
    assert unit_c.variants is not None
    lp = logp(unit_c.prompt_ids[-1])
    masses = [math.exp(lp[v[0]]) for v in unit_c.variants]
    expected = {o: m / sum(masses) for o, m in zip(["x", "y", "z"], masses, strict=True)}
    assert resp.answers["c"].probs == pytest.approx(expected, rel=1e-5)
    assert resp.error is None
    assert resp.usage is not None
    assert set(resp.usage["candidate_mass"]) == {"b", "c"}


def test_tree_scores_sum_along_paths_and_crop_the_cache() -> None:
    r = readout()
    (unit,) = r.prepare([request(m=MANY)])
    assert unit.paths is not None
    r.score_units([unit])
    assert unit.scores is not None
    last = unit.prompt_ids[-1]
    for path, got in zip(unit.paths, unit.scores, strict=True):
        assert got == pytest.approx(float(logp(last)[path[0]] + logp(path[0])[path[1]]), abs=1e-4)
    firsts = {p[0] for p in unit.paths}
    assert r.model.calls == 1 + len(firsts)  # the prompt, then one step per internal node


def test_batching_and_padding_do_not_change_scores() -> None:
    reqs = [
        request(c=CHOICE),
        DecisionRequest(state={"text": "a much longer state " * 20}, questions={"c": CHOICE}),
    ]
    together = readout(max_batch_rows=8).decide_many(reqs)
    alone = [readout(max_batch_rows=1).decide_many([q])[0] for q in reqs]
    for a, b in zip(together, alone, strict=True):
        assert a.answers["c"].probs == pytest.approx(b.answers["c"].probs, rel=1e-6)
    r = readout(max_batch_tokens=10**6, max_batch_rows=2)
    units = r.prepare([request(c=CHOICE)] * 5)
    assert [len(b) for b in r._batches(units)] == [2, 2, 1]


def test_refused_questions_are_reported() -> None:
    (resp,) = readout(context=150).decide_many([request(c=CHOICE)])
    assert resp.answers == {}
    assert resp.error is not None
    assert resp.error.startswith("rejected: c: prompt longer")


def test_combine() -> None:
    assert combine([0.0, 0.0], BOOL) == {"true": 0.5, "false": 0.5}
    probs = combine([math.log(1), math.log(3), -1e9], CHOICE)
    assert probs == pytest.approx({"x": 0.25, "y": 0.75, "z": 0.0})


# --- runner -----------------------------------------------------------------------------------------


class Scripted:
    """Answers 'x' with certainty; fails transiently on chosen states."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.asked: list[str] = []

    @property
    def model_id(self) -> str:
        return "toy/scripted:v1"

    def decide_many(self, requests: list[DecisionRequest]) -> list[ModelResponse]:
        out = []
        for r in requests:
            self.asked.append(r.state["k"])
            if r.state["k"] in self.fail:
                out.append(ModelResponse(error="CUDA out of memory"))
            else:
                out.append(ModelResponse(answers={"c": {"probs": {"x": 1.0}}}))  # type: ignore[dict-item]
        return out


def items(n: int) -> list[Item]:
    return [
        Item(
            leaf="d/l",
            task="d/l/t",
            request=DecisionRequest(state={"k": str(i)}, questions={"c": CHOICE}),
            gold={"c": Gold(label="x")},
        )
        for i in range(n)
    ]


def test_run_resumes_and_retries_only_transient_failures(tmp_path: Path) -> None:
    data = items(5)
    path = tmp_path / "run.jsonl"
    first = Scripted(fail={"3"})
    assert run(first, data, path, chunk=2, log=lambda _: None) == 5
    second = Scripted()
    assert run(second, data, path, chunk=2, log=lambda _: None) == 1
    assert second.asked == ["3"]
    responses = read_responses(path)
    assert len(responses) == 5
    assert all(r.error is None for r in responses.values())
    gz = tmp_path / "run.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(path.read_text(encoding="utf-8"))
    assert read_responses(gz).keys() == responses.keys()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {"item_id", "sample", "response"}


def test_a_rejection_is_final(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"item_id": items(1)[0].id, "sample": 0, "response": {"error": "rejected: too long"}})
        + "\n",
        encoding="utf-8",
    )
    model = Scripted()
    assert run(model, items(1), path, log=lambda _: None) == 0


def test_slug() -> None:
    assert slug("Qwen/Qwen3-8B@label-think0") == "Qwen__Qwen3-8B@label-think0"
    assert slug("clm:Contrastive-LM/CLM-v0.1-8B") == "clm_Contrastive-LM__CLM-v0.1-8B"
