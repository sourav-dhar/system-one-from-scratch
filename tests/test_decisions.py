"""Typed-decision schema and the pinned System1-Arena loader.

The fixture rows were written by S1-Arena's own schema code (harness commit 28e16b1), so their ids are
an independent reference: our ``content_id`` must reproduce them byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from systemone.decisions import Answer, DecisionRequest, Gold, Item, Question, content_id, normalize, s1arena

FIXTURE = Path(__file__).parent / "fixtures" / "s1arena_rows.jsonl"


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def choice(**criteria: str) -> Question:
    return Question(type="choice", instructions="Pick one.", criteria=criteria)


def item(question: Question, gold: Gold, task: str = "toy/leaf/task") -> Item:
    return Item(
        leaf=task.rsplit("/", 1)[0],
        task=task,
        request=DecisionRequest(state={"text": "s"}, questions={"q": question}),
        gold={"q": gold},
    )


# --- ids --------------------------------------------------------------------------------------------


def test_ids_match_the_reference_implementation() -> None:
    for row in rows():
        assert s1arena.row_to_item(row).id == row["id"]


def test_changed_content_is_rejected() -> None:
    row = rows()[0]
    request = json.loads(row["request"])
    request["questions"]["answer"]["criteria"]["mars"] = "Mars (the fourth planet)"
    row["request"] = json.dumps(request)
    with pytest.raises(ValueError, match="was changed"):
        s1arena.row_to_item(row)


def test_option_order_is_part_of_the_id() -> None:
    a = item(choice(x="X", y="Y"), Gold(label="x"))
    b = item(choice(y="Y", x="X"), Gold(label="x"))
    assert a.id != b.id
    assert a.id.startswith("toy.leaf.task:")
    assert len(a.id.split(":")[1]) == 12


def test_noul_is_an_alias_of_boolean() -> None:
    noul = Question(type="noul", instructions="Is it true?")  # type: ignore[arg-type]
    assert noul.type == "boolean"
    assert noul.option_ids() == ["true", "false"]
    assert (
        item(noul, Gold(label="true")).id
        == item(Question(type="boolean", instructions="Is it true?"), Gold(label="true")).id
    )


# --- validation -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"type": "choice", "criteria": {"only": "one option"}}, "choice questions need"),
        ({"type": "choice", "criteria": ["a", "list", "is", "not", "a", "dict"]}, "choice questions need"),
        ({"type": "choice", "criteria": {str(i): str(i) for i in range(4097)}}, "choice questions need"),
        ({"type": "score", "criteria": ["one level"]}, "score questions need"),
        ({"type": "score", "criteria": [str(i) for i in range(11)]}, "score questions need"),
        ({"type": "boolean", "criteria": ["yes", "no"]}, "describe"),
        ({"type": "boolean", "criteria": {"maybe": "unsure"}}, "describe"),
        ({"type": "ranking", "criteria": None}, "type"),
    ],
)
def test_malformed_questions_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Question(instructions="?", **kwargs)


def test_gold_must_fit_its_question() -> None:
    q = choice(a="A", b="B")
    with pytest.raises(ValueError, match="not an option"):
        item(q, Gold(label="c"))
    with pytest.raises(ValueError, match="unknown options"):
        item(q, Gold(dist={"a": 0.5, "c": 0.5}))
    with pytest.raises(ValueError, match="sums to"):
        Gold(dist={"a": 0.5, "b": 0.4})
    with pytest.raises(ValueError, match="label or a distribution"):
        Gold()
    with pytest.raises(ValueError, match="unknown question"):
        Item(
            leaf="t/l",
            task="t/l/x",
            request=DecisionRequest(state={}, questions={"q": q}),
            gold={"other": Gold(label="a")},
        )


def test_typesafe_states_and_boolean_descriptions() -> None:
    described = Question(type="noul", instructions="On-prem?", criteria={"true": "Must stay on-prem."})  # type: ignore[arg-type]
    assert described.option_texts() == ["Must stay on-prem.", "false"]
    assert Question(type="boolean", instructions="?").option_texts() == ["true", "false"]
    for state in ("a plain string", [{"role": "user", "content": "hi"}], {"k": 1}):
        assert item(described, Gold(label="true")).id  # parses and hashes
        assert DecisionRequest(state=state, questions={"q": described}).state == state


def test_option_ids_and_texts_align() -> None:
    score = Question(type="score", instructions="Rate.", criteria=["low", "mid", "high"])
    assert score.option_ids() == ["0", "1", "2"]
    assert score.option_texts() == ["low", "mid", "high"]
    c = choice(a="Alpha", b="Beta")
    assert list(zip(c.option_ids(), c.option_texts(), strict=True)) == [("a", "Alpha"), ("b", "Beta")]


def test_gold_targets() -> None:
    options = ["0", "1", "2"]
    assert Gold(label="1").target(options) == {"0": 0.0, "1": 1.0, "2": 0.0}
    soft = Gold(label="2", dist={"1": 0.2, "2": 0.8})
    assert soft.target(options) == {"0": 0.0, "1": 0.2, "2": 0.8}
    assert soft.hard() == "2"
    assert Gold(dist={"0": 0.1, "1": 0.6, "2": 0.3}).hard() == "1"


# --- open-set extension -----------------------------------------------------------------------------


def test_absent_gold() -> None:
    absent = Gold(absent=True)
    assert absent.hard() is None
    assert absent.target(["a", "b"]) == {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError, match="no label"):
        Gold(absent=True, label="a")
    with pytest.raises(ValueError, match="only choice"):
        item(Question(type="boolean", instructions="?"), absent)
    q = choice(a="A", b="B")
    assert item(q, absent).id != item(q, Gold(label="a")).id


def test_closed_set_ids_do_not_see_the_extension() -> None:
    gold = Gold(label="a")
    assert gold.wire() == {"label": "a", "dist": None}
    assert Answer(probs={"a": 1.0}).none is None


def test_normalize() -> None:
    assert normalize({"a": 2.0, "b": 2.0, "zzz": 5.0}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert normalize({"a": -1.0, "b": 3.0}, ["a", "b"]) == {"a": 0.0, "b": 1.0}
    assert normalize({}, ["a", "b", "c", "d"]) == {o: 0.25 for o in "abcd"}


# --- loader -----------------------------------------------------------------------------------------


def write_parquet(path: Path, data: list[dict[str, Any]]) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist(data), path)
    return path


def test_read_parquet_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_parquet(tmp_path / "test.parquet", rows())
    items = list(s1arena.read_parquet(path))
    assert [i.task for i in items] == [
        "toy/choice/planets",
        "toy/multilabel/emotions",
        "toy/score/helpfulness",
    ]
    assert items[1].gold["fear"].dist == {"true": 0.75, "false": 0.25}
    assert items[1].request.state["lang"] == "fr"

    monkeypatch.setattr(s1arena, "config_file", lambda domain, revision=s1arena.REVISION: str(path))
    assert len(s1arena.load(["know"])) == 3
    assert [i.task for i in s1arena.load(["know"], tasks=["toy/score"])] == ["toy/score/helpfulness"]


def test_unknown_domain() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        s1arena.config_file("nope")


def test_domain_bookkeeping() -> None:
    assert len(s1arena.STATIC_DOMAINS) == 13 == len(set(s1arena.STATIC_DOMAINS))
    assert not set(s1arena.STATIC_DOMAINS) & set(s1arena.AGENT_DOMAINS)
    assert {t.split("/", 1)[0] for t in s1arena.RESTRICTED_TASKS} <= set(s1arena.DOMAINS)
    assert sorted(s1arena.INCOMPLETE_DOMAINS) == ["basic_cls", "fore", "pro", "ret", "social"]
    assert len(s1arena.REVISION) == len(s1arena.HARNESS_COMMIT) == 40


@pytest.mark.network
def test_pinned_release_verifies() -> None:
    """Every row of one real config hashes to its id (the full set takes ~6 s; see the PR)."""
    items = s1arena.load(["fore"])
    assert len(items) == 920
    assert {i.leaf for i in items} == {"fore/binary_events", "fore/multi_outcome", "fore/numeric_bins"}
    assert all(content_id(i) == i.id for i in items)
