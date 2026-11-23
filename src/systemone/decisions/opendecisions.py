"""OpenDecisions, pinned: 2,352 typed decisions in TypeSafe's format (jgeuter/OpenDecisions, CC-BY-SA-4.0).

The items are 1,178 choice, 586 yes/no and 588 score questions, in 11 families and 4 tiers (easy,
standard, judge, hard). Labels come from code (599), from an LLM author confirmed by a labeller panel
(1,443), or from the panel alone (310). On 129 "calibration" items, the state fixes the true probability
of each option. Those target distributions become soft gold, so Brier and NLL measure calibration
directly there.

Two rows (``jb16-negation_pair_gen-0dc80dc5fa`` and ``-3251a68279``) have identical content, so they
share one content id. A model's cached answer serves both, and both count when scoring, as in the
suite's own scorer, which goes by row.

**The authors ask that the data not be used for training.** Every item carries
``meta["do_not_train"] = True``. The decontamination registry (ADR 0004) also lists this suite.

**Mapping to the typed-decision schema:**
* ``noul`` -> ``boolean``. ``expected`` "yes"/"no" -> gold ``"true"``/``"false"``. The true/false
  descriptions are kept as criteria.
* A score's ``expected`` level (an int) -> its option id ``str(level)``.
* ``target_probs`` -> ``Gold.dist``, with "yes"/"no" keys renamed.
* State strings stay strings, as the TypeSafe format allows.
* Leaf ``opendecisions/<type>``, task ``opendecisions/<type>/<family>``. The mean over leaves is then a
  mean over the three question types, the same unit the suite's own Capability score averages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from systemone.decisions.schema import DecisionRequest, Gold, Item, Question

DATASET = "jgeuter/OpenDecisions"
REVISION = "ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe"  # 2026-10-15
FILE = "opendecisions.jsonl"
LICENSE = "CC-BY-SA-4.0"
YES_NO = {"yes": "true", "no": "false"}
META_KEYS = ("tier", "group", "family", "label_source", "language", "topic", "use_case", "trap_answer")


def _option(question: Question, value: Any) -> str:
    if question.type == "boolean":
        return YES_NO[str(value)]
    return str(value)


def row_to_item(row: Mapping[str, Any]) -> Item:
    question = Question.model_validate(row["question"])
    target = row.get("target_probs")
    dist = {_option(question, k): float(v) for k, v in target.items()} if target else None
    gold = Gold(label=_option(question, row["expected"]), dist=dist)
    kind = question.type
    return Item(
        leaf=f"opendecisions/{kind}",
        task=f"opendecisions/{kind}/{row['family']}",
        request=DecisionRequest(state=row["state"], questions={"answer": question}),
        gold={"answer": gold},
        meta={
            "source_id": row["id"],
            **{k: row.get(k) for k in META_KEYS},
            "license": LICENSE,
            "do_not_train": True,
        },
    )


def load(revision: str = REVISION) -> list[Item]:
    """Every item, in file order (each item has a single question, named ``answer``)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, FILE, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        return [row_to_item(json.loads(line)) for line in f if line.strip()]
