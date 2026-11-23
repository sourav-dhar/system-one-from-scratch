"""typed-decisions, pinned: four enterprise workflows, five typed questions per case (LocalLLaMA, Apache-2.0).

The four workflows are agent-trace observability, customer service, invoice processing and security
incidents. Each case is one state with five questions (choice, yes/no and score). The train split has
1,200 cases (6,000 decisions); the test split has 400 cases (2,000 decisions).

The gold is a teacher's distribution over the options plus its argmax label, so every question has
soft gold. The distributions are rounded to 6 decimals and are renormalised here: some sum to 1 ± 1e-6.
``factors`` (the latent variables the states were generated from) and ``label_agreement`` (how much
two labelling passes agreed) are kept in ``meta`` for analysis. Models never see them.

Train on the train split only. The test split is a reported suite, and the decontamination registry
(ADR 0004) lists it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from systemone.decisions.schema import DecisionRequest, Gold, Item, Question

DATASET = "LocalLLaMA/typed-decisions"
REVISION = "822f44b399bcc4ba87569c0d0e35c2bcbaede672"  # 2026-11-19
LICENSE = "Apache-2.0"
SPLITS = ("train", "test")


def _gold(raw: Mapping[str, Any]) -> Gold:
    probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
    total = sum(probs.values())
    dist = {k: v / total for k, v in probs.items()} if total > 0 else None
    return Gold(label=str(raw["label"]), dist=dist)


def row_to_item(row: Mapping[str, Any]) -> Item:
    questions = {name: Question.model_validate(q) for name, q in json.loads(row["questions"]).items()}
    gold = {name: _gold(g) for name, g in json.loads(row["gold"]).items()}
    workflow, split = row["workflow"], row["split"]
    return Item(
        leaf=f"typed-decisions/{workflow}",
        task=f"typed-decisions/{workflow}/{split}",
        request=DecisionRequest(state=json.loads(row["state"]), questions=questions),
        gold=gold,
        meta={
            "source_id": row["id"],
            "split": split,
            "factors": json.loads(row["factors"]) if row.get("factors") else None,
            "label_agreement": json.loads(row["label_agreement"]) if row.get("label_agreement") else None,
            "license": LICENSE,
        },
    )


def load(split: str = "test", revision: str = REVISION) -> list[Item]:
    """Every case of one split, as items with five questions each."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; known: {SPLITS}")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET, f"all/{split}-00000-of-00001.parquet", repo_type="dataset", revision=revision
    )
    return [row_to_item(row) for row in pq.read_table(path).to_pylist()]
