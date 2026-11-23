"""Jeva-Eval, pinned: the Jeva paper's frozen evaluation inputs (LIQIIIII/Jeva-Eval; arXiv 2610.18773).

The suite has 22 configurations. Candidate text, order and targets are frozen, and file hashes ship
with the data.
* ``typed_test``: 2,000 decisions.
* BANKING77 and MASSIVE at K = 8 and K = 16. The BANKING77 K = 16 slice holds 3,080 requests, 588 of
  them with the answer absent: Jeva's null protocol.
* Mind2Web cross-task, cross-website and cross-domain, at K = 16.
* ToolRet at K = 32 ... 2,048: the original diagnostic, plus the nested one, where the same 83 queries
  appear at every K.
* MLDR long documents, with the 50-query cache workload.

**Access.** The dataset is gated, with automatic approval. Accept its terms on Hugging Face, then make a
token available (``hf auth login``, or ``HF_TOKEN``). Without one, :func:`load` raises the hub's
``GatedRepoError``.

**Record format** (``docs/evaluation.md`` in github.com/LiQiiiii/Jeva). Models see ``state``,
``question``, ``decision_type``, ``options``, ``allow_null``, ``values`` and ``positive_index``.
``target_probs`` puts the null mass first, then one value per option.

**Mapping to the typed-decision schema:**
* Option ids are neutral, ``o1``...; the given order is kept. Retrieval ``candidate_ids`` go to
  ``meta``, with ``qrels`` and ``eligible_grade``.
* A null mass of 1 makes the gold *absent* (the gold-removed protocol). Otherwise the option targets,
  renormalised, become soft gold, and a partial null mass is kept in ``meta["null_mass"]``.
* ``noul`` becomes a boolean question whose ``"true"`` description is the option at
  ``positive_index``.
* ``score`` levels keep their order. Their numeric ``values`` go to ``meta`` (for MAE in rubric units).
* Rows scored with ``accuracy_rule: tied_top_set`` accept any top-target option. Those options are
  listed in ``meta["accept"]``.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from typing import Any

from systemone.decisions.schema import DecisionRequest, Gold, Item, Question

DATASET = "LIQIIIII/Jeva-Eval"
REVISION = "068237c5f9e8c1b66a9e117ee2367b4c699663a2"  # 2026-10-15

CONFIGS: dict[str, str] = {  # config -> split
    "typed_test": "test",
    "banking77_test_K8": "test",
    "banking77_test_K16": "test",
    "massive_test_K8": "test",
    "massive_test_K16": "test",
    "mind2web_cross_task_K16_present": "test",
    "mind2web_cross_website_K16_present": "test",
    "mind2web_cross_domain_K16_present": "test",
    **{f"toolret_K{k}": "evaluation" for k in (32, 128, 256, 512, 1024, 2048)},
    **{f"toolret_nested_K{k}": "evaluation" for k in (32, 128, 256, 512, 1024, 2048)},
    "mldr": "evaluation",
    "mldr_cache50": "evaluation",
}


def task_of(config: str) -> tuple[str, str]:
    """(leaf, task) for a configuration, e.g. ``banking77_test_K16`` -> ``jeva/banking77``, ``.../K16``."""
    if config == "typed_test":
        return "jeva/typed", "jeva/typed/test"
    if config.startswith("mind2web_"):
        part = config.removeprefix("mind2web_").removesuffix("_present")
        return "jeva/mind2web", f"jeva/mind2web/{part}"
    if config.startswith("mldr"):
        return "jeva/mldr", "jeva/mldr/" + ("cache50" if config.endswith("cache50") else "all")
    family, _, k = config.rpartition("_")
    family = family.removesuffix("_test")
    return f"jeva/{family}", f"jeva/{family}/{k}"


def row_to_item(row: Mapping[str, Any], config: str) -> Item:
    options = [str(o) for o in row["options"]]
    ids = [f"o{i + 1}" for i in range(len(options))]
    target = [float(x) for x in row["target_probs"]]
    null, rest = target[0], target[1:]
    kind = row["decision_type"]
    meta: dict[str, Any] = {"source_id": row["id"], "config": config, "allow_null": bool(row["allow_null"])}
    for key in ("candidate_ids", "qrels", "eligible_grade", "values", "accuracy_rule"):
        if key in row:
            meta[key] = row[key]

    if kind == "noul":
        pos = int(row["positive_index"])
        question = Question(
            type="boolean",
            instructions=row["question"],
            criteria={"true": options[pos], "false": options[1 - pos]},
        )
        p_true = rest[pos] / (sum(rest) or 1.0)
        gold = Gold(dist={"true": p_true, "false": 1.0 - p_true}, label="true" if p_true >= 0.5 else "false")
    elif kind == "score":
        question = Question(type="score", instructions=row["question"], criteria=options)
        levels = [str(i) for i in range(len(options))]
        gold = _soft(levels, rest)
    else:
        question = Question(
            type="choice", instructions=row["question"], criteria=dict(zip(ids, options, strict=True))
        )
        if row["allow_null"] and null >= 1.0 - 1e-9:
            gold = Gold(absent=True)
        else:
            gold = _soft(ids, rest)
            top = max(rest)
            if row.get("accuracy_rule") == "tied_top_set":
                meta["accept"] = [o for o, t in zip(ids, rest, strict=True) if t == top]
    if 0.0 < null < 1.0:
        meta["null_mass"] = null
    leaf, task = task_of(config)
    return Item(
        leaf=leaf,
        task=task,
        request=DecisionRequest(state=row["state"], questions={"answer": question}),
        gold={"answer": gold},
        meta=meta,
    )


def _soft(ids: list[str], values: list[float]) -> Gold:
    total = sum(values)
    if total <= 0:
        raise ValueError("a closed-set record needs positive target mass on its options")
    dist = {o: v / total for o, v in zip(ids, values, strict=True)}
    return Gold(dist=dist, label=max(ids, key=dist.__getitem__))


def load(config: str, revision: str = REVISION, token: str | bool | None = None) -> list[Item]:
    """Every record of one configuration (needs an authorised Hugging Face token; see the module doc)."""
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET, f"data/{config}.jsonl.gz", repo_type="dataset", revision=revision, token=token
    )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [row_to_item(json.loads(line), config) for line in f if line.strip()]
