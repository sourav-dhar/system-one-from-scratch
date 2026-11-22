"""System1-Arena, pinned: typed decisions with probabilities, 13 domains plus agent steps (arXiv 2610.21244).

Every row of the public dataset becomes an :class:`~systemone.decisions.schema.Item`, and its
content id is recomputed. A row whose content does not match its id is an error, so a silently
changed item can never be scored.

**Pinning.** Scores are comparable only on the same items, so the loader defaults to one dataset
commit. It also records the harness commit whose leaderboard (snapshot of 2026-10-14) holds the
published numbers.

**What the public dataset lacks.** Fourteen sources forbid redistribution: eleven non-agentic and
three agent-step sources. Upstream rebuilds them from their original loaders
(``s1arena build --local``), and they are listed in :data:`RESTRICTED_TASKS`. Without them, the
public set has 41,183 of the leaderboard's 43,226 non-agentic requests. Every one of the 87 leaves
is present, but 9 of them hold fewer tasks; all 9 are in ``basic_cls``, ``fore``, ``pro``, ``ret``
and ``social``. The other eight domains are complete, so their scores compare one to one with the
published per-domain numbers.

**Domains.** The leaderboard's static accuracy averages the 13 domains in :data:`GROUPS`, as a mean
over leaves of the mean over each leaf's tasks. The ``step``, ``embodied`` and ``game`` configs
hold recorded agent steps, scored separately as step accuracy; they form our W2 quality suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from systemone.decisions.schema import DecisionRequest, Gold, Item, content_id

DATASET = "C-Tianyu/S1-Arena"
REVISION = "ad05ed573294482ab4f76d501617e7d53b2f9e7a"  # dataset commit of 2026-10-16
HARNESS_REPO = "https://github.com/TianyuCodings/S1-Arena"
HARNESS_COMMIT = "28e16b16cdf68800a4cba148c3c6ea036f75bda3"  # 2026-10-15; leaderboard snapshot 2026-10-14

GROUPS: dict[str, tuple[str, ...]] = {
    "text": ("basic_cls", "know", "social"),
    "applied": ("pro", "ret", "struct", "code", "fore", "dialog", "batch"),
    "control": ("judge", "safety", "route"),
}
STATIC_DOMAINS: tuple[str, ...] = tuple(d for domains in GROUPS.values() for d in domains)
AGENT_DOMAINS: tuple[str, ...] = ("step", "embodied", "game")
DOMAINS: tuple[str, ...] = STATIC_DOMAINS + AGENT_DOMAINS

#: Tasks of the leaderboard whose sources forbid redistribution: absent from the public dataset.
RESTRICTED_TASKS: frozenset[str] = frozenset(
    {
        "basic_cls/multilabel/semeval2018_ec",
        "basic_cls/sentiment/semeval2018_voc",
        "basic_cls/spam_phishing/enron_spam",
        "basic_cls/stance_sarcasm_humor/semeval2016_stance",
        "basic_cls/stance_sarcasm_humor/humicroedit",
        "fore/multi_outcome/polymarket",
        "fore/numeric_bins/polymarket",
        "pro/answer_grading/ellipse",
        "ret/relevance_grading/trec_covid",
        "ret/relevance_grading/antique",
        "social/opinion_distribution/subpop_eval",
        "step/action_critic/webrewardbench",
        "step/failure_attribution/trail",
        "step/next_action/mind2web",
    }
)
#: Static domains the restricted tasks leave incomplete; the other eight compare one to one.
INCOMPLETE_DOMAINS: frozenset[str] = frozenset(
    t.split("/", 1)[0] for t in RESTRICTED_TASKS if t.split("/", 1)[0] in STATIC_DOMAINS
)


def row_to_item(row: Mapping[str, Any]) -> Item:
    """One dataset row (request, gold and meta are JSON strings) as a verified item.

    Raises ``ValueError`` if the content does not hash to the row's id.
    """
    item = Item(
        id=row["id"],
        leaf=row["leaf"],
        task=row["task"],
        request=DecisionRequest.model_validate_json(row["request"]),
        gold={name: Gold(**gold) for name, gold in json.loads(row["gold"]).items()},
        meta=json.loads(row["meta"]) if row.get("meta") else {},
    )
    expected = content_id(item)
    if expected != item.id:
        raise ValueError(f"{item.id}: the content hashes to {expected}; the item was changed")
    return item


def read_parquet(path: str | Path) -> Iterator[Item]:
    """Verified items from one parquet file of the dataset's layout."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            yield row_to_item(row)


def config_file(domain: str, revision: str = REVISION) -> str:
    """Local path of one domain's parquet file at ``revision``, downloaded on first use (Hub cache)."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; known: {', '.join(DOMAINS)}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        DATASET, f"{domain}/test-00000-of-00001.parquet", repo_type="dataset", revision=revision
    )


def load(
    domains: Iterable[str] = STATIC_DOMAINS,
    revision: str = REVISION,
    tasks: Iterable[str] | None = None,
) -> list[Item]:
    """Verified items of the given domains (default: the 13 static ones), in dataset order.

    ``tasks`` keeps only items whose task starts with one of the given prefixes.
    """
    prefixes = tuple(tasks) if tasks is not None else None
    items: list[Item] = []
    for domain in domains:
        for item in read_parquet(config_file(domain, revision)):
            if prefixes is None or item.task.startswith(prefixes):
                items.append(item)
    return items
