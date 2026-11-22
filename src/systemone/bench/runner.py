"""Run a decision model over items, with a resumable cache of responses.

Responses are appended to a JSONL file, one row per item: ``{"item_id", "sample", "response"}``. This is
S1-Arena's run-row layout, so its tools and ours read each other's runs. Re-running skips every item
with a final row, so a job that dies halfway resumes where it stopped and pays only for what is missing.

A row is *final* when it has no error, or when its error starts with ``rejected:``. A rejection means
the model refuses the input, for example because it is longer than the context; re-asking would give
the same answer. Any other error (transport, out of memory) is retried on the next run.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, Protocol

from systemone.decisions.schema import DecisionRequest, Item, ModelResponse

REJECTED = "rejected: "


class DecisionModel(Protocol):
    @property
    def model_id(self) -> str: ...

    def decide_many(self, requests: Sequence[DecisionRequest]) -> list[ModelResponse]: ...


def slug(model_id: str) -> str:
    """A file-system-safe name for a model id (S1-Arena's convention)."""
    return model_id.replace("/", "__").replace(":", "_")


def is_final(response: ModelResponse) -> bool:
    return response.error is None or response.error.startswith(REJECTED)


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_rows(path: str | Path) -> Iterator[tuple[str, ModelResponse]]:
    """(item id, response) for every sample-0 row of a run file (.jsonl or .jsonl.gz)."""
    path = Path(path)
    if not path.exists():
        return
    with _open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("sample", 0) == 0:
                    yield row["item_id"], ModelResponse.model_validate(row["response"])


def read_responses(path: str | Path) -> dict[str, ModelResponse]:
    """The latest response per item, preferring a final one over a later retryable failure."""
    out: dict[str, ModelResponse] = {}
    for item_id, response in iter_rows(path):
        if is_final(response) or item_id not in out or not is_final(out[item_id]):
            out[item_id] = response
    return out


def write_row(f: IO[str], item_id: str, response: ModelResponse) -> None:
    row = {"item_id": item_id, "sample": 0, "response": response.model_dump(exclude_none=True)}
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    model: DecisionModel,
    items: Sequence[Item],
    path: str | Path,
    chunk: int = 2000,
    log: Callable[[str], None] = print,
    on_chunk: Callable[[], None] | None = None,
) -> int:
    """Answer every item without a final row in ``path``; returns how many were asked.

    ``on_chunk`` runs after each chunk is written (for example, to commit a cloud volume).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {i for i, r in read_responses(path).items() if is_final(r)}
    todo = [item for item in items if item.id not in done]
    log(f"{model.model_id}: {len(todo):,} of {len(items):,} items to answer ({len(done):,} cached)")
    started = time.time()
    for start in range(0, len(todo), chunk):
        part = todo[start : start + chunk]
        responses = model.decide_many([item.request for item in part])
        with path.open("a", encoding="utf-8") as f:
            for item, response in zip(part, responses, strict=True):
                write_row(f, item.id, response)
        if on_chunk is not None:
            on_chunk()
        n = start + len(part)
        rate = n / max(time.time() - started, 1e-9)
        log(f"  {n:,}/{len(todo):,} answered, {rate:.1f}/s, ~{(len(todo) - n) / rate / 60:.0f} min left")
    return len(todo)
