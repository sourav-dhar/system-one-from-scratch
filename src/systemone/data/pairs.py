"""In-memory (state, action) pairs and loading them from a local mirror of the Hub layout."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from systemone.data.catalog import ChunkFiles


@dataclass
class PairSet:
    """Row-aligned arrays: ``states[i]`` is answered by ``actions[i]``.

    Attributes:
        states: ``[N, D]`` frozen-LLM state vectors (float16 or float32).
        actions: ``[N, D]`` frozen-LLM action vectors.
        doc: ``[N]`` int64 source-document ids, used to mask same-document pairs.
        negatives: optional ``[N, K, D]`` mined hard negatives (actions) per state.
        neg_count: optional ``[N]`` number of valid negatives per row.
    """

    states: np.ndarray
    actions: np.ndarray
    doc: np.ndarray
    negatives: np.ndarray | None = None
    neg_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.states)
        if len(self.actions) != n or len(self.doc) != n:
            raise ValueError("states, actions and doc must have the same length")
        if (self.negatives is None) != (self.neg_count is None):
            raise ValueError("negatives and neg_count come together")
        if self.negatives is not None and (len(self.negatives) != n or len(self.neg_count) != n):  # type: ignore[arg-type]
            raise ValueError("negatives must be row-aligned with states")

    def __len__(self) -> int:
        return len(self.states)

    @property
    def dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def has_negatives(self) -> bool:
        return self.negatives is not None

    def subset(self, idx: np.ndarray | slice) -> PairSet:
        return PairSet(
            self.states[idx],
            self.actions[idx],
            self.doc[idx],
            None if self.negatives is None else self.negatives[idx],
            None if self.neg_count is None else self.neg_count[idx],
        )

    def without_negatives(self) -> PairSet:
        return replace(self, negatives=None, neg_count=None)

    @staticmethod
    def concat(parts: Sequence[PairSet]) -> PairSet:
        if not parts:
            raise ValueError("nothing to concatenate")
        negs = all(p.has_negatives for p in parts)
        return PairSet(
            np.concatenate([p.states for p in parts]),
            np.concatenate([p.actions for p in parts]),
            np.concatenate([p.doc for p in parts]),
            np.concatenate([p.negatives for p in parts]) if negs else None,
            np.concatenate([p.neg_count for p in parts]) if negs else None,
        )

    @property
    def nbytes(self) -> int:
        arrays = [self.states, self.actions, self.doc, self.negatives, self.neg_count]
        return sum(a.nbytes for a in arrays if a is not None)


def doc_ids_from_strings(ids: Sequence[str]) -> np.ndarray:
    """Stable 32-bit hashes of document-id strings (CRC32, as in the reference trainer)."""
    return np.fromiter((zlib.crc32(x.encode()) for x in ids), dtype=np.int64, count=len(ids))


def _read_doc_ids(root: Path, files: ChunkFiles, rows: int | None) -> np.ndarray:
    cache = root / "qa" / f"{files.chunk}_doc.npy"
    if cache.exists():
        doc = np.load(cache)
    else:
        import pyarrow.parquet as pq

        ids = pq.read_table(root / "qa" / files.meta, columns=["doc_id"]).column("doc_id").to_pylist()
        doc = doc_ids_from_strings(ids)
    return doc[:rows] if rows else doc


def _load(path: Path, rows: int | None, mmap: bool) -> np.ndarray:
    arr = np.load(path, mmap_mode="r" if (mmap or rows) else None)
    if rows:
        arr = arr[:rows]
    return arr if mmap else np.ascontiguousarray(arr)


def load_chunks(
    root: str | Path,
    chunks: Sequence[str],
    negatives: bool = False,
    rows_per_chunk: int | None = None,
    mmap: bool = False,
) -> PairSet:
    """Load chunks from a local mirror (``root/qa/...`` and ``root/neg/...``)."""
    root = Path(root)
    parts = []
    for c in chunks:
        f = ChunkFiles(c)
        q = _load(root / "qa" / f.questions, rows_per_chunk, mmap)
        a = _load(root / "qa" / f.answers, rows_per_chunk, mmap)
        doc = _read_doc_ids(root, f, rows_per_chunk)
        n = nc = None
        if negatives:
            n = _load(root / "neg" / f.negatives, rows_per_chunk, mmap)
            nc = _load(root / "neg" / f.negative_counts, rows_per_chunk, False).astype(np.int64)
            n = n.reshape(len(q), -1, q.shape[1])
        parts.append(PairSet(q, a, doc, n, nc))
    return parts[0] if len(parts) == 1 else PairSet.concat(parts)
