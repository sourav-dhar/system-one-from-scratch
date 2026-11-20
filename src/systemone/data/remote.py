"""Fetch published vectors from the Hugging Face Hub.

Two modes:

* :func:`download_chunk` mirrors whole files (what the cloud runner does, ~1.6 GB per chunk);
* :class:`RemoteNpy` reads only the first ``rows`` of a remote ``.npy`` with HTTP range requests,
  so a laptop can experiment on, say, 20k real pairs per chunk without downloading 1.6 GB.
"""

from __future__ import annotations

import ast
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from systemone.data.catalog import NEG_REPO, NEG_REVISION, QA_REPO, QA_REVISION, ChunkFiles

Fetch = Callable[[int, int], bytes]
"""``fetch(start, end_inclusive) -> bytes`` for one remote file."""

HUB = "https://huggingface.co"


def hub_url(repo: str, path: str, revision: str) -> str:
    return f"{HUB}/datasets/{repo}/resolve/{revision}/{path}"


def http_range_fetcher(url: str, timeout: float = 120.0) -> Fetch:
    import requests

    session = requests.Session()
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def fetch(start: int, end: int) -> bytes:
        r = session.get(url, headers={**headers, "Range": f"bytes={start}-{end}"}, timeout=timeout)
        r.raise_for_status()
        if r.status_code != 206:
            raise OSError(f"server ignored the Range header for {url}")
        return r.content

    return fetch


def file_range_fetcher(path: str | Path) -> Fetch:
    """Local-file equivalent of :func:`http_range_fetcher` (used by tests)."""

    def fetch(start: int, end: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    return fetch


@dataclass(frozen=True)
class NpyHeader:
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int

    @property
    def row_bytes(self) -> int:
        return int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize


def parse_npy_header(head: bytes) -> NpyHeader:
    """Parse the header of a ``.npy`` file (format versions 1-3) from its first bytes."""
    if head[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    major = head[6]
    if major == 1:
        (hlen,) = struct.unpack("<H", head[8:10])
        start = 10
    else:
        (hlen,) = struct.unpack("<I", head[8:12])
        start = 12
    if len(head) < start + hlen:
        raise ValueError("header truncated; fetch more bytes")
    meta = ast.literal_eval(head[start : start + hlen].decode("latin1"))
    if meta["fortran_order"]:
        raise ValueError("Fortran-ordered arrays are not supported")
    return NpyHeader(np.dtype(meta["descr"]), tuple(meta["shape"]), start + hlen)


class RemoteNpy:
    """Row-slice access to a remote ``.npy`` without downloading the whole file."""

    def __init__(self, fetch: Fetch) -> None:
        self._fetch = fetch
        self.header = parse_npy_header(fetch(0, 4095))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.header.shape

    def rows(self, start: int, stop: int) -> np.ndarray:
        stop = min(stop, self.header.shape[0])
        if stop <= start:
            return np.empty((0, *self.header.shape[1:]), dtype=self.header.dtype)
        rb = self.header.row_bytes
        raw = self._fetch(self.header.offset + start * rb, self.header.offset + stop * rb - 1)
        return (
            np.frombuffer(raw, dtype=self.header.dtype).reshape(stop - start, *self.header.shape[1:]).copy()
        )


def mirror_subset(root: str | Path, chunk: str, rows: int, negatives: bool = False) -> None:
    """Write the first ``rows`` of a chunk into ``root`` (same layout as a full mirror).

    Writes ``_q.npy``, ``_a.npy`` and a ``_doc.npy`` cache of document-id hashes, plus
    ``_n.npy``/``_ncnt.npy`` when ``negatives`` is set.
    """
    import pyarrow.parquet as pq
    import requests

    from systemone.data.pairs import doc_ids_from_strings

    root = Path(root)
    f = ChunkFiles(chunk)
    (root / "qa" / chunk).parent.mkdir(parents=True, exist_ok=True)
    for name in (f.questions, f.answers):
        arr = RemoteNpy(http_range_fetcher(hub_url(QA_REPO, name, QA_REVISION))).rows(0, rows)
        np.save(root / "qa" / name, arr)
    meta = root / "qa" / f.meta
    with requests.get(hub_url(QA_REPO, f.meta, QA_REVISION), timeout=120) as r:
        r.raise_for_status()
        meta.write_bytes(r.content)
    ids = pq.read_table(meta, columns=["doc_id"]).column("doc_id").to_pylist()[:rows]
    np.save(root / "qa" / f"{chunk}_doc.npy", doc_ids_from_strings(ids))
    if negatives:
        (root / "neg" / chunk).parent.mkdir(parents=True, exist_ok=True)
        n = RemoteNpy(http_range_fetcher(hub_url(NEG_REPO, f.negatives, NEG_REVISION))).rows(0, rows)
        np.save(root / "neg" / f.negatives, n)
        c = RemoteNpy(http_range_fetcher(hub_url(NEG_REPO, f.negative_counts, NEG_REVISION))).rows(0, rows)
        np.save(root / "neg" / f.negative_counts, c)


def download_chunk(root: str | Path, chunk: str, negatives: bool = False) -> int:
    """Mirror a chunk's full files from the Hub into ``root``. Skips files already complete.

    Returns the number of bytes newly written.
    """
    from huggingface_hub import hf_hub_download

    root = Path(root)
    f = ChunkFiles(chunk)
    jobs = [(QA_REPO, QA_REVISION, "qa", name) for name in (f.questions, f.answers, f.meta)]
    if negatives:
        jobs += [(NEG_REPO, NEG_REVISION, "neg", name) for name in (f.negatives, f.negative_counts)]
    written = 0
    for repo, rev, kind, name in jobs:
        dst = root / kind / name
        if dst.exists():
            continue
        # local_dir downloads straight into the mirror (resumable, no second copy in the HF cache)
        hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=root / kind)
        written += dst.stat().st_size
    return written
