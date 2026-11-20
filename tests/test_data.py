import io

import numpy as np
import pytest
import torch

from systemone.config import SyntheticConfig
from systemone.data import BatchSampler, PairSet, load_chunks, make_synthetic
from systemone.data.catalog import HELD_OUT, hard_negative_chunks, spread_chunks
from systemone.data.pairs import doc_ids_from_strings
from systemone.data.remote import RemoteNpy, file_range_fetcher, parse_npy_header


def tiny(n: int = 10, d: int = 4, negs: bool = False) -> PairSet:
    rng = np.random.default_rng(0)
    return PairSet(
        rng.standard_normal((n, d)).astype(np.float16),
        rng.standard_normal((n, d)).astype(np.float16),
        np.arange(n, dtype=np.int64),
        rng.standard_normal((n, 3, d)).astype(np.float16) if negs else None,
        np.full(n, 3, dtype=np.int64) if negs else None,
    )


def test_pairset_validation() -> None:
    p = tiny()
    with pytest.raises(ValueError, match="same length"):
        PairSet(p.states, p.actions[:5], p.doc)
    with pytest.raises(ValueError, match="together"):
        PairSet(p.states, p.actions, p.doc, negatives=np.zeros((10, 3, 4)))


def test_subset_concat_and_negatives() -> None:
    p = tiny(negs=True)
    both = PairSet.concat([p.subset(slice(0, 4)), p.subset(slice(4, None))])
    assert np.array_equal(both.states, p.states)
    assert both.has_negatives
    assert not PairSet.concat([p, p.without_negatives()]).has_negatives
    assert p.nbytes > p.without_negatives().nbytes


def write_mirror(root, chunk: str, n: int = 6, d: int = 4, negs: bool = True) -> PairSet:
    rng = np.random.default_rng(1)
    q = rng.standard_normal((n, d)).astype(np.float16)
    a = rng.standard_normal((n, d)).astype(np.float16)
    (root / "qa" / chunk).parent.mkdir(parents=True, exist_ok=True)
    np.save(root / "qa" / f"{chunk}_q.npy", q)
    np.save(root / "qa" / f"{chunk}_a.npy", a)
    doc = doc_ids_from_strings([f"doc{i // 2}" for i in range(n)])
    np.save(root / "qa" / f"{chunk}_doc.npy", doc)
    n_arr = c_arr = None
    if negs:
        (root / "neg" / chunk).parent.mkdir(parents=True, exist_ok=True)
        n_arr = rng.standard_normal((n, 10 * d)).astype(np.float16)  # stored flat, as on the Hub
        c_arr = np.arange(n) % 11
        np.save(root / "neg" / f"{chunk}_n.npy", n_arr)
        np.save(root / "neg" / f"{chunk}_ncnt.npy", c_arr.astype(np.int32))
    return PairSet(q, a, doc, None if n_arr is None else n_arr.reshape(n, 10, d), c_arr)


def test_load_chunks_from_mirror(tmp_path) -> None:
    a = write_mirror(tmp_path, "shard_00/chunk_0000")
    b = write_mirror(tmp_path, "shard_00/chunk_0001")
    got = load_chunks(tmp_path, ["shard_00/chunk_0000", "shard_00/chunk_0001"], negatives=True)
    assert len(got) == 12
    assert np.array_equal(got.states[:6], a.states)
    assert got.negatives is not None
    assert got.negatives.shape == (12, 10, 4)
    assert np.array_equal(got.negatives[6:], b.negatives)  # type: ignore[arg-type]
    assert got.doc[0] == got.doc[1] != got.doc[2]
    head = load_chunks(tmp_path, ["shard_00/chunk_0000"], rows_per_chunk=3)
    assert len(head) == 3


def test_load_chunks_reads_doc_ids_from_parquet(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    write_mirror(tmp_path, "shard_01/chunk_0000", negs=False)
    (tmp_path / "qa" / "shard_01" / "chunk_0000_doc.npy").unlink()
    pq.write_table(
        pa.table({"doc_id": [f"d{i // 3}" for i in range(6)]}),
        tmp_path / "qa/shard_01/chunk_0000_meta.parquet",
    )
    got = load_chunks(tmp_path, ["shard_01/chunk_0000"])
    assert np.array_equal(got.doc, doc_ids_from_strings([f"d{i // 3}" for i in range(6)]))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32])
def test_npy_header_and_range_reads(tmp_path, dtype) -> None:
    arr = (np.arange(50 * 6).reshape(50, 6)).astype(dtype)
    path = tmp_path / "x.npy"
    np.save(path, arr)
    buf = io.BytesIO()
    np.save(buf, arr)
    h = parse_npy_header(buf.getvalue()[:4096])
    assert h.shape == (50, 6)
    assert h.dtype == np.dtype(dtype)
    r = RemoteNpy(file_range_fetcher(path))
    assert np.array_equal(r.rows(0, 7), arr[:7])
    assert np.array_equal(r.rows(45, 999), arr[45:])
    assert r.rows(10, 10).shape == (0, 6)


def test_npy_header_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="npy"):
        parse_npy_header(b"not a numpy file")


def test_spread_chunks_avoid_held_out_and_are_distinct() -> None:
    chunks = spread_chunks(40)
    assert len(set(chunks)) == 40
    assert not set(chunks) & HELD_OUT
    assert {c.split("/")[0] for c in chunks} == {"shard_01", "shard_02", "shard_03", "shard_04"}
    hn = hard_negative_chunks(30)
    assert "shard_00/chunk_0028" not in hn
    assert hn[-1] == "shard_00/chunk_0031"
    with pytest.raises(ValueError, match="only"):
        hard_negative_chunks(100)


def test_synthetic_is_structured_and_deterministic() -> None:
    cfg = SyntheticConfig(n_train=500, n_eval=100, dim=32, latent_dim=8)
    tr, ev = make_synthetic(cfg)
    tr2, _ = make_synthetic(cfg)
    assert np.array_equal(tr.states, tr2.states)
    assert (len(tr), len(ev), tr.dim) == (500, 100, 32)
    assert tr.negatives is not None
    assert tr.negatives.shape == (500, 10, 32)
    assert np.allclose(np.linalg.norm(tr.states.astype(np.float32), axis=1), 1.0, atol=1e-2)
    assert len(np.unique(tr.doc)) < len(tr)  # some pairs share a document
    assert not set(tr.doc) & set(ev.doc)


@pytest.mark.parametrize("prefetch", [0, 2])
def test_batch_sampler_covers_each_row_once_per_epoch(prefetch: int) -> None:
    p = tiny(n=20, negs=True)
    s = BatchSampler(p, batch_size=6, seed=3, prefetch=prefetch)
    assert s.steps_per_epoch == 3
    it = iter(s)
    rows = [next(it) for _ in range(3)]
    docs = torch.cat([b["doc"] for b in rows])
    assert len(set(docs.tolist())) == 18  # 3 batches of 6 distinct rows, ragged tail dropped
    assert set(rows[0]) == {"states", "actions", "doc", "negatives", "neg_count"}
    assert rows[0]["negatives"].shape == (6, 3, 4)


def test_batch_sampler_rejects_oversized_batch() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        BatchSampler(tiny(n=4), batch_size=8)


def test_resolve_chunk_specs() -> None:
    from systemone.data.catalog import resolve_chunks

    got = resolve_chunks(["spread:3", "hardneg:2", "shard_05/chunk_0001", "shard_00/chunk_0000"])
    assert got[:3] == ["shard_01/chunk_0000", "shard_02/chunk_0000", "shard_03/chunk_0000"]
    assert got[3:] == ["shard_00/chunk_0000", "shard_00/chunk_0001", "shard_05/chunk_0001"]
    with pytest.raises(ValueError, match="held-out"):
        resolve_chunks(["shard_05/chunk_0084"])


def test_limit_negatives() -> None:
    p = tiny(negs=True)
    assert p.neg_count is not None
    p.neg_count[:] = [0, 1, 2, 3, 3, 3, 1, 2, 3, 0]
    k1 = p.limit_negatives(1)
    assert k1.negatives is not None
    assert k1.neg_count is not None
    assert k1.negatives.shape[1] == 1
    assert k1.neg_count.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert not p.limit_negatives(0).has_negatives
    with pytest.raises(ValueError, match="no negatives"):
        tiny().limit_negatives(2)
