"""Where the public data lives, and which chunks are held out.

The pre-training vectors (62.45M question/answer pairs embedded with frozen Qwen3-8B, last-token
pooling, L2-normalized, float16) are split into 625 chunks of 100,000 rows across six shards.
Hard negatives (up to 10 per pair) exist for shard_00 only.

Held-out chunks follow the authors' notes, so our numbers are comparable to theirs:

* ``shard_05/chunk_0083-0084`` - general in-batch retrieval (200k pairs);
* ``shard_00/chunk_0028-0029`` - the hard-negative ("tricky Q&A") test (191,161 scorable pairs).
"""

from __future__ import annotations

from dataclasses import dataclass

QA_REPO = "Contrastive-LM/CLM-v0.1-Pretrain-Nemotron"
QA_REVISION = "05ca7d05c043350326c5e86c589b9542962747b4"
NEG_REPO = "jackyk02/nemotron_dqa_qwen3_8b_neg_embeddings"
NEG_REVISION = "main"
AUTHORS_HEADS_REPO = "jackyk02/qwen3_8b_head"
AUTHORS_HEADS = {
    "pre": "qwen3_8b_pretrained_head.pt",
    "mid": "qwen3_8b_midtrained_head.pt",
    "post": "qwen3_8b_posttrained_head.pt",
}

CHUNKS_PER_SHARD = {
    "shard_00": 60,
    "shard_01": 120,
    "shard_02": 120,
    "shard_03": 120,
    "shard_04": 120,
    "shard_05": 85,
}
ROWS_PER_CHUNK = 100_000
DIM = 4096
MAX_NEGATIVES = 10

HELD_OUT_GENERAL = ("shard_05/chunk_0083", "shard_05/chunk_0084")
HELD_OUT_HARD = ("shard_00/chunk_0028", "shard_00/chunk_0029")
HELD_OUT = frozenset(HELD_OUT_GENERAL + HELD_OUT_HARD)


@dataclass(frozen=True)
class ChunkFiles:
    """Relative paths of one chunk's files, valid both on the Hub and in a local mirror."""

    chunk: str

    @property
    def questions(self) -> str:
        return f"{self.chunk}_q.npy"

    @property
    def answers(self) -> str:
        return f"{self.chunk}_a.npy"

    @property
    def meta(self) -> str:
        return f"{self.chunk}_meta.parquet"

    @property
    def negatives(self) -> str:
        return f"{self.chunk}_n.npy"

    @property
    def negative_counts(self) -> str:
        return f"{self.chunk}_ncnt.npy"


def chunk_id(shard: int | str, index: int) -> str:
    shard = shard if isinstance(shard, str) else f"shard_{shard:02d}"
    return f"{shard}/chunk_{index:04d}"


def spread_chunks(
    n: int, shards: tuple[str, ...] = ("shard_01", "shard_02", "shard_03", "shard_04")
) -> list[str]:
    """``n`` training chunks spread round-robin over ``shards``, never touching held-out data.

    Spreading matters: each shard was built from a different slice of the source corpus, so a
    subset drawn from one shard alone is less diverse than the full dataset.
    """
    out: list[str] = []
    i = 0
    while len(out) < n:
        for s in shards:
            if len(out) == n:
                break
            if i < CHUNKS_PER_SHARD[s] and chunk_id(s, i) not in HELD_OUT:
                out.append(chunk_id(s, i))
        i += 1
        if i > max(CHUNKS_PER_SHARD.values()):
            raise ValueError(f"cannot pick {n} chunks from {shards}")
    return out


def hard_negative_chunks(n: int) -> list[str]:
    """First ``n`` shard_00 chunks that carry hard negatives, skipping the held-out ones."""
    out = [chunk_id(0, i) for i in range(CHUNKS_PER_SHARD["shard_00"]) if chunk_id(0, i) not in HELD_OUT]
    if n > len(out):
        raise ValueError(f"only {len(out)} hard-negative training chunks exist")
    return out[:n]


def resolve_chunks(specs: list[str]) -> list[str]:
    """Expand chunk specs used in configs.

    * ``spread:N`` - :func:`spread_chunks` (pre-training subsets);
    * ``hardneg:N`` - :func:`hard_negative_chunks` (mid-training subsets);
    * anything else is taken as a literal chunk id such as ``shard_01/chunk_0007``.

    Duplicates are removed (first occurrence wins) and held-out chunks are rejected.
    """
    out: list[str] = []
    for spec in specs:
        kind, _, n = spec.partition(":")
        if kind == "spread" and n:
            out += spread_chunks(int(n))
        elif kind == "hardneg" and n:
            out += hard_negative_chunks(int(n))
        else:
            out.append(spec)
    leaked = HELD_OUT.intersection(out)
    if leaked:
        raise ValueError(f"training chunks include held-out data: {sorted(leaked)}")
    return list(dict.fromkeys(out))
