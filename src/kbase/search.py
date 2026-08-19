"""Embed the query, score every chunk in the collection, apply the floor.

On SQLite this is a scan, which is what the corpus this service is built for
actually needs. The interface is the seam a pgvector implementation slots into
later without any caller noticing.

Two things keep that scan from taking the process down with it as a collection
grows.

Rows arrive a partition at a time rather than all at once, so the memory a search
holds is set by the partition and not by the collection. Read in one list, 3,000
chunks of 1,536 dimensions cost about 180 MB of Python floats to return five
hits, and 6,000 cost twice that; a partition at a time it is 16 MB for any of
them.

And the arithmetic runs in a worker thread, because it is the one genuinely
CPU-bound stretch in an otherwise IO-bound process. Scored inline, a collection
of 8,000 chunks stalled every other request in the process -- `/healthz`
included, which is what the container's healthcheck polls -- for 0.65 s at a
time. Off the loop it is 0.08 s.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

#: Rows pulled, and scored, per round trip -- and so the thing that decides what
#: a search costs in memory: about 125 KB per row held, whatever the collection
#: holds in total. Measured over 10,000 chunks, 128 and 512 answer in the same
#: time to within 6%, so the larger buffer buys nothing and costs four times the
#: memory on a path that runs on every conversational turn.
PARTITION_SIZE = 128

#: Past this many chunks in one collection, a scan is the wrong shape and the
#: operator should hear about it once. The fix is a vector index, not a bigger
#: number here.
SCAN_WARN_CHUNKS = 50_000

_warned: set[str] = set()


def cosine(a: list[float], b: list[float]) -> float:
    """0.0 for anything that cannot be compared, including a dimension mismatch.

    Vectors of different length mean the embedding model changed under a corpus
    that was indexed with the old one. Letting `zip` truncate would score those
    chunks anyway, and the score would look entirely reasonable -- so a stale
    chunk would keep winning top-k against queries it has nothing to do with.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _score_batch(
    qvec: np.ndarray,
    qnorm: float,
    batch: list[tuple[list[float], dict]],
    min_score: float,
    limit: int,
) -> list[tuple[float, dict]]:
    """Cosine against every row of one partition, floor applied, top-`limit` kept.

    Rows whose embedding has a different width are dropped rather than scored
    zero. `cosine` returns 0.0 for them, which reads as "no similarity" -- but
    `min_score` is allowed to be 0.0, and at that floor a chunk left over from a
    replaced embedding model would be admitted as a hit.
    """
    usable = [(vec, payload) for vec, payload in batch if len(vec) == qvec.shape[0]]
    if not usable:
        return []
    matrix = np.array([vec for vec, _ in usable], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    scores = np.zeros(len(usable), dtype=np.float64)
    live = norms > 0.0
    scores[live] = (matrix[live] @ qvec) / (norms[live] * qnorm)

    keep = np.flatnonzero(scores >= min_score)
    if keep.size > limit:
        keep = keep[np.argpartition(-scores[keep], limit)[:limit]]
    return [(float(scores[i]), usable[i][1]) for i in keep]


def warn_if_large(collection_id: str, scanned: int) -> None:
    """Say it once per collection, then stop. The fix is an index, not a
    bigger machine."""
    if scanned >= SCAN_WARN_CHUNKS and collection_id not in _warned:
        _warned.add(collection_id)
        logger.warning(
            "collection %s holds %d+ chunks; every search scans all of them. "
            "This store is a linear scan -- move to a vector index.",
            collection_id,
            scanned,
        )
