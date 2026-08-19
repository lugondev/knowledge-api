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

import asyncio
import logging
import math

import numpy as np
from sqlalchemy import select

from kbase.db import Database
from kbase.embedding import Embedder
from kbase.models import Chunk, Document

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


async def search_collection(
    db: Database,
    collection_id: str,
    query: str,
    *,
    embed: Embedder,
    limit: int = 5,
    min_score: float = 0.35,
) -> tuple[list[dict], int]:
    if not query.strip():
        return [], 0
    vectors, tokens = await embed([query])
    if not vectors:
        return [], tokens
    qvec = np.array(vectors[0], dtype=np.float64)
    qnorm = float(np.linalg.norm(qvec))
    if qnorm == 0.0:
        return [], tokens

    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        # No `status` predicate: a chunk only exists for an indexed document
        # (see store.mark_pending / indexer's failure path). The join is here
        # for the title and filename in the payload, not to filter.
        .where(Chunk.collection_id == collection_id)
        .execution_options(yield_per=PARTITION_SIZE)
    )

    best: list[tuple[float, int, dict]] = []
    scanned = 0
    async with db.session() as s:
        result = await s.stream(stmt)
        async for partition in result.partitions(PARTITION_SIZE):
            batch = [
                (
                    list(chunk.embedding or []),
                    {
                        "text": chunk.text,
                        "document_id": doc.id,
                        "title": doc.title,
                        "filename": doc.filename,
                        "heading": chunk.heading,
                    },
                )
                for chunk, doc in partition
            ]
            hits = await asyncio.to_thread(_score_batch, qvec, qnorm, batch, min_score, limit)
            # The scan position breaks ties, so equal scores come back in the
            # order they were read and the same query answers the same way twice.
            best.extend((score, scanned + i, payload) for i, (score, payload) in enumerate(hits))
            scanned += len(batch)
            if len(best) > limit:
                best.sort(key=lambda h: (-h[0], h[1]))
                del best[limit:]

    if scanned >= SCAN_WARN_CHUNKS and collection_id not in _warned:
        _warned.add(collection_id)
        logger.warning(
            "collection %s holds %d+ chunks; every search scans all of them. "
            "This store is a linear scan -- move to a vector index.",
            collection_id,
            scanned,
        )

    best.sort(key=lambda h: (-h[0], h[1]))
    return [{**payload, "score": score} for score, _order, payload in best[:limit]], tokens
