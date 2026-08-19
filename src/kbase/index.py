"""Everything that touches an embedding, behind one interface.

The write methods take the caller's session rather than opening their own: a
collection delete removes chunks, documents and the collection in a single
transaction today, and an index that opened its own connection would break that
into three. `query` owns its session, because a read joins nothing else.

Both are honest consequences of this seam being scoped to SQL backends. An
external store (Qdrant, Vectorize) shares no transaction with the metadata
tables and would need a different shape -- along with the reconcile pass that a
shared transaction makes unnecessary here.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kbase.db import Database
from kbase.models import Chunk, Document
from kbase.search import PARTITION_SIZE, _score_batch, warn_if_large


class ChunkIndex(Protocol):
    async def create_schema(self) -> None: ...

    async def replace(
        self, s: AsyncSession, document_id: str, collection_id: str, rows: list[dict]
    ) -> None: ...

    async def drop_document(self, s: AsyncSession, document_id: str) -> None: ...

    async def drop_where_document_in(self, s: AsyncSession, doc_id_select) -> None: ...

    async def chunks(self, document_id: str) -> list[dict]: ...

    async def query(
        self, collection_id: str, qvec: list[float], *, limit: int, min_score: float
    ) -> list[dict]: ...


class SqlScanIndex:
    """Cosine against every chunk in the collection, a partition at a time."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create_schema(self) -> None:
        """Nothing beyond the tables `Database.create_all` already makes."""
        return None

    async def replace(
        self, s: AsyncSession, document_id: str, collection_id: str, rows: list[dict]
    ) -> None:
        await s.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))
        for r in rows:
            s.add(
                Chunk(
                    document_id=document_id,
                    collection_id=collection_id,
                    ordinal=r["ordinal"],
                    text=r["text"],
                    heading=r["heading"],
                    char_count=len(r["text"]),
                    embedding=r["embedding"],
                )
            )

    async def drop_document(self, s: AsyncSession, document_id: str) -> None:
        await s.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))

    async def drop_where_document_in(self, s: AsyncSession, doc_id_select) -> None:
        await s.execute(sa_delete(Chunk).where(Chunk.document_id.in_(doc_id_select)))

    async def chunks(self, document_id: str) -> list[dict]:
        async with self._db.session() as s:
            rows = (
                (
                    await s.execute(
                        select(Chunk)
                        .where(Chunk.document_id == document_id)
                        .order_by(Chunk.ordinal)
                    )
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": c.id,
                    "ordinal": c.ordinal,
                    "text": c.text,
                    "heading": c.heading,
                    "embedding": list(c.embedding or []),
                }
                for c in rows
            ]

    async def query(
        self, collection_id: str, qvec: list[float], *, limit: int, min_score: float
    ) -> list[dict]:
        vec = np.array(qvec, dtype=np.float64)
        qnorm = float(np.linalg.norm(vec))
        if qnorm == 0.0:
            return []

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
        async with self._db.session() as s:
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
                hits = await asyncio.to_thread(_score_batch, vec, qnorm, batch, min_score, limit)
                # The scan position breaks ties, so equal scores come back in the
                # order they were read and the same query answers the same way twice.
                best.extend(
                    (score, scanned + i, payload) for i, (score, payload) in enumerate(hits)
                )
                scanned += len(batch)
                if len(best) > limit:
                    best.sort(key=lambda h: (-h[0], h[1]))
                    del best[limit:]

        warn_if_large(collection_id, scanned)
        best.sort(key=lambda h: (-h[0], h[1]))
        return [{**payload, "score": score} for score, _order, payload in best[:limit]]
