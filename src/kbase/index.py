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
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kbase.db import Database
from kbase.errors import SchemaError
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
        """Nothing to build -- but a corpus this backend cannot read is fatal.

        A Postgres database whose `chunks.embedding` `PgVectorIndex` already
        converted to `vector(n)` reads, through the ORM's `JSON` declaration,
        as the string '[1,0,0]'. `list()` of that is a list of characters,
        `_score_batch` drops it as a width mismatch, and every search answers
        200 with nothing over a corpus that is entirely intact -- `/healthz`
        green, `kb doctor` clean. Unsetting `KB_EMBED_DIM` must therefore be as
        fatal as setting it wrong (`PgVectorIndex.create_schema` already
        refuses that direction), which is what this closes.

        The check lives here rather than in `choose_index` because it needs a
        live query and `choose_index` is synchronous and connectionless -- it
        is called against URLs with no server behind them. This is the one
        boot-time hook every deployment awaits before serving
        (`server/app.py`), so nothing routes around it.
        """
        # SQLite has no `vector` type to find and no `pg_attribute` to ask, and
        # a SQLite deployment must boot without the `postgres` extra installed.
        if self._db.engine.dialect.name != "postgresql":
            return None
        async with self._db.engine.connect() as conn:
            kind = (
                await conn.execute(
                    text(
                        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                        "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding' "
                        "AND NOT attisdropped"
                    )
                )
            ).scalar_one_or_none()
        if kind and kind.startswith("vector"):
            width = kind.partition("(")[2].rstrip(")")
            raise SchemaError(
                f"chunks.embedding is {kind}, but KB_EMBED_DIM is unset, so this process "
                "selected the scanning backend -- which cannot read that column and would "
                "answer every search with nothing. Set KB_EMBED_DIM"
                + (f"={width}" if width else " to the column's width")
                + ", or drop the column and reindex."
            )
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


def choose_index(db: Database, settings) -> ChunkIndex:
    """Postgres plus a dimension means pgvector; anything else scans.

    Making the dimension the switch is what keeps this upgrade non-breaking: a
    Postgres deployment that does not set it keeps the behaviour it has, and
    `kb doctor` is what tells the operator they are still scanning.
    """
    if settings.database_url.startswith("postgresql") and settings.embed_dim > 0:
        from kbase.pgindex import PgVectorIndex

        return PgVectorIndex(db, dim=settings.embed_dim)
    return SqlScanIndex(db)
