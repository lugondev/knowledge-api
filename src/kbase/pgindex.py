"""The pgvector backend.

The ORM never loads a `Chunk` here. `models.Chunk.embedding` is declared `JSON`,
which is what SQLite stores and what `create_all` builds before this class
converts it; a `select(Chunk)` against the converted column would hand
SQLAlchemy a vector literal to JSON-decode. Every statement below is explicit
SQL for that reason, and that constraint is load-bearing -- adding an ORM read
of `Chunk` to this file will fail at runtime, not at import.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete as sa_delete
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kbase.db import Database
from kbase.errors import SchemaError
from kbase.models import Chunk
from kbase.settings import HNSW_MAX_DIMENSIONS

logger = logging.getLogger(__name__)


def _literal(vec: list[float]) -> str:
    """pgvector's text input form, bound as a parameter and cast in SQL."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgVectorIndex:
    def __init__(self, db: Database, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError("PgVectorIndex needs a positive dimension")
        self._db = db
        # Interpolated into DDL below, so it must be an int and nothing else.
        self._dim = int(dim)

    async def create_schema(self) -> None:
        """Convert the JSON column to `vector(n)`, once.

        The guard is not decoration. Without it a second call adds a fresh
        empty `embedding_vec`, drops the converted `embedding` that holds every
        vector in the corpus, and renames the empty column over it -- silently,
        and on an ordinary restart.
        """
        async with self._db.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            existing = (
                await conn.execute(
                    text(
                        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding' "
                        "AND NOT attisdropped"
                    )
                )
            ).scalar_one_or_none()

            if existing and existing.startswith("vector"):
                if existing != f"vector({self._dim})":
                    # Migrating on a guess would destroy a corpus to satisfy a
                    # typo. The operator restores the setting, or reindexes
                    # deliberately.
                    raise SchemaError(
                        f"chunks.embedding is {existing} but KB_EMBED_DIM is {self._dim}; "
                        "refusing to migrate. Restore the old value, or drop the column "
                        "and reindex."
                    )
                await self._build_index(conn)
                return

            await conn.execute(
                text(
                    f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_vec vector({self._dim})"
                )
            )
            # Task 7 adds the backfill and the mismatch handling here. Until
            # then this path is correct only for a database with no chunks in
            # it, which is what this task's tests cover.
            await conn.execute(text("ALTER TABLE chunks DROP COLUMN embedding"))
            await conn.execute(text("ALTER TABLE chunks RENAME COLUMN embedding_vec TO embedding"))
            await self._build_index(conn)

    async def _build_index(self, conn) -> None:
        if self._dim > HNSW_MAX_DIMENSIONS:
            logger.warning(
                "KB_EMBED_DIM is %d; pgvector builds no HNSW index above %d dimensions, "
                "so every search scans. Reduce the model's output width.",
                self._dim,
                HNSW_MAX_DIMENSIONS,
            )
            return
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )

    async def replace(
        self, s: AsyncSession, document_id: str, collection_id: str, rows: list[dict]
    ) -> None:
        await s.execute(
            text("DELETE FROM chunks WHERE document_id = :doc"), {"doc": document_id}
        )
        for r in rows:
            await s.execute(
                text(
                    "INSERT INTO chunks (id, document_id, collection_id, ordinal, text, "
                    "heading, char_count, embedding) VALUES (gen_random_uuid()::text, :doc, "
                    ":col, :ordinal, :text, :heading, :chars, CAST(:emb AS vector))"
                ),
                {
                    "doc": document_id,
                    "col": collection_id,
                    "ordinal": r["ordinal"],
                    "text": r["text"],
                    "heading": r["heading"],
                    "chars": len(r["text"]),
                    "emb": _literal(r["embedding"]),
                },
            )

    async def drop_document(self, s: AsyncSession, document_id: str) -> None:
        await s.execute(text("DELETE FROM chunks WHERE document_id = :doc"), {"doc": document_id})

    async def drop_where_document_in(self, s: AsyncSession, doc_id_select) -> None:
        # A DELETE never selects `embedding`, so the ORM's stale JSON
        # declaration is not consulted and reusing the caller's `select()` here
        # is safe -- unlike a `select(Chunk)`, which is why this file reads
        # every other statement as raw SQL.
        await s.execute(sa_delete(Chunk).where(Chunk.document_id.in_(doc_id_select)))

    async def chunks(self, document_id: str) -> list[dict]:
        async with self._db.session() as s:
            rows = (
                await s.execute(
                    text(
                        "SELECT id, ordinal, text, heading, embedding::text AS emb FROM chunks "
                        "WHERE document_id = :doc ORDER BY ordinal"
                    ),
                    {"doc": document_id},
                )
            ).mappings().all()
        return [
            {
                "id": r["id"],
                "ordinal": r["ordinal"],
                "text": r["text"],
                "heading": r["heading"],
                "embedding": [float(x) for x in r["emb"].strip("[]").split(",") if x],
            }
            for r in rows
        ]

    async def query(
        self, collection_id: str, qvec: list[float], *, limit: int, min_score: float
    ) -> list[dict]:
        raise NotImplementedError  # Task 6
