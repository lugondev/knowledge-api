"""One engine, one sessionmaker, and a context manager that always closes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kbase.models import Base


class Database:
    def __init__(self, url: str) -> None:
        self._engine = create_async_engine(url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def engine(self):
        """For DDL that SQLAlchemy's metadata cannot express -- the vector
        extension, the typed column, the HNSW index."""
        return self._engine

    async def create_all(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(self._ensure_chunk_collection_id)

    @staticmethod
    def _ensure_chunk_collection_id(conn) -> None:
        """`create_all` creates missing tables, never missing columns.

        A deployment that indexed anything before `chunks.collection_id`
        existed keeps a table the ORM no longer matches, and every insert
        fails against a column that is not there.
        """
        inspector = sa_inspect(conn)
        if "chunks" not in inspector.get_table_names():
            return
        if "collection_id" in {c["name"] for c in inspector.get_columns("chunks")}:
            return
        conn.execute(text("ALTER TABLE chunks ADD COLUMN collection_id VARCHAR(36) DEFAULT ''"))
        conn.execute(
            text(
                "UPDATE chunks SET collection_id = (SELECT collection_id FROM documents "
                "WHERE documents.id = chunks.document_id) "
                "WHERE collection_id IS NULL OR collection_id = ''"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_chunks_collection_id ON chunks (collection_id)")
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as s:
            yield s

    async def dispose(self) -> None:
        await self._engine.dispose()
