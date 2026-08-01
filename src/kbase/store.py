"""Every read and write, each one scoped by tenant at the query, never after it."""

from __future__ import annotations

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select

from kbase.db import Database
from kbase.models import Chunk, Collection, Document


class CollectionStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, tenant: str, name: str) -> dict:
        """Idempotent: creating an existing collection returns it untouched."""
        async with self._db.session() as s:
            row = (
                await s.execute(
                    select(Collection).where(
                        Collection.tenant == tenant, Collection.name == name
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = Collection(tenant=tenant, name=name)
                s.add(row)
                await s.commit()
            return {"name": row.name, "document_count": await self._count(s, row.id)}

    async def list(self, tenant: str) -> list[dict]:
        async with self._db.session() as s:
            rows = (
                await s.execute(
                    select(Collection)
                    .where(Collection.tenant == tenant)
                    .order_by(Collection.name)
                )
            ).scalars().all()
            return [
                {"name": r.name, "document_count": await self._count(s, r.id)} for r in rows
            ]

    async def get(self, tenant: str, name: str) -> dict | None:
        async with self._db.session() as s:
            row = await self._row(s, tenant, name)
            if row is None:
                return None
            return {"name": row.name, "document_count": await self._count(s, row.id)}

    async def resolve_id(self, tenant: str, name: str) -> str | None:
        """The internal id, for callers that need to hang documents off it."""
        async with self._db.session() as s:
            row = await self._row(s, tenant, name)
            return row.id if row else None

    async def delete(self, tenant: str, name: str) -> bool:
        async with self._db.session() as s:
            row = await self._row(s, tenant, name)
            if row is None:
                return False
            doc_ids = (
                await s.execute(
                    select(Document.id).where(Document.collection_id == row.id)
                )
            ).scalars().all()
            if doc_ids:
                await s.execute(sa_delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
                await s.execute(sa_delete(Document).where(Document.id.in_(doc_ids)))
            await s.delete(row)
            await s.commit()
            return True

    @staticmethod
    async def _row(s, tenant: str, name: str) -> Collection | None:
        return (
            await s.execute(
                select(Collection).where(Collection.tenant == tenant, Collection.name == name)
            )
        ).scalar_one_or_none()

    @staticmethod
    async def _count(s, collection_id: str) -> int:
        return int(
            (
                await s.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.collection_id == collection_id)
                )
            ).scalar_one()
        )
