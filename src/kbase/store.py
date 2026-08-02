"""Every read and write, each one scoped by tenant at the query, never after it."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError

from kbase.db import Database
from kbase.models import Chunk, Collection, Document, utcnow


class CollectionStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, tenant: str, name: str) -> dict:
        """Idempotent: creating an existing collection returns it untouched."""
        async with self._db.session() as s:
            row = (
                await s.execute(
                    select(Collection).where(Collection.tenant == tenant, Collection.name == name)
                )
            ).scalar_one_or_none()
            if row is None:
                row = Collection(tenant=tenant, name=name)
                s.add(row)
                try:
                    await s.commit()
                except IntegrityError:
                    # Two creates of the same name raced past the lookup above.
                    # The endpoint is documented as idempotent, so the loser
                    # returns the winner's row instead of a 500.
                    await s.rollback()
                    row = await self._row(s, tenant, name)
                    if row is None:  # pragma: no cover - only if the constraint changed
                        raise
            return {"name": row.name, "document_count": await self._count(s, row.id)}

    async def list(self, tenant: str) -> list[dict]:
        async with self._db.session() as s:
            rows = (
                (
                    await s.execute(
                        select(Collection)
                        .where(Collection.tenant == tenant)
                        .order_by(Collection.name)
                    )
                )
                .scalars()
                .all()
            )
            return [{"name": r.name, "document_count": await self._count(s, r.id)} for r in rows]

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
                (await s.execute(select(Document.id).where(Document.collection_id == row.id)))
                .scalars()
                .all()
            )
            if doc_ids:
                await s.execute(sa_delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
                await s.execute(sa_delete(Document).where(Document.id.in_(doc_ids)))
            await s.delete(row)
            await s.commit()
            return True

    async def owns(self, tenant: str, collection_id: str) -> bool:
        """Whether this tenant owns that collection id -- one query.

        The alternative, listing the tenant's collections and resolving each one,
        costs a query per collection on a path that runs on every document read.
        """
        async with self._db.session() as s:
            found = (
                await s.execute(
                    select(Collection.id).where(
                        Collection.id == collection_id, Collection.tenant == tenant
                    )
                )
            ).scalar_one_or_none()
            return found is not None

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


def _iso_utc(dt: datetime | None) -> str | None:
    """Always an offset-bearing UTC string, whatever the driver handed back.

    SQLite stores no timezone, so a row written as aware comes back naive. Left
    alone, the same field is offset-bearing in the response that created it and
    naive in every later read of it, and a client that parses the naive one as
    local time is wrong by its own UTC offset.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _doc_dict(d: Document) -> dict:
    return {
        "id": d.id,
        "title": d.title,
        "filename": d.filename,
        "mime": d.mime,
        "sha256": d.sha256,
        "bytes_len": d.bytes_len,
        "status": d.status,
        "error": d.error,
        "chunk_count": d.chunk_count,
        "created_at": _iso_utc(d.created_at),
        "indexed_at": _iso_utc(d.indexed_at),
    }


class DocumentStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self,
        collection_id: str,
        *,
        title: str,
        filename: str,
        mime: str,
        sha256: str,
        data: bytes,
    ) -> tuple[dict, bool]:
        """Returns (document, created). `created is False` means these exact bytes
        were already in this collection -- a retried upload, not a second copy."""
        async with self._db.session() as s:
            existing = (
                await s.execute(
                    select(Document).where(
                        Document.collection_id == collection_id,
                        Document.sha256 == sha256,
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                return _doc_dict(existing), False
            row = Document(
                collection_id=collection_id,
                title=title,
                filename=filename,
                mime=mime,
                sha256=sha256,
                bytes_len=len(data),
                data=data,
                status="pending",
            )
            s.add(row)
            try:
                await s.commit()
            except IntegrityError:
                # Two uploads of the same bytes in flight at once: both looked,
                # both found nothing, both inserted. The unique constraint is what
                # actually decides, and the loser reads the winner's row rather
                # than turning an idempotent call into a 500.
                await s.rollback()
                existing = (
                    await s.execute(
                        select(Document).where(
                            Document.collection_id == collection_id,
                            Document.sha256 == sha256,
                        )
                    )
                ).scalar_one()
                return _doc_dict(existing), False
            return _doc_dict(row), True

    async def get(self, document_id: str) -> dict | None:
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            return _doc_dict(row) if row else None

    async def owner_collection_id(self, document_id: str) -> str | None:
        async with self._db.session() as s:
            return (
                await s.execute(select(Document.collection_id).where(Document.id == document_id))
            ).scalar_one_or_none()

    async def raw_bytes(self, document_id: str) -> bytes | None:
        """The only caller that wants the file itself. `data` is a deferred
        column, so it is selected explicitly here and nowhere else."""
        async with self._db.session() as s:
            row = (
                await s.execute(select(Document.data).where(Document.id == document_id))
            ).scalar_one_or_none()
            return bytes(row) if row is not None else None

    async def list(self, collection_id: str, status: str | None = None) -> list[dict]:
        async with self._db.session() as s:
            stmt = select(Document).where(Document.collection_id == collection_id)
            if status:
                stmt = stmt.where(Document.status == status)
            rows = (
                (await s.execute(stmt.order_by(Document.created_at.desc(), Document.id)))
                .scalars()
                .all()
            )
            return [_doc_dict(r) for r in rows]

    async def delete(self, document_id: str) -> bool:
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return False
            await s.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))
            await s.delete(row)
            await s.commit()
            return True

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

    async def drop_chunks(self, document_id: str) -> None:
        async with self._db.session() as s:
            await s.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))
            await s.commit()

    async def replace_chunks(self, document_id: str, rows: list[dict]) -> None:
        async with self._db.session() as s:
            await s.execute(sa_delete(Chunk).where(Chunk.document_id == document_id))
            for r in rows:
                s.add(
                    Chunk(
                        document_id=document_id,
                        ordinal=r["ordinal"],
                        text=r["text"],
                        heading=r["heading"],
                        char_count=len(r["text"]),
                        embedding=r["embedding"],
                    )
                )
            await s.commit()

    async def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return
            row.status = "indexed"
            row.error = ""
            row.chunk_count = chunk_count
            row.indexed_at = utcnow()
            await s.commit()

    async def fail_stale_pending(self, reason: str) -> int:
        """Move every `pending` document to `failed`. Returns how many moved.

        Called at startup only: nothing can legitimately be pending before the
        first request of a process, so anything found here is work a previous
        process was in the middle of when it died.
        """
        async with self._db.session() as s:
            result = await s.execute(
                sa_update(Document)
                .where(Document.status == "pending")
                .values(status="failed", error=reason[:2000], chunk_count=0)
            )
            await s.commit()
            return result.rowcount or 0

    async def mark_failed(self, document_id: str, reason: str) -> None:
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return
            row.status = "failed"
            row.error = reason[:2000]
            row.chunk_count = 0
            row.indexed_at = None
            await s.commit()
