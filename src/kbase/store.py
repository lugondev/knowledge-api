"""Every read and write, each one scoped by tenant at the query, never after it."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.exc import IntegrityError

from kbase.db import Database
from kbase.index import ChunkIndex
from kbase.models import Collection, Document, utcnow


class CollectionStore:
    def __init__(self, db: Database, index: ChunkIndex) -> None:
        self._db = db
        self._index = index

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
            # Subqueries rather than a list of ids: a collection with more
            # documents than the driver's bind-parameter limit would otherwise
            # fail to delete, and the limit is low enough on older SQLite to
            # reach with an ordinary corpus.
            owned = select(Document.id).where(Document.collection_id == row.id)
            await self._index.drop_where_document_in(s, owned)
            await s.execute(sa_delete(Document).where(Document.collection_id == row.id))
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


def _reset(d: Document) -> None:
    """Back to the state a freshly uploaded document is in."""
    d.status = "pending"
    d.error = ""
    d.chunk_count = 0
    d.indexed_at = None


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
    def __init__(self, db: Database, index: ChunkIndex) -> None:
        self._db = db
        self._index = index

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
        """Returns (document, accepted). `accepted is False` means these exact
        bytes are already in this collection and nothing needs indexing.

        A `failed` row is the exception: re-uploading the same bytes is a retry,
        not a duplicate, and it comes back accepted so the caller schedules the
        work again. Without that, the instruction this service writes onto its
        own interrupted documents -- "upload the document again" -- did nothing,
        and one transient outage at the provider stranded a document for good.
        """
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
                return await self._retry_if_failed(s, existing)
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
                return await self._retry_if_failed(s, existing)
            return _doc_dict(row), True

    async def _retry_if_failed(self, s, existing: Document) -> tuple[dict, bool]:
        if existing.status != "failed":
            return _doc_dict(existing), False
        # A failed row can carry chunks from whatever left it failed -- this
        # resets the row for another pass, and the chunks have to go with it,
        # not stay behind for a caller elsewhere to remember to drop.
        await self._index.drop_document(s, existing.id)
        _reset(existing)
        await s.commit()
        return _doc_dict(existing), True

    async def mark_pending(self, document_id: str) -> dict | None:
        """Queue an existing document for another pass over its stored bytes.

        Returns None if the document is gone, and the row untouched if it is
        already pending -- two indexers racing over one document would each
        replace the other's chunks and leave `chunk_count` describing neither.
        """
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return None
            if row.status == "pending":
                return _doc_dict(row)
            # The chunks go with the status. Left behind, they would be a
            # complete, searchable copy of a document that is about to be
            # re-embedded -- invisible today only because search filters on a
            # joined `status`, which the pgvector backend cannot afford to do.
            await self._index.drop_document(s, document_id)
            _reset(row)
            await s.commit()
            return _doc_dict(row)

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
            await self._index.drop_document(s, document_id)
            await s.delete(row)
            await s.commit()
            return True

    async def drop_chunks(self, document_id: str) -> None:
        async with self._db.session() as s:
            await self._index.drop_document(s, document_id)
            await s.commit()

    async def replace_chunks(
        self, document_id: str, collection_id: str, rows: list[dict]
    ) -> bool:
        """False when the document was deleted while it was being indexed.

        Nothing else refuses the write. The schema carries no cascade on purpose
        (see `models.py`) and SQLite does not enforce the foreign key, so the
        insert would succeed against a document row that no longer exists. Search
        joins Document, so those chunks never surface; deleting the collection
        only reaches chunks whose document it can still see, so they are never
        collected either. They would sit in the database for its whole life.

        The check and the insert share one transaction, which is what makes it
        an answer rather than a narrower window.
        """
        async with self._db.session() as s:
            still_there = (
                await s.execute(select(Document.id).where(Document.id == document_id))
            ).scalar_one_or_none()
            if still_there is None:
                return False
            await self._index.replace(s, document_id, collection_id, rows)
            await s.commit()
            return True

    async def finish_indexing(
        self, document_id: str, collection_id: str, rows: list[dict]
    ) -> bool:
        """`replace_chunks` and `mark_indexed`, fused into one commit.

        Done as two separate transactions, there is a real window between them
        where the chunks exist and are searchable by `collection_id` alone but
        the row still reads `pending` -- a concurrent search would either miss
        them (joined-status filter) or wrongly surface a document mid-reindex
        (chunk-only filter, which is what the pgvector backend requires). One
        commit means an external reader never observes the row and its chunks
        disagreeing about whether this document is indexed.

        Returns False when the document was deleted while it was being
        indexed, exactly like `replace_chunks` -- nothing is written.
        """
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return False
            await self._index.replace(s, document_id, collection_id, rows)
            row.status = "indexed"
            row.error = ""
            row.chunk_count = len(rows)
            row.indexed_at = utcnow()
            await s.commit()
            return True

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

        Their chunks go too. A process that died between writing chunks and
        marking the row leaves both behind, and the row is about to be told it
        holds none -- so without this the count says zero while the rows sit
        there, reachable by nothing and deleted by nothing.
        """
        async with self._db.session() as s:
            stranded = select(Document.id).where(Document.status == "pending")
            await self._index.drop_where_document_in(s, stranded)
            result = await s.execute(
                sa_update(Document)
                .where(Document.status == "pending")
                .values(status="failed", error=reason[:2000], chunk_count=0, indexed_at=None)
            )
            await s.commit()
            return result.rowcount or 0

    async def mark_failed(self, document_id: str, reason: str) -> None:
        async with self._db.session() as s:
            row = await s.get(Document, document_id)
            if row is None:
                return
            # `failed` and having chunks would mean a document that couldn't
            # be indexed still answers searches -- self-enforced here rather
            # than left to every caller to have dropped them first.
            await self._index.drop_document(s, document_id)
            row.status = "failed"
            row.error = reason[:2000]
            row.chunk_count = 0
            row.indexed_at = None
            await s.commit()
