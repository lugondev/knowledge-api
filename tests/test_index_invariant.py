"""Chunks exist only for documents whose status is `indexed`."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import update as sa_update

from kbase.db import Database
from kbase.index import SqlScanIndex
from kbase.indexer import index_document
from kbase.models import Document
from kbase.store import CollectionStore, DocumentStore


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


@pytest.fixture
def index(db):
    return SqlScanIndex(db)


def fake_embedder():
    async def embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts], len(texts)

    return embed


async def _indexed_document(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db, index)
    body = b"# Heading\n\nsome text"
    doc, _ = await docs.create(
        cid,
        title="t",
        filename="f.md",
        mime="text/markdown",
        sha256=hashlib.sha256(body).hexdigest(),
        data=body,
    )
    await index_document(db, index, doc["id"], embed=fake_embedder())
    return docs, doc["id"]


async def test_marking_pending_removes_the_chunks(db, index):
    docs, doc_id = await _indexed_document(db, index)
    assert await index.chunks(doc_id) != []

    await docs.mark_pending(doc_id)

    # Not merely hidden from search -- gone. A chunk that outlives its
    # document's `indexed` status is reachable by a query that filters on
    # chunks alone, which is what the pgvector backend does.
    assert await index.chunks(doc_id) == []


async def test_marking_pending_twice_is_still_empty(db, index):
    docs, doc_id = await _indexed_document(db, index)
    await docs.mark_pending(doc_id)
    await docs.mark_pending(doc_id)
    assert await index.chunks(doc_id) == []


async def _pending_document(db, index):
    """A document that exists but has never been indexed: still `pending`,
    with no chunks -- the state `finish_indexing` is meant to move on from."""
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db, index)
    body = b"# Heading\n\nsome text"
    doc, _ = await docs.create(
        cid,
        title="t",
        filename="f.md",
        mime="text/markdown",
        sha256=hashlib.sha256(body).hexdigest(),
        data=body,
    )
    return docs, doc["id"], cid


async def test_finish_indexing_flips_status_and_writes_chunks_together(db, index):
    docs, doc_id, cid = await _pending_document(db, index)

    # Before: pending, no chunks. `finish_indexing` never runs on anything
    # else, so this is the only state a document is in beforehand.
    before = await docs.get(doc_id)
    assert before["status"] == "pending"
    assert await index.chunks(doc_id) == []

    rows = [
        {"ordinal": 0, "text": "some text", "heading": "Heading", "embedding": [1.0, 0.0, 0.0]}
    ]
    ok = await docs.finish_indexing(doc_id, cid, rows)
    assert ok is True

    # After: status and chunks flip together, in the one commit
    # `finish_indexing` makes -- there is no window in between where chunks
    # exist under a non-indexed status for a concurrent search to reach.
    after = await docs.get(doc_id)
    assert after["status"] == "indexed"
    assert after["chunk_count"] == 1
    assert len(await index.chunks(doc_id)) == 1


async def test_finish_indexing_returns_false_when_document_was_deleted(db, index):
    docs, doc_id, cid = await _pending_document(db, index)
    await docs.delete(doc_id)

    rows = [{"ordinal": 0, "text": "x", "heading": None, "embedding": [1.0, 0.0, 0.0]}]
    ok = await docs.finish_indexing(doc_id, cid, rows)

    assert ok is False
    assert await index.chunks(doc_id) == []


async def test_mark_failed_removes_chunks(db, index):
    docs, doc_id = await _indexed_document(db, index)
    assert await index.chunks(doc_id) != []

    await docs.mark_failed(doc_id, "boom")

    assert await index.chunks(doc_id) == []


async def test_reuploading_a_failed_document_leaves_no_chunks(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db, index)
    body = b"# Heading\n\nsome text"
    sha = hashlib.sha256(body).hexdigest()
    doc, _ = await docs.create(
        cid, title="t", filename="f.md", mime="text/markdown", sha256=sha, data=body
    )
    doc_id = doc["id"]
    await index_document(db, index, doc_id, embed=fake_embedder())
    assert await index.chunks(doc_id) != []

    # Force the row to `failed` directly, bypassing `mark_failed`, so this
    # proves `_retry_if_failed` cleans up chunks on its own rather than
    # relying on a caller (or `mark_failed`) having already done it.
    async with db.session() as s:
        await s.execute(sa_update(Document).where(Document.id == doc_id).values(status="failed"))
        await s.commit()
    assert await index.chunks(doc_id) != []  # confirms the forced setup stuck

    doc2, accepted = await docs.create(
        cid, title="t", filename="f.md", mime="text/markdown", sha256=sha, data=body
    )

    assert accepted is True
    assert doc2["status"] == "pending"
    assert await index.chunks(doc_id) == []
