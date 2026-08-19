"""Chunks exist only for documents whose status is `indexed`."""

from __future__ import annotations

import hashlib

import pytest

from kbase.db import Database
from kbase.indexer import index_document
from kbase.store import CollectionStore, DocumentStore


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


def fake_embedder():
    async def embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts], len(texts)

    return embed


async def _indexed_document(db):
    cols = CollectionStore(db)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db)
    body = b"# Heading\n\nsome text"
    doc, _ = await docs.create(
        cid,
        title="t",
        filename="f.md",
        mime="text/markdown",
        sha256=hashlib.sha256(body).hexdigest(),
        data=body,
    )
    await index_document(db, doc["id"], embed=fake_embedder())
    return docs, doc["id"]


async def test_marking_pending_removes_the_chunks(db):
    docs, doc_id = await _indexed_document(db)
    assert await docs.chunks(doc_id) != []

    await docs.mark_pending(doc_id)

    # Not merely hidden from search -- gone. A chunk that outlives its
    # document's `indexed` status is reachable by a query that filters on
    # chunks alone, which is what the pgvector backend does.
    assert await docs.chunks(doc_id) == []


async def test_marking_pending_twice_is_still_empty(db):
    docs, doc_id = await _indexed_document(db)
    await docs.mark_pending(doc_id)
    await docs.mark_pending(doc_id)
    assert await docs.chunks(doc_id) == []
