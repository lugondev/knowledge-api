"""The pgvector backend, held to the same contract as the scan.

Skipped unless KB_TEST_POSTGRES_URL points at a database with the `vector`
extension available -- `docker compose --profile pg up -d postgres`.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from kbase.db import Database
from kbase.models import Base
from kbase.pgindex import PgVectorIndex
from kbase.store import CollectionStore, DocumentStore

POSTGRES_URL = os.environ.get("KB_TEST_POSTGRES_URL", "")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="KB_TEST_POSTGRES_URL is not set"
)

DIM = 3


@pytest.fixture
async def db():
    d = Database(POSTGRES_URL)
    async with d.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await d.create_all()
    yield d
    await d.dispose()


@pytest.fixture
async def index(db):
    ix = PgVectorIndex(db, dim=DIM)
    await ix.create_schema()
    return ix


async def _document(db, index, cid, sha):
    docs = DocumentStore(db, index)
    doc, _ = await docs.create(
        cid, title="T", filename="f.md", mime="text/markdown", sha256=sha, data=b"x"
    )
    return docs, doc["id"]


async def test_create_schema_makes_a_vector_column(db, index):
    async with db.session() as s:
        kind = (
            await s.execute(
                text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
                )
            )
        ).scalar_one()
    assert kind == f"vector({DIM})"


async def test_a_written_chunk_reads_back_as_its_vector(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs, doc_id = await _document(db, index, cid, "a" * 64)

    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "hi", "heading": "H", "embedding": [1.0, 0.0, 0.0]}]
    )

    rows = await index.chunks(doc_id)
    assert [r["embedding"] for r in rows] == [[1.0, 0.0, 0.0]]
    assert rows[0]["heading"] == "H"


async def test_deleting_a_document_removes_its_chunks(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs, doc_id = await _document(db, index, cid, "b" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "hi", "heading": "", "embedding": [1.0, 0.0, 0.0]}]
    )

    await docs.delete(doc_id)

    assert await index.chunks(doc_id) == []


async def test_deleting_a_collection_removes_its_chunks(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs, doc_id = await _document(db, index, cid, "c" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "hi", "heading": "", "embedding": [1.0, 0.0, 0.0]}]
    )

    await cols.delete("acme", "faq")

    assert await index.chunks(doc_id) == []


async def test_create_schema_runs_twice_without_losing_the_vectors(db, index):
    """A restart calls this again. It must not be how a corpus disappears."""
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs, doc_id = await _document(db, index, cid, "9" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "hi", "heading": "", "embedding": [1.0, 0.0, 0.0]}]
    )

    await index.create_schema()

    assert [r["embedding"] for r in await index.chunks(doc_id)] == [[1.0, 0.0, 0.0]]


async def test_a_different_dimension_refuses_to_migrate(db, index):
    from kbase.errors import SchemaError

    with pytest.raises(SchemaError) as caught:
        await PgVectorIndex(db, dim=DIM + 1).create_schema()

    assert str(DIM) in str(caught.value)
