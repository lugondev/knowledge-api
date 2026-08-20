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


async def _seeded(db, index, rows, tenant="acme", name="faq", sha="d" * 64):
    cols = CollectionStore(db, index)
    await cols.create(tenant, name)
    cid = await cols.resolve_id(tenant, name)
    docs, doc_id = await _document(db, index, cid, sha)
    await docs.replace_chunks(doc_id, cid, rows)
    await docs.mark_indexed(doc_id, len(rows))
    return cid


async def test_query_returns_the_nearer_chunk_first(db, index):
    cid = await _seeded(
        db,
        index,
        [
            {"ordinal": 0, "text": "far", "heading": "", "embedding": [0.0, 1.0, 0.0]},
            {"ordinal": 1, "text": "near", "heading": "", "embedding": [1.0, 0.0, 0.0]},
        ],
    )

    hits = await index.query(cid, [1.0, 0.0, 0.0], limit=5, min_score=0.0)

    assert [h["text"] for h in hits] == ["near", "far"]
    assert hits[0]["title"] == "T"
    assert hits[0]["score"] == pytest.approx(1.0)


async def test_min_score_is_a_floor(db, index):
    cid = await _seeded(
        db,
        index,
        [
            {"ordinal": 0, "text": "orthogonal", "heading": "", "embedding": [0.0, 1.0, 0.0]},
            {"ordinal": 1, "text": "same", "heading": "", "embedding": [1.0, 0.0, 0.0]},
        ],
    )

    hits = await index.query(cid, [1.0, 0.0, 0.0], limit=5, min_score=0.5)

    assert [h["text"] for h in hits] == ["same"]


async def test_query_never_crosses_collections(db, index):
    await _seeded(
        db,
        index,
        [{"ordinal": 0, "text": "secret", "heading": "", "embedding": [1.0, 0.0, 0.0]}],
        tenant="globex",
        name="theirs",
        sha="e" * 64,
    )
    cols = CollectionStore(db, index)
    await cols.create("acme", "mine")
    mine = await cols.resolve_id("acme", "mine")

    assert await index.query(mine, [1.0, 0.0, 0.0], limit=5, min_score=0.0) == []


async def test_a_zero_vector_scores_nothing_rather_than_NaN(db, index):
    cid = await _seeded(
        db,
        index,
        [{"ordinal": 0, "text": "empty", "heading": "", "embedding": [0.0, 0.0, 0.0]}],
        sha="f" * 64,
    )

    # The scan path returns 0.0 for an uncomparable vector; `<=>` returns NaN,
    # and NaN passes a `>= min_score` test in some comparisons. Neither
    # backend may admit it as a hit.
    assert await index.query(cid, [1.0, 0.0, 0.0], limit=5, min_score=0.0) == []


async def test_limit_caps_the_result(db, index):
    cid = await _seeded(
        db,
        index,
        [
            {"ordinal": i, "text": f"c{i}", "heading": "", "embedding": [1.0, 0.0, 0.0]}
            for i in range(5)
        ],
        sha="0" * 64,
    )

    assert len(await index.query(cid, [1.0, 0.0, 0.0], limit=2, min_score=0.0)) == 2


async def test_existing_json_vectors_are_backfilled_without_re_embedding(db):
    """The JSON column holds vectors someone already paid for."""
    cols_index = PgVectorIndex(db, dim=DIM)  # not yet migrated
    cols = CollectionStore(db, cols_index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    async with db.session() as s:
        await s.execute(
            text(
                "INSERT INTO documents (id, collection_id, title, filename, mime, sha256, "
                "bytes_len, data, status, error, chunk_count, created_at) VALUES "
                "('d1', :cid, 'T', 'f.md', 'text/markdown', :sha, 1, '\\x78'::bytea, "
                "'indexed', '', 1, now())"
            ),
            {"cid": cid, "sha": "a" * 64},
        )
        await s.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, ordinal, text, heading, "
                "char_count, embedding) VALUES ('c1', 'd1', :cid, 0, 'hi', '', 2, "
                "'[1.0, 0.0, 0.0]'::json)"
            ),
            {"cid": cid},
        )
        await s.commit()

    await cols_index.create_schema()

    assert [r["embedding"] for r in await cols_index.chunks("d1")] == [[1.0, 0.0, 0.0]]


async def test_a_chunk_of_the_wrong_width_fails_its_document_instead_of_the_boot(db):
    index = PgVectorIndex(db, dim=DIM)
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    async with db.session() as s:
        await s.execute(
            text(
                "INSERT INTO documents (id, collection_id, title, filename, mime, sha256, "
                "bytes_len, data, status, error, chunk_count, created_at) VALUES "
                "('d2', :cid, 'T', 'f.md', 'text/markdown', :sha, 1, '\\x78'::bytea, "
                "'indexed', '', 1, now())"
            ),
            {"cid": cid, "sha": "b" * 64},
        )
        await s.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, ordinal, text, heading, "
                "char_count, embedding) VALUES ('c2', 'd2', :cid, 0, 'hi', '', 2, "
                "'[1.0, 0.0]'::json)"
            ),
            {"cid": cid},
        )
        await s.commit()

    await index.create_schema()

    doc = await DocumentStore(db, index).get("d2")
    assert doc["status"] == "failed"
    assert "embedding model" in doc["error"]
    assert await index.chunks("d2") == []


async def test_a_wrong_width_query_vector_returns_nothing(db, index):
    """The scan calls a width mismatch "not comparable" and returns `[]`.

    `<=>` calls it an error, and the route turns that into a 500. Two backends
    behind one seam cannot disagree about what an ordinary caller mistake is.
    """
    cid = await _seeded(
        db,
        index,
        [{"ordinal": 0, "text": "hi", "heading": "", "embedding": [1.0, 0.0, 0.0]}],
        sha="1" * 64,
    )

    assert await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0) == []
    assert await index.query(cid, [1.0, 0.0, 0.0, 1.0], limit=5, min_score=0.0) == []


async def test_dropping_the_dimension_refuses_to_start(db, index):
    """`KB_EMBED_DIM` unset on a migrated database selects the scan backend.

    The scan then reads `vector(3)` through a `JSON` attribute, gets the string
    '[1,0,0]', turns it into a list of characters, and drops every chunk as a
    width mismatch: search answers 200 with nothing, for a corpus that is
    entirely intact. That must be a refusal to start, not a silent emptying.
    """
    from kbase.errors import SchemaError
    from kbase.index import SqlScanIndex

    with pytest.raises(SchemaError) as caught:
        await SqlScanIndex(db).create_schema()

    assert "KB_EMBED_DIM" in str(caught.value)
    assert str(DIM) in str(caught.value)


async def test_a_null_embedding_reads_back_as_no_vector(db, index):
    """The migrated column is nullable, so `chunks()` must survive a NULL."""
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs, doc_id = await _document(db, index, cid, "2" * 64)
    async with db.session() as s:
        await s.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, ordinal, text, heading, "
                "char_count, embedding) VALUES ('cnull', :doc, :cid, 0, 'hi', '', 2, NULL)"
            ),
            {"doc": doc_id, "cid": cid},
        )
        await s.commit()

    rows = await index.chunks(doc_id)

    assert [r["embedding"] for r in rows] == [[]]


async def test_a_failed_document_keeps_none_of_its_chunks(db):
    """One good chunk and one of the wrong width under the same document.

    Deleting only the unconverted chunk leaves the good one searchable under a
    document that is `failed` with `chunk_count = 0` -- the exact violation of
    the chunk-exists-only-for-an-indexed-document invariant that the pgvector
    single-table filter is built on.
    """
    index = PgVectorIndex(db, dim=DIM)
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    async with db.session() as s:
        await s.execute(
            text(
                "INSERT INTO documents (id, collection_id, title, filename, mime, sha256, "
                "bytes_len, data, status, error, chunk_count, created_at) VALUES "
                "('d3', :cid, 'T', 'f.md', 'text/markdown', :sha, 1, '\\x78'::bytea, "
                "'indexed', '', 2, now())"
            ),
            {"cid": cid, "sha": "3" * 64},
        )
        await s.execute(
            text(
                "INSERT INTO chunks (id, document_id, collection_id, ordinal, text, heading, "
                "char_count, embedding) VALUES ('c3a', 'd3', :cid, 0, 'good', '', 4, "
                "'[1.0, 0.0, 0.0]'::json), ('c3b', 'd3', :cid, 1, 'narrow', '', 6, "
                "'[1.0, 0.0]'::json)"
            ),
            {"cid": cid},
        )
        await s.commit()

    await index.create_schema()

    doc = await DocumentStore(db, index).get("d3")
    assert doc["status"] == "failed"
    assert await index.chunks("d3") == []
    assert await index.query(cid, [1.0, 0.0, 0.0], limit=5, min_score=0.0) == []


async def test_the_scan_backend_still_boots_on_an_unmigrated_postgres(db):
    """The refusal is about a `vector` column, not about Postgres.

    A Postgres deployment that never set `KB_EMBED_DIM` has a JSON column the
    scan reads correctly, and must keep starting.
    """
    from kbase.index import SqlScanIndex

    await SqlScanIndex(db).create_schema()
