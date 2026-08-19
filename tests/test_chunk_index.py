"""The seam's contract, stated once so both backends can be held to it."""

from __future__ import annotations

import pytest

from kbase.db import Database
from kbase.index import SqlScanIndex
from kbase.search import PARTITION_SIZE
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


async def _collection(db, index, tenant="acme", name="faq"):
    cols = CollectionStore(db, index)
    await cols.create(tenant, name)
    return await cols.resolve_id(tenant, name)


async def _document(db, index, cid, sha):
    docs = DocumentStore(db, index)
    doc, _ = await docs.create(
        cid, title="T", filename="f.md", mime="text/markdown", sha256=sha, data=b"x"
    )
    return docs, doc["id"]


async def test_query_returns_the_nearer_chunk_first(db, index):
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "a" * 64)
    await docs.replace_chunks(
        doc_id,
        cid,
        [
            {"ordinal": 0, "text": "far", "heading": "", "embedding": [0.0, 1.0]},
            {"ordinal": 1, "text": "near", "heading": "", "embedding": [1.0, 0.0]},
        ],
    )
    await docs.mark_indexed(doc_id, 2)

    hits = await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0)

    assert [h["text"] for h in hits] == ["near", "far"]
    assert hits[0]["title"] == "T"


async def test_min_score_is_a_floor(db, index):
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "b" * 64)
    await docs.replace_chunks(
        doc_id,
        cid,
        [
            {"ordinal": 0, "text": "orthogonal", "heading": "", "embedding": [0.0, 1.0]},
            {"ordinal": 1, "text": "same", "heading": "", "embedding": [1.0, 0.0]},
        ],
    )
    await docs.mark_indexed(doc_id, 2)

    hits = await index.query(cid, [1.0, 0.0], limit=5, min_score=0.5)

    assert [h["text"] for h in hits] == ["same"]


async def test_query_never_crosses_collections(db, index):
    mine = await _collection(db, index, "acme", "faq")
    theirs = await _collection(db, index, "globex", "faq")
    docs, doc_id = await _document(db, index, theirs, "c" * 64)
    await docs.replace_chunks(
        doc_id, theirs, [{"ordinal": 0, "text": "secret", "heading": "", "embedding": [1.0, 0.0]}]
    )
    await docs.mark_indexed(doc_id, 1)

    assert await index.query(mine, [1.0, 0.0], limit=5, min_score=0.0) == []


async def test_deleting_the_document_removes_its_chunks(db, index):
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "d" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "gone", "heading": "", "embedding": [1.0, 0.0]}]
    )
    await docs.mark_indexed(doc_id, 1)

    await docs.delete(doc_id)

    assert await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0) == []
    assert await index.chunks(doc_id) == []


async def test_deleting_the_collection_removes_its_chunks(db, index):
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "e" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "gone", "heading": "", "embedding": [1.0, 0.0]}]
    )
    await docs.mark_indexed(doc_id, 1)

    await CollectionStore(db, index).delete("acme", "faq")

    assert await index.chunks(doc_id) == []


async def test_a_zero_vector_query_never_touches_the_database(db, index, monkeypatch):
    # A zero vector has no direction to be close to anything, and dividing by
    # its norm is undefined -- the early return has to fire before a session
    # is ever opened. Asserting only the empty result would not catch its
    # removal: every score comes back NaN in that case, and NaN >= min_score
    # is False regardless of the floor, so the query would still (accidentally)
    # answer empty -- just after opening a session and scanning every chunk.
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "f" * 64)
    await docs.replace_chunks(
        doc_id, cid, [{"ordinal": 0, "text": "anything", "heading": "", "embedding": [1.0, 0.0]}]
    )
    await docs.mark_indexed(doc_id, 1)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a zero-vector query must return before opening a session")

    monkeypatch.setattr(db, "session", refuse)

    assert await index.query(cid, [0.0, 0.0], limit=5, min_score=0.0) == []


async def test_tied_scores_break_by_scan_position_the_same_way_twice(db, index):
    # Equal scores must not come back in an arbitrary order: the same query
    # against the same corpus has to answer the same way every time, which is
    # only true if ties are broken by something stable -- the order the rows
    # were read in, not, say, an unstable sort or a reversed tiebreaker.
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "g" * 64)
    await docs.replace_chunks(
        doc_id,
        cid,
        [
            {"ordinal": i, "text": f"tied-{i}", "heading": "", "embedding": [1.0, 0.0]}
            for i in range(5)
        ],
    )
    await docs.mark_indexed(doc_id, 5)

    first = await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0)
    second = await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0)

    order = [h["text"] for h in first]
    assert order == [h["text"] for h in second]
    assert order == [f"tied-{i}" for i in range(5)]


async def test_limit_holds_across_partitions_not_only_within_one(db, index):
    # PARTITION_SIZE is 128: fewer chunks than that exercises only one
    # partition, where `_score_batch`'s own per-partition cap already trims
    # to `limit` and would hide a broken cross-partition cap entirely.
    cid = await _collection(db, index)
    docs, doc_id = await _document(db, index, cid, "h" * 64)
    n = PARTITION_SIZE * 2 + 50
    rows = [
        {"ordinal": i, "text": f"c{i}", "heading": "", "embedding": [1.0, 0.0]} for i in range(n)
    ]
    await docs.replace_chunks(doc_id, cid, rows)
    await docs.mark_indexed(doc_id, n)

    hits = await index.query(cid, [1.0, 0.0], limit=5, min_score=0.0)

    assert len(hits) == 5
