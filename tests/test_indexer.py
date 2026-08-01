import hashlib
import json

import httpx
import pytest

from kbase.db import Database
from kbase.embedding import make_embedder
from kbase.errors import EmbeddingError
from kbase.indexer import index_document
from kbase.store import CollectionStore, DocumentStore


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


@pytest.fixture
async def collection_id(db):
    store = CollectionStore(db)
    await store.create("acme", "faq")
    return await store.resolve_id("acme", "faq")


def fake_embedder(dim: int = 3):
    async def embed(texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts], len(texts)

    return embed


def failing_embedder():
    async def embed(texts):
        raise EmbeddingError("provider said no")

    return embed


def short_reply_embedder():
    """Returns fewer vectors than it was given -- a provider that partially
    succeeded. Zipping this onto the chunk list would silently index a prefix."""

    async def embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts[:1]], len(texts)

    return embed


def half_failing_embedder():
    """The real batching client, with a provider that dies after the first batch.

    Batching lives inside `make_embedder`, not in the indexer -- the indexer makes
    exactly one `embed()` call -- so a fake that counts its own invocations would
    never see a second one. This drives the actual client instead.
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] > 1:
            return httpx.Response(429, json={"error": "slow down"})
        body = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [1.0, 0.0, 0.0]} for _ in body["input"]],
                "usage": {"prompt_tokens": 1},
            },
        )

    return make_embedder(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        batch_size=32,
        transport=httpx.MockTransport(handler),
    )


async def _add(db, collection_id, text: str, filename="a.md"):
    data = text.encode()
    docs = DocumentStore(db)
    doc, _created = await docs.create(
        collection_id,
        title="t",
        filename=filename,
        mime="text/markdown",
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )
    return doc["id"]


async def test_happy_path_marks_indexed_with_chunk_count(db, collection_id):
    doc_id = await _add(db, collection_id, "## Bảo hành\n\nMười hai tháng.\n")
    await index_document(db, doc_id, embed=fake_embedder())
    doc = await DocumentStore(db).get(doc_id)
    assert doc["status"] == "indexed"
    assert doc["chunk_count"] == 1
    assert doc["error"] == ""
    assert doc["indexed_at"] is not None


async def test_chunks_carry_heading_and_embedding(db, collection_id):
    doc_id = await _add(db, collection_id, "## Bảo hành\n\nMười hai tháng.\n")
    await index_document(db, doc_id, embed=fake_embedder())
    rows = await DocumentStore(db).chunks(doc_id)
    assert rows[0]["heading"] == "Bảo hành"
    assert len(rows[0]["embedding"]) == 3


async def test_embedding_failure_marks_failed_and_leaves_no_chunks(db, collection_id):
    doc_id = await _add(db, collection_id, "## A\n\nbody\n")
    await index_document(db, doc_id, embed=failing_embedder())
    docs = DocumentStore(db)
    doc = await docs.get(doc_id)
    assert doc["status"] == "failed"
    assert "provider said no" in doc["error"]
    assert doc["chunk_count"] == 0
    assert await docs.chunks(doc_id) == []


async def test_failure_midway_leaves_no_partial_index(db, collection_id):
    # A document that reports `indexed` while holding only the first third of the
    # manual answers questions from that third and never says it is incomplete.
    body = "\n\n".join(f"Đoạn {i} " + "x" * 700 for i in range(80))
    doc_id = await _add(db, collection_id, body)
    await index_document(db, doc_id, embed=half_failing_embedder())
    docs = DocumentStore(db)
    doc = await docs.get(doc_id)
    assert doc["status"] == "failed"
    assert await docs.chunks(doc_id) == []


async def test_short_vector_reply_fails_instead_of_indexing_a_prefix(db, collection_id):
    body = "\n\n".join(f"Đoạn {i} " + "x" * 700 for i in range(5))
    doc_id = await _add(db, collection_id, body)
    await index_document(db, doc_id, embed=short_reply_embedder())
    docs = DocumentStore(db)
    assert (await docs.get(doc_id))["status"] == "failed"
    assert await docs.chunks(doc_id) == []


async def test_unsupported_file_fails_that_document_only(db, collection_id):
    bad = await _add(db, collection_id, "irrelevant", filename="manual.pdf")
    good = await _add(db, collection_id, "## A\n\nbody\n", filename="ok.md")
    await index_document(db, bad, embed=fake_embedder())
    await index_document(db, good, embed=fake_embedder())
    docs = DocumentStore(db)
    assert (await docs.get(bad))["status"] == "failed"
    assert (await docs.get(good))["status"] == "indexed"


async def test_empty_document_indexes_with_zero_chunks(db, collection_id):
    doc_id = await _add(db, collection_id, "   \n\n  ")
    await index_document(db, doc_id, embed=fake_embedder())
    doc = await DocumentStore(db).get(doc_id)
    assert doc["status"] == "indexed"
    assert doc["chunk_count"] == 0


async def test_reindex_replaces_rather_than_appends(db, collection_id):
    doc_id = await _add(db, collection_id, "## A\n\nbody\n")
    await index_document(db, doc_id, embed=fake_embedder())
    await index_document(db, doc_id, embed=fake_embedder())
    docs = DocumentStore(db)
    assert (await docs.get(doc_id))["chunk_count"] == 1
    assert len(await docs.chunks(doc_id)) == 1


async def test_missing_document_is_a_no_op_not_a_crash(db):
    await index_document(db, "does-not-exist", embed=fake_embedder())
