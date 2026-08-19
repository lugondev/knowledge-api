"""One test per defect the audit found, each failing against the code as it was.

The names say what goes wrong without the fix, not which function was touched:
these are here to stop a regression, and a regression will be reintroduced by
someone who has never read the patch that fixed it.
"""

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from kbase.db import Database
from kbase.errors import EmbeddingError
from kbase.index import SqlScanIndex
from kbase.indexer import index_document
from kbase.models import Chunk
from kbase.search import PARTITION_SIZE
from kbase.server.app import create_app
from kbase.settings import Settings
from kbase.store import CollectionStore, DocumentStore
from tests.conftest import keyword_embedder


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/audit.db")
    await d.create_all()
    yield d
    await d.dispose()


@pytest.fixture
def index(db):
    return SqlScanIndex(db)


@pytest.fixture
async def collection_id(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    return await cols.resolve_id("acme", "faq")


async def _add(db, index, collection_id, text: str, filename: str = "a.md") -> str:
    data = text.encode()
    doc, _ = await DocumentStore(db, index).create(
        collection_id,
        title="t",
        filename=filename,
        mime="text/markdown",
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )
    return doc["id"]


async def _chunk_count(db) -> int:
    async with db.session() as s:
        return int((await s.execute(select(func.count()).select_from(Chunk))).scalar_one())


def _held_embedder(gate: asyncio.Event, dim: int = 2):
    async def embed(texts):
        await gate.wait()
        return [[1.0] + [0.0] * (dim - 1) for _ in texts], len(texts)

    return embed


def _failing_embedder(message: str = "Client error '401' for url 'https://embed.internal/v1'"):
    async def embed(texts):
        raise EmbeddingError(f"embedding request failed: {message}")

    return embed


# --- 1. a failed document must be retryable ---------------------------------


async def test_reuploading_the_same_bytes_retries_a_failed_document(db, index, collection_id):
    # The message this service writes onto its own interrupted documents is
    # "upload it again". That has to actually do something.
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await index_document(db, index, doc_id, embed=_failing_embedder())
    docs = DocumentStore(db, index)
    assert (await docs.get(doc_id))["status"] == "failed"

    same_bytes = b"## A\n\nbody"
    doc, accepted = await docs.create(
        collection_id,
        title="t",
        filename="a.md",
        mime="text/markdown",
        sha256=hashlib.sha256(same_bytes).hexdigest(),
        data=same_bytes,
    )
    assert accepted is True
    assert doc["id"] == doc_id  # the same row, not a second copy
    assert doc["status"] == "pending"
    assert doc["error"] == ""


async def test_reuploading_an_indexed_document_is_still_a_duplicate(db, index, collection_id):
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await index_document(db, index, doc_id, embed=keyword_embedder())
    docs = DocumentStore(db, index)
    same_bytes = b"## A\n\nbody"
    _doc, accepted = await docs.create(
        collection_id,
        title="t",
        filename="a.md",
        mime="text/markdown",
        sha256=hashlib.sha256(same_bytes).hexdigest(),
        data=same_bytes,
    )
    assert accepted is False


def test_failed_upload_over_http_comes_back_202_and_indexes(tmp_path, acme):
    state = {"down": True}

    async def embed(texts):
        if state["down"]:
            raise EmbeddingError("provider timed out")
        return [[1.0, 0.0] for _ in texts], len(texts)

    settings = Settings.from_env(
        {
            "KB_API_KEYS": "acme-key:acme",
            "KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/http.db",
            "KB_EMBED_BASE_URL": "http://embed.invalid/v1",
            "KB_EMBED_MODEL": "fake",
        }
    )
    with TestClient(create_app(settings, embedder=embed)) as c:
        c.post("/v1/collections", json={"name": "faq"}, headers=acme)
        body = {"collection": "faq", "text": "## A\n\nbody"}
        first = c.post("/v1/documents", json=body, headers=acme)
        doc_id = first.json()["id"]
        assert c.get(f"/v1/documents/{doc_id}", headers=acme).json()["status"] == "failed"

        state["down"] = False
        second = c.post("/v1/documents", json=body, headers=acme)
        assert second.status_code == 202
        after = c.get(f"/v1/documents/{doc_id}", headers=acme).json()
        assert after["status"] == "indexed"
        assert after["chunk_count"] == 1


def test_reindex_endpoint_reuses_the_stored_bytes(tmp_path, acme, globex):
    state = {"down": True}

    async def embed(texts):
        if state["down"]:
            raise EmbeddingError("provider timed out")
        return [[1.0, 0.0] for _ in texts], len(texts)

    settings = Settings.from_env(
        {
            "KB_API_KEYS": "acme-key:acme, globex-key:globex",
            "KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/reindex.db",
            "KB_EMBED_BASE_URL": "http://embed.invalid/v1",
            "KB_EMBED_MODEL": "fake",
        }
    )
    with TestClient(create_app(settings, embedder=embed)) as c:
        c.post("/v1/collections", json={"name": "faq"}, headers=acme)
        doc_id = c.post(
            "/v1/documents", json={"collection": "faq", "text": "## A\n\nbody"}, headers=acme
        ).json()["id"]
        assert c.get(f"/v1/documents/{doc_id}", headers=acme).json()["status"] == "failed"

        state["down"] = False
        # Nothing is uploaded a second time: the bytes were kept for this.
        r = c.post(f"/v1/documents/{doc_id}/reindex", headers=acme)
        assert r.status_code == 202
        assert c.get(f"/v1/documents/{doc_id}", headers=acme).json()["status"] == "indexed"

        # And it is someone else's document as far as another tenant is concerned.
        assert c.post(f"/v1/documents/{doc_id}/reindex", headers=globex).status_code == 404


# --- 3. no chunk outlives the document it belongs to ------------------------


async def test_deleting_a_document_mid_index_leaves_no_orphan_chunks(db, index, collection_id):
    doc_id = await _add(db, index, collection_id, "## A\n\nbody one\n\n## B\n\nbody two")
    gate = asyncio.Event()
    task = asyncio.create_task(index_document(db, index, doc_id, embed=_held_embedder(gate)))
    await asyncio.sleep(0.05)
    await DocumentStore(db, index).delete(doc_id)
    gate.set()
    await task
    assert await _chunk_count(db) == 0


async def test_deleting_a_collection_mid_index_leaves_no_orphan_chunks(db, index, collection_id):
    doc_id = await _add(db, index, collection_id, "## A\n\nbody one\n\n## B\n\nbody two")
    gate = asyncio.Event()
    task = asyncio.create_task(index_document(db, index, doc_id, embed=_held_embedder(gate)))
    await asyncio.sleep(0.05)
    assert await CollectionStore(db, index).delete("acme", "faq") is True
    gate.set()
    await task
    assert await _chunk_count(db) == 0


async def test_the_startup_sweep_takes_the_chunks_with_it(db, index, collection_id):
    # A process that died between writing chunks and marking the row leaves both.
    # The row is about to be told it holds none, so the chunks have to go.
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    docs = DocumentStore(db, index)
    await docs.replace_chunks(
        doc_id,
        collection_id,
        [{"ordinal": 0, "text": "body", "heading": "A", "embedding": [1.0, 0.0]}],
    )
    assert await _chunk_count(db) == 1

    moved = await docs.fail_stale_pending("interrupted")
    assert moved == 1
    assert await _chunk_count(db) == 0
    assert (await docs.get(doc_id))["chunk_count"] == 0


# --- 4. the provider's words stay out of the tenant's row -------------------


async def test_the_embedding_host_never_reaches_the_document_row(db, index, collection_id):
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await index_document(db, index, doc_id, embed=_failing_embedder())
    error = (await DocumentStore(db, index).get(doc_id))["error"]
    assert "embed.internal" not in error
    assert "401" not in error
    assert error == "the embedding provider could not be reached or refused the request"


async def test_an_unexpected_failure_is_not_quoted_to_the_tenant(db, index, collection_id):
    async def exploding(texts):
        raise RuntimeError("connection to postgresql://kb:hunter2@db.internal/kbase lost")

    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await index_document(db, index, doc_id, embed=exploding)
    error = (await DocumentStore(db, index).get(doc_id))["error"]
    assert "hunter2" not in error
    assert "db.internal" not in error


async def test_a_failure_about_the_callers_own_file_is_still_reported(db, index, collection_id):
    # The other half of the rule: an ExtractError describes what the caller sent,
    # so hiding it would leave them with nothing to act on.
    doc_id = await _add(db, index, collection_id, "irrelevant", filename="manual.pdf")
    await index_document(db, index, doc_id, embed=keyword_embedder())
    error = (await DocumentStore(db, index).get(doc_id))["error"]
    assert "pdf" in error.lower()


# --- 5. the size limit applies before the bytes are held --------------------


def test_a_body_without_a_declared_length_is_refused(client, acme):
    # KB_MAX_UPLOAD_BYTES is 1000 in this fixture, and a generator body is sent
    # chunked, so there is no Content-Length for the cheap check to read.
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)

    def chunked():
        payload = b'{"collection":"faq","text":"' + b"A" * 200_000 + b'"}'
        for i in range(0, len(payload), 8192):
            yield payload[i : i + 8192]

    r = client.post(
        "/v1/documents", content=chunked(), headers={**acme, "Content-Type": "application/json"}
    )
    assert r.status_code == 413


async def test_the_limit_stops_the_read_rather_than_measuring_it_afterwards(tmp_path):
    """The point is when the refusal happens, not that it happens.

    Driven through the ASGI interface directly, because `TestClient` serialises
    the whole request body before the application is ever called -- so over that
    transport a limit applied after the last byte and a limit applied at the
    first one look exactly alike, and only one of them keeps the bytes out of
    the process.
    """
    settings = Settings.from_env(
        {
            "KB_API_KEYS": "acme-key:acme",
            "KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/asgi.db",
            "KB_EMBED_BASE_URL": "http://embed.invalid/v1",
            "KB_EMBED_MODEL": "fake",
            "KB_MAX_UPLOAD_BYTES": "1000",
        }
    )
    app = create_app(settings, embedder=keyword_embedder())

    pulled = 0

    async def receive():
        # A 16 MB chunked body against a 1 KB limit. It ends eventually only so
        # that a broken limit fails this test instead of hanging it.
        nonlocal pulled
        pulled += 1
        return {
            "type": "http.request",
            "body": b"A" * 8192,
            "more_body": pulled < 2000,
        }

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/documents",
        "raw_path": b"/v1/documents",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer acme-key"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }

    async with app.router.lifespan_context(app):
        await app(scope, receive, send)

    assert sent[0]["status"] == 413
    # The limit is 1000 bytes and each pull is 8 KB, so one is already over it.
    assert pulled == 1


def test_an_oversized_body_on_a_declared_route_is_413_not_a_parse_error(client, acme):
    # FastAPI turns anything that escapes body parsing into "400 There was an
    # error parsing the body", which is the wrong status and an untrue reason.
    r = client.post("/v1/search", json={"collection": "faq", "query": "x" * 5000}, headers=acme)
    assert r.status_code == 413
    assert "KB_MAX_UPLOAD_BYTES" in r.json()["detail"]


# --- 6-9. the request and response contract ---------------------------------


def test_the_search_response_is_declared_in_the_schema(client):
    schema = client.get("/openapi.json").json()
    search = schema["paths"]["/v1/search"]["post"]["responses"]["200"]
    ref = search["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("SearchResponse")


def test_a_collection_is_found_under_the_name_it_was_created_with(client, acme):
    assert client.post("/v1/collections", json={"name": "  faq  "}, headers=acme).status_code == 201
    assert (
        client.post(
            "/v1/documents",
            json={"collection": "  faq  ", "text": "## A\n\nbảo hành"},
            headers=acme,
        ).status_code
        == 202
    )
    assert (
        client.post(
            "/v1/search",
            json={"collection": " faq", "query": "bảo hành", "min_score": 0.1},
            headers=acme,
        ).status_code
        == 200
    )
    assert (
        client.get("/v1/documents", params={"collection": "faq "}, headers=acme).status_code == 200
    )
    assert client.delete("/v1/collections/%20faq%20", headers=acme).status_code == 204


def test_listing_an_unknown_collection_is_404_not_an_empty_list(client, acme):
    r = client.get("/v1/documents", params={"collection": "nope"}, headers=acme)
    assert r.status_code == 404


def test_an_overlong_title_is_refused_rather_than_stored_past_its_column(client, acme):
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)
    r = client.post(
        "/v1/documents", json={"collection": "faq", "title": "T" * 600, "text": "x"}, headers=acme
    )
    assert r.status_code == 422


def test_an_overlong_filename_is_truncated_to_its_column(client, acme):
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)
    name = "n" * 600 + ".md"
    r = client.post(
        "/v1/documents",
        data={"collection": "faq"},
        files={"file": (name, b"## A\n\nbody", "text/markdown")},
        headers=acme,
    )
    assert r.status_code == 202
    assert len(r.json()["filename"]) == 512


# --- 2. one search does not cost the whole collection ------------------------


async def test_a_search_holds_one_partition_not_the_collection(db, index, collection_id):
    import tracemalloc

    dim = 512
    docs = DocumentStore(db, index)
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await docs.replace_chunks(
        doc_id,
        collection_id,
        [
            {"ordinal": i, "text": f"c{i}", "heading": "A", "embedding": [0.01] * dim}
            for i in range(PARTITION_SIZE * 8)
        ],
    )
    await docs.mark_indexed(doc_id, PARTITION_SIZE * 8)

    async def embed(texts):
        return [[0.01] * dim for _ in texts], len(texts)

    tracemalloc.start()
    vectors, _tokens = await embed(["q"])
    hits = await index.query(collection_id, vectors[0], limit=5, min_score=0.35)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(hits) == 5
    # Eight partitions' worth of rows, and nothing close to eight partitions of
    # memory. The old code read every row into one list before scoring any.
    one_partition = PARTITION_SIZE * dim * 8  # bytes, as a float64 matrix
    assert peak < one_partition * 12


async def test_a_chunk_from_a_replaced_model_is_never_a_hit(db, index, collection_id):
    # Even at min_score=0.0, where a zero score would otherwise be admitted.
    docs = DocumentStore(db, index)
    doc_id = await _add(db, index, collection_id, "## A\n\nbody")
    await docs.replace_chunks(
        doc_id,
        collection_id,
        [
            {"ordinal": 0, "text": "stale", "heading": "A", "embedding": [1.0, 0.0, 0.0]},
            {"ordinal": 1, "text": "current", "heading": "A", "embedding": [1.0, 0.0]},
        ],
    )
    await docs.mark_indexed(doc_id, 2)

    async def embed(texts):
        return [[1.0, 0.0]], 1

    vectors, _tokens = await embed(["q"])
    hits = await index.query(collection_id, vectors[0], limit=5, min_score=0.0)
    assert [h["text"] for h in hits] == ["current"]
