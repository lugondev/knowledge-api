"""The failure modes a green suite still had, found by reading the code and by
driving the running service rather than the TestClient."""

import asyncio
import hashlib

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from kbase.db import Database
from kbase.embedding import make_embedder
from kbase.models import Document
from kbase.server.app import create_app
from kbase.settings import Settings
from kbase.store import CollectionStore, DocumentStore

# --- a malformed JSON body is a bad request, not a service fault -------------


def test_missing_required_field_is_422_not_500(client, acme):
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)
    r = client.post("/v1/documents", json={"collection": "faq"}, headers=acme)
    assert r.status_code == 422
    assert "text" in r.text


def test_wrong_field_type_is_422(client, acme):
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)
    r = client.post(
        "/v1/documents", json={"collection": "faq", "text": {"not": "a string"}}, headers=acme
    )
    assert r.status_code == 422


# --- the size limit applies before the body is read --------------------------


def test_oversized_body_is_refused_on_its_declared_length(client, acme):
    client.post("/v1/collections", json={"name": "faq"}, headers=acme)
    # Content-Length says 5000, over the fixture's KB_MAX_UPLOAD_BYTES of 1000.
    r = client.post(
        "/v1/documents",
        content=b"x" * 5000,
        headers={**acme, "Content-Type": "application/json"},
    )
    assert r.status_code == 413


# --- metadata reads must not drag the file through memory --------------------


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


async def _seed_doc(db, blob: bytes) -> tuple[str, str]:
    cols = CollectionStore(db)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    doc, _ = await DocumentStore(db).create(
        cid,
        title="t",
        filename="a.md",
        mime="text/markdown",
        sha256=hashlib.sha256(blob).hexdigest(),
        data=blob,
    )
    return cid, doc["id"]


async def test_listing_documents_does_not_load_their_bytes(db):
    cid, _ = await _seed_doc(db, b"y" * 100_000)
    async with db.session() as s:
        rows = (
            (await s.execute(select(Document).where(Document.collection_id == cid))).scalars().all()
        )
        # `data` is deferred: it is absent from the loaded state until asked for.
        assert "data" in inspect(rows[0]).unloaded


async def test_raw_bytes_still_returns_the_file(db):
    _cid, doc_id = await _seed_doc(db, b"y" * 1000)
    assert await DocumentStore(db).raw_bytes(doc_id) == b"y" * 1000


async def test_owner_collection_id_still_resolves(db):
    cid, doc_id = await _seed_doc(db, b"y" * 1000)
    assert await DocumentStore(db).owner_collection_id(cid) is None
    assert await DocumentStore(db).owner_collection_id(doc_id) == cid


# --- one HTTP client for the process, not one per call -----------------------


async def test_embedder_reuses_one_connection_pool():
    # Two calls, one pool: rebuilding the client per call throws away every
    # kept-alive connection, and this service embeds once per conversational turn.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"embedding": [1.0]}], "usage": {"prompt_tokens": 1}}
        )

    embed = make_embedder(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    await embed(["a"])
    first = embed._client
    await embed(["b"])
    assert embed._client is first
    assert not first.is_closed
    await embed.aclose()
    assert first.is_closed


def test_app_shutdown_closes_the_embedding_client(tmp_path):
    s = Settings.from_env(
        {
            "KB_API_KEYS": "k:acme",
            "KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/x.db",
            "KB_EMBED_BASE_URL": "http://x/v1",
            "KB_EMBED_MODEL": "m",
        }
    )
    app = create_app(s)
    with TestClient(app) as c:
        c.get("/healthz")
    assert app.state.embedder._client.is_closed


# --- a restart must not leave documents pending forever ----------------------


async def test_startup_fails_documents_left_pending_by_a_restart(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/restart.db"
    db = Database(url)
    await db.create_all()
    _cid, doc_id = await _seed_doc(db, b"## A\n\nbody")
    assert (await DocumentStore(db).get(doc_id))["status"] == "pending"
    await db.dispose()

    s = Settings.from_env(
        {
            "KB_API_KEYS": "k:acme",
            "KB_DATABASE_URL": url,
            "KB_EMBED_BASE_URL": "http://x/v1",
            "KB_EMBED_MODEL": "m",
        }
    )
    with TestClient(create_app(s)) as c:
        r = c.get(f"/v1/documents/{doc_id}", headers={"Authorization": "Bearer k"})
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "interrupted" in r.json()["error"]


# --- an idempotent create stays idempotent under concurrency -----------------


async def test_concurrent_identical_collection_creates_do_not_raise(db):
    store = CollectionStore(db)
    results = await asyncio.gather(
        *(store.create("acme", "faq") for _ in range(4)), return_exceptions=True
    )
    assert all(not isinstance(r, Exception) for r in results), results
    assert len(await store.list("acme")) == 1


async def test_concurrent_identical_document_uploads_do_not_raise(db):
    cols = CollectionStore(db)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db)
    blob = b"## A\n\nbody"
    digest = hashlib.sha256(blob).hexdigest()
    results = await asyncio.gather(
        *(
            docs.create(
                cid,
                title="t",
                filename="a.md",
                mime="text/markdown",
                sha256=digest,
                data=blob,
            )
            for _ in range(4)
        ),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), results
    assert len(await docs.list(cid)) == 1
    assert sum(1 for _doc, created in results if created) == 1
