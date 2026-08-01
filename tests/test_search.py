import hashlib

import pytest

from kbase.db import Database
from kbase.indexer import index_document
from kbase.search import cosine, search_collection
from kbase.store import CollectionStore, DocumentStore


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


def keyword_embedder(vocabulary: list[str]):
    """A deterministic stand-in: one dimension per vocabulary word, set when the
    word appears. Cosine over it behaves like real embeddings for these tests
    without a network call or a model."""

    async def embed(texts):
        vectors = []
        for t in texts:
            low = t.lower()
            vectors.append([1.0 if w in low else 0.0 for w in vocabulary])
        return vectors, len(texts)

    return embed


VOCAB = ["bảo hành", "đổi trả", "giao hàng", "mèo"]


async def _seed(db, tenant: str, name: str, body: str) -> str:
    cols = CollectionStore(db)
    await cols.create(tenant, name)
    cid = await cols.resolve_id(tenant, name)
    data = body.encode()
    doc, _ = await DocumentStore(db).create(
        cid, title="Sổ tay", filename="s.md", mime="text/markdown",
        sha256=hashlib.sha256(data).hexdigest(), data=data,
    )
    await index_document(db, doc["id"], embed=keyword_embedder(VOCAB))
    return cid


def test_cosine_basics():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


async def test_finds_the_relevant_chunk(db):
    cid = await _seed(
        db, "acme", "faq",
        "## Bảo hành\n\nbảo hành mười hai tháng.\n\n## Giao hàng\n\ngiao hàng 3 ngày.\n",
    )
    hits, tokens = await search_collection(
        db, cid, "bảo hành", embed=keyword_embedder(VOCAB), limit=5, min_score=0.1
    )
    assert hits
    assert "mười hai tháng" in hits[0]["text"]
    assert hits[0]["heading"] == "Bảo hành"
    assert hits[0]["title"] == "Sổ tay"
    assert tokens == 1


async def test_unrelated_query_returns_nothing_thanks_to_the_floor(db):
    # Without a floor top-k always returns something, and the assistant reads
    # the warranty policy out loud in answer to a question about cats.
    cid = await _seed(db, "acme", "faq", "## Bảo hành\n\nbảo hành mười hai tháng.\n")
    hits, _ = await search_collection(
        db, cid, "mèo", embed=keyword_embedder(VOCAB), limit=5, min_score=0.35
    )
    assert hits == []


async def test_limit_is_respected(db):
    body = "\n\n".join(f"## M{i}\n\nbảo hành mục {i}" for i in range(10))
    cid = await _seed(db, "acme", "faq", body)
    hits, _ = await search_collection(
        db, cid, "bảo hành", embed=keyword_embedder(VOCAB), limit=3, min_score=0.1
    )
    assert len(hits) == 3


async def test_results_are_sorted_by_descending_score(db):
    cid = await _seed(
        db, "acme", "faq",
        "## A\n\nbảo hành đổi trả\n\n## B\n\nbảo hành\n",
    )
    hits, _ = await search_collection(
        db, cid, "bảo hành đổi trả", embed=keyword_embedder(VOCAB), limit=5, min_score=0.1
    )
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_search_never_crosses_collections(db):
    acme = await _seed(db, "acme", "faq", "## Bảo hành\n\nbảo hành của acme\n")
    globex = await _seed(db, "globex", "faq", "## Bảo hành\n\nbảo hành của globex\n")
    hits, _ = await search_collection(
        db, acme, "bảo hành", embed=keyword_embedder(VOCAB), limit=5, min_score=0.1
    )
    assert all("globex" not in h["text"] for h in hits)
    assert globex != acme


async def test_empty_collection_returns_no_hits(db):
    cols = CollectionStore(db)
    await cols.create("acme", "empty")
    cid = await cols.resolve_id("acme", "empty")
    hits, _ = await search_collection(
        db, cid, "bảo hành", embed=keyword_embedder(VOCAB), limit=5, min_score=0.1
    )
    assert hits == []
