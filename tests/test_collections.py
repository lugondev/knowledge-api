import pytest

from kbase.db import Database
from kbase.index import SqlScanIndex
from kbase.store import CollectionStore


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    await d.create_all()
    yield d
    await d.dispose()


@pytest.fixture
def index(db):
    return SqlScanIndex(db)


async def test_create_then_list(db, index):
    store = CollectionStore(db, index)
    await store.create("acme", "faq")
    assert await store.list("acme") == [{"name": "faq", "document_count": 0}]


async def test_same_name_under_two_tenants_are_two_rows(db, index):
    # The whole point of keeping tenant a separate dimension: `faq` belonging to
    # acme and `faq` belonging to globex must not be the same collection.
    store = CollectionStore(db, index)
    await store.create("acme", "faq")
    await store.create("globex", "faq")
    assert await store.list("acme") == [{"name": "faq", "document_count": 0}]
    assert await store.list("globex") == [{"name": "faq", "document_count": 0}]


async def test_create_is_idempotent_within_a_tenant(db, index):
    store = CollectionStore(db, index)
    first = await store.create("acme", "faq")
    again = await store.create("acme", "faq")
    assert first == again
    assert len(await store.list("acme")) == 1


async def test_get_is_scoped_to_the_tenant(db, index):
    store = CollectionStore(db, index)
    await store.create("acme", "faq")
    assert await store.get("acme", "faq") is not None
    assert await store.get("globex", "faq") is None


async def test_delete_reports_whether_anything_was_deleted(db, index):
    store = CollectionStore(db, index)
    await store.create("acme", "faq")
    assert await store.delete("acme", "faq") is True
    assert await store.delete("acme", "faq") is False
    assert await store.list("acme") == []


async def test_delete_is_scoped_to_the_tenant(db, index):
    store = CollectionStore(db, index)
    await store.create("acme", "faq")
    assert await store.delete("globex", "faq") is False
    assert await store.get("acme", "faq") is not None
