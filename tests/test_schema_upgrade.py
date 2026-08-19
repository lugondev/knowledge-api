"""A database written before a column existed still has to work."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from kbase.db import Database
from kbase.index import SqlScanIndex
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


async def test_missing_collection_id_column_is_added_and_backfilled(db, index):
    cols = CollectionStore(db, index)
    await cols.create("acme", "faq")
    cid = await cols.resolve_id("acme", "faq")
    docs = DocumentStore(db, index)
    doc, _ = await docs.create(
        cid, title="t", filename="f.md", mime="text/markdown", sha256="a" * 64, data=b"x"
    )
    await docs.replace_chunks(
        doc["id"], cid, [{"ordinal": 0, "text": "hi", "heading": "", "embedding": [1.0]}]
    )

    # Rewind to the schema as it was before this column existed. SQLite refuses
    # to drop an indexed column, so the index goes first.
    async with db.session() as s:
        await s.execute(text("DROP INDEX IF EXISTS ix_chunks_collection_id"))
        await s.execute(text("ALTER TABLE chunks DROP COLUMN collection_id"))
        await s.commit()

    await db.create_all()

    async with db.session() as s:
        got = (await s.execute(text("SELECT collection_id FROM chunks"))).scalars().all()
    assert got == [cid]
