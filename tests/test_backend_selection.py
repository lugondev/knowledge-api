"""Which index a configuration gets, decided in one place."""

from __future__ import annotations

import pytest

from kbase.db import Database
from kbase.index import SqlScanIndex, choose_index
from kbase.settings import Settings


@pytest.fixture
async def db(tmp_path):
    d = Database(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    yield d
    await d.dispose()


def _settings(**env):
    return Settings.from_env(
        {"KB_API_KEYS": "k:acme", "KB_EMBED_BASE_URL": "http://e/v1", "KB_EMBED_MODEL": "m", **env}
    )


async def test_sqlite_always_scans(db):
    s = _settings(KB_DATABASE_URL="sqlite+aiosqlite:///x.db", KB_EMBED_DIM="1536")
    assert isinstance(choose_index(db, s), SqlScanIndex)


async def test_postgres_without_a_dimension_scans(db):
    s = _settings(KB_DATABASE_URL="postgresql+asyncpg://u:p@h/db")
    assert isinstance(choose_index(db, s), SqlScanIndex)


async def test_postgres_with_a_dimension_uses_pgvector(db):
    from kbase.pgindex import PgVectorIndex

    s = _settings(KB_DATABASE_URL="postgresql+asyncpg://u:p@h/db", KB_EMBED_DIM="1536")
    assert isinstance(choose_index(db, s), PgVectorIndex)


async def test_the_scan_backend_asks_sqlite_nothing(db):
    """The migrated-column guard is a Postgres query; SQLite has no such type.

    Boot on SQLite must not run it -- and must not need the `postgres` extra
    installed to find that out.
    """
    await SqlScanIndex(db).create_schema()
