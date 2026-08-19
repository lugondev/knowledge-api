"""The application object, and the state every route reads from."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from kbase.db import Database
from kbase.embedding import Embedder, make_embedder
from kbase.index import SqlScanIndex
from kbase.server.limits import MaxBodySize
from kbase.server.routes import router
from kbase.settings import Settings
from kbase.store import DocumentStore

logger = logging.getLogger(__name__)


def create_app(settings: Settings, *, embedder: Embedder | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_url)
        await db.create_all()
        app.state.db = db
        index = SqlScanIndex(db)
        await index.create_schema()
        app.state.index = index
        # Indexing runs in a background task, so a restart mid-index leaves a
        # document `pending` with nothing left to move it along. Left alone it
        # stays pending forever: the operator sees "still working" and waits for
        # something that is never going to happen. Assumes one instance per
        # database -- with several, a restarting instance would fail work that
        # another is still doing.
        swept = await DocumentStore(db, index).fail_stale_pending(
            "indexing was interrupted by a restart; upload it again or POST "
            "/v1/documents/{id}/reindex"
        )
        if swept:
            logger.warning("marked %d interrupted document(s) as failed", swept)
        try:
            yield
        finally:
            closer = getattr(app.state.embedder, "aclose", None)
            if closer is not None:
                await closer()
            await db.dispose()

    app = FastAPI(
        title="kbase",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    # Outside the routes, because by the time a route runs, the body it is about
    # to measure has already been read into the process. It raises an
    # HTTPException, so the 413 needs no handler of its own.
    app.add_middleware(MaxBodySize, max_bytes=settings.max_upload_bytes)

    app.state.settings = settings
    app.state.embedder = embedder or make_embedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    app.include_router(router)
    return app
