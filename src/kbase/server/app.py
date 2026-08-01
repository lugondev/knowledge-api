"""The application object, and the state every route reads from."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from kbase.db import Database
from kbase.embedding import Embedder, make_embedder
from kbase.server.routes import router
from kbase.settings import Settings


def create_app(settings: Settings, *, embedder: Embedder | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_url)
        await db.create_all()
        app.state.db = db
        try:
            yield
        finally:
            await db.dispose()

    app = FastAPI(
        title="kbase",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.embedder = embedder or make_embedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    app.include_router(router)
    return app
