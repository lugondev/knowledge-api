"""Every endpoint. A collection is always resolved through the caller's tenant
before anything else happens, so no handler can be reached with someone else's
row in hand."""

from __future__ import annotations

import hashlib

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
)

# Starlette's, not FastAPI's: `request.form()` yields the base class, and
# `fastapi.UploadFile` is a subclass of it, so an isinstance check against the
# subclass rejects every real upload.
from starlette.datastructures import UploadFile

from kbase.indexer import index_document
from kbase.search import search_collection
from kbase.server.auth import require_tenant
from kbase.store import CollectionStore, DocumentStore
from kbase.types import CreateCollection, CreateTextDocument, SearchRequest

router = APIRouter()


def _stores(request: Request) -> tuple[CollectionStore, DocumentStore]:
    db = request.app.state.db
    return CollectionStore(db), DocumentStore(db)


async def _collection_id_or_404(request: Request, tenant: str, name: str) -> str:
    cols, _ = _stores(request)
    cid = await cols.resolve_id(tenant, name)
    if cid is None:
        raise HTTPException(status_code=404, detail="collection not found")
    return cid


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@router.post("/v1/collections", status_code=201)
async def create_collection(
    body: CreateCollection, request: Request, tenant: str = Depends(require_tenant)
) -> dict:
    cols, _ = _stores(request)
    return await cols.create(tenant, body.name)


@router.get("/v1/collections")
async def list_collections(
    request: Request, tenant: str = Depends(require_tenant)
) -> list[dict]:
    cols, _ = _stores(request)
    return await cols.list(tenant)


@router.delete("/v1/collections/{name}", status_code=204)
async def delete_collection(
    name: str, request: Request, tenant: str = Depends(require_tenant)
) -> Response:
    cols, _ = _stores(request)
    if not await cols.delete(tenant, name):
        raise HTTPException(status_code=404, detail="collection not found")
    return Response(status_code=204)


@router.post("/v1/documents")
async def create_document(
    request: Request,
    background: BackgroundTasks,
    response: Response,
    tenant: str = Depends(require_tenant),
) -> dict:
    """Accepts multipart (a file) or JSON (pasted text).

    The content type is dispatched by hand rather than by declaring both `File`
    and `Form` parameters. With optional form parameters declared, FastAPI parses
    every request body as a form first -- including the JSON ones -- and the two
    shapes end up fighting over the same body. Reading the header and choosing is
    both shorter and unambiguous.
    """
    settings = request.app.state.settings
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="file is required")
        coll_name = str(form.get("collection") or "")
        if not coll_name:
            raise HTTPException(status_code=422, detail="collection is required")
        data = await upload.read()
        filename = upload.filename or ""
        mime = upload.content_type or ""
        doc_title = str(form.get("title") or "") or filename
    else:
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(
                status_code=422, detail="body must be JSON or multipart"
            ) from None
        body = CreateTextDocument.model_validate(payload)
        data = body.text.encode("utf-8")
        filename = "text.md"
        mime = "text/markdown"
        doc_title = body.title
        coll_name = body.collection

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"document exceeds KB_MAX_UPLOAD_BYTES ({settings.max_upload_bytes})",
        )

    cid = await _collection_id_or_404(request, tenant, coll_name)
    _, docs = _stores(request)
    doc, created = await docs.create(
        cid,
        title=doc_title,
        filename=filename,
        mime=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )
    if not created:
        response.status_code = 200
        return doc
    background.add_task(
        index_document, request.app.state.db, doc["id"], embed=request.app.state.embedder
    )
    response.status_code = 202
    return doc


@router.get("/v1/documents")
async def list_documents(
    collection: str,
    request: Request,
    status: str | None = None,
    tenant: str = Depends(require_tenant),
) -> list[dict]:
    cols, docs = _stores(request)
    cid = await cols.resolve_id(tenant, collection)
    if cid is None:
        return []
    return await docs.list(cid, status=status)


async def _owned_document_or_404(request: Request, tenant: str, document_id: str) -> dict:
    cols, docs = _stores(request)
    doc = await docs.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    cid = await docs.owner_collection_id(document_id)
    if cid is None or not await cols.owns(tenant, cid):
        # 404 rather than 403: confirming a document exists but belongs to
        # someone else is itself a leak.
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@router.get("/v1/documents/{document_id}")
async def get_document(
    document_id: str, request: Request, tenant: str = Depends(require_tenant)
) -> dict:
    return await _owned_document_or_404(request, tenant, document_id)


@router.delete("/v1/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str, request: Request, tenant: str = Depends(require_tenant)
) -> Response:
    await _owned_document_or_404(request, tenant, document_id)
    _, docs = _stores(request)
    await docs.delete(document_id)
    return Response(status_code=204)


@router.post("/v1/search")
async def search(
    body: SearchRequest, request: Request, tenant: str = Depends(require_tenant)
) -> dict:
    cid = await _collection_id_or_404(request, tenant, body.collection)
    hits, tokens = await search_collection(
        request.app.state.db,
        cid,
        body.query,
        embed=request.app.state.embedder,
        limit=body.limit,
        min_score=body.min_score,
    )
    return {"chunks": hits, "usage": {"prompt_tokens": tokens}}
