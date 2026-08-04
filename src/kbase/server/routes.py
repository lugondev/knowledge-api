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
from pydantic import ValidationError

# Starlette's, not FastAPI's: `request.form()` yields the base class, and
# `fastapi.UploadFile` is a subclass of it, so an isinstance check against the
# subclass rejects every real upload.
from starlette.datastructures import UploadFile

from kbase.indexer import index_document
from kbase.search import search_collection
from kbase.server.auth import require_tenant
from kbase.store import CollectionStore, DocumentStore
from kbase.types import (
    MAX_FILENAME,
    MAX_MIME,
    MAX_TITLE,
    CollectionOut,
    CreateCollection,
    CreateTextDocument,
    DocumentOut,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()


def _stores(request: Request) -> tuple[CollectionStore, DocumentStore]:
    db = request.app.state.db
    return CollectionStore(db), DocumentStore(db)


def _clean_name(raw: str) -> str:
    """The same trim a request body gets, for the names that arrive elsewhere.

    A collection is named in four places -- a JSON field, a form field, a query
    parameter and a path segment -- and only the first goes through pydantic.
    Trimming in one and not the others is how `" faq "` created a collection that
    every later request spelling it the same way could not find.
    """
    name = raw.strip()
    if not name:
        raise HTTPException(status_code=422, detail="collection is required")
    return name


async def _collection_id_or_404(request: Request, tenant: str, name: str) -> str:
    cols, _ = _stores(request)
    cid = await cols.resolve_id(tenant, name)
    if cid is None:
        raise HTTPException(status_code=404, detail="collection not found")
    return cid


@router.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@router.post("/v1/collections", status_code=201, response_model=CollectionOut)
async def create_collection(
    body: CreateCollection, request: Request, tenant: str = Depends(require_tenant)
) -> dict:
    cols, _ = _stores(request)
    return await cols.create(tenant, body.name)


@router.get("/v1/collections", response_model=list[CollectionOut])
async def list_collections(request: Request, tenant: str = Depends(require_tenant)) -> list[dict]:
    cols, _ = _stores(request)
    return await cols.list(tenant)


@router.delete("/v1/collections/{name}", status_code=204)
async def delete_collection(
    name: str, request: Request, tenant: str = Depends(require_tenant)
) -> Response:
    cols, _ = _stores(request)
    if not await cols.delete(tenant, _clean_name(name)):
        raise HTTPException(status_code=404, detail="collection not found")
    return Response(status_code=204)


@router.post("/v1/documents", response_model=DocumentOut)
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

    202 means the bytes were accepted and indexing is queued; 200 means these
    exact bytes are already indexed here and nothing was scheduled. A previous
    attempt that failed comes back 202: the same upload is how a caller retries.
    """
    settings = request.app.state.settings
    content_type = (request.headers.get("content-type") or "").lower()

    # Refuse on the declared length before reading anything. This is the cheap
    # rejection for an honest client; `MaxBodySize` is what actually holds the
    # line, because Content-Length is absent under chunked transfer and a client
    # is free to lie.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"document exceeds KB_MAX_UPLOAD_BYTES ({settings.max_upload_bytes})",
        )

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="file is required")
        coll_name = _clean_name(str(form.get("collection") or ""))
        data = await upload.read()
        # Truncated to what the column holds. A filename longer than that is
        # accepted silently by SQLite and rejected outright by Postgres, and
        # neither is worth failing an otherwise good upload over.
        filename = (upload.filename or "")[:MAX_FILENAME]
        mime = (upload.content_type or "")[:MAX_MIME]
        doc_title = (str(form.get("title") or "") or filename)[:MAX_TITLE]
    else:
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(status_code=422, detail="body must be JSON or multipart") from None
        try:
            body = CreateTextDocument.model_validate(payload)
        except ValidationError as exc:
            # This route validates by hand, so pydantic's error is not the
            # RequestValidationError FastAPI knows how to turn into a 422 -- left
            # to escape, a body missing `text` is a 500 and reads like a service
            # fault rather than a bad request.
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
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
    doc, accepted = await docs.create(
        cid,
        title=doc_title,
        filename=filename,
        mime=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )
    if not accepted:
        response.status_code = 200
        return doc
    background.add_task(
        index_document, request.app.state.db, doc["id"], embed=request.app.state.embedder
    )
    response.status_code = 202
    return doc


@router.get("/v1/documents", response_model=list[DocumentOut])
async def list_documents(
    collection: str,
    request: Request,
    status: str | None = None,
    tenant: str = Depends(require_tenant),
) -> list[dict]:
    # 404 rather than an empty list, for the same reason search and delete give
    # one: an empty list cannot be told apart from a collection this tenant does
    # not have, and a caller debugging a typo needs to know which it is. It
    # discloses nothing -- the lookup was already scoped to the credential.
    cid = await _collection_id_or_404(request, tenant, _clean_name(collection))
    _, docs = _stores(request)
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


@router.get("/v1/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str, request: Request, tenant: str = Depends(require_tenant)
) -> dict:
    return await _owned_document_or_404(request, tenant, document_id)


@router.post("/v1/documents/{document_id}/reindex", status_code=202, response_model=DocumentOut)
async def reindex_document(
    document_id: str,
    request: Request,
    background: BackgroundTasks,
    tenant: str = Depends(require_tenant),
) -> dict:
    """Index the document again from the bytes already stored.

    This is what the uploaded file is kept for. Re-uploading it works too, but a
    document that failed at 19 MB should not have to cross the network twice to
    be retried, and after a change of embedding model an operator wants the
    corpus re-embedded without holding any of it locally.
    """
    await _owned_document_or_404(request, tenant, document_id)
    _, docs = _stores(request)
    doc = await docs.mark_pending(document_id)
    if doc is None:  # deleted between the ownership check and here
        raise HTTPException(status_code=404, detail="document not found")
    background.add_task(
        index_document, request.app.state.db, document_id, embed=request.app.state.embedder
    )
    return doc


@router.delete("/v1/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str, request: Request, tenant: str = Depends(require_tenant)
) -> Response:
    await _owned_document_or_404(request, tenant, document_id)
    _, docs = _stores(request)
    await docs.delete(document_id)
    return Response(status_code=204)


@router.post("/v1/search", response_model=SearchResponse)
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
