"""extract -> chunk -> embed -> finalize, and never in a half state.

Chunks are written in one shot at the end rather than batch by batch. Writing as
each batch returns would leave a document that failed on batch 7 holding the
first six batches' chunks; those chunks are indistinguishable from a complete
document at search time, so the assistant answers from the first third of the
manual and never mentions that the rest is missing.
"""

from __future__ import annotations

import logging

from kbase.chunker import chunk_markdown
from kbase.db import Database
from kbase.embedding import Embedder
from kbase.errors import EmbeddingError, KbError
from kbase.extract import extract_text
from kbase.index import ChunkIndex
from kbase.store import DocumentStore

logger = logging.getLogger(__name__)

EMPTY_DOCUMENT = "document contains no text to index"


def _tenant_reason(exc: BaseException) -> str:
    """What goes on the row, as opposed to what goes in the log.

    Anything not raised on purpose is reported without its own message: a driver
    or library error carries connection strings, SQL, and hostnames, and every
    one of them would otherwise be readable through `GET /v1/documents/{id}`.
    """
    if isinstance(exc, KbError):
        return exc.safe_message
    return "indexing failed unexpectedly; the service log has the detail"


async def index_document(
    db: Database,
    index: ChunkIndex,
    document_id: str,
    *,
    embed: Embedder,
    max_chars: int = 800,
    overlap: int = 100,
) -> None:
    """Index one document. Never raises: a failure is recorded on the row."""
    docs = DocumentStore(db, index)
    doc = await docs.get(document_id)
    if doc is None:
        logger.warning("index requested for unknown document %s", document_id)
        return
    try:
        data = await docs.raw_bytes(document_id) or b""
        text = extract_text(data, filename=doc["filename"], mime=doc["mime"])
        pieces = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
        if not pieces:
            # `indexed` with zero chunks reads as success and answers nothing:
            # every search against it comes back empty, with no reason given
            # anywhere. A document that cannot be searched was not indexed.
            await docs.drop_chunks(document_id)
            await docs.mark_failed(document_id, EMPTY_DOCUMENT)
            return
        vectors, _tokens = await embed([p.text for p in pieces])
        if len(vectors) != len(pieces):
            raise EmbeddingError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        collection_id = await docs.owner_collection_id(document_id)
        if collection_id is None:
            logger.info("document %s was deleted while it was being indexed", document_id)
            return
        written = await docs.finish_indexing(
            document_id,
            collection_id,
            [
                {
                    "ordinal": p.ordinal,
                    "text": p.text,
                    "heading": p.heading,
                    "embedding": v,
                }
                for p, v in zip(pieces, vectors, strict=True)
            ],
        )
        if not written:
            # Deleted while we were embedding. No row is left to mark, and the
            # chunks were refused rather than written, so nothing is orphaned.
            logger.info("document %s was deleted while it was being indexed", document_id)
            return
    except Exception as exc:  # noqa: BLE001 - the failure belongs on the row
        logger.warning("indexing document %s failed: %s", document_id, exc, exc_info=True)
        await docs.drop_chunks(document_id)
        await docs.mark_failed(document_id, _tenant_reason(exc))
