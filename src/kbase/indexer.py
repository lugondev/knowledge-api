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
from kbase.errors import KbError
from kbase.extract import extract_text
from kbase.store import DocumentStore

logger = logging.getLogger(__name__)


async def index_document(
    db: Database,
    document_id: str,
    *,
    embed: Embedder,
    max_chars: int = 800,
    overlap: int = 100,
) -> None:
    """Index one document. Never raises: a failure is recorded on the row."""
    docs = DocumentStore(db)
    doc = await docs.get(document_id)
    if doc is None:
        logger.warning("index requested for unknown document %s", document_id)
        return
    try:
        data = await docs.raw_bytes(document_id) or b""
        text = extract_text(data, filename=doc["filename"], mime=doc["mime"])
        pieces = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
        if not pieces:
            await docs.drop_chunks(document_id)
            await docs.mark_indexed(document_id, 0)
            return
        vectors, _tokens = await embed([p.text for p in pieces])
        if len(vectors) != len(pieces):
            raise KbError(
                f"embedder returned {len(vectors)} vectors for {len(pieces)} chunks"
            )
        await docs.replace_chunks(
            document_id,
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
        await docs.mark_indexed(document_id, len(pieces))
    except Exception as exc:  # noqa: BLE001 - the failure belongs on the row
        logger.warning("indexing document %s failed: %s", document_id, exc)
        await docs.drop_chunks(document_id)
        await docs.mark_failed(document_id, str(exc))
