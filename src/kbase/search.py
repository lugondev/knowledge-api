"""Embed the query, score every chunk in the collection, apply the floor.

On SQLite this is a scan in Python, which is what the corpus this service is
built for actually needs. The interface is the seam a pgvector implementation
slots into later without any caller noticing.
"""

from __future__ import annotations

import math

from sqlalchemy import select

from kbase.db import Database
from kbase.embedding import Embedder
from kbase.models import Chunk, Document


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def search_collection(
    db: Database,
    collection_id: str,
    query: str,
    *,
    embed: Embedder,
    limit: int = 5,
    min_score: float = 0.35,
) -> tuple[list[dict], int]:
    if not query.strip():
        return [], 0
    vectors, tokens = await embed([query])
    if not vectors:
        return [], tokens
    qvec = vectors[0]

    async with db.session() as s:
        rows = (
            await s.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.collection_id == collection_id, Document.status == "indexed")
            )
        ).all()

    scored: list[dict] = []
    for chunk, doc in rows:
        score = cosine(qvec, list(chunk.embedding or []))
        if score < min_score:
            continue
        scored.append(
            {
                "text": chunk.text,
                "score": score,
                "document_id": doc.id,
                "title": doc.title,
                "filename": doc.filename,
                "heading": chunk.heading,
            }
        )
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored[:limit], tokens
