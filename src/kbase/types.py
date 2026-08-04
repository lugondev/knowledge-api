"""What crosses the wire, in both directions. Tenant is deliberately absent from
every request: it comes from the credential, and a field for it would only invite
a caller to claim one.

The lengths here are not arbitrary -- each one matches the column that ends up
holding the value. SQLite ignores a `VARCHAR(n)`, so an overlong title is stored
happily and the same request is a 500 the day the deployment moves to the
Postgres URL this project already ships an extra for.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

MAX_NAME = 128
MAX_TITLE = 512
MAX_FILENAME = 512
MAX_MIME = 128
MAX_QUERY = 8192


def _clean_name(v: str) -> str:
    v = v.strip()
    if not v or "/" in v:
        raise ValueError("name must be non-empty and must not contain '/'")
    return v


#: Trimmed in one place, so a collection cannot be created under a name that no
#: later request can spell. `" faq "` used to be stored as `faq` and then 404 on
#: every upload that named it the way it was created.
CollectionName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_NAME),
    AfterValidator(_clean_name),
]


class CreateCollection(BaseModel):
    name: CollectionName


class CreateTextDocument(BaseModel):
    collection: CollectionName
    title: str = Field(default="", max_length=MAX_TITLE)
    text: str


class SearchRequest(BaseModel):
    collection: CollectionName
    # Bounded because it is forwarded to a metered provider: an unbounded query
    # is a bill, not just a slow request.
    query: str = Field(max_length=MAX_QUERY)
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)


class CollectionOut(BaseModel):
    name: str
    document_count: int


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    mime: str
    sha256: str
    bytes_len: int
    status: str
    error: str
    chunk_count: int
    created_at: str | None
    indexed_at: str | None


class SearchHit(BaseModel):
    text: str
    score: float
    document_id: str
    title: str
    filename: str
    heading: str


class Usage(BaseModel):
    prompt_tokens: int = 0


class SearchResponse(BaseModel):
    chunks: list[SearchHit]
    usage: Usage
