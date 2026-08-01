"""What crosses the wire. Tenant is deliberately absent: it comes from the
credential, and a field for it would only invite a caller to claim one."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CreateCollection(BaseModel):
    name: str = Field(min_length=1, max_length=128)

    @field_validator("name")
    @classmethod
    def _no_slashes(cls, v: str) -> str:
        v = v.strip()
        if not v or "/" in v:
            raise ValueError("name must be non-empty and must not contain '/'")
        return v


class CreateTextDocument(BaseModel):
    collection: str
    title: str = ""
    text: str


class SearchRequest(BaseModel):
    collection: str
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)


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
