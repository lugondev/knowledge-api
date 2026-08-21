"""Configuration, and the checks `kb doctor` runs before anything depends on it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./kbase.db"
DEFAULT_MAX_UPLOAD_BYTES = 20_000_000

#: Inputs per /embeddings request. Providers cap this and they do not agree:
#: OpenAI takes hundreds, DashScope's text-embedding-v3 rejects anything over
#: 10 with a 400. Hardcoded at 32 this made a whole provider unusable in a way
#: that only showed on indexing -- a one-item query embed still worked, so the
#: service looked healthy while every document failed.
DEFAULT_EMBED_BATCH = 32

#: pgvector will not build an HNSW index on a wider vector. Above this a column
#: still stores and still searches -- by scanning, which is the thing pgvector
#: was brought in to stop doing.
HNSW_MAX_DIMENSIONS = 2000


def _parse_api_keys(raw: str) -> dict[str, str]:
    """`key:tenant,key:tenant` -> {key: tenant}.

    An entry without a colon is dropped rather than guessed at. Treating a bare
    string as "a key with some default tenant" is how one tenant ends up reading
    another's collections, so a malformed entry simply does not grant access.
    """
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        key, _, tenant = entry.partition(":")
        key, tenant = key.strip(), tenant.strip()
        if key and tenant:
            out[key] = tenant
    return out


@dataclass(frozen=True)
class Settings:
    api_keys: dict[str, str] = field(default_factory=dict)
    database_url: str = DEFAULT_DATABASE_URL
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    embed_batch: int = DEFAULT_EMBED_BATCH
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    docs_enabled: bool = True
    embed_dim: int = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        raw_max = env.get("KB_MAX_UPLOAD_BYTES", "").strip()
        try:
            max_upload = int(raw_max) if raw_max else DEFAULT_MAX_UPLOAD_BYTES
        except ValueError:
            max_upload = DEFAULT_MAX_UPLOAD_BYTES
        raw_batch = env.get("KB_EMBED_BATCH", "").strip()
        try:
            embed_batch = int(raw_batch) if raw_batch else DEFAULT_EMBED_BATCH
        except ValueError:
            embed_batch = -1  # invalid, and `check` says so rather than guessing
        raw_dim = env.get("KB_EMBED_DIM", "").strip()
        try:
            embed_dim = int(raw_dim) if raw_dim else 0
        except ValueError:
            embed_dim = -1  # invalid, and `check` says so rather than guessing
        return cls(
            api_keys=_parse_api_keys(env.get("KB_API_KEYS", "")),
            database_url=env.get("KB_DATABASE_URL", "").strip() or DEFAULT_DATABASE_URL,
            embed_base_url=env.get("KB_EMBED_BASE_URL", "").strip(),
            embed_api_key=env.get("KB_EMBED_API_KEY", "").strip(),
            embed_model=env.get("KB_EMBED_MODEL", "").strip(),
            embed_batch=embed_batch,
            max_upload_bytes=max_upload,
            docs_enabled=env.get("KB_DOCS", "").strip().lower() not in {"false", "0", "no"},
            embed_dim=embed_dim,
        )

    def check(self) -> list[str]:
        """Everything wrong that can be known without making a request."""
        problems: list[str] = []
        if not self.api_keys:
            problems.append("KB_API_KEYS is unset: every request will be rejected with 401")
        if not self.embed_base_url:
            problems.append("KB_EMBED_BASE_URL is unset: nothing can be indexed or searched")
        if not self.embed_model:
            problems.append("KB_EMBED_MODEL is unset: nothing can be indexed or searched")
        if self.embed_batch <= 0:
            problems.append("KB_EMBED_BATCH must be a positive integer")
        if self.max_upload_bytes <= 0:
            problems.append("KB_MAX_UPLOAD_BYTES must be a positive integer")
        if self.embed_dim < 0:
            problems.append("KB_EMBED_DIM must be a positive integer")
        return problems

    def warnings(self) -> list[str]:
        """Things worth saying that are not reasons to refuse to start."""
        notes: list[str] = []
        if self.database_url.startswith("postgresql") and self.embed_dim == 0:
            # Not "will scan", flatly: if this database was ever started *with*
            # the variable set, its `chunks.embedding` is a `vector` column the
            # scanning backend cannot read, and the service refuses to start
            # (`SqlScanIndex.create_schema`) rather than searching an intact
            # corpus and finding nothing in it.
            notes.append(
                "KB_EMBED_DIM is unset on a Postgres database: searches will scan the "
                "collection rather than use a vector index -- and if this database was "
                "ever migrated by a previous start with it set, the service will refuse "
                "to start until it is set again"
            )
        if self.embed_dim > HNSW_MAX_DIMENSIONS:
            notes.append(
                f"KB_EMBED_DIM is {self.embed_dim}: pgvector builds no HNSW index above "
                f"{HNSW_MAX_DIMENSIONS} dimensions, so searches will scan. Reduce the "
                "model's output width instead"
            )
        return notes
