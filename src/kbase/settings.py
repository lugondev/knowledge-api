"""Configuration, and the checks `kb doctor` runs before anything depends on it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./kbase.db"
DEFAULT_MAX_UPLOAD_BYTES = 20_000_000


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
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    docs_enabled: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        raw_max = env.get("KB_MAX_UPLOAD_BYTES", "").strip()
        try:
            max_upload = int(raw_max) if raw_max else DEFAULT_MAX_UPLOAD_BYTES
        except ValueError:
            max_upload = DEFAULT_MAX_UPLOAD_BYTES
        return cls(
            api_keys=_parse_api_keys(env.get("KB_API_KEYS", "")),
            database_url=env.get("KB_DATABASE_URL", "").strip() or DEFAULT_DATABASE_URL,
            embed_base_url=env.get("KB_EMBED_BASE_URL", "").strip(),
            embed_api_key=env.get("KB_EMBED_API_KEY", "").strip(),
            embed_model=env.get("KB_EMBED_MODEL", "").strip(),
            max_upload_bytes=max_upload,
            docs_enabled=env.get("KB_DOCS", "").strip().lower() not in {"false", "0", "no"},
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
        if self.max_upload_bytes <= 0:
            problems.append("KB_MAX_UPLOAD_BYTES must be a positive integer")
        return problems
