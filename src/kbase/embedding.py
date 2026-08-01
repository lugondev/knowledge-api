"""The embedding call, batched, order-preserving, and loud about mismatches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx

from kbase.errors import EmbeddingError

Embedder = Callable[[list[str]], Awaitable[tuple[list[list[float]], int]]]

DEFAULT_TIMEOUT = 60.0


def make_embedder(
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int = 32,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Embedder:
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def embed(texts: list[str]) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0
        vectors: list[list[float]] = []
        tokens = 0
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                try:
                    resp = await client.post(
                        url, headers=headers, json={"model": model, "input": batch}
                    )
                    resp.raise_for_status()
                    body = resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise EmbeddingError(f"embedding request failed: {exc}") from exc
                data = body.get("data") or []
                if len(data) != len(batch):
                    # Zipping a short reply onto the batch would attach one chunk's
                    # vector to a different chunk's text, and every later search
                    # would be quietly wrong with no error anywhere.
                    raise EmbeddingError(
                        f"embedding provider returned {len(data)} vectors for {len(batch)} inputs"
                    )
                vectors.extend(d["embedding"] for d in data)
                tokens += int((body.get("usage") or {}).get("prompt_tokens") or 0)
        return vectors, tokens

    return embed
