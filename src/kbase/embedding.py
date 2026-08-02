"""The embedding call, batched, order-preserving, and loud about mismatches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx

from kbase.errors import EmbeddingError

Embedder = Callable[[list[str]], Awaitable[tuple[list[list[float]], int]]]

DEFAULT_TIMEOUT = 60.0


class HttpEmbedder:
    """One client for the process, not one per call.

    A search embeds its query before it can answer, and the caller this service
    exists for issues a search on every conversational turn. Building an
    `AsyncClient` per call throws away the connection pool each time, so every
    turn pays a fresh TCP and TLS handshake to the provider -- tens to hundreds
    of milliseconds, on the one path where latency is audible.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 32,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._batch_size = batch_size
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                resp = await self._client.post(
                    self._url, headers=self._headers, json={"model": self._model, "input": batch}
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

    async def aclose(self) -> None:
        await self._client.aclose()


def make_embedder(
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int = 32,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> HttpEmbedder:
    return HttpEmbedder(
        base_url=base_url,
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        timeout=timeout,
        transport=transport,
    )
