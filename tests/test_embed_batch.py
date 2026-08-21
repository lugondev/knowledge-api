"""KB_EMBED_BATCH: providers cap the input array, and 32 is not universal."""

from __future__ import annotations

import httpx

from kbase.embedding import make_embedder
from kbase.settings import Settings


def test_embed_batch_defaults_to_32():
    assert Settings.from_env({}).embed_batch == 32


def test_embed_batch_is_read_from_the_environment():
    assert Settings.from_env({"KB_EMBED_BATCH": "10"}).embed_batch == 10


def test_a_non_numeric_batch_is_a_problem_not_a_guess():
    s = Settings.from_env({"KB_EMBED_BATCH": "lots"})
    assert any("KB_EMBED_BATCH" in p for p in s.check())


def test_a_zero_or_negative_batch_is_a_problem():
    assert any("KB_EMBED_BATCH" in p for p in Settings.from_env({"KB_EMBED_BATCH": "0"}).check())


async def test_the_embedder_never_sends_more_than_the_batch_size():
    """DashScope's text-embedding-v3 rejects an input array longer than 10 with
    a 400. A hardcoded 32 made that provider unusable for indexing while a
    single-item query embed still worked -- so it failed only on documents."""
    seen: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        import json

        n = len(json.loads(request.read())["input"])
        seen.append(n)
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2]} for _ in range(n)],
                "usage": {"prompt_tokens": n},
            },
        )

    embed = make_embedder(
        base_url="http://e.invalid/v1",
        api_key="k",
        model="m",
        batch_size=10,
        transport=httpx.MockTransport(handler),
    )
    vectors, _ = await embed([f"chunk {i}" for i in range(25)])

    assert len(vectors) == 25
    assert seen == [10, 10, 5]
    assert max(seen) <= 10
