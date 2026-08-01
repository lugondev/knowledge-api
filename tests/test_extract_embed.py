import json

import httpx
import pytest

from kbase.embedding import make_embedder
from kbase.errors import EmbeddingError, ExtractError
from kbase.extract import extract_text


def test_utf8_text_decodes():
    assert extract_text("Bảo hành".encode(), filename="a.txt") == "Bảo hành"


def test_markdown_decodes():
    assert extract_text(b"# Title\n\nbody", filename="a.md").startswith("# Title")


def test_undecodable_bytes_raise_extract_error():
    with pytest.raises(ExtractError):
        extract_text(b"\xff\xfe\x00\x00\xff", filename="a.txt")


def test_unsupported_type_names_itself():
    with pytest.raises(ExtractError) as exc:
        extract_text(b"%PDF-1.4 ...", filename="manual.pdf")
    assert "pdf" in str(exc.value).lower()


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_embedder_batches_and_sums_tokens():
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        seen.append(len(body["input"]))
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2]} for _ in body["input"]],
                "usage": {"prompt_tokens": 7},
            },
        )

    embed = make_embedder(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        batch_size=2,
        transport=_transport(handler),
    )
    vectors, tokens = await embed(["a", "b", "c"])
    assert seen == [2, 1]  # batched, not one giant request
    assert len(vectors) == 3
    assert tokens == 14  # summed across batches, not taken from the last


async def test_embedder_preserves_input_order():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [float(len(t))]} for t in body["input"]],
                "usage": {"prompt_tokens": 1},
            },
        )

    embed = make_embedder(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        batch_size=2,
        transport=_transport(handler),
    )
    vectors, _ = await embed(["a", "bb", "ccc", "dddd"])
    assert vectors == [[1.0], [2.0], [3.0], [4.0]]


async def test_provider_error_becomes_embedding_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"})

    embed = make_embedder(
        base_url="http://x/v1", api_key="k", model="m", transport=_transport(handler)
    )
    with pytest.raises(EmbeddingError):
        await embed(["a"])


async def test_wrong_vector_count_is_an_error_not_a_silent_truncation():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 1}}
        )

    embed = make_embedder(
        base_url="http://x/v1",
        api_key="k",
        model="m",
        batch_size=8,
        transport=_transport(handler),
    )
    with pytest.raises(EmbeddingError):
        await embed(["a", "b"])


async def test_empty_input_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not have been called")

    embed = make_embedder(
        base_url="http://x/v1", api_key="k", model="m", transport=_transport(handler)
    )
    assert await embed([]) == ([], 0)
