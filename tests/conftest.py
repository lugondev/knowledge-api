import pytest
from fastapi.testclient import TestClient

from kbase.server.app import create_app
from kbase.settings import Settings

VOCAB = ["bảo hành", "đổi trả", "giao hàng", "mèo"]


def keyword_embedder(vocabulary=VOCAB):
    async def embed(texts):
        vectors = []
        for t in texts:
            low = t.lower()
            vectors.append([1.0 if w in low else 0.0 for w in vocabulary])
        return vectors, len(texts)

    return embed


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(
        {
            "KB_API_KEYS": "acme-key:acme, globex-key:globex",
            "KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/api.db",
            "KB_EMBED_BASE_URL": "http://embed.invalid/v1",
            "KB_EMBED_MODEL": "fake",
            "KB_MAX_UPLOAD_BYTES": "1000",
        }
    )


@pytest.fixture
def client(settings):
    app = create_app(settings, embedder=keyword_embedder())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def acme():
    return {"Authorization": "Bearer acme-key"}


@pytest.fixture
def globex():
    return {"Authorization": "Bearer globex-key"}
