from fastapi.testclient import TestClient

from kbase.server.app import create_app
from kbase.settings import Settings


def test_no_credential_is_401(client):
    assert client.get("/v1/collections").status_code == 401


def test_wrong_key_is_401(client):
    r = client.get("/v1/collections", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_malformed_header_is_401(client):
    r = client.get("/v1/collections", headers={"Authorization": "acme-key"})
    assert r.status_code == 401


def test_valid_key_is_accepted(client, acme):
    assert client.get("/v1/collections", headers=acme).status_code == 200


def test_healthz_needs_no_credential(client):
    assert client.get("/healthz").status_code == 200


def test_no_configured_keys_rejects_everything(tmp_path):
    # An unset KB_API_KEYS must lock the service, not open it.
    s = Settings.from_env({"KB_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/x.db"})
    with TestClient(create_app(s)) as c:
        assert c.get(
            "/v1/collections", headers={"Authorization": "Bearer any"}
        ).status_code == 401
