def _make_collection(client, headers, name="faq"):
    r = client.post("/v1/collections", json={"name": name}, headers=headers)
    assert r.status_code == 201
    return r.json()


def test_collection_lifecycle(client, acme):
    _make_collection(client, acme)
    assert client.get("/v1/collections", headers=acme).json() == [
        {"name": "faq", "document_count": 0}
    ]
    assert client.delete("/v1/collections/faq", headers=acme).status_code == 204
    assert client.get("/v1/collections", headers=acme).json() == []


def test_two_tenants_do_not_see_each_others_collections(client, acme, globex):
    _make_collection(client, acme)
    _make_collection(client, globex)
    client.post(
        "/v1/documents",
        json={"collection": "faq", "title": "acme doc", "text": "## Bảo hành\n\nbảo hành acme"},
        headers=acme,
    )
    assert client.get("/v1/documents?collection=faq", headers=globex).json() == []


def test_tenant_in_the_body_cannot_widen_access(client, acme, globex):
    # The body is a claim; the credential is the decision.
    _make_collection(client, globex)
    r = client.post(
        "/v1/documents",
        json={"collection": "faq", "text": "x", "tenant": "globex"},
        headers=acme,
    )
    assert r.status_code == 404


def test_text_document_indexes_synchronously_enough_to_search(client, acme):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents",
        json={"collection": "faq", "title": "Sổ tay", "text": "## Bảo hành\n\nbảo hành 12 tháng"},
        headers=acme,
    )
    assert r.status_code == 202
    doc_id = r.json()["id"]
    assert r.json()["status"] == "pending"
    # FastAPI runs background tasks before TestClient returns, so by now it is done.
    doc = client.get(f"/v1/documents/{doc_id}", headers=acme).json()
    assert doc["status"] == "indexed"
    assert doc["chunk_count"] == 1


def test_search_returns_hits_with_usage(client, acme):
    _make_collection(client, acme)
    client.post(
        "/v1/documents",
        json={"collection": "faq", "title": "Sổ tay", "text": "## Bảo hành\n\nbảo hành 12 tháng"},
        headers=acme,
    )
    r = client.post(
        "/v1/search",
        json={"collection": "faq", "query": "bảo hành", "min_score": 0.1},
        headers=acme,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"][0]["heading"] == "Bảo hành"
    assert body["usage"]["prompt_tokens"] == 1


def test_search_on_a_missing_collection_is_404(client, acme):
    r = client.post("/v1/search", json={"collection": "nope", "query": "x"}, headers=acme)
    assert r.status_code == 404


def test_upload_file(client, acme):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents",
        data={"collection": "faq", "title": "Sổ tay"},
        files={"file": ("s.md", "## Bảo hành\n\nbảo hành 12 tháng".encode(), "text/markdown")},
        headers=acme,
    )
    assert r.status_code == 202
    doc = client.get(f"/v1/documents/{r.json()['id']}", headers=acme).json()
    assert doc["status"] == "indexed"


def test_identical_upload_returns_the_existing_document(client, acme):
    _make_collection(client, acme)
    payload = {"collection": "faq", "title": "Sổ tay", "text": "## A\n\nbảo hành"}
    first = client.post("/v1/documents", json=payload, headers=acme)
    second = client.post("/v1/documents", json=payload, headers=acme)
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get("/v1/documents?collection=faq", headers=acme).json()) == 1


def test_oversized_upload_is_rejected(client, acme):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents",
        json={"collection": "faq", "text": "x" * 2000},  # KB_MAX_UPLOAD_BYTES=1000
        headers=acme,
    )
    assert r.status_code == 413


def test_unsupported_file_type_is_recorded_as_failed(client, acme):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents",
        data={"collection": "faq"},
        files={"file": ("manual.pdf", b"%PDF-1.4", "application/pdf")},
        headers=acme,
    )
    doc = client.get(f"/v1/documents/{r.json()['id']}", headers=acme).json()
    assert doc["status"] == "failed"
    assert "pdf" in doc["error"].lower()


def test_document_of_another_tenant_is_404_not_403(client, acme, globex):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents", json={"collection": "faq", "text": "## A\n\nbảo hành"}, headers=acme
    )
    doc_id = r.json()["id"]
    assert client.get(f"/v1/documents/{doc_id}", headers=globex).status_code == 404
    assert client.delete(f"/v1/documents/{doc_id}", headers=globex).status_code == 404


def test_delete_document(client, acme):
    _make_collection(client, acme)
    r = client.post(
        "/v1/documents", json={"collection": "faq", "text": "## A\n\nbảo hành"}, headers=acme
    )
    assert client.delete(f"/v1/documents/{r.json()['id']}", headers=acme).status_code == 204
    assert client.get("/v1/documents?collection=faq", headers=acme).json() == []


def test_document_into_a_missing_collection_is_404(client, acme):
    r = client.post("/v1/documents", json={"collection": "ghost", "text": "x"}, headers=acme)
    assert r.status_code == 404
