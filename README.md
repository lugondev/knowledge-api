# kbase — Knowledge Base Service

Documents in, retrievable chunks out. One collection per body of knowledge, one
bearer key per tenant, and a semantic search that refuses to answer with
something irrelevant.

## Run it

```bash
pip install -e ".[dev]"

export KB_API_KEYS=pick-a-long-random-string:acme
export KB_EMBED_BASE_URL=https://api.openai.com/v1
export KB_EMBED_API_KEY=sk-...
export KB_EMBED_MODEL=text-embedding-3-small

kb doctor     # says what is missing, and nothing else
kb serve      # refuses to start on a configuration doctor already failed
```

```bash
AUTH="Authorization: Bearer pick-a-long-random-string"

curl -X POST localhost:8090/v1/collections -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"name":"faq"}'

curl -X POST localhost:8090/v1/documents -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"collection":"faq","title":"Sổ tay","text":"## Bảo hành\n\nMười hai tháng."}'

curl -X POST localhost:8090/v1/search -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"collection":"faq","query":"bảo hành bao lâu"}'
```

Or the whole thing in one file:

```bash
cp .env.example .env      # put your keys in it
docker compose up -d
```

## How it works

An upload is stored and returns `202 pending` immediately; extraction, chunking,
and embedding happen in the background. Embedding a large document takes minutes,
and a synchronous upload would time out before it finished. Poll
`GET /v1/documents/{id}` for `status`.

Chunks are cut on markdown headings first and length second (800 characters, 100
of overlap), and each one keeps the heading path it came from, so an answer can
cite *Bảo hành > Đổi trả* rather than a floating paragraph.

A document either indexes completely or is marked `failed` with a reason. There
is no partial index: a document holding only its first third would answer
questions from that third and never say the rest is missing.

## Configuration

| | |
| --- | --- |
| `KB_API_KEYS` | `key:tenant,key:tenant`. Unset means every request is a 401 |
| `KB_DATABASE_URL` | SQLite by default; a Postgres URL switches the store |
| `KB_EMBED_BASE_URL` | OpenAI-compatible `/embeddings` endpoint |
| `KB_EMBED_API_KEY` | credential for the above |
| `KB_EMBED_MODEL` | embedding model id |
| `KB_MAX_UPLOAD_BYTES` | rejected above this (default 20 MB) |
| `KB_DOCS` | `false` closes `/docs` and `/redoc`, which need no credential |

## API

```
POST   /v1/collections            {name}
GET    /v1/collections
DELETE /v1/collections/{name}

POST   /v1/documents              multipart (file, collection, title?)
                                  or JSON {collection, title, text}
GET    /v1/documents?collection=&status=
GET    /v1/documents/{id}
DELETE /v1/documents/{id}

POST   /v1/search                 {collection, query, limit, min_score}
GET    /healthz
```

## Not here yet

PDF and DOCX ingestion, URL crawling, hybrid keyword search, and re-ranking.
Gateway integration is specified in the parent repository's spec and deliberately
not built yet.

## Tests

```bash
.venv/bin/pytest -q
```
