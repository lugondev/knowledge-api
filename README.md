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
questions from that third and never say the rest is missing. A document with no
text in it fails too, rather than reporting `indexed` with nothing to search.

`failed` is never the end of the road. Uploading the same bytes again retries
that document rather than reporting a duplicate, and `POST
/v1/documents/{id}/reindex` runs it again from the copy already stored — which
is also how to re-embed a corpus after changing `KB_EMBED_MODEL`, without
holding any of it locally.

The `error` a document carries is written for the tenant reading it. Anything
describing the file they sent is quoted; anything describing this deployment —
the embedding host, a driver's connection string — is not, and stays in the
service log.

Search has two backends. On SQLite — and on Postgres with no `KB_EMBED_DIM` —
it scores every chunk in the collection, in a worker thread, a partition at
a time: a search costs about 16 MB whether the collection holds a thousand chunks
or a hundred thousand. It is a linear scan, so it slows as the corpus grows,
around 1.2 s over 10,000 chunks.

Set `KB_EMBED_DIM` on Postgres and chunks are stored in a pgvector column with
an HNSW index, and the ordering happens in the database. Vectors already indexed
are copied across on the next start — no re-embedding, no provider spend — and
any chunk whose width does not match marks its document `failed` with a reason,
recoverable with `POST /v1/documents/{id}/reindex`. HNSW does not go above 2000
dimensions; wider than that stores fine and searches by scanning, and `kb
doctor` says so. Results from the two backends are not identical: HNSW is
approximate, which is the ordinary price of a vector index.

## Configuration

| | |
| --- | --- |
| `KB_API_KEYS` | `key:tenant,key:tenant`. Unset means every request is a 401 |
| `KB_DATABASE_URL` | SQLite by default; a Postgres URL switches the store |
| `KB_EMBED_DIM` | Postgres only. Setting it stores vectors in a pgvector column with an HNSW index; unset means the linear scan. Must match the model's width |
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
                                  202 queued, 200 already indexed here
GET    /v1/documents?collection=&status=
GET    /v1/documents/{id}
POST   /v1/documents/{id}/reindex index again from the stored bytes
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
