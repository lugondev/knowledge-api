FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# `[postgres]` because compose ships a pgvector service and KB_DATABASE_URL is
# an env var: without the driver in the image, pointing this container at
# Postgres is a ModuleNotFoundError on the first connection, not a config error.
RUN pip install --no-cache-dir ".[postgres]"

ENV KB_DATABASE_URL=sqlite+aiosqlite:////data/kbase.db
VOLUME ["/data"]
EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz').status==200 else 1)"

CMD ["kb", "serve", "--host", "0.0.0.0", "--port", "8090"]
