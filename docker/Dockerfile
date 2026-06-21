# Memgram API + worker image. Single image, two entrypoints (compose picks).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY memgram ./memgram
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Bake the local embedding model into the image so the first request doesn't pay
# a download (and offline deploys work). ~80MB; harmless if MEMGRAM_EMBEDDER!=local.
ARG MEMGRAM_LOCAL_EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${MEMGRAM_LOCAL_EMBED_MODEL}')"

# Default: API. The worker service overrides `command` in docker-compose.
EXPOSE 8000
CMD ["uvicorn", "memgram.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
