# Memgram API + worker image. Single image, two entrypoints (compose picks).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt uvicorn

COPY memgram ./memgram
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Default: API. The worker service overrides `command` in docker-compose.
EXPOSE 8000
CMD ["uvicorn", "memgram.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
