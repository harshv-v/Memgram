"""Observability: Prometheus metrics + structured JSON logs + request IDs.

Debugging by grep is what forces rework later — this closes that gap:
  GET /metrics            Prometheus exposition (latency, jobs, queue, memories)
  MEMGRAM_JSON_LOGS=1     one JSON object per log line (request_id included)

prometheus-client is a backend dependency; if it's absent (SDK-only installs
importing shared modules) every metric degrades to a no-op — observability must
never be the thing that crashes the system it observes.
"""
import contextvars
import json
import logging
import time
import uuid

request_id_var = contextvars.ContextVar("memgram_request_id", default="-")

try:
    from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge,
                                   Histogram, generate_latest)
    HAVE_PROM = True
except ImportError:  # pragma: no cover
    HAVE_PROM = False

    class _Noop:
        def labels(self, *a, **k): return self
        def observe(self, *a): pass
        def inc(self, *a): pass
        def set(self, *a): pass
    Counter = Gauge = Histogram = lambda *a, **k: _Noop()  # noqa: E731
    CONTENT_TYPE_LATEST = "text/plain"
    def generate_latest(): return b"# prometheus-client not installed\n"


REQUEST_SECONDS = Histogram(
    "memgram_request_seconds", "API request latency", ["route", "method", "status"],
    buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
JOBS_TOTAL = Counter("memgram_jobs_total", "Background jobs", ["type", "status"])
JOB_SECONDS = Histogram("memgram_job_seconds", "Job duration", ["type"])
MEMORIES_TOTAL = Counter(
    "memgram_memories_total", "Memory writes by outcome",
    ["action"])  # created | reinforced | updated | superseded
QUEUE_DEPTH = Gauge("memgram_queue_depth", "Jobs outstanding on the stream")
DLQ_DEPTH = Gauge("memgram_dlq_depth", "Jobs in the dead-letter stream")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": round(time.time(), 3), "level": record.levelname,
            "logger": record.name, "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(json_logs: bool | None = None) -> None:
    """Call once per process (API startup, worker startup)."""
    import os
    if json_logs is None:
        json_logs = os.environ.get("MEMGRAM_JSON_LOGS") == "1"
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    if json_logs:
        for h in root.handlers:
            h.setFormatter(JsonFormatter())
    root.setLevel(os.environ.get("MEMGRAM_LOG_LEVEL", "INFO"))


def new_request_id(incoming: str | None = None) -> str:
    rid = incoming or uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid
