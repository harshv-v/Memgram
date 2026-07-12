-- Migration 012 — monitor findings.
-- The monitor agents watch STORED memory (not the pipeline) and report
-- anomalies here: suspected injections, PII at rest, duplicates, bloat,
-- staleness. Surfaced via GET /v1/findings and Prometheus.

CREATE TABLE IF NOT EXISTS memory_findings (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id  TEXT NOT NULL,
  agent_id    TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  monitor     TEXT NOT NULL,       -- safety | hygiene | drift
  kind        TEXT NOT NULL,       -- injection_suspect | pii_at_rest | duplicate_content | memory_bloat | stale_active | review_stuck
  severity    TEXT NOT NULL,       -- info | warn | critical
  detail      TEXT NOT NULL,
  memory_id   UUID,
  resolved    BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS findings_lookup
  ON memory_findings (project_id, resolved, created_at);

GRANT SELECT, INSERT, UPDATE, DELETE ON memory_findings TO memgram_app;
