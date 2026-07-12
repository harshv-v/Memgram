-- Migration 010 — transactional outbox.
-- Closes the ingest dual-write gap: the episodic rows and the "a job must run
-- for this turn" intent now commit in ONE transaction. A relay moves pending
-- outbox rows onto the queue; idempotency keys make dispatch exactly-once
-- from the consumer's point of view even if the relay races the API.

CREATE TABLE IF NOT EXISTS outbox (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  job_type      TEXT NOT NULL,
  payload       JSONB NOT NULL,
  dispatched_at TIMESTAMPTZ,             -- NULL = not yet on the queue
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS outbox_pending
  ON outbox (created_at) WHERE dispatched_at IS NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON outbox TO memgram_app;
