-- Migration 011 — usage accounting.
-- Every token the memory layer consumes or injects, attributable per
-- (project, agent, user): background LLM calls, embeddings, and the tokens
-- injected into the developer's prompts. Powers GET /v1/usage.

CREATE TABLE IF NOT EXISTS usage_events (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id  TEXT NOT NULL,
  agent_id    TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  kind        TEXT NOT NULL,      -- 'llm:ExtractorAgent' | 'injection' | 'embedding' | ...
  model       TEXT,
  tokens_in   INT DEFAULT 0,
  tokens_out  INT DEFAULT 0,
  cost_usd    NUMERIC(12, 8) DEFAULT 0,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS usage_lookup
  ON usage_events (project_id, user_id, created_at);

GRANT SELECT, INSERT ON usage_events TO memgram_app;
