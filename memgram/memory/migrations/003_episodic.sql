-- Memgram migration 003 — episodic memory (Pillar 3)
-- Plain Postgres with a composite index; TimescaleDB partitioning comes at
-- scale (month 4-6 in the plan) — schema needs no change for it.

CREATE TABLE IF NOT EXISTS episodic_logs (
  id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id       TEXT NOT NULL,
  agent_id         TEXT NOT NULL,
  user_id          TEXT NOT NULL,
  role             TEXT NOT NULL,      -- 'user' | 'assistant' | 'compressed_session'
  content          TEXT NOT NULL,
  emotional_weight FLOAT DEFAULT 1.0,
  reflected        BOOLEAN DEFAULT FALSE,  -- consumed by a reflection run yet?
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS episodic_lookup
  ON episodic_logs (project_id, user_id, agent_id, created_at);

ALTER TABLE episodic_logs ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_isolation_episodic ON episodic_logs
  USING (user_id = current_setting('app.current_user_id', true));
