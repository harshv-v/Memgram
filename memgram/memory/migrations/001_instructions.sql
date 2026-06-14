-- Memgram migration 001 — instruction store (Week 1 thin slice)
-- NOTE: user_id is TEXT, not UUID. Developers manage user identity mapping
-- (section 12 of the design doc); demo user ids like "u1" must work.

CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, unused this week but enabled per plan

CREATE TABLE IF NOT EXISTS instructions (
  id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id        TEXT NOT NULL,
  agent_id          TEXT NOT NULL,
  user_id           TEXT NOT NULL,
  content           TEXT NOT NULL,
  priority          INT DEFAULT 2,         -- 1=always, 2=strong, 3=soft
  source            TEXT NOT NULL,         -- 'user' | 'agent_proposed'
  status            TEXT DEFAULT 'active', -- 'active' | 'pending' | 'rejected'
  confidence        FLOAT DEFAULT 1.0,
  last_confirmed_at TIMESTAMPTZ DEFAULT NOW(),
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS instructions_lookup
  ON instructions (project_id, agent_id, user_id, status, priority);

-- Trust model: agents can never write status='active' rows. Enforced at the
-- API layer (no endpoint accepts source='agent_proposed' with status='active').
ALTER TABLE instructions ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_isolation ON instructions
  USING (user_id = current_setting('app.current_user_id', true));
