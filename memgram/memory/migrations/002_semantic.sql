-- Memgram migration 002 — semantic memory (Pillar 2)
-- HNSW index instead of ivfflat: better recall, no `lists` tuning, the right
-- pgvector default at our scale.

CREATE TABLE IF NOT EXISTS semantic_memories (
  id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id          TEXT NOT NULL,
  agent_id            TEXT NOT NULL,
  user_id             TEXT NOT NULL,
  content             TEXT NOT NULL,
  memory_type         TEXT,            -- 'fact' | 'preference' | 'entity' | 'correction'
  source              TEXT,            -- 'extractor' | 'reflection' | 'user'
  embedding           VECTOR({EMBED_DIMS}),  -- default 1536 (text-embedding-3-small); set MEMGRAM_EMBED_DIMS before first migrate for local models
  stability           FLOAT DEFAULT 2.0,
  reinforcement_count INT DEFAULT 1,
  retention_score     FLOAT DEFAULT 1.0,
  emotional_weight    FLOAT DEFAULT 1.0,
  memory_tier         TEXT DEFAULT 'active',  -- active|fading|archived|promoted
  last_accessed_at    TIMESTAMPTZ DEFAULT NOW(),
  scope               TEXT DEFAULT 'project', -- global|project|private
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS semantic_embedding_hnsw
  ON semantic_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS semantic_lookup
  ON semantic_memories (project_id, user_id, agent_id, memory_tier);

ALTER TABLE semantic_memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_isolation_semantic ON semantic_memories
  USING (user_id = current_setting('app.current_user_id', true));
