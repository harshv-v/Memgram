-- Migration 013 — runtime project settings (the on/off toggles).
-- First tenant: pii_redact. DB value wins over env; env remains the default
-- for projects with no row here.

CREATE TABLE IF NOT EXISTS project_settings (
  project_id  TEXT NOT NULL,
  key         TEXT NOT NULL,
  value       TEXT NOT NULL,
  updated_at  TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (project_id, key)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON project_settings TO memgram_app;
