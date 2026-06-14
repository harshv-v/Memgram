-- Memgram migration 005 — a dedicated, non-superuser application role.
--
-- Why this exists: migration 004 turned on FORCE ROW LEVEL SECURITY to make
-- tenant isolation a real second layer. But a SUPERUSER (and any role with
-- BYPASSRLS) bypasses RLS unconditionally — FORCE does not apply to them. The
-- default Postgres role created by the image (POSTGRES_USER=memgram) is a
-- superuser, so when the API/worker connect as it, RLS is silently inert and
-- isolation rests entirely on query WHERE clauses — exactly what 004 set out to
-- stop relying on.
--
-- Fix: the runtime (API + worker) connects as `memgram_app`, a NOSUPERUSER /
-- NOBYPASSRLS role. Only the migrate step keeps the superuser (it needs DDL).
-- The worker still sees all rows because it never sets app.current_user_id, and
-- the policy is permissive-when-unset; the API sets the GUC per request and is
-- restricted to one tenant — now genuinely enforced by the engine.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'memgram_app') THEN
    CREATE ROLE memgram_app LOGIN PASSWORD 'memgram_app'
      NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  END IF;
END $$;

GRANT USAGE ON SCHEMA public TO memgram_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO memgram_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO memgram_app;

-- Tables created by future migrations (run as the owner) should be reachable by
-- the app role automatically.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO memgram_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO memgram_app;
