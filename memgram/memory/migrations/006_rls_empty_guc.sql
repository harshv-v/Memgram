-- Memgram migration 006 — make the RLS policies robust to an *empty* GUC, and
-- set WITH CHECK explicitly.
--
-- The bug 005 exposed: once a pooled connection has run
--   set_config('app.current_user_id', <uid>, true)   -- transaction-local
-- for a tenant-scoped read (the retriever), the parameter does NOT revert to
-- "undefined" on that connection — current_setting('app.current_user_id', true)
-- subsequently returns the EMPTY STRING '' rather than NULL. The 004 policy only
-- treated NULL as "unset → permissive", so on the next INSERT reusing that
-- connection (ingest, instruction create) the WITH CHECK saw '' , took neither
-- branch, and Postgres raised:
--   new row violates row-level security policy
-- A superuser bypassed RLS entirely, so this only surfaced once 005 made the
-- runtime connect as a non-superuser and RLS actually applied.
--
-- Fix: collapse NULL and '' to the same "unset" meaning via NULLIF, and declare
-- WITH CHECK explicitly (so reads AND writes share one rule). Worker (GUC never
-- set) stays permissive; API (GUC = real uid) stays restricted.

DROP POLICY IF EXISTS tenant_isolation ON instructions;
DROP POLICY IF EXISTS tenant_isolation ON semantic_memories;
DROP POLICY IF EXISTS tenant_isolation ON episodic_logs;

CREATE POLICY tenant_isolation ON instructions
  USING      (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true))
  WITH CHECK (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true));

CREATE POLICY tenant_isolation ON semantic_memories
  USING      (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true))
  WITH CHECK (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true));

CREATE POLICY tenant_isolation ON episodic_logs
  USING      (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true))
  WITH CHECK (NULLIF(current_setting('app.current_user_id', true), '') IS NULL
              OR user_id = current_setting('app.current_user_id', true));
