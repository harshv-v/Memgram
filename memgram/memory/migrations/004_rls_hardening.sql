-- Memgram migration 004 — make row-level security actually enforce.
--
-- Migrations 001-003 ENABLEd RLS, but the API connects as the table owner, and
-- the owner bypasses RLS unless FORCE is set — so isolation rested entirely on
-- the WHERE clauses in queries. This migration makes RLS a real second layer.
--
-- The mechanism: FORCE ROW LEVEL SECURITY (so even the owner is subject), plus a
-- policy that is PERMISSIVE when `app.current_user_id` is unset and RESTRICTIVE
-- when it is set. That split is deliberate:
--   * The background WORKER never sets the GUC -> sees all rows. It legitimately
--     needs cross-user access (decay sweeps every user; reflection reads a user's
--     logs). Permissive-when-unset preserves that.
--   * The API sets `app.current_user_id` per request (see retriever / memory
--     routes) -> the engine restricts every row to that user, even if a query
--     forgot its WHERE clause. Defense in depth for the multi-tenant hot path.
--
-- WITH CHECK is omitted, so it defaults to the USING expression — INSERTs from
-- the worker (GUC unset) remain permitted.

DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY['instructions', 'semantic_memories', 'episodic_logs']
  LOOP
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
  END LOOP;
END $$;

DROP POLICY IF EXISTS user_isolation ON instructions;
DROP POLICY IF EXISTS user_isolation_semantic ON semantic_memories;
DROP POLICY IF EXISTS user_isolation_episodic ON episodic_logs;

CREATE POLICY tenant_isolation ON instructions
  USING (current_setting('app.current_user_id', true) IS NULL
         OR user_id = current_setting('app.current_user_id', true));

CREATE POLICY tenant_isolation ON semantic_memories
  USING (current_setting('app.current_user_id', true) IS NULL
         OR user_id = current_setting('app.current_user_id', true));

CREATE POLICY tenant_isolation ON episodic_logs
  USING (current_setting('app.current_user_id', true) IS NULL
         OR user_id = current_setting('app.current_user_id', true));
