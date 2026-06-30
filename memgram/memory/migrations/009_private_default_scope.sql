-- Memgram migration 009 — isolated-by-default multi-agent memory.
--
-- New memories default to `private` (visible only to the agent that wrote them).
-- Sharing across agents is opt-in: set a memory's scope to 'project' (shared
-- within the project) or 'global' (shared everywhere) for the same user. The
-- retriever surfaces own + shared memories; private stays with its owner.

ALTER TABLE semantic_memories ALTER COLUMN scope SET DEFAULT 'private';
