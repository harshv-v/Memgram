-- Memgram migration 008 — contradiction resolution / supersession.
--
-- When a new fact replaces an old one (user moved Berlin -> Munich, switched
-- Rust -> Go), the old memory must be deprecated, not left to outrank the new
-- reality via a higher reinforcement_count. We mark the stale row with the id of
-- the memory that replaced it and archive it.
--
-- `superseded_by IS NOT NULL` means "deprecated by a newer fact". Retrieval and
-- dedup already skip archived rows; decay must ALSO skip these so the nightly
-- re-tier can't resurrect a freshly-superseded fact.

ALTER TABLE semantic_memories ADD COLUMN IF NOT EXISTS superseded_by UUID;
