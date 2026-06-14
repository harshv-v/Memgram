-- Memgram migration 007 — provenance for semantic memories.
--
-- Every extracted memory should be auditable back to what produced it. We store
-- a short excerpt of the source conversation so a human (or the decay/faithfulness
-- machinery) can see *why* a memory exists. This is also the anti-hallucination
-- backstop's paper trail: if a memory was wrongly stored, its provenance shows
-- whether the source actually supports it.

ALTER TABLE semantic_memories ADD COLUMN IF NOT EXISTS provenance TEXT;
