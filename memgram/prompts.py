"""Overridable prompt layer.

Every agent prompt resolves through `get_prompt(key, default)` at call time, so a
deployment can override any prompt via an environment variable WITHOUT editing
code. Prompts become *configuration*, not hardcoded constants — and they're
resolved per call, so an override takes effect immediately.

PROMPT POLICY (read before "improving" any prompt):
1. ZERO few-shot examples in extraction-class prompts, by design. Few-shot
   demonstrations leak: on empty/thin input, models emit near-verbatim copies
   of example content (majority-label/common-token bias — Zhao et al. 2021,
   "Calibrate Before Use"). In a MEMORY system a leaked example becomes a
   stored false fact that poisons every future prompt. Contract + rules +
   an explicit "empty is a valid answer" beats demonstrations here.
2. Data is always fenced in XML tags (<conversation>, <logs>, <new_fact>, ...)
   and every prompt instructs the model to use ONLY tagged content — the
   instruction/data boundary is part of the injection defense.
3. No behavior hardcoded in Python that belongs in a prompt or a setting.
   Prompts are configuration: override via MEMGRAM_PROMPT_<KEY>, extend via
   MEMGRAM_PROMPT_SUFFIX_<KEY>. Any prompt change must pass the quality eval
   (bench/quality) before it ships as a default.

Override a prompt by setting `MEMGRAM_PROMPT_<KEY>`, where <KEY> is the dotted key
upper-cased with dots → underscores. Examples:
    MEMGRAM_PROMPT_EXTRACTOR_SYSTEM="You extract ..."
    MEMGRAM_PROMPT_REFLECTION_SYSTEM="..."
    MEMGRAM_PROMPT_SUMMARIZER_SYSTEM="..."

The built-in defaults still live next to each agent (so the agent is readable on
its own); this layer just makes them all swappable from one mechanism.
"""
import os

# Known override keys — for discoverability, docs, and a future dashboard editor.
PROMPT_KEYS = (
    "extractor.system",
    "extractor.verify",
    "extractor.operation",
    "reflection.system",
    "proposer.system",
    "summarizer.system",
    "procedural.system",
)


def _env_key(key: str) -> str:
    return "MEMGRAM_PROMPT_" + key.upper().replace(".", "_")


def get_prompt(key: str, default: str) -> str:
    """Resolve a prompt: full env override wins; otherwise the built-in default,
    optionally extended by an additive suffix (MEMGRAM_PROMPT_SUFFIX_<KEY>) —
    the light-touch way to adapt wording to a specific model family without
    rewriting the whole prompt. Tune suffixes from eval results, not vibes."""
    override = os.environ.get(_env_key(key))
    if override:
        return override
    suffix = os.environ.get(_env_key(key).replace("MEMGRAM_PROMPT_", "MEMGRAM_PROMPT_SUFFIX_"))
    return default + ("\n" + suffix if suffix else "")
