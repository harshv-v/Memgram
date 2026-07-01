"""Overridable prompt layer.

Every agent prompt resolves through `get_prompt(key, default)` at call time, so a
deployment can override any prompt via an environment variable WITHOUT editing
code. Prompts become *configuration*, not hardcoded constants — and they're
resolved per call, so an override takes effect immediately.

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
    """Return the env override for `key` if set, else the built-in `default`."""
    return os.environ.get(_env_key(key)) or default
