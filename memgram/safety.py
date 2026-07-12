"""Safety layer.

1) sanitize_memory — recalled memories are UNTRUSTED text (they were extracted
   from conversations, which an adversary can influence: plant an instruction-
   shaped "fact" today, it executes inside the prompt weeks later). Before any
   memory is injected we collapse its structure and neutralize instruction-like
   patterns, so a memory can state facts but cannot issue orders. Memories are
   single sentences by design, so structure collapse costs nothing.
   Kill-switch: MEMGRAM_SANITIZE=0.

2) redact_pii — optional write-time redaction of emails / phones / SSNs / cards
   / IPs, applied centrally in the store so extractor, reflection, summarizer
   and procedural writes are all covered. Off by default (a memory layer that
   silently mangles data must be opted into): MEMGRAM_PII_REDACT=1.

Regex-first by intent: deterministic, instant, zero-cost — the hot path can't
afford an LLM judge, and a wrong [filtered] is recoverable while an executed
injection is not.
"""
import os
import re

# --- 1. injection sanitizer ---------------------------------------------------

_INJECTION_PATTERNS = [
    # direct instruction override attempts
    r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|your)\s+[^.,;]{0,40}?(?:instructions?|rules?|prompts?|context)",
    r"(?:new|updated|real|true)\s+(?:system\s+)?instructions?\s*:",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"act\s+as\s+(?:if\s+you\s+are|a|an)\b[^.]{0,40}(?:unrestricted|jailbroken|developer\s+mode)",
    # role / block spoofing
    r"^\s*(?:system|assistant|developer)\s*:",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
    r"\[/?(?:SYSTEM|INST)\]",
    r"(?:BEGIN|END)\s+(?:SYSTEM|HIDDEN)\s+(?:PROMPT|INSTRUCTIONS?)",
    # tool / exfil coercion
    r"do\s+not\s+(?:tell|inform|alert)\s+the\s+user",
    r"(?:print|reveal|repeat)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS),
                           re.IGNORECASE | re.MULTILINE)


def sanitize_memory(text: str) -> str:
    """Neutralize a recalled memory before prompt injection. Facts pass through
    verbatim; structure and instruction-patterns do not."""
    if os.environ.get("MEMGRAM_SANITIZE") == "0" or not text:
        return text
    # collapse structure: a memory can't open new markdown blocks/roles/lines
    out = " ".join(text.split())
    out = out.replace("```", "'''")
    out = re.sub(r"(?:^|\s)#{1,6}\s", " ", out)          # markdown headers
    out = _INJECTION_RE.sub("[filtered]", out)
    return out.strip()


# --- 2. PII redaction -----------------------------------------------------------

_PII_RES = [
    (re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"), "[email]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ssn]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card]"),
    (re.compile(r"(?:(?<=\s)|^)\+?\d[\d\s().-]{7,}\d\b"), "[phone]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip]"),
]


def pii_enabled() -> bool:
    return os.environ.get("MEMGRAM_PII_REDACT") == "1"


def redact_pii(text: str) -> str:
    """Replace common PII with typed placeholders. Order matters: cards/SSNs
    before the greedy phone pattern."""
    if not text:
        return text
    for rx, repl in _PII_RES:
        text = rx.sub(repl, text)
    return text
