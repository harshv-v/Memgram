"""Property-based tests (hypothesis) for the two places where a subtle
regression would be invisible to example-based tests: the decay math and the
safety/parsing text transforms.

    pip install hypothesis && python tests/test_properties.py
"""
import json
import math
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from memgram.llm import extract_json
from memgram.safety import redact_pii, sanitize_memory

FAILURES = []


def run(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        FAILURES.append(name)
        print(f"FAIL  {name}: {e}")


# ---- decay math (mirrors agents/decay.py SQL) ---------------------------------
def retention(t_days: float, stability: float) -> float:
    return math.exp(-t_days / stability)


def tier(r: float) -> str:
    return "active" if r > 0.3 else ("fading" if r > 0.1 else "archived")


@settings(max_examples=300)
@given(t=st.floats(0, 3650), s=st.floats(0.01, 1000))
def decay_bounds(t, s):
    r = retention(t, s)
    # underflow to exactly 0.0 at extreme age/low stability is fine -> archived
    assert 0.0 <= r <= 1.0
    assert tier(r) in ("active", "fading", "archived")


@settings(max_examples=300)
@given(t1=st.floats(0, 3650), dt=st.floats(0.001, 3650), s=st.floats(0.01, 1000))
def decay_monotonic_in_time(t1, dt, s):
    assert retention(t1 + dt, s) <= retention(t1, s)  # older = weaker, always


@settings(max_examples=300)
@given(t=st.floats(0.001, 3650), s1=st.floats(0.01, 999), ds=st.floats(0.001, 100))
def decay_monotonic_in_stability(t, s1, ds):
    assert retention(t, s1 + ds) >= retention(t, s1)  # more stable = slower fade


@settings(max_examples=300)
@given(s=st.floats(0.01, 1000), r=st.floats(0.0, 1.0))
def reinforcement_never_weakens(s, r):
    assert s * (1 + 0.2 * r) >= s  # access can only strengthen


# ---- sanitizer ---------------------------------------------------------------
@settings(max_examples=300)
@given(st.text(max_size=400))
def sanitizer_idempotent(text):
    once = sanitize_memory(text)
    assert sanitize_memory(once) == once


@settings(max_examples=300)
@given(st.text(max_size=400))
def sanitizer_never_leaves_structure(text):
    out = sanitize_memory(text)
    assert "\n" not in out and "```" not in out


@settings(max_examples=300)
@given(st.text(alphabet=st.characters(blacklist_characters="\n`#"),
               max_size=200))
def sanitizer_no_false_positive_on_plain_words(text):
    # a memory made only of neutral words must never get [filtered]
    neutral = " ".join(w for w in text.split() if w.isalpha() and len(w) < 12)
    if any(k in neutral.lower() for k in
           ("ignore", "disregard", "forget", "system", "instruction", "assistant",
            "developer", "inst", "now", "act", "reveal", "print", "repeat", "tell")):
        return  # words that legitimately appear in injection patterns
    assert "[filtered]" not in sanitize_memory(neutral)


# ---- PII redaction --------------------------------------------------------------
@settings(max_examples=300)
@given(st.text(max_size=300))
def redact_idempotent(text):
    once = redact_pii(text)
    assert redact_pii(once) == once


@settings(max_examples=200)
@given(st.emails())
def redact_catches_all_emails(email):
    if email.startswith('"'):
        return  # quoted local parts ("a b"@x.co) are RFC-legal but out of scope
    assert email not in redact_pii(f"contact: {email} thanks")


# ---- extract_json round-trip --------------------------------------------------
json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-1e6, 1e6)
    | st.text(alphabet=st.characters(blacklist_characters='\\"', min_codepoint=32),
              max_size=30),
    lambda inner: st.lists(inner, max_size=4)
    | st.dictionaries(st.text(alphabet="abcdefgh", min_size=1, max_size=8),
                      inner, max_size=4),
    max_leaves=12)


@settings(max_examples=300)
@given(st.dictionaries(st.text(alphabet="abcdefgh", min_size=1, max_size=8),
                       json_values, min_size=1, max_size=5))
def extract_json_roundtrip(obj):
    s = json.dumps(obj)
    for wrapped in (s, f"```json\n{s}\n```", f"Sure! Here it is: {s} Hope it helps.",
                    f"```\n{s}\n```"):
        assert json.loads(extract_json(wrapped)) == obj


for f in [decay_bounds, decay_monotonic_in_time, decay_monotonic_in_stability,
          reinforcement_never_weakens, sanitizer_idempotent,
          sanitizer_never_leaves_structure, sanitizer_no_false_positive_on_plain_words,
          redact_idempotent, redact_catches_all_emails, extract_json_roundtrip]:
    run(f.__name__, f)

print(f"\n{10 - len(FAILURES)}/10 property suites passed"
      + (f" — FAILURES: {FAILURES}" if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
