"""Multi-provider LLM layer for the background brain — NO frameworks.

Every provider is a thin adapter exposing the OpenAI `chat.completions.create`
surface, so BaseAgent, FakeLLM, and every test keep working unchanged. The
adapter's job is MECHANICS, not prompt wording: JSON-mode enforcement differs
per provider (OpenAI `response_format` / Anthropic none / instruct-and-parse),
and that difference is handled here, once.

Pick a brain with one env var:

    MEMGRAM_BRAIN=openai        gpt-4o-mini / gpt-4o             (default)
    MEMGRAM_BRAIN=deepseek      deepseek-chat        ~$0.14/M in  (cheapest)
    MEMGRAM_BRAIN=gemini        gemini flash-lite    ~$0.10/M in  (Google, via
                                Google's own OpenAI-compatible endpoint)
    MEMGRAM_BRAIN=groq          llama-3.3-70b        (fast open models)
    MEMGRAM_BRAIN=anthropic     claude haiku / sonnet (native adapter)
    MEMGRAM_BRAIN=watsonx       IBM granite (needs `pip install memgram[watsonx]`)

Model ids inside a preset are defaults, not law: MEMGRAM_FAST_MODEL /
MEMGRAM_QUALITY_MODEL always win, and MEMGRAM_LLM_BASE_URL + OPENAI_API_KEY
still works as the fully-custom escape hatch (any OpenAI-compatible server:
Ollama, vLLM, Together, Mistral, ...).
"""
import logging
import os

logger = logging.getLogger("memgram.llm")

_JSON_STRICT = ("\nYou MUST respond with a single valid JSON object and nothing "
                "else — no prose, no markdown, no code fences.")

# brain -> (kind, base_url, api_key_env, fast_model, quality_model)
BRAINS = {
    "openai":    ("openai-compat", None, "OPENAI_API_KEY", "gpt-4o-mini", "gpt-4o"),
    "deepseek":  ("openai-compat", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
                  "deepseek-chat", "deepseek-chat"),
    "gemini":    ("openai-compat",
                  "https://generativelanguage.googleapis.com/v1beta/openai/",
                  "GEMINI_API_KEY", "gemini-2.5-flash-lite", "gemini-2.5-flash"),
    "groq":      ("openai-compat", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                  "llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    "anthropic": ("anthropic", None, "ANTHROPIC_API_KEY",
                  "claude-haiku-4-5", "claude-sonnet-4-6"),
    "watsonx":   ("watsonx", None, "WATSONX_APIKEY",
                  "ibm/granite-3-8b-instruct", "ibm/granite-3-8b-instruct"),
}


def brain_spec() -> dict:
    name = os.environ.get("MEMGRAM_BRAIN", "openai").lower()
    if name not in BRAINS:
        raise ValueError(f"Unknown MEMGRAM_BRAIN={name!r}. One of: {', '.join(BRAINS)}")
    kind, base_url, key_env, fast, quality = BRAINS[name]
    return {"name": name, "kind": kind,
            # explicit base_url env wins even inside a preset
            "base_url": os.environ.get("MEMGRAM_LLM_BASE_URL") or base_url,
            "key_env": key_env, "fast": fast, "quality": quality}


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may wrap it in prose or
    fences. Returns the original text if nothing better is found (json.loads
    will then raise and the agent retry loop takes over)."""
    if not text:
        return text
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    if "```" in t:  # fenced block, ```json or plain
        for chunk in t.split("```"):
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                t = c
                break
    start = t.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
    return text


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        self.usage = usage  # normalized by memgram.usage.norm_usage


class _Create:
    """Binds an async create(**kw) function into the .chat.completions shape."""
    def __init__(self, fn): self.create = fn


class _Chat:
    def __init__(self, fn): self.completions = _Create(fn)


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint. Uses native JSON mode when the server
    accepts it; transparently falls back to instruct-and-parse when it doesn't
    (some open-model servers reject `response_format`)."""

    def __init__(self, base_url=None, api_key_env="OPENAI_API_KEY"):
        from openai import AsyncOpenAI
        key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(base_url=base_url, api_key=key)
        self._json_mode_ok = True
        self.chat = _Chat(self._create)

    async def _create(self, model=None, messages=None, response_format=None, **kw):
        if response_format and self._json_mode_ok:
            try:
                return await self._client.chat.completions.create(
                    model=model, messages=messages,
                    response_format=response_format, **kw)
            except Exception as e:
                if "response_format" not in str(e):
                    raise
                logger.info("server rejects response_format; instruct-and-parse from now on")
                self._json_mode_ok = False
        msgs = _with_json_instruction(messages) if response_format else messages
        r = await self._client.chat.completions.create(model=model, messages=msgs, **kw)
        return _Resp(extract_json(r.choices[0].message.content),
                     usage=getattr(r, "usage", None))


class AnthropicLLM:
    """Native Anthropic adapter. Translates the OpenAI call shape: system
    messages -> the `system` param, JSON mode -> strict instruction + robust
    extraction. Requires `pip install memgram[anthropic]`."""

    def __init__(self, client=None):
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "MEMGRAM_BRAIN=anthropic needs the anthropic SDK: "
                    "pip install 'memgram[anthropic]'") from e
            client = AsyncAnthropic()
        self._client = client
        self.chat = _Chat(self._create)

    async def _create(self, model=None, messages=None, response_format=None, **kw):
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        if response_format:
            system_parts.append(_JSON_STRICT.strip())
        turns = [m for m in messages if m["role"] in ("user", "assistant")]
        if not turns:  # Anthropic requires at least one user turn
            turns = [{"role": "user", "content": "Proceed."}]
        r = await self._client.messages.create(
            model=model, max_tokens=int(os.environ.get("MEMGRAM_MAX_TOKENS", "2048")),
            system="\n\n".join(system_parts) or "",
            messages=turns)
        text = "".join(getattr(b, "text", "") for b in r.content)
        return _Resp(extract_json(text) if response_format else text,
                     usage=getattr(r, "usage", None))


class WatsonxLLM:
    """IBM watsonx.ai adapter — via IBM's own SDK (sync, run in a thread), no
    LangChain. Requires `pip install memgram[watsonx]` plus WATSONX_APIKEY,
    WATSONX_URL, WATSONX_PROJECT_ID."""

    def __init__(self):
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError as e:
            raise ImportError(
                "MEMGRAM_BRAIN=watsonx needs IBM's SDK: "
                "pip install 'memgram[watsonx]'") from e
        self._ModelInference = ModelInference
        self._creds = Credentials(
            url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.environ["WATSONX_APIKEY"])
        self._project = os.environ["WATSONX_PROJECT_ID"]
        self._models: dict = {}
        self.chat = _Chat(self._create)

    async def _create(self, model=None, messages=None, response_format=None, **kw):
        import asyncio
        if model not in self._models:
            self._models[model] = self._ModelInference(
                model_id=model, credentials=self._creds, project_id=self._project)
        msgs = _with_json_instruction(messages) if response_format else messages
        r = await asyncio.to_thread(
            self._models[model].chat, messages=msgs,
            params={"max_tokens": int(os.environ.get("MEMGRAM_MAX_TOKENS", "2048"))})
        text = r["choices"][0]["message"]["content"]
        return _Resp(extract_json(text) if response_format else text,
                     usage=r.get("usage"))


def _with_json_instruction(messages: list[dict]) -> list[dict]:
    """Append the JSON-strict line to the system message (or prepend one)."""
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m["role"] == "system":
            m["content"] = m["content"] + _JSON_STRICT
            return msgs
    return [{"role": "system", "content": _JSON_STRICT.strip()}] + msgs


def get_brain_llm():
    """The worker's LLM, per MEMGRAM_BRAIN (see module docstring)."""
    spec = brain_spec()
    if spec["kind"] == "anthropic":
        return AnthropicLLM()
    if spec["kind"] == "watsonx":
        return WatsonxLLM()
    return OpenAICompatLLM(base_url=spec["base_url"], api_key_env=spec["key_env"])
