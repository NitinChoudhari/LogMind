"""
LogMind - LM Studio chat streaming with raw reasoning capture.

LangChain's ChatOpenAI silently DROPS any streamed delta field it doesn't know about -
confirmed directly: LM Studio's OpenAI-compatible API streams a `reasoning` delta field
(separate from `content`) for reasoning-capable local models (e.g. gpt-oss-20b's harmony
format), but ChatOpenAI's chunk objects show `content` as completely empty during the
whole reasoning phase - the text isn't merged or skipped, it's thrown away outright,
since ChatOpenAI only ever surfaces the field names it explicitly knows about.

This module talks to LM Studio's HTTP API directly instead, so the reasoning text is
never lost. It yields `_Chunk` objects with the exact same `content`/`kind` contract as
llm/huggingface.py's `HFChatModel` (kind="thinking" for reasoning, kind="answer" for the
real response) - agents/crew.py's stream_query()/`_general_answer` already branch on
`kind` generically, so no changes were needed there to make lmstudio-served reasoning
models show up in the same `thinking`/`thinking_done` SSE events HF thinking models use.
Models with no `reasoning` field (the common case - most LM Studio models aren't
reasoning models) just never emit a kind="thinking" chunk, identical to today's behavior.
"""

import json

import requests


class _Chunk:
    __slots__ = ("content", "kind")

    def __init__(self, content: str, kind: str = "answer"):
        self.content = content
        self.kind = kind


class LMStudioChatModel:
    """`.stream()`-only chat wrapper for LM Studio's OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.1):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def stream(self, messages):
        chat_messages = [{"role": role, "content": content} for role, content in messages]
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": chat_messages,
                "stream": True,
                "temperature": self.temperature,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
        # Decode each line as UTF-8 explicitly - `iter_lines(decode_unicode=True)`
        # decodes using `resp.encoding`, which `requests` guesses from headers and
        # falls back to a Latin-1-family default when the server doesn't send an
        # explicit charset (LM Studio doesn't). That silently mangled every
        # multi-byte UTF-8 character (₹, em dashes, apostrophes, ...) into mojibake.
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning")
            if reasoning:
                yield _Chunk(reasoning, kind="thinking")
            content = delta.get("content")
            if content:
                yield _Chunk(content, kind="answer")
