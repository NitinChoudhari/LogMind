"""
LogMind - local Hugging Face transformers backend (PROVIDER=huggingface and/or
ROUTER_PROVIDER=huggingface).

Loads quantized causal LMs in-process (no separate inference server) via
`_load(model_path)`, cached per model path so the main pipeline's model
(LLM_MODEL, resolved to config.LLM_IDENTIFIER) and the Manager router's
model (ROUTER_MODEL, resolved to config.ROUTER_MODEL_PATH) can stay
resident simultaneously without evicting each other:

  - `HFTransformersLLM`  - a crewAI BaseLLM for the agent pipeline (Strategist /
    Researcher / Synthesizer). It does NOT implement `supports_function_calling`,
    so crewAI's step executor (see `check_native_tool_support` in
    crewai.agents.step_executor) falls back to its text-parsed ReAct loop:
    crewAI itself injects the tool descriptions/format into the prompt and
    parses "Action:"/"Final Answer:" out of our plain-text completions. We
    never need to parse or execute tool calls ourselves.
  - `HFChatModel`        - a minimal `.stream()`-only chat wrapper for the
    token-streamed synthesis path in agents/crew.py's `stream_query()`.
  - `generate_once()`    - a synchronous, non-streamed, explicit-model-path
    helper used by agents/manager.py's classify() for the huggingface router
    branch (HFChatModel is deliberately .stream()-only, so the router needs
    its own one-shot call shape rather than reusing it).

The main pipeline's model (LLM_MODEL) is typically a "thinking" model that
always emits a `<think>...</think>` block before its real answer (e.g.
Qwen3-4B-Thinking-2507). That block is stripped from non-streamed output;
during streaming it is instead yielded as `kind="thinking"` chunks so the UI
can show it as a live reasoning trace (see `HFChatModel.stream()` below). The
router's model (ROUTER_MODEL) is typically a non-thinking instruct model
instead, since classification needs to be fast, not deliberative.
"""

import re
import threading
from functools import lru_cache

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM

import config

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@lru_cache(maxsize=2)
def _load(model_path: str):
    """Load the tokenizer + quantized model once per (process, model_path).

    maxsize=2 lets the main pipeline's model (HF_LLM_PATH) and the router's
    classification model (ROUTER_HF_LLM_PATH) stay resident simultaneously -
    a zero-arg cache would key both calls on the same global, so whichever
    model loaded second would silently evict the other on every subsequent
    call.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    quant_config = None
    if config.HF_LOAD_IN_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    device_map = config.HF_DEVICE or ("auto" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if quant_config is None else None,
    )
    model.eval()
    return tokenizer, model


def count_tokens(text: str) -> int:
    """Exact token count via the main pipeline model's own tokenizer (used
    for the tok/sec UI metric)."""
    tokenizer, _ = _load(config.LLM_IDENTIFIER)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _strip_think(text: str) -> str:
    # The chat template already opens <think> in the generation prompt, so the
    # generated text itself usually only contains the closing tag - everything
    # before it is reasoning, take what's after. Fall back to stripping a full
    # <think>...</think> block in case a model ever emits both tags itself.
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def _generate(
    messages: list[dict], max_new_tokens: int, temperature: float, model_path: str
) -> str:
    """Run one full (non-streamed) generation pass against model_path; may
    still contain <think>."""
    import torch

    tokenizer, model = _load(model_path)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_once(
    model_path: str,
    messages: list[dict],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    """One-shot non-streamed generation against an explicit model_path, with
    <think> stripped. Used by agents/manager.py's classify() so the router
    can use a small/fast model independently of whichever model the main
    pipeline has loaded."""
    raw = _generate(
        messages, max_new_tokens=max_new_tokens, temperature=temperature, model_path=model_path
    )
    return _strip_think(raw)


class HFTransformersLLM(BaseLLM):
    """crewAI LLM backed by the local transformers model (no LiteLLM)."""

    llm_type: str = "huggingface_local"

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, provider="huggingface", **kwargs)

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        formatted = self._format_messages(messages)
        self._emit_call_started_event(
            messages=formatted,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
        )
        try:
            max_new = int(self.max_tokens) if self.max_tokens else config.HF_MAX_NEW_TOKENS
            temperature = self.temperature if self.temperature is not None else 0.1
            raw = _generate(
                formatted, max_new_tokens=max_new, temperature=temperature, model_path=self.model
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_call_failed_event(
                error=f"{type(exc).__name__}: {exc}", from_task=from_task, from_agent=from_agent
            )
            raise

        text = self._apply_stop_words(_strip_think(raw))
        result = self._validate_structured_output(text, response_model)

        self._emit_call_completed_event(
            response=text,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=formatted,
        )
        return result

    def get_context_window_size(self) -> int:
        return 32768


class _Chunk:
    """Minimal stand-in for a LangChain message chunk - needs `.content` plus a
    `kind` tag ("thinking" | "answer") so callers can surface the model's
    reasoning separately from its final answer."""

    __slots__ = ("content", "kind")

    def __init__(self, content: str, kind: str = "answer"):
        self.content = content
        self.kind = kind


class HFChatModel:
    """`.stream()`-only chat wrapper for the streamed synthesis path.

    Accepts the same `[(role, content), ...]` tuple form rag_crew.py already
    builds for ChatOpenAI. Reasoning inside `<think>...</think>`
    (the Qwen3-Thinking chat template opens `<think>` for us in the generation
    prompt) is yielded as `kind="thinking"` chunks so the UI can show it as a
    live reasoning trace; text after the closing tag is yielded as the normal
    `kind="answer"` chunks.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
        model_path: str | None = None,
    ):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens or config.HF_MAX_NEW_TOKENS
        self.model_path = model_path or config.LLM_IDENTIFIER

    def stream(self, messages):
        import torch
        from transformers import TextIteratorStreamer

        tokenizer, model = _load(self.model_path)
        chat_messages = [{"role": role, "content": content} for role, content in messages]
        prompt = tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        in_think = True  # the chat template already opened <think> for us
        buffer = ""
        # Keep a short tail unflushed so a "</think>" tag split across two
        # streamer pieces is never missed mid-flush.
        _TAIL = len("</think>") - 1
        for piece in streamer:
            if not in_think:
                if piece:
                    yield _Chunk(piece, kind="answer")
                continue
            buffer += piece
            if "</think>" in buffer:
                in_think = False
                before, after = buffer.split("</think>", 1)
                if before:
                    yield _Chunk(before, kind="thinking")
                buffer = ""
                if after:
                    yield _Chunk(after, kind="answer")
            elif len(buffer) > _TAIL:
                emit, buffer = buffer[:-_TAIL], buffer[-_TAIL:]
                yield _Chunk(emit, kind="thinking")
        thread.join()

        if in_think and buffer:
            # Ran out of tokens before the model closed </think> - surface the
            # raw buffer rather than silently returning an empty answer.
            yield _Chunk(buffer, kind="thinking")
