"""
LogMind - Manager/router: a single lightweight classification call, NOT a
crewAI Agent. Decides whether a question needs the indexed knowledge base
("kb"), live web search ("web"), or can be answered directly from the model's
general knowledge ("general"). Runs before the Strategist in both
run_query() and stream_query().

Three-way routing. kb/web are mutually exclusive (a question never triggers
both in one request) - the simplest extension of what was originally a
binary kb/general label. The router always offers "web" as a possible
classification regardless of whether TAVILY_API_KEY is configured; a missing
key is handled per-request in agents/crew.py (graceful degradation to the
general-knowledge answer), not by hiding the route here.

Uses a DEDICATED model selection (ROUTER_MODEL, a models.yaml registry name,
independent of LLM_MODEL/EMBEDDING_MODEL/RERANKER_MODEL) so the
classification call can use a small/fast model even when the main pipeline
uses a large/slow one (e.g. a "thinking" HF model for the main LLM).
config.ROUTER_LOADER ("chat" vs "zeroshot") is checked first: "chat" prompts
a generative model (provider openai|huggingface|lmstudio, via
config.ROUTER_PROVIDER) for one word; "zeroshot" instead scores the question
via NLI entailment against short label descriptions using a dedicated
zero-shot classification model (e.g. deberta-v3-large-zeroshot) - see
llm/zeroshot.py.
"""

import os
import re

import config

_VALID_ROUTES = {"kb", "general", "web"}

_SYSTEM_PROMPT = (
    "You are a routing classifier for a question-answering system backed by a "
    "knowledge base of indexed documents and a live web search tool. Decide which "
    "of three ways to answer the user's question.\n\n"
    "Respond with EXACTLY one word, nothing else:\n"
    "  kb       - the question likely needs facts from the indexed documents\n"
    "  web      - the question needs current, external, or real-time information "
    "unlikely to be in the indexed documents or general training knowledge (e.g. "
    "today's exchange rate, a recent notification or circular, current news)\n"
    "  general  - the question is general knowledge, conversational, or unrelated "
    "to the indexed domain, and does not need current/external information either\n\n"
    'When in doubt, prefer "kb" (a false negative silently gives an ungrounded '
    "answer; a false positive just costs one extra retrieval pass)."
)


def _parse_label(raw: str) -> str:
    """Robustly pull "kb"/"web"/"general" out of a raw completion. Defaults to
    "kb" on anything ambiguous/unparseable - the safe failure direction, since
    treating an answerable question as needing retrieval just costs latency,
    while treating a kb-needing question as "general"/"web" silently produces
    an ungrounded or wrongly-grounded answer that looks just as confident as a
    correctly grounded one."""
    if not raw:
        return "kb"
    text = raw.strip().lower()
    if text in _VALID_ROUTES:
        return text
    m = re.search(r"\b(kb|web|general)\b", text)
    if m:
        return m.group(1)
    return "kb"


def _classify_openai_compatible(question: str, model: str, base_url: str | None, api_key: str) -> str:
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=model, api_key=api_key, temperature=0.0, max_tokens=8)
    if base_url:
        kwargs["base_url"] = base_url
    chat = ChatOpenAI(**kwargs)
    result = chat.invoke([("system", _SYSTEM_PROMPT), ("user", question)])
    return _parse_label(getattr(result, "content", "") or "")


def _classify_huggingface(question: str) -> str:
    from llm.huggingface import generate_once

    raw = generate_once(
        model_path=config.ROUTER_MODEL_PATH,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_new_tokens=8,
        temperature=0.0,
    )
    return _parse_label(raw)


def _classify_zeroshot(question: str) -> str:
    from llm.zeroshot import classify as zs_classify

    labels = {
        config.ROUTER_ZEROSHOT_KB_LABEL: "kb",
        config.ROUTER_ZEROSHOT_WEB_LABEL: "web",
        config.ROUTER_ZEROSHOT_GENERAL_LABEL: "general",
    }
    # "This example is {}." (the pipeline's usual default) measurably
    # outperforms passing these labels through unwrapped - see config.py's
    # ROUTER_ZEROSHOT_KB_LABEL comment.
    top_label, _score = zs_classify(
        question, candidate_labels=list(labels), hypothesis_template="This example is {}."
    )
    return labels.get(top_label, "kb")


def classify(question: str) -> dict:
    """Returns {"route": "kb"|"general"|"web", "reason": str}. Never raises -
    any failure in the classification call itself falls back to "kb" (the
    safe default) rather than taking down the whole request."""
    try:
        if config.ROUTER_LOADER == "zeroshot":
            route = _classify_zeroshot(question)
        elif config.ROUTER_PROVIDER == "lmstudio":
            route = _classify_openai_compatible(
                question,
                model=config.ROUTER_MODEL_PATH,
                base_url=config.LMSTUDIO_BASE_URL,
                api_key=config.LMSTUDIO_API_KEY,
            )
        elif config.ROUTER_PROVIDER == "huggingface":
            route = _classify_huggingface(question)
        else:
            route = _classify_openai_compatible(
                question,
                model=config.ROUTER_MODEL_PATH,
                base_url=None,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        return {
            "route": route,
            "reason": f"classified by {config.ROUTER_PROVIDER}:{config.get_router_llm_label()}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "route": "kb",
            "reason": f"classification failed ({type(exc).__name__}: {exc}); defaulting to kb",
        }
