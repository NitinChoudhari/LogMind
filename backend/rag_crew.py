"""
LogMind - the agentic RAG crew.

Three agents, each placed where an LLM genuinely adds value:

  1. Query Strategist  - turns the question into focused sub-queries (or leaves
     a simple question alone). Replaces the original 3-call decompose/dedup chain.
  2. Researcher        - gathers grounded evidence via the HybridSearch tool.
  3. Answer Synthesizer- writes the final answer with [n] citations and flags
     insufficient context.

Default run is orchestrated so citation numbers are deterministic and always
match the sources shown in the UI. Set AGENTIC_RETRIEVAL=1 to let the Researcher
agent drive retrieval via the tool instead (better with capable models).
"""

import os
import re
import io
import sys
import json
from contextlib import redirect_stdout
from typing import List, Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

import config
from retrieval import Retriever, SourceCollector

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee:
    """Write to several streams at once (keep terminal output AND capture it)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


# Build the retriever once (loads the persisted index + BM25 corpus).
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# --------------------------------------------------------------------------- #
# HybridSearch as a crewAI tool
# --------------------------------------------------------------------------- #
class _SearchArgs(BaseModel):
    query: str = Field(..., description="A focused search query")


class HybridSearchTool(BaseTool):
    name: str = "hybrid_search"
    description: str = (
        "Search the indexed documents for passages relevant to a query. "
        "Returns numbered passages with their source. Call once per sub-query."
    )
    args_schema: Type[BaseModel] = _SearchArgs

    # carries per-request retrieval state
    _retriever: Retriever
    _collector: SourceCollector

    def __init__(self, retriever: Retriever, collector: SourceCollector, **kw):
        super().__init__(**kw)
        self._retriever = retriever
        self._collector = collector

    def _run(self, query: str) -> str:
        entries = self._collector.add(self._retriever.search(query))
        if not entries:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{e.n}] (source: {e.source}) {e.snippet}" for e in entries
        )


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
def _strategist(llm) -> Agent:
    return Agent(
        role="Query Strategist",
        goal="Break a user question into the minimum set of focused retrieval sub-queries.",
        backstory=(
            "You optimise questions for retrieval. Simple questions stay as one query; "
            "multi-part questions become 2-4 distinct sub-queries that each target a "
            "different aspect using concrete keywords from the domain."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def _researcher(llm, tools) -> Agent:
    return Agent(
        role="Researcher",
        goal="Gather the most relevant grounded passages for each sub-query.",
        backstory="You retrieve evidence and never rely on prior knowledge - only the documents.",
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
    )


def _synthesizer(llm) -> Agent:
    return Agent(
        role="Answer Synthesizer",
        goal="Write a precise, grounded answer that cites its sources by number.",
        backstory=(
            "You answer strictly from the provided numbered context. You cite every claim "
            "with its [n]. You never invent information. If the context is insufficient, you "
            "say so plainly rather than guessing."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


SYNTH_RULES = (
    "Answer the question using ONLY the numbered context. Rules:\n"
    "- Cite every fact with its source number like [2]. Cite multiple as [1][3].\n"
    "- Procedures: numbered steps, copied faithfully. Lists: bullets. Else: short paragraphs.\n"
    "- Action/answer first, explanation second. Be concise; omit anything not asked.\n"
    "- If the context does not contain the answer, reply exactly: "
    "\"Not found in the indexed documents.\""
)

# System prompt for the streamed synthesis path (mirrors the Synthesizer agent).
SYNTH_SYSTEM = (
    "You answer strictly from the provided numbered context. You cite every claim "
    "with its [n]. You never invent information. If the context is insufficient, you "
    "say so plainly rather than guessing."
)


# --------------------------------------------------------------------------- #
# Sub-query parsing
# --------------------------------------------------------------------------- #
def _clean_subqueries(raw: str, fallback: str) -> List[str]:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip "1.", "1)", "- ", "* ", "Q:" style prefixes
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*\u2022]|Q\s*\d*\s*[:.])\s*", "", line).strip()
        # drop obvious preamble lines
        if re.match(r"(?i)^(here|these|sub-?quer|the following|sure|certainly)\b", line):
            continue
        if len(line) < 3:
            continue
        out.append(line)
    seen, deduped = set(), []
    for q in out:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(q)
    return deduped[:4] or [fallback]


def _run_agent(agent: Agent, description: str, expected: str, inputs: dict, trace: list | None = None) -> str:
    crew = Crew(agents=[agent], tasks=[
        Task(description=description, expected_output=expected, agent=agent)
    ], process=Process.sequential, verbose=True)
    buf = io.StringIO()
    real_stdout = sys.stdout
    # Tee: keep printing to the terminal AND capture for the UI trace.
    with redirect_stdout(_Tee(real_stdout, buf)):
        out = crew.kickoff(inputs=inputs)
    if trace is not None:
        trace.append(_ANSI.sub("", buf.getvalue()))
    return str(out.raw).strip()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_query(question: str) -> dict:
    llm = config.get_crew_llm()
    retriever = get_retriever()
    collector = SourceCollector()
    trace: list = []

    # 1) Strategist -> sub-queries
    raw = _run_agent(
        _strategist(llm),
        description=(
            "User question: {question}\n\n"
            "Output the retrieval sub-queries, ONE PER LINE, no numbering, no preamble. "
            "A simple question should produce a single line."
        ),
        expected="One sub-query per line.",
        inputs={"question": question},
        trace=trace,
    )
    sub_queries = _clean_subqueries(raw, fallback=question)

    # 2) Retrieval (deterministic by default; agentic if requested)
    if os.getenv("AGENTIC_RETRIEVAL") == "1":
        tool = HybridSearchTool(retriever, collector)
        _run_agent(
            _researcher(llm, [tool]),
            description=(
                "Use hybrid_search once for each of these sub-queries and report what you "
                "found:\n{subqs}"
            ),
            expected="A short note of the passages gathered.",
            inputs={"subqs": "\n".join(sub_queries)},
            trace=trace,
        )
    else:
        for sq in sub_queries:
            collector.add(retriever.search(sq))

    context = collector.numbered_context()
    if not context:
        return {
            "answer": "Not found in the indexed documents.",
            "sub_queries": sub_queries,
            "sources": [],
            "trace": "\n".join(trace),
            "provider": config.PROVIDER,
            "model": config.active_model_label(),
        }

    # 3) Synthesizer -> grounded answer with [n] citations
    answer = _run_agent(
        _synthesizer(llm),
        description=(
            "Question: {question}\n\n"
            "Numbered context:\n{context}\n\n" + SYNTH_RULES
        ),
        expected="A grounded, cited answer.",
        inputs={"question": question, "context": context},
        trace=trace,
    )

    return {
        "answer": answer,
        "sub_queries": sub_queries,
        "sources": collector.as_payload(),
        "trace": "\n".join(trace),
        "provider": config.PROVIDER,
        "model": config.active_model_label(),
    }


# --------------------------------------------------------------------------- #
# Streaming entry point (Server-Sent Events)
# --------------------------------------------------------------------------- #
def stream_query(question: str):
    """Yield SSE events: live trace, planned sub-queries, sources, then the
    answer token-by-token. Planning stays a crewAI agent; the final synthesis is
    streamed directly from the LLM so the answer appears as it's generated."""

    def ev(obj) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    try:
        llm = config.get_crew_llm()
        retriever = get_retriever()
        collector = SourceCollector()

        # 1) Strategist (crewAI agent) -> sub-queries
        yield ev({"type": "trace", "line": "> Strategist: planning sub-queries..."})
        raw = _run_agent(
            _strategist(llm),
            description=(
                "User question: {question}\n\n"
                "Output the retrieval sub-queries, ONE PER LINE, no numbering, no preamble. "
                "A simple question should produce a single line."
            ),
            expected="One sub-query per line.",
            inputs={"question": question},
        )
        sub_queries = _clean_subqueries(raw, fallback=question)
        yield ev({"type": "subqueries", "items": sub_queries})

        # 2) Retrieval (hybrid + rerank) per sub-query
        for sq in sub_queries:
            yield ev({"type": "trace", "line": f"> Researcher: hybrid search + rerank -> '{sq}'"})
            entries = collector.add(retriever.search(sq))
            yield ev({"type": "trace", "line": f"  {len(entries)} passages kept"})

        context = collector.numbered_context()
        yield ev({"type": "sources", "sources": collector.as_payload()})

        if not context:
            yield ev({"type": "token", "text": "Not found in the indexed documents."})
            yield ev({"type": "done", "provider": config.PROVIDER, "model": config.active_model_label()})
            return

        # 3) Synthesizer -> streamed answer tokens
        yield ev({"type": "trace", "line": "> Synthesizer: writing grounded answer..."})
        chat = config.get_chat_model()
        messages = [
            ("system", SYNTH_SYSTEM),
            ("user", f"Question: {question}\n\nNumbered context:\n{context}\n\n" + SYNTH_RULES),
        ]
        for chunk in chat.stream(messages):
            text = getattr(chunk, "content", "") or ""
            if text:
                yield ev({"type": "token", "text": text})

        yield ev({"type": "done", "provider": config.PROVIDER, "model": config.active_model_label()})

    except Exception as exc:  # noqa: BLE001
        yield ev({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
