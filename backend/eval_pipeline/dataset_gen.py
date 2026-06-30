"""
LogMind - eval dataset generator.

Samples chunks from the already-built index, stratified roughly evenly across
source documents, and asks an LLM to write one specific Q/A pair per sampled
chunk. Output is a *candidate* file for human review - nothing here is
auto-trusted into eval_set.json.

    python -m eval_pipeline.dataset_gen                       # ~30 candidates (default)
    python -m eval_pipeline.dataset_gen --n 60 --out eval_pipeline/data/candidates.json

Each candidate carries `gold_passage` (the full sampled chunk text) - the real
passage-level relevance field used by eval_pipeline/retrieval_metrics.py. Two
mechanical gates run before a candidate is written: duplicate questions are
dropped, and a self-retrieval check confirms the source chunk is actually
retrievable for its own generated question (drops ambiguous/unanswerable
questions before a human ever sees them).

After reviewing eval_pipeline/data/candidates.json against each gold_passage,
copy the good items into eval_pipeline/data/eval_set.json as-is - candidates are
already in the final schema (nothing to strip).
"""

import argparse
import json
import os
import random

from pydantic import BaseModel, Field

from agents.crew import get_retriever
from eval_pipeline.retrieval_metrics import passage_match

GEN_MODEL = os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini")
MIN_CHUNK_LEN = 400  # skip short/boilerplate chunks (headers, page footers, etc.)

GEN_PROMPT = """Here is a passage from an Indian income-tax study document (source: {source}):

\"\"\"
{passage}
\"\"\"

Write ONE specific, self-contained question that this passage directly and fully answers.
The question must stand on its own - someone asking it should NOT need to have read the
passage (don't write things like "according to the passage..."). Also write a short
reference answer using ONLY facts present in the passage, and 1-3 short lowercase
terms/phrases that a correct answer should contain."""


class GeneratedQA(BaseModel):
    question: str = Field(description="A specific, self-contained question this passage directly answers")
    reference_answer: str = Field(description="A one-to-two sentence answer using only facts in the passage")
    expect_terms: list[str] = Field(description="1-3 short lowercase terms/phrases the answer should contain")


def sample_chunks(n: int, seed: int = 42):
    """Stratified sample: roughly n / (number of source files) chunks per source,
    so generation coverage spans the whole corpus rather than the largest PDFs."""
    retriever = get_retriever()
    by_source: dict[str, list] = {}
    for doc in retriever.all_chunks():
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(doc)

    n_sources = max(len(by_source), 1)
    per_source = max(1, n // n_sources)
    rng = random.Random(seed)

    sampled = []
    for src, docs in by_source.items():
        candidates = [d for d in docs if len(d.page_content) > MIN_CHUNK_LEN]
        if not candidates:
            continue
        sampled.extend((src, d) for d in rng.sample(candidates, min(per_source, len(candidates))))

    rng.shuffle(sampled)
    return sampled[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="total candidates to generate")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "candidates.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=GEN_MODEL, api_key=os.environ["OPENAI_API_KEY"], temperature=0.4)
    structured = llm.with_structured_output(GeneratedQA)

    retriever = get_retriever()  # singleton; sample_chunks uses the same one
    sampled = sample_chunks(args.n, seed=args.seed)
    candidates = []
    seen_questions: set[str] = set()
    for i, (source, doc) in enumerate(sampled, 1):
        gen = structured.invoke(GEN_PROMPT.format(source=source, passage=doc.page_content))

        # Gate 1: drop duplicate questions (normalized).
        qnorm = " ".join(gen.question.lower().split())
        if qnorm in seen_questions:
            print(f"[{i}/{len(sampled)}] skip (duplicate): {gen.question}")
            continue
        seen_questions.add(qnorm)

        # Gate 2: the source chunk must actually be retrievable for its own
        # generated question - otherwise the question is ambiguous/unanswerable.
        pool, _ = retriever._hybrid_pool(gen.question)
        if not any(passage_match([doc.page_content], p.page_content) for p in pool):
            print(f"[{i}/{len(sampled)}] skip (source chunk not retrievable): {gen.question}")
            continue

        candidates.append({
            "id": f"gen-{len(candidates) + 1:03d}",
            "question": gen.question,
            "gold_passage": doc.page_content,
            "expected_sources": [source],
            "expect_terms": [t.lower() for t in gen.expect_terms],
            "reference_answer": gen.reference_answer,
        })
        print(f"[{i}/{len(sampled)}] keep: {source}: {gen.question}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(candidates)} candidates (from {len(sampled)} sampled) to {args.out}")
    print("Review each one against its gold_passage, then copy the good ones into "
          "eval_pipeline/data/eval_set.json as-is (already in the final schema).")


if __name__ == "__main__":
    main()
