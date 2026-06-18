"""
LogMind - tiny eval harness.

    python eval.py

Measures, against a small labelled set:
  * retrieval hit-rate  - did the chunk from the expected source get retrieved?
  * answer coverage      - does the final answer contain the expected key term?

This is the thing that turns "I think it's good" into a number you can tune
against (chunk size, k, weights, MMR). Extend EVAL_SET with your own Q/A pairs.
Run it twice - once with PROVIDER=openai, once with =ollama - to compare.
"""

from rag_crew import run_query

# Each item: question, the source file the answer should come from,
# and a term the answer should contain (lowercased match).
EVAL_SET = [
    {
        "q": "What does error code on the washing machine guide indicate?",
        "expected_source": "error_codes__manual__production__guide__05-04.2026.txt",
        "expect_term": "error",
    },
    {
        "q": "What is the policy when an incident is put on hold?",
        "expected_source": "policy_on_hold__incident__production__troubleshooting__29-03-2026__.txt",
        "expect_term": "hold",
    },
    {
        "q": "How do I troubleshoot the washing machine?",
        "expected_source": "washing_machine__user_manual__production__troubleshooting__05-04-2026__.txt",
        "expect_term": "machine",
    },
    {
        "q": "Explain what RAG is.",
        "expected_source": "explain_rag__summary__W3School__Manual__05-04-2026__.txt",
        "expect_term": "retrieval",
    },
]


def main():
    hits, covered = 0, 0
    print(f"{'Q':<48} {'retrieval':<10} {'coverage':<9}")
    print("-" * 70)
    for item in EVAL_SET:
        res = run_query(item["q"])
        sources = {s["source"] for s in res.get("sources", [])}
        retrieved = item["expected_source"] in sources
        answer = (res.get("answer") or "").lower()
        has_term = item["expect_term"].lower() in answer
        hits += retrieved
        covered += has_term
        print(f"{item['q'][:46]:<48} {'HIT' if retrieved else 'miss':<10} {'ok' if has_term else '--':<9}")

    n = len(EVAL_SET)
    print("-" * 70)
    print(f"retrieval hit-rate: {hits}/{n} = {hits / n:.0%}")
    print(f"answer coverage:    {covered}/{n} = {covered / n:.0%}")


if __name__ == "__main__":
    main()
