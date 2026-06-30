"""
LogMind - LLM-as-judge scoring for the eval pipeline.

Two independent judgments per eval item:
  * faithfulness - is every claim in the generated answer actually supported by the
    numbered context that was handed to the synthesizer? Catches hallucination even
    when retrieval itself succeeded.
  * correctness  - does the generated answer convey the same key facts as the
    human-written reference_answer? Catches "retrieval was fine but the answer is
    still wrong" cases that a keyword/source check can't see.

The judge is always a fixed OpenAI model (EVAL_JUDGE_MODEL, default gpt-4o-mini),
regardless of which PROVIDER is under test (including matrix-mode runs). A judge
that changed along with the system under test couldn't be trusted to show real
differences between combos - it could just as easily be grading itself.
"""

import os
from functools import lru_cache

from pydantic import BaseModel, Field


JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0.0 = completely fails, 1.0 = fully satisfies")
    reasoning: str = Field(description="One or two sentences explaining the score")


FAITHFULNESS_PROMPT = """You are grading whether an AI-generated answer is faithful to its source context.

Source context (numbered passages the answer was allowed to use):
\"\"\"
{context}
\"\"\"

Generated answer:
\"\"\"
{answer}
\"\"\"

Score 1.0 if every factual claim in the answer is directly supported by the context above.
Score 0.0 if the answer contains claims not supported by (or contradicting) the context.
Score proportionally for partial support. Ignore style or completeness - only judge factual
grounding in the given context."""

CORRECTNESS_PROMPT = """You are grading whether an AI-generated answer conveys the same key facts as a reference answer.

Question: {question}

Reference answer (ground truth):
\"\"\"
{reference_answer}
\"\"\"

Generated answer:
\"\"\"
{generated_answer}
\"\"\"

Score 1.0 if the generated answer conveys the same key facts as the reference answer (wording
can differ freely). Score 0.0 if it misses or contradicts the key facts. Score proportionally
for partial matches."""


@lru_cache(maxsize=1)
def _judge_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=JUDGE_MODEL, api_key=os.environ["OPENAI_API_KEY"], temperature=0)


def score_faithfulness(context: str, answer: str) -> JudgeScore:
    if not context.strip() or not answer.strip():
        return JudgeScore(score=0.0, reasoning="Empty context or answer - nothing to grade.")
    structured = _judge_llm().with_structured_output(JudgeScore)
    return structured.invoke(FAITHFULNESS_PROMPT.format(context=context, answer=answer))


def score_correctness(question: str, reference_answer: str, generated_answer: str) -> JudgeScore:
    if not generated_answer.strip():
        return JudgeScore(score=0.0, reasoning="Empty generated answer - nothing to grade.")
    structured = _judge_llm().with_structured_output(JudgeScore)
    return structured.invoke(CORRECTNESS_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    ))
