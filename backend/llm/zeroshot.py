"""
LogMind - zero-shot NLI classifier backend (ROUTER_MODEL resolving to a
models.yaml entry with loader: zeroshot).

Backs the Manager's router (agents/manager.py) with a dedicated NLI
sequence-classification model (e.g. MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
via transformers' zero-shot-classification pipeline, instead of a generative
LLM. This is a different model class than llm/huggingface.py's causal LMs
(AutoModelForSequenceClassification, no chat template, no token generation) -
NLI entailment scoring against short candidate-label descriptions is a more
direct fit for a binary routing decision than prompting a generative model to
emit exactly one word, and is typically much faster.
"""

from functools import lru_cache

import config


@lru_cache(maxsize=2)
def _load(model_path: str):
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=model_path, device=device)


def classify(
    text: str,
    candidate_labels: list[str],
    hypothesis_template: str = "{}",
) -> tuple[str, float]:
    """Returns (top_label, top_score) - the highest-scoring candidate label and
    its entailment score. hypothesis_template defaults to "{}" (no wrapping)
    since LogMind's candidate labels are already full sentences, not the short
    noun phrases the pipeline's own default template ("This example is {}.")
    is designed to wrap."""
    clf = _load(config.ROUTER_MODEL_PATH)
    result = clf(
        text,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
    )
    return result["labels"][0], result["scores"][0]
