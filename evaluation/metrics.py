# evaluation/metrics.py
import re
from typing import List, Dict


# -----------------------------
# Retrieval-level metrics
# -----------------------------

def mean_reciprocal_rank(results: List[Dict]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).

    results: list of dicts with keys:
      - retrieved: list[str]
      - expected: list[str]
    """
    reciprocal_ranks = []

    for r in results:
        retrieved = r["retrieved"]
        expected = set(r["expected"])

        rr = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in expected:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def top1_accuracy(results: List[Dict]) -> float:
    """
    Percentage of queries where the top retrieved document is relevant.
    """
    correct = 0

    for r in results:
        if not r["retrieved"]:
            continue
        if r["retrieved"][0] in r["expected"]:
            correct += 1

    return correct / len(results)


# -----------------------------
# Answer-level RAG metrics
# -----------------------------

def citation_present_rate(outputs: List[Dict]) -> float:
    """
    Percentage of answers that include at least one citation marker.
    """
    count = 0
    for o in outputs:
        if o.get("answer") and "[source:" in o["answer"].lower():
            count += 1
    return count / len(outputs)


def keyword_coverage(outputs: List[Dict]) -> float:
    """
    Average fraction of expected keywords present in the answer.
    Only applied when expected_keywords exist.
    """
    scores = []

    for o in outputs:
        keywords = o.get("expected_keywords", [])
        answer = o.get("answer")

        if not keywords or not answer:
            continue

        answer_lc = answer.lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lc)
        scores.append(hits / len(keywords))

    return sum(scores) / len(scores) if scores else None


# -----------------------------
# Context overlap
# -----------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "this", "that", "from", "as",
    "it", "at", "which"
}


def _normalize(text: str) -> List[str]:
    """
    Lowercase, remove punctuation, split, and drop stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS]


def context_overlap_score(outputs: List[Dict]) -> float:
    """
    Measures lexical grounding between answer and retrieved context.

    Score = |answer_tokens ∩ context_tokens| / |answer_tokens|
    """
    scores = []

    for o in outputs:
        answer = o.get("answer")
        chunks = o.get("chunks", [])

        if not answer or not chunks:
            continue

        answer_tokens = set(_normalize(answer))
        context_text = " ".join(c.text for c in chunks)
        context_tokens = set(_normalize(context_text))

        if not answer_tokens:
            continue

        overlap = answer_tokens & context_tokens
        scores.append(len(overlap) / len(answer_tokens))

    return sum(scores) / len(scores) if scores else 0.0
