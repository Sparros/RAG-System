# evaluation/metrics.py
import re
from typing import List, Dict, Set


# -----------------------------
# Retrieval-level metrics
# -----------------------------

def mean_reciprocal_rank(results: List[Dict]) -> float:
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
    count = 0
    for o in outputs:
        if o.get("answer") and "[source:" in o["answer"].lower():
            count += 1
    return count / len(outputs)


def keyword_coverage(outputs: List[Dict]) -> float:
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
# Context overlap (grounding)
# -----------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "this", "that", "from", "as",
    "it", "at", "which", "also", "has", "have"
}


def _normalize(text: str) -> Set[str]:
    """
    Normalize text into content-bearing tokens.
    """
    # Remove citations like [1], (source: ...)
    text = re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = set()
    for tok in text.split():
        if tok in _STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        tokens.add(tok)

    return tokens


def context_overlap_score(outputs: List[Dict]) -> float:
    """
    Measures lexical grounding between generated answers and retrieved context.

    Score = |answer_tokens ∩ context_tokens| / |answer_tokens|
    """
    scores = []

    for o in outputs:
        answer = o.get("answer")
        chunks = o.get("chunks", [])

        # Integrity check: grounding requires evidence
        if answer is not None:
            assert chunks, "Chunks missing — grounding impossible"

        if not answer or not chunks:
            continue

        answer_tokens = _normalize(answer)
        context_text = " ".join(c.text for c in chunks)
        context_tokens = _normalize(context_text)

        if not answer_tokens:
            continue

        overlap = answer_tokens & context_tokens
        scores.append(len(overlap) / len(answer_tokens))

    return sum(scores) / len(scores) if scores else 0.0

