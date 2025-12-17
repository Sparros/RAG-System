# evaluation/metrics.py
def mean_reciprocal_rank(results):
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

def top1_accuracy(results):
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
