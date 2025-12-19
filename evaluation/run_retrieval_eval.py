# evaluation/run_retrieval_eval.py
import json
from typing import Dict, List

from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.retrieval.dense_retriever import DenseRetriever
from app.rag.pipeline import RAGPipeline


def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(questions: List[Dict], pipeline: RAGPipeline, k: int = 5) -> Dict:
    """
    Evaluate retrieval performance and explicitly track failure cases.

    A failure is defined as retrieving ZERO relevant documents in top-k.
    """
    results = []
    failures = []

    for q in questions:
        query = q["question"]
        relevant = set(q["relevant_docs"])

        output = pipeline.retrieve_only(query, k=k)

        retrieved_sources = {
            chunk.metadata.source for chunk in output["chunks"]
        }

        retrieved_docs = {
            src.replace(".txt", "") for src in retrieved_sources
        }

        intersection = relevant & retrieved_docs

        precision = (
            len(intersection) / len(retrieved_docs)
            if retrieved_docs else 0.0
        )
        recall = len(intersection) / len(relevant)

        result = {
            "question": query,
            "precision@k": precision,
            "recall@k": recall,
            "retrieved": list(retrieved_docs),
            "expected": list(relevant),
        }

        results.append(result)

        # 🚨 Explicit failure detection
        if len(intersection) == 0:
            failures.append(result)

    return {
        "results": results,
        "failures": failures,
    }


if __name__ == "__main__":
    # This block is for quick, local debugging only.
    # The benchmark runner is the primary entry point.

    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = VectorStore(dim=embedder.embedding_dimension)
    vector_store.load("data/faiss.index", "data/faiss_meta.json")

    retriever = DenseRetriever(embedder, vector_store)

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_generator=None,
    )

    single = load_questions("evaluation/questions_single.json")
    cross = load_questions("evaluation/questions_cross.json")

    print("Single-document evaluation")
    print(evaluate(single, pipeline))

    print("\nCross-document evaluation")
    print(evaluate(cross, pipeline))
