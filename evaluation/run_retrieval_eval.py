# evaluation/run_retrieval_eval.py
import json
from app.api.dependencies import get_embedder, get_vector_store
from app.retrieval.retriever import Retriever
from app.rag.pipeline import RAGPipeline

def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(questions, pipeline, k=5):
    results = []

    for q in questions:
        query = q["question"]
        relevant = set(q["relevant_docs"])

        output = pipeline.retrieve_only(query, k=k)
        
        retrieved_sources = {
            chunk.metadata.source for chunk in output["chunks"]
        }

        retrieved_docs = set(
            src.replace(".txt", "") for src in retrieved_sources
        )

        precision = (
            len(relevant & retrieved_docs) / len(retrieved_docs)
            if retrieved_docs else 0.0
        )
        recall = len(relevant & retrieved_docs) / len(relevant)

        results.append({
            "question": query,
            "precision@k": precision,
            "recall@k": recall,
            "retrieved": list(retrieved_docs),
            "expected": list(relevant)
        })

    return results


if __name__ == "__main__":
    embedder = get_embedder()
    vector_store = get_vector_store()
    retriever = Retriever(embedder, vector_store)

    pipeline = RAGPipeline(retriever=retriever, llm_generator=None)

    single = load_questions("evaluation/questions_single.json")
    cross = load_questions("evaluation/questions_cross.json")

    print("Single-document evaluation")
    print(evaluate(single, pipeline))

    print("\nCross-document evaluation")
    print(evaluate(cross, pipeline))
