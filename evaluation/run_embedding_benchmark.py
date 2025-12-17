# evaluation/run_embedding_benchmark.py

from evaluation.embedding_models import EMBEDDING_MODELS
from evaluation.run_retrieval_eval import evaluate, load_questions
from evaluation.ingest_corpus import ingest_all

from app.embedding.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.rag.pipeline import RAGPipeline

from evaluation.metrics import mean_reciprocal_rank, top1_accuracy

def summarize(results):
    return {
        "avg_precision@k": sum(r["precision@k"] for r in results) / len(results),
        "avg_recall@k": sum(r["recall@k"] for r in results) / len(results),
        "MRR": mean_reciprocal_rank(results),
        "Top-1 Accuracy": top1_accuracy(results),
    }


if __name__ == "__main__":
    single_questions = load_questions("evaluation/questions_single.json")
    cross_questions = load_questions("evaluation/questions_cross.json")

    all_results = {}

    for name, model_name in EMBEDDING_MODELS.items():
        print(f"\n=== Evaluating embedding model: {name} ===")

        # Fresh ingestion per model
        vector_store = ingest_all(model_name)

        embedder = Embedder(model_name=model_name)
        retriever = Retriever(embedder, vector_store)
        pipeline = RAGPipeline(retriever=retriever, llm_generator=None)

        single_results = evaluate(single_questions, pipeline)
        cross_results = evaluate(cross_questions, pipeline)

        all_results[name] = {
            "single": summarize(single_results),
            "cross": summarize(cross_results),
        }

    print("\n=== FINAL SUMMARY ===")
    for model, metrics in all_results.items():
        print(f"\nModel: {model}")
        print("  Single-doc:", metrics["single"])
        print("  Cross-doc :", metrics["cross"])
