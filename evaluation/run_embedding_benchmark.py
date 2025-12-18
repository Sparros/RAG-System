# evaluation/run_embedding_benchmark.py
import argparse

from evaluation.embedding_models import EMBEDDING_MODELS
from evaluation.run_retrieval_eval import evaluate, load_questions
from evaluation.ingest_corpus import ingest_all

from app.embedding.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker

from evaluation.metrics import mean_reciprocal_rank, top1_accuracy


def summarize(eval_output):
    results = eval_output["results"]

    return {
        "avg_precision@k": sum(r["precision@k"] for r in results) / len(results),
        "avg_recall@k": sum(r["recall@k"] for r in results) / len(results),
        "MRR": mean_reciprocal_rank(results),
        "Top-1 Accuracy": top1_accuracy(results),
        "num_failures": len(eval_output.get("failures", [])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding retrieval benchmark")
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking",
    )

    args = parser.parse_args()
    USE_RERANKER = args.reranker

    single_questions = load_questions("evaluation/questions_single.json")
    cross_questions = load_questions("evaluation/questions_cross.json")
    adversarial_questions = load_questions("evaluation/questions_adversarial.json")
    noisy_questions = load_questions("evaluation/questions_noisy.json")

    KS = [1, 2, 3, 5]
    results = {}

    for name, model_name in EMBEDDING_MODELS.items():
        print(f"\n=== Evaluating embedding model: {name} ===")

        vector_store = ingest_all(model_name)
        retriever = Retriever(Embedder(model_name), vector_store)

        reranker = CrossEncoderReranker() if USE_RERANKER else None

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_generator=None,
            reranker=reranker,
        )

        results[name] = {}

        for k in KS:
            results[name][f"k={k}"] = {
                "single": summarize(evaluate(single_questions, pipeline, k=k)),
                "cross": summarize(evaluate(cross_questions, pipeline, k=k)),
                "adversarial": summarize(evaluate(adversarial_questions, pipeline, k=k)),
                "noisy": summarize(evaluate(noisy_questions, pipeline, k=k)),
            }

    print("\n=== FINAL SUMMARY ===")
    print(f"Reranker enabled: {USE_RERANKER}")

    for model, ks in results.items():
        print(f"\nModel: {model}")
        for k, metrics in ks.items():
            print(f"  {k}")
            print("    Single-doc :", metrics["single"])
            print("    Cross-doc  :", metrics["cross"])
            print("    Adversarial:", metrics["adversarial"])
            print("    Noisy      :", metrics["noisy"])
