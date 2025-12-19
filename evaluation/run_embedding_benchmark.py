# evaluation/run_embedding_benchmark.py
import argparse

from evaluation.embedding_models import EMBEDDING_MODELS
from evaluation.run_retrieval_eval import evaluate, load_questions
from evaluation.ingest_corpus import ingest_all

from app.embedding.embedder import Embedder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
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

    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retriever type to use",
    )

    args = parser.parse_args()

    USE_RERANKER = args.reranker
    RETRIEVER_TYPE = args.retriever

    single_questions = load_questions("evaluation/questions_single.json")
    cross_questions = load_questions("evaluation/questions_cross.json")
    adversarial_questions = load_questions("evaluation/questions_adversarial.json")
    noisy_questions = load_questions("evaluation/questions_noisy.json")

    KS = [1, 2, 3, 5]
    results = {}

    for name, model_name in EMBEDDING_MODELS.items():
        print(f"\n=== Evaluating embedding model: {name} ===")
        print(f"Retriever: {RETRIEVER_TYPE}")

        vector_store = ingest_all(model_name)

        # ----------------------------
        # Retriever selection
        # ----------------------------
        if RETRIEVER_TYPE == "dense":
            retriever = DenseRetriever(
                Embedder(model_name),
                vector_store,
            )

        elif RETRIEVER_TYPE == "bm25":
            retriever = BM25Retriever(vector_store.chunks)

        elif RETRIEVER_TYPE == "hybrid":
            dense = DenseRetriever(
                Embedder(model_name),
                vector_store,
            )
            bm25 = BM25Retriever(vector_store.chunks)
            retriever = HybridRetriever(dense, bm25)

        else:
            raise ValueError(f"Unknown retriever type: {RETRIEVER_TYPE}")

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
    print(f"Retriever enabled: {RETRIEVER_TYPE}")
    print(f"Reranker enabled : {USE_RERANKER}")

    for model, ks in results.items():
        print(f"\nModel: {model}")
        for k, metrics in ks.items():
            print(f"  {k}")
            print("    Single-doc :", metrics["single"])
            print("    Cross-doc  :", metrics["cross"])
            print("    Adversarial:", metrics["adversarial"])
            print("    Noisy      :", metrics["noisy"])
