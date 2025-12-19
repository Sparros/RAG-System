# evaluation/run_rag_eval.py
import argparse

from evaluation.run_retrieval_eval import load_questions
from evaluation.ingest_corpus import ingest_all
from evaluation.metrics import (
    citation_present_rate,
    keyword_coverage,
    context_overlap_score,
    sentence_grounding_rate,
    hallucination_rate,
)

from app.embedding.embedder import Embedder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import LLMGenerator
from app.llm.ollama_client import OllamaClient
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker


def run_rag_eval(questions, pipeline, k=3):
    outputs = []

    for q in questions:
        result = pipeline.answer_query(q["question"], k=k)

        outputs.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "chunks": result.get("chunks", []),
            "expected_keywords": q.get("expected_keywords", []),
        })

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation")

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

    single = load_questions("evaluation/questions_single.json")
    cross = load_questions("evaluation/questions_cross.json")

    # Ingest corpus (dense embeddings still needed for hybrid)
    vector_store = ingest_all("sentence-transformers/all-MiniLM-L6-v2")

    # ----------------------------
    # Retriever selection
    # ----------------------------
    if RETRIEVER_TYPE == "dense":
        retriever = DenseRetriever(
            Embedder("sentence-transformers/all-MiniLM-L6-v2"),
            vector_store,
        )

    elif RETRIEVER_TYPE == "bm25":
        retriever = BM25Retriever(vector_store.chunks)

    elif RETRIEVER_TYPE == "hybrid":
        dense = DenseRetriever(
            Embedder("sentence-transformers/all-MiniLM-L6-v2"),
            vector_store,
        )
        bm25 = BM25Retriever(vector_store.chunks)
        retriever = HybridRetriever(dense, bm25)

    else:
        raise ValueError(f"Unknown retriever type: {RETRIEVER_TYPE}")

    reranker = CrossEncoderReranker() if USE_RERANKER else None
    llm = LLMGenerator(OllamaClient(model="phi3"))

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_generator=llm,
        reranker=reranker,
    )

    print(f"\nRetriever enabled: {RETRIEVER_TYPE}")
    print(f"Reranker enabled : {USE_RERANKER}")

    for name, questions in [
        ("single", single),
        ("cross", cross),
    ]:
        outputs = run_rag_eval(questions, pipeline)

        print(f"\n=== RAG Evaluation ({name}) ===")
        print("Citation rate        :", citation_present_rate(outputs))
        print("Keyword coverage     :", keyword_coverage(outputs))
        print("Context overlap      :", context_overlap_score(outputs))
        print("Sentence grounding   :", sentence_grounding_rate(outputs))
        print("Hallucination rate   :", hallucination_rate(outputs))
