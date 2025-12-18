# evaluation/run_rag_eval.py
import argparse

from evaluation.run_retrieval_eval import load_questions
from evaluation.ingest_corpus import ingest_all
from evaluation.metrics import (
    citation_present_rate,
    keyword_coverage,
    context_overlap_score,
)

from app.embedding.embedder import Embedder
from app.retrieval.retriever import Retriever
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

    args = parser.parse_args()
    USE_RERANKER = args.reranker

    single = load_questions("evaluation/questions_single.json")
    cross = load_questions("evaluation/questions_cross.json")

    vector_store = ingest_all("sentence-transformers/all-MiniLM-L6-v2")
    embedder = Embedder()
    retriever = Retriever(embedder, vector_store)

    reranker = CrossEncoderReranker() if USE_RERANKER else None
    llm = LLMGenerator(OllamaClient(model="phi3"))

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_generator=llm,
        reranker=reranker,
    )

    print(f"\nReranker enabled: {USE_RERANKER}")

    for name, questions in [
        ("single", single),
        ("cross", cross),
    ]:
        outputs = run_rag_eval(questions, pipeline)

        print(f"\n=== RAG Evaluation ({name}) ===")
        print("Citation rate   :", citation_present_rate(outputs))
        print("Keyword coverage:", keyword_coverage(outputs))
        print("Context overlap :", context_overlap_score(outputs))
