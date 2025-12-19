# app/rag/pipeline.py

from typing import Dict, Any, Optional
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.retrieval.reranker import CrossEncoderReranker


class RAGPipeline:
    """
    Orchestrates:
      1. Retrieval
      2. Optional reranking
      3. Optional LLM generation
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_generator: Optional[LLMGenerator] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator
        self.reranker = reranker

    def retrieve_only(self, query: str, k: int = 5):
        """
        Retrieval-only path (used for evaluation).
        """
        # Over-retrieve if reranking is enabled
        retrieve_k = max(k * 4, 10) if self.reranker else k

        retrieved = self.retriever.retrieve(query, k=retrieve_k)
        chunks = [chunk for _, chunk in retrieved]

        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, k)

        return {
            "query": query,
            "chunks": chunks,
        }

    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Full RAG flow with optional reranking and LLM generation.
        """
        retrieve_k = max(k * 4, 10) if self.reranker else k

        retrieved = self.retriever.retrieve(query, k=retrieve_k)
        chunks = [chunk for _, chunk in retrieved]

        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, k)

        sources = list({c.metadata.source for c in chunks})

        if self.llm_generator is None:
            return {
                "answer": None,
                "sources": sources,
                "chunks": chunks,
            }

        answer = self.llm_generator.generate_answer(query, chunks)

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
        }
