# app/rag/pipeline.py

from typing import Dict, Any, List, Optional
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.models.document_models import DocumentChunk


class RAGPipeline:
    """
    Orchestrates:
      1. Retrieval
      2. Optional LLM generation
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_generator: Optional[LLMGenerator] = None,
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

    def retrieve_only(self, query: str, k: int = 5):
        """
        Retrieval-only path for evaluation.
        """
        retrieved = self.retriever.retrieve(query, k=k)
        chunks = [chunk for _, chunk in retrieved]

        return {
            "query": query,
            "chunks": chunks,
        }

    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Full RAG flow if LLM is enabled.
        Falls back to retrieval-only if not.
        """
        retrieved = self.retriever.retrieve(query, k=k)
        chunks = [chunk for _, chunk in retrieved]

        sources = list({c.metadata.source for c in chunks})

        # Retrieval-only fallback
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
        }
    

