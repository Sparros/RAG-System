# app/rag/pipeline.py

from typing import Dict, Any
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator


class RAGPipeline:
    """
    Orchestrates:
      1. Retrieval
      2. Prompt building
      3. LLM answer generation
    """

    def __init__(self, retriever: Retriever, llm_generator: LLMGenerator):
        self.retriever = retriever
        self.llm_generator = llm_generator

    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Full RAG flow: retrieve → generate answer → return with sources.
        """
        # 1. Retrieve top-k semantic matches
        retrieved = self.retriever.retrieve(query, k=k)
        chunks = [chunk for _, chunk in retrieved]

        # 2. Generate LLM answer using context chunks
        answer = self.llm_generator.generate_answer(query, chunks)

        # 3. Collect document sources
        sources = list({chunk.metadata.source for chunk in chunks})

        return {
            "answer": answer,
            "sources": sources,
        }
