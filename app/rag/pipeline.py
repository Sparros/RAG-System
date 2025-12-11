# app/rag/pipeline.py

from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.models.document_models import DocumentChunk
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.llm.ollama_client import OllamaClient
from app.llm.generator import LLMGenerator
from app.retrieval.retriever import Retriever


class RAGPipeline:
    """
    Main RAG orchestration class.
    """

    def __init__(self, retriever: Retriever, llm_generator: LLMGenerator):
        self.retriever = retriever
        self.llm_generator = llm_generator

    def answer_query(self, query: str, k: int = 5):
        # 1. Retrieve the relevant chunks
        results = self.retriever.retrieve(query, k=k)

        # Build context string
        context = "\n\n".join(chunk.text for _, chunk in results)

        # Construct prompt
        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        # 2. Generate answer from LLM
        answer = self.llm_generator.generate(prompt)

        # 3. Build source metadata
        sources = [
            {"chunk_id": chunk.chunk_id, "source": chunk.metadata.source}
            for _, chunk in results
        ]

        return {"answer": answer, "sources": sources}

def get_rag_pipeline():
    embedder = Embedder()
    vector_store = VectorStore(dim=embedder.embedding_dimension)
    retriever = Retriever(embedder, vector_store)

    llm = LLMGenerator(OllamaClient("phi3"))

    return RAGPipeline(retriever, llm)

