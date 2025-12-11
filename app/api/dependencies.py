from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.llm.ollama_client import OllamaClient
from app.rag.pipeline import RAGPipeline


# Global singletons (optional but recommended)
_embedder = None
_vector_store = None
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Builds or returns a cached RAGPipeline instance.
    FastAPI will call this using dependency injection.
    """

    global _embedder, _vector_store, _rag_pipeline

    # Create embedder once
    if _embedder is None:
        _embedder = Embedder()

    # Create vector store once
    if _vector_store is None:
        dim = _embedder.embedding_dimension
        _vector_store = VectorStore(dim=dim)

        # If you have saved indexes, load them here:
        # _vector_store.load("faiss.index", "metadata.pkl")

    # Create retriever
    retriever = Retriever(embedder=_embedder, vector_store=_vector_store)

    # Create Ollama LLM client
    ollama_client = OllamaClient(model="phi3")
    llm_generator = LLMGenerator(client=ollama_client)

    # Create RAG pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(retriever=retriever, llm_generator=llm_generator)

    return _rag_pipeline
