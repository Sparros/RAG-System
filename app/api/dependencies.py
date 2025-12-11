# app/api/dependencies.py
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.llm.ollama_client import OllamaClient
from app.rag.pipeline import RAGPipeline


# Global singletons (optional but recommended)
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_rag_pipeline: RAGPipeline | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        embedder = get_embedder()
        _vector_store = VectorStore(dim=embedder.embedding_dimension)
        # Optional: load existing index here
        # _vector_store.load("faiss.index", "metadata.pkl")
    return _vector_store

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
