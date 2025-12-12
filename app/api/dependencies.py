# app/api/dependencies.py
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.retrieval.retriever import Retriever
from app.llm.generator import LLMGenerator
from app.llm.ollama_client import OllamaClient
from app.rag.pipeline import RAGPipeline
from app.core.settings import ensure_dirs, FAISS_INDEX_PATH, FAISS_META_PATH

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
        if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
            _vector_store.load(str(FAISS_INDEX_PATH), str(FAISS_META_PATH))
    return _vector_store

def get_rag_pipeline() -> RAGPipeline:
    """
    Builds or returns a cached RAGPipeline instance.
    FastAPI will call this using dependency injection.
    """
    global _rag_pipeline

    if _rag_pipeline is None:
        embedder = get_embedder()
        vector_store = get_vector_store()  # ensures persistence load happens

        retriever = Retriever(embedder=embedder, vector_store=vector_store)

        ollama_client = OllamaClient(model="phi3")
        llm_generator = LLMGenerator(client=ollama_client)

        _rag_pipeline = RAGPipeline(retriever=retriever, llm_generator=llm_generator)

    return _rag_pipeline