from app.llm.ollama_client import OllamaClient
from app.llm.generator import LLMGenerator
from app.retrieval.retriever import Retriever
from app.vector_store.store import VectorStore
from app.embedding.embedder import Embedder

def get_rag_pipeline():
    # --- 1. Create embedder ---
    embedder = Embedder()

    # --- 2. Create vector store ---
    vector_store = VectorStore(dim=embedder.embedding_dimension)

    # Load FAISS index + metadata if you want:
    # vector_store.load("faiss.index", "metadata.pkl")

    # --- 3. Create retriever ---
    retriever = Retriever(embedder, vector_store)

    # --- 4. LLM via Ollama ---
    ollama_client = OllamaClient(model="phi3")
    llm_generator = LLMGenerator(ollama_client)

    # --- 5. Build RAG pipeline ---
    from app.rag.pipeline import RAGPipeline
    return RAGPipeline(retriever, llm_generator)
