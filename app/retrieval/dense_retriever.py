# app/retrieval/retriever.py
from typing import List, Tuple
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.models.document_models import DocumentChunk


class DenseRetriever:
    """
    Dense (embedding-based) retriever using vector similarity search.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        """
        Retrieve top-k chunks relevant to a query string.
        Returns list of (distance, DocumentChunk).
        """
        q_vec = self.embedder.embed_text(query)
        return self.vector_store.search(q_vec, k=k)
