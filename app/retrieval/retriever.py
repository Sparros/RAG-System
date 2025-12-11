from typing import List, Tuple
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.models.document_models import DocumentChunk

print("LOADED RETRIEVER FROM:", __file__)
print("Retriever defined?", "Retriever" in globals())

class Retriever:
    """
    Combines embedding + vector search.
    Takes a query string, embeds it, and retrieves the top-k similar chunks.
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        """
        Retrieve top-k chunks relevant to a query string.
        Returns list of (distance, DocumentChunk).
        """
        # Embed query into vector
        q_vec = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(q_vec, k=k)

        return results

print("After class, Retriever defined?", "Retriever" in globals())
print("Module path:", __name__)