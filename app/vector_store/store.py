# app/vector_store/store.py
import faiss
import numpy as np
from typing import List, Tuple
from app.models.document_models import DocumentChunk


class VectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # Simple L2 distance index
        self.metadata_store: List[DocumentChunk] = []

    @property
    def size(self) -> int:
        """Return number of vectors stored in FAISS."""
        return self.index.ntotal

    @property
    def chunks(self) -> List[DocumentChunk]:
        """
        Return all stored document chunks.
        Useful for sparse and hybrid retrieval.
        """
        return self.metadata_store

    def add(self, vectors: np.ndarray, chunks: List[DocumentChunk]):
        """
        Add vectors + associated metadata.
        """
        if len(vectors) != len(chunks):
            raise ValueError("Vectors and chunks must be the same length")

        self.index.add(vectors.astype("float32"))
        self.metadata_store.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        """
        Retrieve top-k nearest chunks for a query vector.
        Returns (distance, chunk) pairs.
        """
        distances, indices = self.index.search(
            query_vector[np.newaxis, :].astype("float32"),
            k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(dist), self.metadata_store[idx]))

        return results

    def save(self, index_path: str, metadata_path: str):
        """
        Save FAISS index + metadata.
        """
        faiss.write_index(self.index, index_path)

        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

    def load(self, index_path: str, metadata_path: str):
        """
        Load FAISS index + metadata.
        """
        import pickle

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata_store = pickle.load(f)
