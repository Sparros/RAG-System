# app/embedding/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.document_models import DocumentChunk


class Embedder:
    """
    Wrapper around a SentenceTransformer embedding model.
    Loads once at startup and provides embedding utilities.
    Embeddings are L2-normalized for cosine-style similarity.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string into a normalized vector."""
        vector = self.model.encode(text, convert_to_numpy=True)
        return self._normalize(vector)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple strings into normalized vectors."""
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return self._normalize(vectors)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Embed DocumentChunk objects by extracting their text."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize vectors.
        Required for BGE models and safe for all others.
        """
        if vectors.ndim == 1:
            return vectors / np.linalg.norm(vectors)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-10, None)
