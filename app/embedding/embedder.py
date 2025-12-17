from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from app.models.document_models import DocumentChunk


class Embedder:
    """
    Wrapper around a SentenceTransformer embedding model.
    Loads once at startup and provides several embedding utilities.
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
        """Embed a single string into a vector."""
        return np.array(self.model.encode(text, convert_to_numpy=True))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple strings at once."""
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Embed DocumentChunk objects by extracting their text."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)
