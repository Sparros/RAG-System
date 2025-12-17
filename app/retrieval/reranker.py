from typing import List, Tuple
from sentence_transformers import CrossEncoder
from app.models.document_models import DocumentChunk


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int
    ) -> List[DocumentChunk]:
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        return [c for _, c in ranked[:top_k]]
