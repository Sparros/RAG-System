# app/retrieval/hybrid_retriever.py
from typing import List, Tuple, Dict
from app.models.document_models import DocumentChunk
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def _chunk_key(self, chunk: DocumentChunk) -> str:
        """
        Stable identifier for a document chunk.
        """
        return chunk.chunk_id

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        dense_results = self.dense.retrieve(query, k=k)
        bm25_results = self.bm25.retrieve(query, k=k)

        scores: Dict[str, float] = {}
        chunks: Dict[str, DocumentChunk] = {}

        # Dense contribution
        for rank, (_, chunk) in enumerate(dense_results):
            key = self._chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank + 1)
            chunks[key] = chunk

        # BM25 contribution
        for rank, (_, chunk) in enumerate(bm25_results):
            key = self._chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank + 1)
            chunks[key] = chunk

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (score, chunks[key])
            for key, score in ranked[:k]
        ]
