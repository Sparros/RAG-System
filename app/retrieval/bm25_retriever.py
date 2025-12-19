# app/retrieval/bm25_retriever.py
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from app.models.document_models import DocumentChunk
import re


class BM25Retriever:
    """
    Sparse lexical retriever using BM25.
    """

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.text) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )

        return ranked[:k]
