# app/processing/ingestion_service.py
from pathlib import Path
from typing import List, Literal

from app.processing.loader import load_document
from app.processing.chunker import chunk_document_pages
from app.models.document_models import DocumentChunk

ChunkStrategy = Literal["fixed", "sentence", "section"]


def ingest_document(
    path: Path,
    document_id: str,
    source: str,
    *,
    strategy: ChunkStrategy = "fixed",
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """
    Full ingestion pipeline with configurable chunking strategy.
    """

    pages = load_document(path)

    chunks = chunk_document_pages(
        pages=pages,
        document_id=document_id,
        source=source,
        strategy=strategy,
        max_chars=max_chars,
        overlap=overlap,
    )

    return chunks
