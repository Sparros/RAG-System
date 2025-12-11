from pathlib import Path
from typing import List
import uuid

from app.processing.loader import load_document
from app.processing.chunker import chunk_document_pages
from app.models.document_models import DocumentChunk


def ingest_document(
    path: Path,
    max_chars: int = 1000,
    overlap: int = 200
) -> List[DocumentChunk]:
    """
    Full ingestion pipeline:
    - load document (PDF or TXT)
    - clean + extract text
    - chunk into overlapping windows
    - attach metadata (document ID, page, source)
    """

    # Give each uploaded file its own UUID
    document_id = str(uuid.uuid4())

    pages = load_document(path)
    source = str(path)

    chunks = chunk_document_pages(
        pages=pages,
        document_id=document_id,
        source=source,
        max_chars=max_chars,
        overlap=overlap,
    )

    return chunks
