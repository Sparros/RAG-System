# app/processing/chunker.py
from typing import List, Tuple
import uuid

from app.models.document_models import DocumentChunk, DocumentMetadata


def chunk_text(
    text: str,
    document_id: str,
    source: str,
    page: int | None = None,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:

    chunks = []
    if not text:
        return chunks

    start = 0
    index = 0

    while start < len(text):
        end = start + max_chars
        chunk_str = text[start:end]

        metadata = DocumentMetadata(
            document_id=document_id,
            source=source,
            page=page
        )

        chunk = DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_index=index,
            text=chunk_str,
            metadata=metadata
        )

        chunks.append(chunk)

        index += 1
        start = end - overlap  # sliding window with overlap

        if start < 0:
            start = 0

    return chunks

def safe_page(page_number):
    if isinstance(page_number, int):
        return page_number
    if isinstance(page_number, str):
        if page_number.isdigit():
            return int(page_number)
        return None
    return None

def chunk_document_pages(
    pages: List[Tuple[int | str, str]],
    document_id: str,
    source: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:

    output = []
    for page_number, text in pages:
        p = safe_page(page_number)

        page_chunks = chunk_text(
            text=text,
            document_id=document_id,
            source=source,
            page=p,
            max_chars=max_chars,
            overlap=overlap
        )

        output.extend(page_chunks)

    return output
