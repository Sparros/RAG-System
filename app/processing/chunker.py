from typing import List, Tuple
import uuid
import re

from app.models.document_models import DocumentChunk, DocumentMetadata


# -----------------------------
# Utilities
# -----------------------------

def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.
    Deterministic, dependency-free.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_sections(text: str) -> List[str]:
    """
    Split text into sections using simple structural heuristics.
    """
    if not text:
        return []

    lines = text.splitlines()
    sections = []
    current = []

    def is_header(line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if line.isupper() and len(line) > 3:
            return True
        if line.endswith(":"):
            return True
        if line.startswith("#"):
            return True
        return False

    for line in lines:
        if is_header(line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if s]


# -----------------------------
# Chunking strategies
# -----------------------------

def chunk_fixed(
    text: str,
    document_id: str,
    source: str,
    page: int | None,
    max_chars: int,
    overlap: int,
) -> List[DocumentChunk]:
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + max_chars
        chunk_str = text[start:end]

        metadata = DocumentMetadata(
            document_id=document_id,
            source=source,
            page=page,
        )

        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=index,
                text=chunk_str,
                metadata=metadata,
            )
        )

        index += 1
        start = max(end - overlap, 0)

    return chunks


def chunk_sentences(
    text: str,
    document_id: str,
    source: str,
    page: int | None,
    max_chars: int,
) -> List[DocumentChunk]:
    sentences = split_sentences(text)
    chunks = []

    buffer = []
    buffer_len = 0
    index = 0

    for sent in sentences:
        if buffer_len + len(sent) > max_chars and buffer:
            metadata = DocumentMetadata(
                document_id=document_id,
                source=source,
                page=page,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    text=" ".join(buffer),
                    metadata=metadata,
                )
            )

            index += 1
            buffer = []
            buffer_len = 0

        buffer.append(sent)
        buffer_len += len(sent)

    if buffer:
        metadata = DocumentMetadata(
            document_id=document_id,
            source=source,
            page=page,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=index,
                text=" ".join(buffer),
                metadata=metadata,
            )
        )

    return chunks


def chunk_sections(
    text: str,
    document_id: str,
    source: str,
    page: int | None,
    max_chars: int,
) -> List[DocumentChunk]:
    sections = split_sections(text)
    chunks = []
    index = 0

    for section in sections:
        if len(section) <= max_chars:
            metadata = DocumentMetadata(
                document_id=document_id,
                source=source,
                page=page,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    text=section,
                    metadata=metadata,
                )
            )
            index += 1
        else:
            # Fallback to sentence chunking inside large sections
            sub_chunks = chunk_sentences(
                section,
                document_id,
                source,
                page,
                max_chars,
            )
            for sc in sub_chunks:
                sc.chunk_index = index
                chunks.append(sc)
                index += 1

    return chunks


# -----------------------------
# Public API
# -----------------------------

def safe_page(page_number):
    if isinstance(page_number, int):
        return page_number
    if isinstance(page_number, str) and page_number.isdigit():
        return int(page_number)
    return None


def chunk_document_pages(
    pages: List[Tuple[int | str, str]],
    document_id: str,
    source: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """
    Dispatch chunking strategy based on CHUNK_STRATEGY env var.
    """
    import os

    strategy = os.getenv("CHUNK_STRATEGY", "fixed")
    output = []

    for page_number, text in pages:
        page = safe_page(page_number)

        if strategy == "sentence":
            page_chunks = chunk_sentences(
                text, document_id, source, page, max_chars
            )

        elif strategy == "section":
            page_chunks = chunk_sections(
                text, document_id, source, page, max_chars
            )

        else:
            page_chunks = chunk_fixed(
                text, document_id, source, page, max_chars, overlap
            )

        output.extend(page_chunks)

    return output
