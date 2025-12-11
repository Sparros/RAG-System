from pydantic import BaseModel
from typing import Optional, Dict


class DocumentMetadata(BaseModel):
    document_id: str    # e.g., "doc1" or a UUID for the uploaded file
    source: str         # e.g., "manual.pdf" or "/path/to/manual.pdf"
    page: Optional[int] = None  # which page in the PDF (if available)
    extra: Dict[str, str] = {}  # any additional metadata as key-value pairs


class DocumentChunk(BaseModel):
    chunk_id: str          # unique ID for this chunk (UUID)
    document_id: str       # ID of the original document
    chunk_index: int       # 0, 1, 2, ... in order
    text: str              # the actual text content of this chunk
    metadata: DocumentMetadata  # where it came from

