from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

from app.processing.doc_registry import list_documents

router = APIRouter()

class DocumentRecord(BaseModel):
    document_id: str
    source: str
    num_chunks: int
    ingested_at: str

@router.get("/", response_model=List[DocumentRecord])
def list_docs():
    return list_documents()
