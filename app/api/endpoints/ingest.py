# app/api/endpoints/ingest.py

from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from app.api.dependencies import get_embedder, get_vector_store
from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.processing.ingestion_service import ingest_document  
from app.processing.doc_registry import register_document
from app.core.settings import ensure_dirs, FAISS_INDEX_PATH, FAISS_META_PATH

router = APIRouter()


class IngestResponse(BaseModel):
    document_id: str
    source: str
    num_chunks: int
    vector_store_size: int


@router.post("/load_documents", response_model=IngestResponse)
async def load_documents(
    file: UploadFile = File(...),
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
):
    # Basic content-type guard; extend as needed
    if file.content_type not in ("text/plain", "application/pdf"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save upload to a temp folder
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename

    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Generate a document_id
    document_id = str(uuid.uuid4())

    # 1) Ingest into chunks (using your loader + chunker)
    chunks = ingest_document(
        temp_path,
        document_id=document_id,
        source=file.filename,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document")

    # 2) Embed chunks
    vectors = embedder.embed_chunks(chunks)

    # 3) Add to vector store
    vector_store.add(vectors, chunks)

    ensure_dirs()
    vector_store.save(str(FAISS_INDEX_PATH), str(FAISS_META_PATH))

    
    register_document(
        document_id=document_id,
        source=file.filename,
        num_chunks=len(chunks),
    )

    return IngestResponse(
        document_id=document_id,
        source=file.filename,
        num_chunks=len(chunks),
        vector_store_size=vector_store.size
    )
