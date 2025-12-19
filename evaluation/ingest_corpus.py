# evaluation/ingest_corpus.py
from pathlib import Path
import uuid
import os

from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.processing.ingestion_service import ingest_document

CORPUS_DIR = Path("evaluation/corpus")


def ingest_all(model_name: str) -> VectorStore:
    """
    Ingest the evaluation corpus using the chunking strategy
    defined by environment variables.

    Environment variables:
      - CHUNK_STRATEGY: fixed | sentence | section
      - CHUNK_MAX_CHARS: int (only for fixed)
      - CHUNK_OVERLAP: int (only for fixed)
    """

    # -----------------------------
    # Read chunking configuration
    # -----------------------------
    chunk_strategy = os.getenv("CHUNK_STRATEGY", "fixed")

    max_chars_env = os.getenv("CHUNK_MAX_CHARS", "")
    overlap_env = os.getenv("CHUNK_OVERLAP", "")

    max_chars = int(max_chars_env) if max_chars_env.isdigit() else 1000
    overlap = int(overlap_env) if overlap_env.isdigit() else 200

    print(
        f"Chunking strategy: {chunk_strategy} "
        f"(max_chars={max_chars}, overlap={overlap})"
    )

    # -----------------------------
    # Build vector store
    # -----------------------------
    embedder = Embedder(model_name=model_name)
    vector_store = VectorStore(dim=embedder.embedding_dimension)

    for file_path in CORPUS_DIR.glob("*.txt"):
        document_id = str(uuid.uuid4())
        source = file_path.name

        print(f"Ingesting {source} (id={document_id})")

        # NOTE: sentence/section strategies will be implemented
        # inside ingest_document() later
        chunks = ingest_document(
            path=file_path,
            document_id=document_id,
            source=source,
            max_chars=max_chars,
            overlap=overlap,
        )

        vectors = embedder.embed_chunks(chunks)
        vector_store.add(vectors, chunks)

    print("\nIngestion complete.")
    print(f"Total vectors stored: {vector_store.size}")

    return vector_store
