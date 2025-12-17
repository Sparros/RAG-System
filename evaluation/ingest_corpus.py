# evaluation/ingest_corpus.py

from pathlib import Path
import uuid

from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.processing.ingestion_service import ingest_document

CORPUS_DIR = Path("evaluation/corpus")


def ingest_all(model_name: str) -> VectorStore:
    embedder = Embedder(model_name=model_name)
    vector_store = VectorStore(dim=embedder.embedding_dimension)

    for file_path in CORPUS_DIR.glob("*.txt"):
        document_id = str(uuid.uuid4())
        source = file_path.name

        print(f"Ingesting {source} (id={document_id})")

        chunks = ingest_document(
            path=file_path,
            document_id=document_id,
            source=source,
        )

        vectors = embedder.embed_chunks(chunks)
        vector_store.add(vectors, chunks)

    print("\nIngestion complete.")
    print(f"Total vectors stored: {vector_store.size}")

    return vector_store
