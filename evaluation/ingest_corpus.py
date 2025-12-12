# evaluation/ingest_corpus.py
from pathlib import Path
import uuid
from app.core.settings import FAISS_INDEX_PATH, FAISS_META_PATH
from app.core.settings import ensure_dirs

from app.api.dependencies import get_embedder, get_vector_store
from app.processing.ingestion_service import ingest_document

CORPUS_DIR = Path("evaluation/corpus")

def ingest_all():
    embedder = get_embedder()
    vector_store = get_vector_store()

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

    # Persist index after ingestion
    ensure_dirs()
    vector_store.save(
        str(FAISS_INDEX_PATH),
        str(FAISS_META_PATH),
    )


    print("\nIngestion complete.")
    print(f"Total vectors stored: {vector_store.size}")

if __name__ == "__main__":
    ingest_all()
