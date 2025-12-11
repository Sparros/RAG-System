from app.embedding.embedder import Embedder
from app.vector_store.store import VectorStore
from app.retrieval.retriever import Retriever
from app.models.document_models import DocumentChunk, DocumentMetadata
import numpy as np

def test_retriever_basic():
    embedder = Embedder()
    dim = embedder.embedding_dimension

    store = VectorStore(dim)
    retriever = Retriever(embedder, store)

    metadata = DocumentMetadata(document_id="doc1", source="manual", page=1)
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        chunk_index=0,
        text="Aspirin is used to reduce pain and inflammation.",
        metadata=metadata
    )

    vec = embedder.embed_text(chunk.text)
    store.add(np.array([vec]), [chunk])

    results = retriever.retrieve("What is aspirin used for?", k=1)
    assert len(results) == 1
    assert "Aspirin" in results[0][1].text
