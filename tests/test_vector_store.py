import numpy as np
from app.vector_store.store import VectorStore
from app.models.document_models import DocumentChunk, DocumentMetadata


def test_vector_store_add_and_search():
    store = VectorStore(dim=3)

    metadata = DocumentMetadata(document_id="doc1", source="test")
    chunk = DocumentChunk(
        chunk_id="1",
        document_id="doc1",
        chunk_index=0,
        text="hello",
        metadata=metadata,
    )

    vectors = np.array([[1.0, 2.0, 3.0]])
    store.add(vectors, [chunk])

    query = np.array([1.0, 2.0, 3.1])
    results = store.search(query, k=1)

    assert len(results) == 1
    assert results[0][1].text == "hello"
