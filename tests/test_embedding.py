from app.embedding.embedder import Embedder
from app.models.document_models import DocumentChunk, DocumentMetadata


def test_embed_single_text():
    embedder = Embedder()
    vec = embedder.embed_text("hello world")

    assert vec is not None
    assert len(vec.shape) == 1  # 1D vector


def test_embed_chunks():
    embedder = Embedder()

    metadata = DocumentMetadata(
        document_id="doc1",
        source="test.txt"
    )
    chunk = DocumentChunk(
        chunk_id="1",
        document_id="doc1",
        chunk_index=0,
        text="This is a test chunk.",
        metadata=metadata
    )

    vecs = embedder.embed_chunks([chunk])
    assert vecs.shape[0] == 1
