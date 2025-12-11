from app.processing.chunker import chunk_text, chunk_document_pages
from app.models.document_models import DocumentChunk


def test_chunk_text_basic():
    text = "A" * 1200  # 1200 characters
    chunks = chunk_text(
        text=text,
        document_id="doc1",
        source="test.txt",
        page=None,
        max_chars=500,
        overlap=100,
    )

    # Expect chunks of approx: 500, 500, 200
    assert len(chunks) == 3
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_document_pages():
    pages = [(1, "A" * 600), (2, "B" * 600)]
    chunks = chunk_document_pages(
        pages,
        document_id="doc1",
        source="test.pdf",
        max_chars=300,
        overlap=50,
    )

    # Each page: 600 chars → chunks: 300, 300, 50 → 3 chunks per page
    assert len(chunks) == 6
    assert chunks[0].metadata.page == 1
    assert chunks[-1].metadata.page == 2
