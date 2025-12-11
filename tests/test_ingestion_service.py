from pathlib import Path
from app.processing.ingestion_service import ingest_document

def test_ingest_txt_document(tmp_path):
    # Create sample TXT file
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a test document used for ingestion.")

    chunks = ingest_document(file_path)

    assert len(chunks) > 0
    assert chunks[0].document_id is not None
    assert "test document" in chunks[0].text
