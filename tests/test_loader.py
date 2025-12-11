from app.processing.loader import load_document
from pathlib import Path


def test_load_text_document(tmp_path):
    # Create a temporary .txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello world\nThis is a test.")

    pages = load_document(test_file)

    assert isinstance(pages, list)
    assert len(pages) == 1
    assert pages[0][0] == "full"
    assert "Hello world" in pages[0][1]
    assert "This is a test." in pages[0][1]
