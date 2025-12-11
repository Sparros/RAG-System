# app/processing/loader.py
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from app.processing.cleaner import clean_text


def load_text_from_txt(path: Path) -> str:
    """Load text from a .txt file."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def load_text_from_pdf(path: Path) -> List[Tuple[int, str]]:
    """
    Load text from a PDF.
    Returns: list of (page_num, cleaned_text)
    """
    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        pages.append((i + 1, clean_text(raw_text)))

    return pages


def load_document(path: Path):
    """
    Load a document and return:
      - TXT: [("full", cleaned_text)]
      - PDF: [(page_number, cleaned_text), ...]
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".txt":
        text = load_text_from_txt(path)
        return [("full", text)]

    elif ext == ".pdf":
        return load_text_from_pdf(path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
