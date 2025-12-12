from pathlib import Path

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_META_PATH = INDEX_DIR / "metadata.pkl"
DOC_REGISTRY_PATH = DATA_DIR / "documents.json"

def ensure_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
