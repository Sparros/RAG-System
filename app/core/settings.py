from pathlib import Path

DATA_DIR = Path("data")


FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_META_PATH = DATA_DIR / "metadata.pkl"
DOC_REGISTRY_PATH = DATA_DIR / "documents.json"

def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
