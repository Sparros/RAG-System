import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.settings import ensure_dirs, DOC_REGISTRY_PATH

def _load_registry(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

def _save_registry(path: Path, data: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def register_document(document_id: str, source: str, num_chunks: int) -> Dict[str, Any]:
    """
    Append a document record to the registry (data/documents.json).
    """
    ensure_dirs()
    record = {
        "document_id": document_id,
        "source": source,
        "num_chunks": num_chunks,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    data = _load_registry(DOC_REGISTRY_PATH)
    data.append(record)
    _save_registry(DOC_REGISTRY_PATH, data)
    return record

def list_documents() -> List[Dict[str, Any]]:
    ensure_dirs()
    return _load_registry(DOC_REGISTRY_PATH)

def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    for rec in list_documents():
        if rec["document_id"] == document_id:
            return rec
    return None
