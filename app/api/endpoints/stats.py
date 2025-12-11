# app/api/endpoints/stats.py

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_vector_store, get_embedder
from app.vector_store.store import VectorStore
from app.embedding.embedder import Embedder

router = APIRouter()


class StatsResponse(BaseModel):
    vector_store_size: int
    embedding_dimension: int


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder)
):
    return StatsResponse(
        vector_store_size=vector_store.size,
        embedding_dimension=embedder.embedding_dimension,
    )
