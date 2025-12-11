# app/api/endpoints/query.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.rag.pipeline import RAGPipeline
from app.api.dependencies import get_rag_pipeline

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    result = pipeline.answer_query(req.query)
    return result
