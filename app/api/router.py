# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import health, query, ingest, stats

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(ingest.router, prefix="/documents", tags=["Ingest"])
api_router.include_router(stats.router, prefix="/documents", tags=["Stats"])
