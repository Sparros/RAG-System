from fastapi import APIRouter
from app.api.endpoints import health
from app.api.endpoints import query

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(query.router, prefix="/query", tags=["query"])