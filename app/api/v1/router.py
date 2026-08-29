from fastapi import APIRouter

from app.api.v1.endpoints import health, lessons

router = APIRouter(prefix="/v1")
router.include_router(health.router)
router.include_router(lessons.router)
