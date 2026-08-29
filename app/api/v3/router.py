from fastapi import APIRouter

from app.api.v3.endpoints import lessons

router = APIRouter(prefix="/v3")
router.include_router(lessons.router)
