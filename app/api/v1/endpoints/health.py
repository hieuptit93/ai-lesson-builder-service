from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ai-lesson-builder-service",
        "version": "2.0.0",
    }
