#!/usr/bin/env python3
"""
Run the AI Lesson Builder Service.

Usage:
    python run.py

Or with uvicorn directly:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("🚀 Starting AI Lesson Builder Service...")
    print(f"   Host: {settings.host}")
    print(f"   Port: {settings.port}")
    print(f"   Debug: {settings.debug}")
    print(f"   AI Provider: {settings.ai_provider}")
    print()
    print(f"📖 API Docs: http://{settings.host}:{settings.port}/docs")
    print()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
