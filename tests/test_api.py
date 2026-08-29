"""
Basic API tests for AI Lesson Builder Service.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    async with client as ac:
        response = await ac.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-lesson-builder"


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint."""
    async with client as ac:
        response = await ac.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "/v1/lessons/generate" in data["endpoints"]


@pytest.mark.asyncio
async def test_v1_generate_validation(client):
    """Test v1 generate endpoint validation."""
    async with client as ac:
        # Missing required fields
        response = await ac.post("/v1/lessons/generate", json={})

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_v3_generate_validation(client):
    """Test v3 generate endpoint validation."""
    async with client as ac:
        # Missing required fields
        response = await ac.post("/v3/lessons/generate", json={})

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_v3_artifact_validation(client):
    """Test v3 artifact endpoint validation."""
    async with client as ac:
        # Missing required fields
        response = await ac.post("/v3/lessons/generate_artifact", json={})

    assert response.status_code == 422  # Validation error
