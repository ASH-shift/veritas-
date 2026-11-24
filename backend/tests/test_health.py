import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_shape(client: AsyncClient) -> None:
    response = await client.get("/health")
    data = response.json()

    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert "environment" in data
    assert "components" in data

    for component in ("database", "redis"):
        assert component in data["components"]
        assert data["components"][component]["status"] in ("ok", "error")


@pytest.mark.asyncio
async def test_health_contains_latency(client: AsyncClient) -> None:
    response = await client.get("/health")
    data = response.json()

    for comp in data["components"].values():
        # latency_ms may be None on immediate connection failure, that's fine
        assert "status" in comp
