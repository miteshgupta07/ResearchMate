import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from backend.main import app


@patch("backend.main.build_portfolio_retriever")
@patch("backend.main.get_llm_registry")
@patch("backend.main.initialize_llm_registry")
def test_health(
    mock_init,
    mock_registry,
    mock_portfolio,
):
    mock_registry.return_value.available_models = []

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"