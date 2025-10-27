from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Generator, Sequence
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def mock_predict_language_safe(_: object, __: object, texts: Sequence[str], **kwargs: object) -> list[str]:
    return ["English", "Italian"][: len(texts)]


@pytest.fixture
def prod_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Generator[ModuleType, None, None]:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    monkeypatch.delenv("SKIP_MODEL_LOADING", raising=False)
    sys.modules.pop("app.main", None)
    import app.main as main

    yield main


@pytest.fixture
def client(prod_app: ModuleType) -> TestClient:
    return TestClient(prod_app.app)


def test_status_endpoint(client: TestClient) -> None:
    """Health check endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_version_endpoint(client: TestClient) -> None:
    """Model version endpoint"""
    response = client.get("/model_version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)


def test_language_detection_with_mock(client: TestClient, prod_app: ModuleType) -> None:
    """Language detection endpoint with mocked model"""
    payload = {"texts": ["Hello world!", "Ciao come state?"]}
    with patch("app.main.predict_language_safe", side_effect=mock_predict_language_safe) as mock_func:
        response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert set(data["predictions"]) == {"English", "Italian"}
    mock_func.assert_called_once()


def test_language_detection_rejects_empty_payload(client: TestClient) -> None:
    """Ensure payload without texts key is rejected by validation."""
    response = client.post("/language_detection", json={})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "texts"


def test_language_detection_returns_typed_structure(client: TestClient) -> None:
    """Check shape of response when predictions succeed."""
    payload = {"texts": ["Buongiorno", "Bonjour"]}
    with patch("app.main.predict_language_safe", return_value=["it", "fr"]):
        response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["predictions"], list)
    assert all(isinstance(item, str) for item in data["predictions"])
    assert isinstance(data["model_version"], str)


def test_prod_environment_skips_model_loading(prod_app: ModuleType) -> None:
    """Ensure prod environment does not load model artifacts during tests."""
    assert prod_app.APP_ENV == "prod"
    assert prod_app.model is None
    assert prod_app.le is None
