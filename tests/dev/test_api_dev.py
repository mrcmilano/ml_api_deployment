from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Generator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def dev_app(monkeypatch: pytest.MonkeyPatch) -> Generator[ModuleType, None, None]:
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("MODEL_VERSION", "v2")
    monkeypatch.delenv("SKIP_MODEL_LOADING", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    sys.modules.pop("app.main", None)
    import app.main as main

    if main.model is None or main.le is None:
        pytest.fail("Model artifacts must be available in dev environment for these tests.")
    yield main


@pytest.fixture
def client(dev_app: ModuleType) -> TestClient:
    return TestClient(dev_app.app)


def test_status(client: TestClient) -> None:
    """Testa che l'endpoint /status risponda correttamente"""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_version(client: TestClient, dev_app: ModuleType) -> None:
    """Testa che l'endpoint /model_version ritorni la versione modello"""
    response = client.get("/model_version")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == dev_app.MODEL_VERSION
    assert dev_app.MODEL_VERSION == "v2"
    assert os.path.normpath(str(dev_app.MODEL_DIR)).split(os.sep)[-1] == "v2"


def test_language_detection(client: TestClient) -> None:
    """Testa che l'endpoint /language_detection ritorni predizioni"""
    payload = {"texts": ["Hello world!", "Ciao come state?"]}
    response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == len(payload["texts"])
    assert all(isinstance(item, str) for item in data["predictions"])


def test_language_detection_rejects_non_list_payload(client: TestClient) -> None:
    """Verifica che l'endpoint validi il payload usando typing Pydantic."""
    response = client.post("/language_detection", json={"texts": "not-a-list"})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["loc"][-1] == "texts"


def test_language_detection_rejects_non_string_items(client: TestClient) -> None:
    """Verifica che l'endpoint rifiuti testi non stringa."""
    response = client.post("/language_detection", json={"texts": ["buongiorno", 42]})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "texts" in detail[0]["loc"]
