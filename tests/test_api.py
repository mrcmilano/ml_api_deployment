import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_status():
    """Testa che l'endpoint /status risponda correttamente"""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_version():
    """Testa che l'endpoint /model_version ritorni la versione modello"""
    response = client.get("/model_version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)


def test_language_detection():
    """Testa che l'endpoint /language_detection ritorni predizioni"""
    payload = {"texts": ["Hello world!", "Ciao come state?"]}
    response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_language_detection_empty_list():
    """Testa che una lista vuota venga gestita senza crash"""
    payload = {"texts": []}
    response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["predictions"] == []
