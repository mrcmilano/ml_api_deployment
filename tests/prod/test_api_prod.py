import os
from fastapi.testclient import TestClient
from unittest.mock import patch

# Skip heavy artifact loading during unit tests
os.environ.setdefault("SKIP_MODEL_LOADING", "true")

from app.main import app

client = TestClient(app)


# mock function to replace the actual model prediction
def mock_predict_language_safe(pipeline, le, texts, **kwargs):
    return ["English", "Italian"]


def test_status_endpoint():
    """Health check endpoint"""
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_version_endpoint():
    """Model version endpoint"""
    response = client.get("/model_version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)


@patch("app.main.predict_language_safe", side_effect=mock_predict_language_safe)
def test_language_detection(mock_func):
    """Language detection endpoint with mocked model"""
    payload = {"texts": ["Hello world!", "Ciao come state?"]}
    response = client.post("/language_detection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert "English" in data["predictions"]
    assert "Italian" in data["predictions"]
    
    mock_func.assert_called_once() # controlla che la funzione mock sia stata chiamata 
