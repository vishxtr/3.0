from fastapi.testclient import TestClient
from app.main import app
import os
from app.core.model_loader import ModelLoader

client = TestClient(app)

def test_verdict_without_model():
    # If model not loaded, endpoint should raise 503
    # We simulate by clearing model in app state
    app.state.model = None
    r = client.post("/api/verdict", json={"url": "https://example.com/login"})
    assert r.status_code == 503