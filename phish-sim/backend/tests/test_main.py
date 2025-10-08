# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns correct response"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Phish-Sim API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "phish-sim-backend"
    assert "timestamp" in data
    assert "version" in data

def test_analyze_url():
    """Test URL analysis endpoint"""
    response = client.post("/analyze", json={
        "url": "https://example.com",
        "content_type": "url"
    })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert "confidence" in data
    assert "reasons" in data
    assert "processing_time_ms" in data
    assert "timestamp" in data

def test_analyze_text():
    """Test text analysis endpoint"""
    response = client.post("/analyze", json={
        "text": "This is a test message",
        "content_type": "text"
    })
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert "confidence" in data
    assert "reasons" in data
    assert "processing_time_ms" in data
    assert "timestamp" in data

def test_analyze_invalid_request():
    """Test analysis endpoint with invalid request"""
    response = client.post("/analyze", json={
        "content_type": "url"
    })
    assert response.status_code == 400

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "average_processing_time_ms" in data
    assert "uptime_seconds" in data