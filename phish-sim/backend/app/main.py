# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
FastAPI backend application for Phish-Sim
Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phish-Sim API",
    description="Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    service: str

class PhishingAnalysisRequest(BaseModel):
    url: str = None
    text: str = None
    content_type: str = "url"  # url, text, email

class PhishingAnalysisResponse(BaseModel):
    score: float
    decision: str  # "phish", "benign", "suspicious"
    confidence: float
    reasons: list
    processing_time_ms: float
    timestamp: float

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Phish-Sim API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        service="phish-sim-backend"
    )

@app.post("/analyze", response_model=PhishingAnalysisResponse)
async def analyze_phishing(request: PhishingAnalysisRequest):
    """
    Analyze content for phishing indicators
    This is a placeholder implementation for T001
    """
    start_time = time.time()
    
    # Placeholder analysis logic
    if request.content_type == "url" and request.url:
        # Simulate URL analysis
        score = 0.3  # Placeholder score
        decision = "benign"
        confidence = 0.7
        reasons = [
            {"type": "url_structure", "value": 0.2, "description": "URL structure appears normal"},
            {"type": "domain_age", "value": 0.1, "description": "Domain age check passed"}
        ]
    elif request.content_type == "text" and request.text:
        # Simulate text analysis
        score = 0.1  # Placeholder score
        decision = "benign"
        confidence = 0.8
        reasons = [
            {"type": "text_analysis", "value": 0.1, "description": "Text content appears legitimate"},
            {"type": "language_model", "value": 0.05, "description": "No suspicious patterns detected"}
        ]
    else:
        raise HTTPException(status_code=400, detail="Invalid request: provide either URL or text content")
    
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return PhishingAnalysisResponse(
        score=score,
        decision=decision,
        confidence=confidence,
        reasons=reasons,
        processing_time_ms=processing_time,
        timestamp=time.time()
    )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics (placeholder)"""
    return {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_processing_time_ms": 0.0,
        "uptime_seconds": 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)