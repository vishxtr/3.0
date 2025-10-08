# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Unified Backend API for Phish-Sim
Integrates NLP, Visual, and Real-time Inference services
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import service modules
try:
    from ml.inference.inference_api import PhishingInferenceAPI
    from visual.api.visual_analysis_api import perform_visual_analysis
    from inference.api.realtime_api import (
        get_redis_manager, get_queue_manager, get_metrics_collector, 
        get_pipeline_orchestrator, RateLimiter, generate_request_id
    )
    from inference.config import get_config as get_inference_config
    from inference.redis.redis_manager import RedisManager
    from inference.queue.queue_manager import QueueManager
    from inference.monitoring.metrics import MetricsCollector
    from inference.orchestration.pipeline_orchestrator import PipelineOrchestrator
except ImportError as e:
    logging.warning(f"Some services not available: {e}")
    # Create mock classes for missing services
    class MockService:
        def __init__(self, *args, **kwargs): pass
        async def initialize(self): pass
        async def close(self): pass
        def is_healthy(self): return True
    
    PhishingInferenceAPI = MockService
    perform_visual_analysis = lambda *args, **kwargs: {"error": "Visual analysis not available"}
    get_redis_manager = lambda: MockService()
    get_queue_manager = lambda: MockService()
    get_metrics_collector = lambda: MockService()
    get_pipeline_orchestrator = lambda: MockService()
    RateLimiter = MockService
    generate_request_id = lambda: f"req_{uuid.uuid4().hex[:12]}"
    get_inference_config = lambda x: MockService()

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phish-Sim Unified API",
    description="Unified API for Real-Time AI/ML-Based Phishing Detection & Prevention",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global service instances
nlp_service = None
visual_service = None
redis_manager = None
queue_manager = None
metrics_collector = None
pipeline_orchestrator = None
rate_limiter = None

# Request/Response Models
class AnalysisRequest(BaseModel):
    """Unified analysis request"""
    content: str = Field(..., description="Content to analyze (URL, email, text)")
    content_type: str = Field(..., description="Type of content: url, email, text")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    force_reanalyze: bool = Field(False, description="Force re-analysis, bypass cache")
    enable_nlp: bool = Field(True, description="Enable NLP analysis")
    enable_visual: bool = Field(True, description="Enable visual analysis")
    enable_realtime: bool = Field(True, description="Enable real-time processing")
    return_features: bool = Field(False, description="Return extracted features")
    return_explanation: bool = Field(True, description="Return detailed explanation")
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = ['url', 'email', 'text']
        if v not in allowed_types:
            raise ValueError(f'content_type must be one of {allowed_types}')
        return v

class BatchAnalysisRequest(BaseModel):
    """Batch analysis request"""
    requests: List[AnalysisRequest] = Field(..., description="List of analysis requests")
    batch_id: Optional[str] = Field(None, description="Optional batch ID")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent processing")

class AnalysisResponse(BaseModel):
    """Unified analysis response"""
    request_id: str
    content: str
    content_type: str
    prediction: str  # "phish", "benign", "suspicious", "unknown"
    confidence: float
    risk_score: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    explanation: Dict[str, Any]
    processing_time_ms: float
    cached: bool
    timestamp: str
    components: Dict[str, Any]  # Results from each component
    features: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # "ok", "degraded", "error"
    service: str
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    performance: Dict[str, float]

class ModelInfoResponse(BaseModel):
    """Model information response"""
    nlp_model: Dict[str, Any]
    visual_model: Dict[str, Any]
    thresholds: Dict[str, float]

class SystemStatus(BaseModel):
    """System status response"""
    backend_api: str
    ml_pipeline: str
    database: str
    redis: str
    websocket: str

class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_scans: int
    threats_detected: int
    avg_response_time_ms: float
    cache_hit_rate: float
    cache_hits: int
    cache_misses: int

class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str
    payload: Any

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = "anonymous"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str = "anonymous"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections and websocket in self.user_connections[user_id]:
            self.user_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

    async def send_to_user(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    self.user_connections[user_id].remove(connection)

manager = ConnectionManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global nlp_service, visual_service, redis_manager, queue_manager, metrics_collector, pipeline_orchestrator, rate_limiter
    
    try:
        logger.info("Initializing unified API services...")
        
        # Initialize NLP service
        try:
            nlp_service = PhishingInferenceAPI(
                model_path="/workspace/phish-sim/ml/models",
                tokenizer_path="/workspace/phish-sim/ml/models",
                config={}
            )
            logger.info("NLP service initialized")
        except Exception as e:
            logger.warning(f"NLP service initialization failed: {e}")
            nlp_service = None
        
        # Initialize Redis manager
        try:
            redis_manager = await get_redis_manager()
            logger.info("Redis manager initialized")
        except Exception as e:
            logger.warning(f"Redis manager initialization failed: {e}")
            redis_manager = None
        
        # Initialize queue manager
        try:
            queue_manager = await get_queue_manager()
            logger.info("Queue manager initialized")
        except Exception as e:
            logger.warning(f"Queue manager initialization failed: {e}")
            queue_manager = None
        
        # Initialize metrics collector
        try:
            metrics_collector = await get_metrics_collector()
            logger.info("Metrics collector initialized")
        except Exception as e:
            logger.warning(f"Metrics collector initialization failed: {e}")
            metrics_collector = None
        
        # Initialize pipeline orchestrator
        try:
            pipeline_orchestrator = await get_pipeline_orchestrator()
            logger.info("Pipeline orchestrator initialized")
        except Exception as e:
            logger.warning(f"Pipeline orchestrator initialization failed: {e}")
            pipeline_orchestrator = None
        
        # Initialize rate limiter
        try:
            rate_limiter = RateLimiter(max_requests=100, window=60)
            logger.info("Rate limiter initialized")
        except Exception as e:
            logger.warning(f"Rate limiter initialization failed: {e}")
            rate_limiter = None
        
        logger.info("Unified API services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if redis_manager:
            await redis_manager.close()
        if queue_manager:
            await queue_manager.close()
        if metrics_collector:
            await metrics_collector.close()
        if pipeline_orchestrator:
            await pipeline_orchestrator.close()
        
        logger.info("Unified API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for metrics"""
    start_time = time.time()
    request_id = generate_request_id()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Check rate limiting
    if rate_limiter:
        client_ip = request.client.host
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "request_id": request_id}
            )
    
    # Process request
    try:
        response = await call_next(request)
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        if metrics_collector:
            await metrics_collector.record_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                processing_time_ms=processing_time
            )
        
        return response
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Request {request_id} failed: {e}")
        
        # Record error metrics
        if metrics_collector:
            await metrics_collector.record_error(
                request_id=request_id,
                error=str(e),
                processing_time_ms=processing_time
            )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "request_id": request_id}
        )

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Phish-Sim Unified API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        start_time = time.time()
        
        # Check component health
        components = {
            "nlp_service": {
                "status": "healthy" if nlp_service else "unavailable",
                "details": "NLP inference service"
            },
            "visual_service": {
                "status": "healthy",  # Visual service is always available as function
                "details": "Visual analysis service"
            },
            "redis": {
                "status": "healthy" if redis_manager and redis_manager.is_healthy() else "unavailable",
                "details": "Redis cache and session management"
            },
            "queue": {
                "status": "healthy" if queue_manager and queue_manager.is_healthy() else "unavailable",
                "details": "Request queue management"
            },
            "metrics": {
                "status": "healthy" if metrics_collector and metrics_collector.is_healthy() else "unavailable",
                "details": "Metrics collection and monitoring"
            },
            "pipeline": {
                "status": "healthy" if pipeline_orchestrator and pipeline_orchestrator.is_healthy() else "unavailable",
                "details": "Pipeline orchestration"
            }
        }
        
        # Get performance metrics
        performance = {}
        if metrics_collector:
            try:
                performance = await metrics_collector.get_performance_metrics()
            except:
                performance = {"error": "Metrics unavailable"}
        
        # Determine overall health
        healthy_components = sum(1 for comp in components.values() if comp["status"] == "healthy")
        total_components = len(components)
        
        if healthy_components == total_components:
            overall_status = "ok"
        elif healthy_components >= total_components * 0.7:
            overall_status = "degraded"
        else:
            overall_status = "error"
        
        return HealthResponse(
            status=overall_status,
            service="phish-sim-unified-api",
            version="1.0.0",
            uptime_seconds=time.time() - start_time,
            components=components,
            performance=performance
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            service="phish-sim-unified-api",
            version="1.0.0",
            uptime_seconds=0.0,
            components={},
            performance={}
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    try:
        nlp_info = {
            "name": "PhishingClassifier",
            "version": "1.0.0",
            "status": "loaded" if nlp_service else "failed",
            "path": "/workspace/phish-sim/ml/models"
        }
        
        visual_info = {
            "name": "VisualClassifier",
            "version": "1.0.0",
            "status": "loaded",
            "path": "/workspace/phish-sim/visual/models"
        }
        
        thresholds = {
            "phishing": 0.7,
            "suspicious": 0.5
        }
        
        return ModelInfoResponse(
            nlp_model=nlp_info,
            visual_model=visual_info,
            thresholds=thresholds
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Unified content analysis endpoint"""
    try:
        start_time = time.time()
        request_id = generate_request_id()
        
        logger.info(f"Processing analysis request {request_id}")
        
        # Check cache if enabled and not forcing re-analysis
        cache_hit = False
        cached_result = None
        if not request.force_reanalyze and redis_manager:
            try:
                cached_result = await redis_manager.get_cached_result(request.content, request.content_type)
                if cached_result:
                    cache_hit = True
                    processing_time = (time.time() - start_time) * 1000
                    
                    return AnalysisResponse(
                        request_id=request_id,
                        content=request.content,
                        content_type=request.content_type,
                        prediction=cached_result["prediction"],
                        confidence=cached_result["confidence"],
                        risk_score=cached_result["risk_score"],
                        risk_level=cached_result["risk_level"],
                        explanation=cached_result.get("explanation", {}),
                        processing_time_ms=processing_time,
                        cached=True,
                        timestamp=datetime.utcnow().isoformat(),
                        components=cached_result.get("components", {}),
                        features=cached_result.get("features")
                    )
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        # Perform analysis
        analysis_result = await perform_unified_analysis(
            content=request.content,
            content_type=request.content_type,
            enable_nlp=request.enable_nlp,
            enable_visual=request.enable_visual,
            enable_realtime=request.enable_realtime,
            return_features=request.return_features,
            return_explanation=request.return_explanation
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result if enabled
        if redis_manager and not cache_hit:
            background_tasks.add_task(
                cache_analysis_result,
                request.content,
                request.content_type,
                analysis_result,
                ttl=300  # 5 minutes
            )
        
        return AnalysisResponse(
            request_id=request_id,
            content=request.content,
            content_type=request.content_type,
            prediction=analysis_result["prediction"],
            confidence=analysis_result["confidence"],
            risk_score=analysis_result["risk_score"],
            risk_level=analysis_result["risk_level"],
            explanation=analysis_result.get("explanation", {}),
            processing_time_ms=processing_time,
            cached=False,
            timestamp=datetime.utcnow().isoformat(),
            components=analysis_result.get("components", {}),
            features=analysis_result.get("features")
        )
        
    except Exception as e:
        logger.error(f"Analysis request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=List[AnalysisResponse])
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Batch analysis endpoint"""
    try:
        batch_id = request.batch_id or generate_request_id()
        logger.info(f"Processing batch analysis {batch_id} with {len(request.requests)} requests")
        
        # Process requests concurrently
        results = []
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def process_single_request(analysis_request: AnalysisRequest):
            async with semaphore:
                return await analyze_content(analysis_request, background_tasks)
        
        tasks = [process_single_request(req) for req in request.requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AnalysisResponse(
                    request_id=generate_request_id(),
                    content=request.requests[i].content,
                    content_type=request.requests[i].content_type,
                    prediction="error",
                    confidence=0.0,
                    risk_score=0.0,
                    risk_level="UNKNOWN",
                    explanation={"error": str(result)},
                    processing_time_ms=0.0,
                    cached=False,
                    timestamp=datetime.utcnow().isoformat(),
                    components={}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Batch analysis {batch_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    try:
        # Check backend API
        backend_status = "healthy"
        
        # Check ML pipeline
        ml_status = "healthy" if nlp_service else "down"
        
        # Check database (Redis)
        database_status = "healthy" if redis_manager and redis_manager.is_healthy() else "down"
        
        # Check Redis
        redis_status = "healthy" if redis_manager and redis_manager.is_healthy() else "down"
        
        # Check WebSocket
        websocket_status = "healthy" if len(manager.active_connections) >= 0 else "down"
        
        return SystemStatus(
            backend_api=backend_status,
            ml_pipeline=ml_status,
            database=database_status,
            redis=redis_status,
            websocket=websocket_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Get metrics from collector
        if metrics_collector:
            metrics_data = await metrics_collector.get_metrics()
            return DashboardStats(
                total_scans=metrics_data.get("requests_total", 0),
                threats_detected=metrics_data.get("threats_detected", 0),
                avg_response_time_ms=metrics_data.get("average_response_time_ms", 0.0),
                cache_hit_rate=metrics_data.get("cache_hit_rate", 0.0),
                cache_hits=metrics_data.get("cache_hits", 0),
                cache_misses=metrics_data.get("cache_misses", 0)
            )
        else:
            # Return default stats
            return DashboardStats(
                total_scans=0,
                threats_detected=0,
                avg_response_time_ms=0.0,
                cache_hit_rate=0.0,
                cache_hits=0,
                cache_misses=0
            )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process message
            if data.get("type") == "analysis":
                # Process analysis request
                try:
                    analysis_request = AnalysisRequest(**data.get("payload", {}))
                    result = await analyze_content(analysis_request, BackgroundTasks())
                    
                    # Send result
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "analysis_complete",
                            "payload": result.dict()
                        }),
                        websocket
                    )
                except Exception as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "analysis_error",
                            "payload": {"error": str(e)}
                        }),
                        websocket
                    )
            
            elif data.get("type") == "ping":
                # Respond to ping
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}),
                    websocket
                )
            
            else:
                # Unknown message type
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "payload": {"message": "Unknown message type"}
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)

# Helper functions
async def perform_unified_analysis(
    content: str,
    content_type: str,
    enable_nlp: bool = True,
    enable_visual: bool = True,
    enable_realtime: bool = True,
    return_features: bool = False,
    return_explanation: bool = True
) -> Dict[str, Any]:
    """Perform unified analysis using all available services"""
    components = {}
    predictions = []
    confidences = []
    risk_scores = []
    
    # NLP Analysis
    if enable_nlp and nlp_service:
        try:
            nlp_result = nlp_service.analyze_text(
                text=content,
                content_type=content_type,
                return_features=return_features
            )
            components["nlp"] = nlp_result
            predictions.append(nlp_result["prediction"])
            confidences.append(nlp_result["confidence"])
            risk_scores.append(1.0 - nlp_result["confidence"] if nlp_result["prediction"] == "benign" else nlp_result["confidence"])
        except Exception as e:
            logger.warning(f"NLP analysis failed: {e}")
            components["nlp"] = {"error": str(e)}
    
    # Visual Analysis (for URLs)
    if enable_visual and content_type == "url":
        try:
            visual_result = await perform_visual_analysis(
                url=content,
                enable_screenshot=True,
                enable_dom_analysis=True,
                enable_cnn_analysis=True,
                enable_template_matching=True
            )
            components["visual"] = visual_result
            if "overall_risk_score" in visual_result:
                risk_scores.append(visual_result["overall_risk_score"])
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            components["visual"] = {"error": str(e)}
    
    # Real-time Pipeline Analysis
    if enable_realtime and pipeline_orchestrator:
        try:
            realtime_result = await pipeline_orchestrator.process_request(
                content=content,
                content_type=content_type,
                request_id=generate_request_id(),
                return_features=return_features,
                return_explanation=return_explanation
            )
            components["realtime"] = realtime_result
            predictions.append(realtime_result.get("prediction", "unknown"))
            confidences.append(realtime_result.get("confidence", 0.0))
            risk_scores.append(realtime_result.get("risk_score", 0.0))
        except Exception as e:
            logger.warning(f"Real-time analysis failed: {e}")
            components["realtime"] = {"error": str(e)}
    
    # Combine results
    if predictions:
        # Use majority voting for prediction
        prediction_counts = {}
        for pred in predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        final_prediction = max(prediction_counts, key=prediction_counts.get)
    else:
        final_prediction = "unknown"
    
    # Calculate average confidence
    final_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Calculate average risk score
    final_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    
    # Determine risk level
    if final_risk_score >= 0.8:
        risk_level = "CRITICAL"
    elif final_risk_score >= 0.6:
        risk_level = "HIGH"
    elif final_risk_score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Generate explanation
    explanation = {}
    if return_explanation:
        explanation = {
            "overall_assessment": f"Content classified as {final_prediction} with {final_confidence:.2%} confidence",
            "risk_level": risk_level,
            "components_analyzed": list(components.keys()),
            "details": components
        }
    
    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "risk_score": final_risk_score,
        "risk_level": risk_level,
        "explanation": explanation,
        "components": components
    }

async def cache_analysis_result(
    content: str,
    content_type: str,
    result: Dict[str, Any],
    ttl: int = 300
):
    """Cache analysis result"""
    try:
        if redis_manager:
            await redis_manager.cache_result(content, content_type, result, ttl)
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")

if __name__ == "__main__":
    # Run the unified API
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1,
        log_level="info"
    )