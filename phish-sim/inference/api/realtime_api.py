# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
FastAPI real-time inference endpoints
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import configuration and utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, APIConfig, PERFORMANCE_TARGETS, FEATURE_FLAGS
from redis.redis_manager import RedisManager
from queue.queue_manager import QueueManager
from monitoring.metrics import MetricsCollector
from orchestration.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phish-Sim Real-time Inference API",
    description="Real-time phishing detection inference pipeline with WebSocket support",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get configuration
api_config = get_config("api")

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.allow_origins,
    allow_credentials=api_config.allow_credentials,
    allow_methods=api_config.allow_methods,
    allow_headers=api_config.allow_headers
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global instances
redis_manager = None
queue_manager = None
metrics_collector = None
pipeline_orchestrator = None

# Request models
class InferenceRequest(BaseModel):
    """Request model for real-time inference"""
    content: str = Field(..., description="Content to analyze (URL, email, text)")
    content_type: str = Field(..., description="Type of content: url, email, text")
    request_id: Optional[str] = Field(None, description="Optional request ID")
    priority: int = Field(1, ge=1, le=5, description="Request priority (1-5)")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="Request timeout in seconds")
    enable_caching: bool = Field(True, description="Enable result caching")
    return_features: bool = Field(False, description="Return extracted features")
    return_explanation: bool = Field(True, description="Return explanation")
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = ['url', 'email', 'text']
        if v not in allowed_types:
            raise ValueError(f'content_type must be one of {allowed_types}')
        return v

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference"""
    requests: List[InferenceRequest] = Field(..., description="List of inference requests")
    batch_id: Optional[str] = Field(None, description="Optional batch ID")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent processing")
    enable_streaming: bool = Field(False, description="Enable streaming results")

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    request_id: str
    prediction: str
    confidence: float
    risk_score: float
    risk_level: str
    processing_time_ms: float
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    pipeline_components: Optional[Dict[str, Any]] = None
    cache_hit: bool = False

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, bool]
    performance: Dict[str, float]

class MetricsResponse(BaseModel):
    """Metrics response"""
    timestamp: datetime
    requests_total: int
    requests_successful: int
    requests_failed: int
    average_response_time_ms: float
    throughput_per_second: float
    error_rate: float
    active_connections: int
    queue_depth: int
    cache_hit_rate: float

# Rate limiting
from collections import defaultdict, deque
import time

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests
        while client_requests and client_requests[0] <= now - self.window:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        
        return False

# Initialize rate limiter
rate_limiter = RateLimiter(
    max_requests=api_config.rate_limit_requests,
    window=api_config.rate_limit_window
)

# Request ID generator
def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}_{int(time.time())}"

# Dependency functions
async def get_redis_manager() -> RedisManager:
    """Get Redis manager instance"""
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager()
        await redis_manager.initialize()
    return redis_manager

async def get_queue_manager() -> QueueManager:
    """Get queue manager instance"""
    global queue_manager
    if queue_manager is None:
        queue_manager = QueueManager()
        await queue_manager.initialize()
    return queue_manager

async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
        await metrics_collector.initialize()
    return metrics_collector

async def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get pipeline orchestrator instance"""
    global pipeline_orchestrator
    if pipeline_orchestrator is None:
        pipeline_orchestrator = PipelineOrchestrator()
        await pipeline_orchestrator.initialize()
    return pipeline_orchestrator

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for metrics"""
    start_time = time.time()
    request_id = generate_request_id()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Check rate limiting
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

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global redis_manager, queue_manager, metrics_collector, pipeline_orchestrator
    
    try:
        logger.info("Initializing real-time inference API components...")
        
        # Initialize components
        redis_manager = await get_redis_manager()
        queue_manager = await get_queue_manager()
        metrics_collector = await get_metrics_collector()
        pipeline_orchestrator = await get_pipeline_orchestrator()
        
        logger.info("Real-time inference API components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
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
        
        logger.info("Real-time inference API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Phish-Sim Real-time Inference API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "websocket": "/ws"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check component health
        components = {
            "redis": redis_manager.is_healthy() if redis_manager else False,
            "queue": queue_manager.is_healthy() if queue_manager else False,
            "metrics": metrics_collector.is_healthy() if metrics_collector else False,
            "pipeline": pipeline_orchestrator.is_healthy() if pipeline_orchestrator else False
        }
        
        # Get performance metrics
        performance = {}
        if metrics_collector:
            performance = await metrics_collector.get_performance_metrics()
        
        # Determine overall health
        overall_health = all(components.values())
        
        return HealthResponse(
            status="healthy" if overall_health else "degraded",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            components=components,
            performance=performance
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            components={},
            performance={}
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API metrics"""
    try:
        if not metrics_collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        
        metrics_data = await metrics_collector.get_metrics()
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            requests_total=metrics_data.get("requests_total", 0),
            requests_successful=metrics_data.get("requests_successful", 0),
            requests_failed=metrics_data.get("requests_failed", 0),
            average_response_time_ms=metrics_data.get("average_response_time_ms", 0.0),
            throughput_per_second=metrics_data.get("throughput_per_second", 0.0),
            error_rate=metrics_data.get("error_rate", 0.0),
            active_connections=metrics_data.get("active_connections", 0),
            queue_depth=metrics_data.get("queue_depth", 0),
            cache_hit_rate=metrics_data.get("cache_hit_rate", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer", response_model=InferenceResponse)
async def infer_realtime(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    redis: RedisManager = Depends(get_redis_manager),
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Real-time inference endpoint"""
    try:
        start_time = time.time()
        request_id = request.request_id or generate_request_id()
        
        logger.info(f"Processing inference request {request_id}")
        
        # Check cache if enabled
        cache_hit = False
        if request.enable_caching and redis:
            cached_result = await redis.get_cached_result(request.content, request.content_type)
            if cached_result:
                cache_hit = True
                processing_time = (time.time() - start_time) * 1000
                
                return InferenceResponse(
                    request_id=request_id,
                    prediction=cached_result["prediction"],
                    confidence=cached_result["confidence"],
                    risk_score=cached_result["risk_score"],
                    risk_level=cached_result["risk_level"],
                    processing_time_ms=processing_time,
                    timestamp=datetime.utcnow(),
                    explanation=cached_result.get("explanation"),
                    features=cached_result.get("features"),
                    pipeline_components=cached_result.get("pipeline_components"),
                    cache_hit=True
                )
        
        # Process through pipeline
        result = await orchestrator.process_request(
            content=request.content,
            content_type=request.content_type,
            request_id=request_id,
            priority=request.priority,
            timeout=request.timeout,
            return_features=request.return_features,
            return_explanation=request.return_explanation
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result if enabled
        if request.enable_caching and redis and not cache_hit:
            background_tasks.add_task(
                redis.cache_result,
                request.content,
                request.content_type,
                result,
                ttl=300  # 5 minutes
            )
        
        return InferenceResponse(
            request_id=request_id,
            prediction=result["prediction"],
            confidence=result["confidence"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            explanation=result.get("explanation"),
            features=result.get("features"),
            pipeline_components=result.get("pipeline_components"),
            cache_hit=cache_hit
        )
        
    except Exception as e:
        logger.error(f"Inference request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer/batch", response_model=List[InferenceResponse])
async def infer_batch(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    queue: QueueManager = Depends(get_queue_manager),
    orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Batch inference endpoint"""
    try:
        batch_id = request.batch_id or generate_request_id()
        logger.info(f"Processing batch inference {batch_id} with {len(request.requests)} requests")
        
        if request.enable_streaming:
            # Return streaming response
            return StreamingResponse(
                process_batch_streaming(request, batch_id, orchestrator),
                media_type="application/json"
            )
        else:
            # Process batch synchronously
            results = []
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def process_single_request(inference_request: InferenceRequest):
                async with semaphore:
                    return await infer_realtime(inference_request, background_tasks)
            
            tasks = [process_single_request(req) for req in request.requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(InferenceResponse(
                        request_id=request.requests[i].request_id or generate_request_id(),
                        prediction="error",
                        confidence=0.0,
                        risk_score=0.0,
                        risk_level="unknown",
                        processing_time_ms=0.0,
                        timestamp=datetime.utcnow(),
                        explanation={"error": str(result)}
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
        
    except Exception as e:
        logger.error(f"Batch inference {batch_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch_streaming(
    request: BatchInferenceRequest,
    batch_id: str,
    orchestrator: PipelineOrchestrator
):
    """Process batch requests with streaming response"""
    try:
        # Send batch start
        yield f"data: {json.dumps({'type': 'batch_start', 'batch_id': batch_id})}\n\n"
        
        # Process requests
        for i, inference_request in enumerate(request.requests):
            try:
                result = await orchestrator.process_request(
                    content=inference_request.content,
                    content_type=inference_request.content_type,
                    request_id=inference_request.request_id or generate_request_id(),
                    priority=inference_request.priority,
                    timeout=inference_request.timeout,
                    return_features=inference_request.return_features,
                    return_explanation=inference_request.return_explanation
                )
                
                # Send result
                yield f"data: {json.dumps({'type': 'result', 'index': i, 'result': result})}\n\n"
                
            except Exception as e:
                # Send error
                yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': str(e)})}\n\n"
        
        # Send batch complete
        yield f"data: {json.dumps({'type': 'batch_complete', 'batch_id': batch_id})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming batch processing failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

@app.get("/infer/status/{request_id}")
async def get_inference_status(
    request_id: str,
    redis: RedisManager = Depends(get_redis_manager)
):
    """Get inference request status"""
    try:
        if not redis:
            raise HTTPException(status_code=503, detail="Redis not available")
        
        status_data = await redis.get_request_status(request_id)
        if not status_data:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return status_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/infer/cache")
async def clear_cache(
    redis: RedisManager = Depends(get_redis_manager)
):
    """Clear inference cache"""
    try:
        if not redis:
            raise HTTPException(status_code=503, detail="Redis not available")
        
        await redis.clear_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/infer/cache/stats")
async def get_cache_stats(
    redis: RedisManager = Depends(get_redis_manager)
):
    """Get cache statistics"""
    try:
        if not redis:
            raise HTTPException(status_code=503, detail="Redis not available")
        
        stats = await redis.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time communication"""
    try:
        await websocket.accept()
        client_id = generate_request_id()
        logger.info(f"WebSocket client {client_id} connected")
        
        # Add to active connections
        if metrics_collector:
            await metrics_collector.add_connection(client_id)
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Process message
                if data.get("type") == "inference":
                    # Process inference request
                    result = await process_websocket_inference(data, client_id)
                    await websocket.send_json({
                        "type": "inference_result",
                        "request_id": data.get("request_id"),
                        "result": result
                    })
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({"type": "pong"})
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type"
                    })
        
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
    
    finally:
        # Remove from active connections
        if metrics_collector:
            await metrics_collector.remove_connection(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")

async def process_websocket_inference(data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Process inference request from WebSocket"""
    try:
        orchestrator = await get_pipeline_orchestrator()
        
        result = await orchestrator.process_request(
            content=data["content"],
            content_type=data["content_type"],
            request_id=data.get("request_id", generate_request_id()),
            priority=data.get("priority", 1),
            timeout=data.get("timeout", 30),
            return_features=data.get("return_features", False),
            return_explanation=data.get("return_explanation", True)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"WebSocket inference failed for client {client_id}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        workers=api_config.workers,
        log_level="info"
    )