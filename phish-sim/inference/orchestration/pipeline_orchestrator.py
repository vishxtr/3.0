# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Inference pipeline orchestration system
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json

from config import get_config, OrchestrationConfig, InferenceMode, CacheStrategy

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    """Component status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"

@dataclass
class PipelineComponent:
    """Pipeline component definition"""
    name: str
    status: ComponentStatus
    response_time_ms: float
    success_rate: float
    last_health_check: datetime
    error_count: int = 0
    total_requests: int = 0

@dataclass
class InferenceRequest:
    """Inference request structure"""
    request_id: str
    content: str
    content_type: str
    priority: int
    timeout: int
    return_features: bool
    return_explanation: bool
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InferenceResult:
    """Inference result structure"""
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

class CircuitBreaker:
    """Circuit breaker for component protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_calls = 0
                return True
            return False
        elif self.state == "half_open":
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == "half_open":
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = "closed"
                self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

class LoadBalancer:
    """Load balancer for component instances"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_index = 0
        self.component_weights = {}
    
    def select_component(self, components: List[PipelineComponent]) -> Optional[PipelineComponent]:
        """Select component based on strategy"""
        if not components:
            return None
        
        # Filter healthy components
        healthy_components = [c for c in components if c.status == ComponentStatus.HEALTHY]
        if not healthy_components:
            # Fall back to degraded components
            healthy_components = [c for c in components if c.status == ComponentStatus.DEGRADED]
        
        if not healthy_components:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(healthy_components)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(healthy_components)
        elif self.strategy == "weighted":
            return self._weighted_selection(healthy_components)
        else:
            return healthy_components[0]
    
    def _round_robin_selection(self, components: List[PipelineComponent]) -> PipelineComponent:
        """Round-robin component selection"""
        component = components[self.round_robin_index % len(components)]
        self.round_robin_index += 1
        return component
    
    def _least_connections_selection(self, components: List[PipelineComponent]) -> PipelineComponent:
        """Select component with least connections"""
        return min(components, key=lambda c: c.total_requests)
    
    def _weighted_selection(self, components: List[PipelineComponent]) -> PipelineComponent:
        """Weighted component selection based on performance"""
        # Weight based on success rate and response time
        weights = []
        for component in components:
            weight = component.success_rate / max(component.response_time_ms, 1.0)
            weights.append(weight)
        
        # Select based on weight
        import random
        total_weight = sum(weights)
        if total_weight == 0:
            return components[0]
        
        rand = random.uniform(0, total_weight)
        current_weight = 0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if rand <= current_weight:
                return components[i]
        
        return components[0]

class PipelineOrchestrator:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or get_config("orchestration")
        
        # Pipeline components
        self.components: Dict[str, List[PipelineComponent]] = {
            "ml_pipeline": [],
            "visual_pipeline": [],
            "dom_pipeline": []
        }
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Load balancers
        self.load_balancers: Dict[str, LoadBalancer] = {}
        
        # Cache
        self.cache = {}
        self.cache_strategy = self.config.cache_strategy
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "component_usage": {}
        }
        
        # Initialize components
        self._initialize_components()
        self._initialize_circuit_breakers()
        self._initialize_load_balancers()
    
    async def initialize(self):
        """Initialize pipeline orchestrator"""
        try:
            # Start health check tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("Pipeline orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline orchestrator: {e}")
            raise
    
    async def close(self):
        """Close pipeline orchestrator"""
        try:
            # Stop background tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("Pipeline orchestrator closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing pipeline orchestrator: {e}")
    
    def is_healthy(self) -> bool:
        """Check if pipeline orchestrator is healthy"""
        # Check if at least one component in each pipeline is healthy
        for pipeline_name, components in self.components.items():
            if not components:
                continue
            
            healthy_components = [c for c in components if c.status == ComponentStatus.HEALTHY]
            if not healthy_components:
                return False
        
        return True
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        # ML Pipeline components
        if self.config.enable_ml_pipeline:
            self.components["ml_pipeline"] = [
                PipelineComponent(
                    name="text_classifier",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=50.0,
                    success_rate=0.95,
                    last_health_check=datetime.utcnow()
                ),
                PipelineComponent(
                    name="feature_extractor",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=30.0,
                    success_rate=0.98,
                    last_health_check=datetime.utcnow()
                )
            ]
        
        # Visual Pipeline components
        if self.config.enable_visual_pipeline:
            self.components["visual_pipeline"] = [
                PipelineComponent(
                    name="screenshot_capture",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=2000.0,
                    success_rate=0.90,
                    last_health_check=datetime.utcnow()
                ),
                PipelineComponent(
                    name="cnn_classifier",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=100.0,
                    success_rate=0.92,
                    last_health_check=datetime.utcnow()
                ),
                PipelineComponent(
                    name="template_matcher",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=80.0,
                    success_rate=0.88,
                    last_health_check=datetime.utcnow()
                )
            ]
        
        # DOM Pipeline components
        if self.config.enable_dom_pipeline:
            self.components["dom_pipeline"] = [
                PipelineComponent(
                    name="dom_analyzer",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=150.0,
                    success_rate=0.94,
                    last_health_check=datetime.utcnow()
                ),
                PipelineComponent(
                    name="content_analyzer",
                    status=ComponentStatus.HEALTHY,
                    response_time_ms=100.0,
                    success_rate=0.96,
                    last_health_check=datetime.utcnow()
                )
            ]
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers"""
        for pipeline_name in self.components.keys():
            self.circuit_breakers[pipeline_name] = CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                half_open_max_calls=self.config.half_open_max_calls
            )
    
    def _initialize_load_balancers(self):
        """Initialize load balancers"""
        for pipeline_name in self.components.keys():
            self.load_balancers[pipeline_name] = LoadBalancer(
                strategy=self.config.load_balancer
            )
    
    async def process_request(
        self,
        content: str,
        content_type: str,
        request_id: str,
        priority: int = 1,
        timeout: int = 30,
        return_features: bool = False,
        return_explanation: bool = True
    ) -> Dict[str, Any]:
        """Process inference request through pipeline"""
        try:
            start_time = time.time()
            
            # Create request
            request = InferenceRequest(
                request_id=request_id,
                content=content,
                content_type=content_type,
                priority=priority,
                timeout=timeout,
                return_features=return_features,
                return_explanation=return_explanation,
                created_at=datetime.utcnow()
            )
            
            # Check cache
            if self.config.enable_prediction_cache:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    cached_result["cache_hit"] = True
                    return cached_result
            
            self.stats["cache_misses"] += 1
            
            # Process through pipeline
            result = await self._execute_pipeline(request)
            
            # Cache result
            if self.config.enable_prediction_cache:
                self._cache_result(request, result)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, True)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for request {request_id}: {e}")
            self._update_stats(0, False)
            raise
    
    async def _execute_pipeline(self, request: InferenceRequest) -> Dict[str, Any]:
        """Execute pipeline for request"""
        try:
            pipeline_results = {}
            overall_risk_score = 0.0
            component_weights = {"ml": 0.4, "visual": 0.3, "dom": 0.3}
            
            # Execute ML Pipeline
            if self.config.enable_ml_pipeline and request.content_type in ["text", "email"]:
                ml_result = await self._execute_component_pipeline("ml_pipeline", request)
                if ml_result:
                    pipeline_results["ml_pipeline"] = ml_result
                    overall_risk_score += ml_result.get("risk_score", 0.0) * component_weights["ml"]
            
            # Execute Visual Pipeline
            if self.config.enable_visual_pipeline and request.content_type == "url":
                visual_result = await self._execute_component_pipeline("visual_pipeline", request)
                if visual_result:
                    pipeline_results["visual_pipeline"] = visual_result
                    overall_risk_score += visual_result.get("risk_score", 0.0) * component_weights["visual"]
            
            # Execute DOM Pipeline
            if self.config.enable_dom_pipeline and request.content_type == "url":
                dom_result = await self._execute_component_pipeline("dom_pipeline", request)
                if dom_result:
                    pipeline_results["dom_pipeline"] = dom_result
                    overall_risk_score += dom_result.get("risk_score", 0.0) * component_weights["dom"]
            
            # Determine final prediction
            prediction, confidence = self._determine_final_prediction(pipeline_results, overall_risk_score)
            risk_level = self._determine_risk_level(overall_risk_score)
            
            return {
                "request_id": request.request_id,
                "prediction": prediction,
                "confidence": confidence,
                "risk_score": overall_risk_score,
                "risk_level": risk_level,
                "processing_time_ms": (datetime.utcnow() - request.created_at).total_seconds() * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "pipeline_components": pipeline_results,
                "cache_hit": False
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def _execute_component_pipeline(self, pipeline_name: str, request: InferenceRequest) -> Optional[Dict[str, Any]]:
        """Execute a component pipeline"""
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(pipeline_name)
            if circuit_breaker and not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker open for {pipeline_name}")
                return None
            
            # Select component
            components = self.components.get(pipeline_name, [])
            load_balancer = self.load_balancers.get(pipeline_name)
            
            if not components or not load_balancer:
                return None
            
            component = load_balancer.select_component(components)
            if not component:
                logger.warning(f"No healthy components available for {pipeline_name}")
                return None
            
            # Execute component
            start_time = time.time()
            result = await self._execute_component(component, request)
            execution_time = (time.time() - start_time) * 1000
            
            # Update component statistics
            component.total_requests += 1
            component.response_time_ms = (component.response_time_ms + execution_time) / 2
            
            if result:
                component.success_rate = (component.success_rate + 1.0) / 2
                circuit_breaker.record_success()
            else:
                component.error_count += 1
                component.success_rate = max(0.0, component.success_rate - 0.1)
                circuit_breaker.record_failure()
            
            return result
            
        except Exception as e:
            logger.error(f"Component pipeline {pipeline_name} failed: {e}")
            circuit_breaker = self.circuit_breakers.get(pipeline_name)
            if circuit_breaker:
                circuit_breaker.record_failure()
            return None
    
    async def _execute_component(self, component: PipelineComponent, request: InferenceRequest) -> Optional[Dict[str, Any]]:
        """Execute a single component"""
        try:
            # Simulate component execution
            await asyncio.sleep(component.response_time_ms / 1000.0)
            
            # Simulate different results based on component type
            if "classifier" in component.name:
                return {
                    "prediction": "phish" if "suspicious" in request.content.lower() else "benign",
                    "confidence": 0.85,
                    "risk_score": 0.7 if "suspicious" in request.content.lower() else 0.2,
                    "component": component.name,
                    "processing_time_ms": component.response_time_ms
                }
            elif "analyzer" in component.name:
                return {
                    "risk_indicators": ["suspicious_domain", "urgent_language"],
                    "risk_score": 0.6,
                    "component": component.name,
                    "processing_time_ms": component.response_time_ms
                }
            elif "capture" in component.name:
                return {
                    "screenshot_path": f"screenshots/{request.request_id}.png",
                    "page_info": {"title": "Test Page", "elements": 100},
                    "risk_score": 0.3,
                    "component": component.name,
                    "processing_time_ms": component.response_time_ms
                }
            else:
                return {
                    "result": "processed",
                    "risk_score": 0.4,
                    "component": component.name,
                    "processing_time_ms": component.response_time_ms
                }
            
        except Exception as e:
            logger.error(f"Component {component.name} execution failed: {e}")
            return None
    
    def _determine_final_prediction(self, pipeline_results: Dict[str, Any], overall_risk_score: float) -> tuple:
        """Determine final prediction from pipeline results"""
        if overall_risk_score >= 0.7:
            return "phish", min(0.95, overall_risk_score + 0.1)
        elif overall_risk_score >= 0.4:
            return "suspicious", overall_risk_score
        else:
            return "benign", max(0.05, 1.0 - overall_risk_score)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.6:
            return "MEDIUM"
        elif risk_score >= 0.4:
            return "LOW"
        else:
            return "SAFE"
    
    # Cache management
    def _get_cached_result(self, request: InferenceRequest) -> Optional[Dict[str, Any]]:
        """Get cached result for request"""
        try:
            cache_key = self._generate_cache_key(request)
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                
                # Check if cache entry is still valid
                if self._is_cache_valid(cached_data):
                    return cached_data["result"]
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def _cache_result(self, request: InferenceRequest, result: Dict[str, Any]):
        """Cache result for request"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_entry = {
                "result": result,
                "cached_at": datetime.utcnow(),
                "ttl": self.config.cache_ttl
            }
            
            self.cache[cache_key] = cache_entry
            
            # Implement cache eviction if needed
            if len(self.cache) > 1000:  # Limit cache size
                self._evict_cache_entries()
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        content_hash = hashlib.md5(f"{request.content_type}:{request.content}".encode()).hexdigest()
        return f"cache:{content_hash}"
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        try:
            cached_at = cached_data["cached_at"]
            ttl = cached_data["ttl"]
            expiry_time = cached_at + timedelta(seconds=ttl)
            return datetime.utcnow() < expiry_time
        except:
            return False
    
    def _evict_cache_entries(self):
        """Evict cache entries based on strategy"""
        try:
            if self.cache_strategy == CacheStrategy.LRU:
                # Remove oldest entries
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1]["cached_at"]
                )
                for key, _ in sorted_entries[:100]:  # Remove 100 oldest entries
                    del self.cache[key]
            elif self.cache_strategy == CacheStrategy.TTL:
                # Remove expired entries
                expired_keys = [
                    key for key, data in self.cache.items()
                    if not self._is_cache_valid(data)
                ]
                for key in expired_keys:
                    del self.cache[key]
            
        except Exception as e:
            logger.error(f"Failed to evict cache entries: {e}")
    
    # Health monitoring
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all components"""
        try:
            for pipeline_name, components in self.components.items():
                for component in components:
                    await self._check_component_health(component)
            
        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")
    
    async def _check_component_health(self, component: PipelineComponent):
        """Check health of a single component"""
        try:
            # Simulate health check
            start_time = time.time()
            await asyncio.sleep(0.01)  # Simulate health check time
            response_time = (time.time() - start_time) * 1000
            
            # Update component status based on response time and success rate
            if response_time < 100 and component.success_rate > 0.9:
                component.status = ComponentStatus.HEALTHY
            elif response_time < 500 and component.success_rate > 0.7:
                component.status = ComponentStatus.DEGRADED
            else:
                component.status = ComponentStatus.UNHEALTHY
            
            component.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Health check failed for component {component.name}: {e}")
            component.status = ComponentStatus.UNHEALTHY
    
    # Cleanup
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._cleanup_expired_cache()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            expired_keys = [
                key for key, data in self.cache.items()
                if not self._is_cache_valid(data)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
    
    # Statistics
    def _update_stats(self, processing_time_ms: float, success: bool):
        """Update pipeline statistics"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average processing time
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "stats": self.stats.copy(),
            "components": {
                pipeline_name: [
                    {
                        "name": component.name,
                        "status": component.status.value,
                        "response_time_ms": component.response_time_ms,
                        "success_rate": component.success_rate,
                        "total_requests": component.total_requests,
                        "error_count": component.error_count,
                        "last_health_check": component.last_health_check.isoformat()
                    }
                    for component in components
                ]
                for pipeline_name, components in self.components.items()
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure_time": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            "cache_stats": {
                "size": len(self.cache),
                "hits": self.stats["cache_hits"],
                "misses": self.stats["cache_misses"],
                "hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            }
        }

def create_pipeline_orchestrator(config: Optional[OrchestrationConfig] = None) -> PipelineOrchestrator:
    """Factory function to create pipeline orchestrator instance"""
    return PipelineOrchestrator(config)