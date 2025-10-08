# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Comprehensive tests for real-time inference system
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

# Import components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import the actual components we created
from api.realtime_api import app, AnalysisRequest, AnalysisResponse
from redis.redis_manager import RedisManager
from websocket.websocket_manager import WebSocketManager
from queue.queue_manager import QueueManager
from monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_LATENCY
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from config import config

class TestRedisManager:
    """Test Redis manager functionality"""
    
    @pytest.fixture
    def redis_manager(self):
        """Create Redis manager instance"""
        return RedisManager()
    
    @pytest.mark.asyncio
    async def test_redis_initialization(self, redis_manager):
        """Test Redis manager initialization"""
        # Mock Redis connection
        with patch('redis.redis_manager.redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            await redis_manager.initialize()
            
            assert redis_manager.is_initialized
            assert redis_manager.redis_client is not None
    
    def test_cache_key_generation(self, redis_manager):
        """Test cache key generation"""
        key1 = redis_manager._generate_cache_key("test content", "text")
        key2 = redis_manager._generate_cache_key("test content", "text")
        key3 = redis_manager._generate_cache_key("different content", "text")
        
        assert key1 == key2  # Same content should generate same key
        assert key1 != key3  # Different content should generate different key
        assert key1.startswith("cache:text:")
    
    def test_data_serialization(self, redis_manager):
        """Test data serialization and deserialization"""
        test_data = {
            "prediction": "phish",
            "confidence": 0.85,
            "risk_score": 0.7,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Serialize
        serialized = redis_manager._serialize_data(test_data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = redis_manager._deserialize_data(serialized)
        assert deserialized == test_data
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, redis_manager):
        """Test cache operations"""
        with patch('redis.redis_manager.redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_client.get.return_value = None
            mock_client.setex.return_value = True
            mock_redis.return_value = mock_client
            
            await redis_manager.initialize()
            
            # Test cache miss
            result = await redis_manager.get_cached_result("test content", "text")
            assert result is None
            
            # Test cache set
            test_result = {"prediction": "benign", "confidence": 0.9}
            success = await redis_manager.cache_result("test content", "text", test_result)
            assert success is True

class TestWebSocketManager:
    """Test WebSocket manager functionality"""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager instance"""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client.host = "127.0.0.1"
        websocket.headers = {"user-agent": "test-client"}
        return websocket
    
    @pytest.mark.asyncio
    async def test_connection_management(self, websocket_manager, mock_websocket):
        """Test connection management"""
        # Test connection
        client_id = await websocket_manager.connection_manager.connect(mock_websocket)
        assert client_id in websocket_manager.connection_manager.active_connections
        assert client_id in websocket_manager.connection_manager.connection_metadata
        
        # Test message sending
        message = {"type": "test", "data": "hello"}
        success = await websocket_manager.connection_manager.send_message(client_id, message)
        assert success is True
        
        # Test disconnection
        await websocket_manager.connection_manager.disconnect(client_id)
        assert client_id not in websocket_manager.connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_room_management(self, websocket_manager, mock_websocket):
        """Test room management"""
        client_id = await websocket_manager.connection_manager.connect(mock_websocket)
        
        # Test joining room
        success = await websocket_manager.connection_manager.join_room(client_id, "test_room")
        assert success is True
        assert "test_room" in websocket_manager.connection_manager.connection_rooms[client_id]
        
        # Test leaving room
        success = await websocket_manager.connection_manager.leave_room(client_id, "test_room")
        assert success is True
        assert "test_room" not in websocket_manager.connection_manager.connection_rooms[client_id]
    
    def test_rate_limiting(self, websocket_manager):
        """Test rate limiting"""
        client_id = "test_client"
        
        # Should not be rate limited initially
        assert not websocket_manager.connection_manager.is_rate_limited(client_id)
        
        # Simulate many requests
        for _ in range(70):  # More than the limit
            websocket_manager.connection_manager.is_rate_limited(client_id)
        
        # Should now be rate limited
        assert websocket_manager.connection_manager.is_rate_limited(client_id)

class TestQueueManager:
    """Test queue manager functionality"""
    
    @pytest.fixture
    def queue_manager(self):
        """Create queue manager instance"""
        return QueueManager()
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, queue_manager):
        """Test queue manager initialization"""
        await queue_manager.initialize()
        assert queue_manager.is_processing
    
    @pytest.mark.asyncio
    async def test_request_enqueue_dequeue(self, queue_manager):
        """Test request enqueue and dequeue"""
        await queue_manager.initialize()
        
        # Enqueue request
        request_id = await queue_manager.enqueue_request(
            content="test content",
            content_type="text",
            priority=Priority.HIGH
        )
        
        assert request_id is not None
        assert request_id in queue_manager.request_status
        assert queue_manager.request_status[request_id] == RequestStatus.PENDING
        
        # Dequeue request
        request = await queue_manager.dequeue_request()
        assert request is not None
        assert request.request_id == request_id
        assert request.priority == Priority.HIGH
    
    @pytest.mark.asyncio
    async def test_worker_management(self, queue_manager):
        """Test worker management"""
        await queue_manager.initialize()
        
        # Register worker
        worker_id = "test_worker"
        success = await queue_manager.register_worker(worker_id)
        assert success is True
        assert worker_id in queue_manager.workers
        assert worker_id in queue_manager.available_workers
        
        # Get worker status
        status = await queue_manager.get_worker_status(worker_id)
        assert status is not None
        assert status["worker_id"] == worker_id
        assert status["status"] == "available"
        
        # Unregister worker
        success = await queue_manager.unregister_worker(worker_id)
        assert success is True
        assert worker_id not in queue_manager.workers
    
    @pytest.mark.asyncio
    async def test_request_processing(self, queue_manager):
        """Test request processing"""
        await queue_manager.initialize()
        
        # Register worker
        worker_id = "test_worker"
        await queue_manager.register_worker(worker_id)
        
        # Create and process request
        request = queue_manager.QueueRequest(
            request_id="test_request",
            content="test content",
            content_type="text",
            priority=Priority.NORMAL,
            created_at=datetime.utcnow(),
            timeout=30
        )
        
        result = await queue_manager.process_request(request, worker_id)
        assert result is not None
        assert "request_id" in result
        assert result["request_id"] == "test_request"

class TestMetricsCollector:
    """Test metrics collector functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector instance"""
        return MetricsCollector()
    
    @pytest.mark.asyncio
    async def test_metrics_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        await metrics_collector.initialize()
        assert metrics_collector.is_healthy()
    
    @pytest.mark.asyncio
    async def test_request_metrics(self, metrics_collector):
        """Test request metrics recording"""
        await metrics_collector.initialize()
        
        # Record request
        await metrics_collector.record_request(
            request_id="test_request",
            method="POST",
            path="/infer",
            status_code=200,
            processing_time_ms=150.0
        )
        
        assert metrics_collector.stats["total_requests"] == 1
        assert metrics_collector.stats["successful_requests"] == 1
        assert metrics_collector.counters["requests_total"] == 1
        assert metrics_collector.counters["requests_200"] == 1
    
    @pytest.mark.asyncio
    async def test_error_metrics(self, metrics_collector):
        """Test error metrics recording"""
        await metrics_collector.initialize()
        
        # Record error
        await metrics_collector.record_error(
            request_id="test_request",
            error="TimeoutError",
            processing_time_ms=5000.0
        )
        
        assert metrics_collector.counters["errors_total"] == 1
        assert metrics_collector.error_counts["TimeoutError"] == 1
    
    def test_alert_evaluation(self, metrics_collector):
        """Test alert evaluation"""
        # Test high response time alert
        metrics_collector.stats["average_response_time"] = 6.0  # Above threshold
        is_triggered = asyncio.run(metrics_collector._evaluate_alert(
            metrics_collector.alerts["high_response_time"]
        ))
        assert is_triggered is True
        
        # Test normal response time
        metrics_collector.stats["average_response_time"] = 1.0  # Below threshold
        is_triggered = asyncio.run(metrics_collector._evaluate_alert(
            metrics_collector.alerts["high_response_time"]
        ))
        assert is_triggered is False
    
    def test_performance_metrics(self, metrics_collector):
        """Test performance metrics calculation"""
        # Add some response times
        metrics_collector.response_times.extend([100, 200, 300, 400, 500])
        
        # Test percentile calculation
        p95 = metrics_collector._calculate_percentile(metrics_collector.response_times, 95)
        assert p95 == 500  # 95th percentile of [100, 200, 300, 400, 500]
        
        p50 = metrics_collector._calculate_percentile(metrics_collector.response_times, 50)
        assert p50 == 300  # 50th percentile (median)

class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality"""
    
    @pytest.fixture
    def pipeline_orchestrator(self):
        """Create pipeline orchestrator instance"""
        return PipelineOrchestrator()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline_orchestrator):
        """Test pipeline orchestrator initialization"""
        await pipeline_orchestrator.initialize()
        assert pipeline_orchestrator.is_healthy()
    
    @pytest.mark.asyncio
    async def test_request_processing(self, pipeline_orchestrator):
        """Test request processing through pipeline"""
        await pipeline_orchestrator.initialize()
        
        # Process request
        result = await pipeline_orchestrator.process_request(
            content="suspicious email content",
            content_type="email",
            request_id="test_request",
            priority=1,
            timeout=30
        )
        
        assert result is not None
        assert "request_id" in result
        assert "prediction" in result
        assert "confidence" in result
        assert "risk_score" in result
        assert "risk_level" in result
        assert result["request_id"] == "test_request"
    
    def test_component_selection(self, pipeline_orchestrator):
        """Test component selection by load balancer"""
        # Create test components
        components = [
            pipeline_orchestrator.PipelineComponent(
                name="component1",
                status=ComponentStatus.HEALTHY,
                response_time_ms=100.0,
                success_rate=0.95,
                last_health_check=datetime.utcnow()
            ),
            pipeline_orchestrator.PipelineComponent(
                name="component2",
                status=ComponentStatus.HEALTHY,
                response_time_ms=150.0,
                success_rate=0.90,
                last_health_check=datetime.utcnow()
            )
        ]
        
        # Test load balancer selection
        load_balancer = pipeline_orchestrator.load_balancers["ml_pipeline"]
        selected = load_balancer.select_component(components)
        assert selected is not None
        assert selected in components
    
    def test_circuit_breaker(self, pipeline_orchestrator):
        """Test circuit breaker functionality"""
        circuit_breaker = pipeline_orchestrator.circuit_breakers["ml_pipeline"]
        
        # Initially should be closed
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.can_execute() is True
        
        # Record failures to open circuit
        for _ in range(6):  # More than failure threshold
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == "open"
        assert circuit_breaker.can_execute() is False
    
    def test_cache_operations(self, pipeline_orchestrator):
        """Test cache operations"""
        # Create test request
        request = pipeline_orchestrator.InferenceRequest(
            request_id="test_request",
            content="test content",
            content_type="text",
            priority=1,
            timeout=30,
            return_features=False,
            return_explanation=True,
            created_at=datetime.utcnow()
        )
        
        # Test cache key generation
        cache_key = pipeline_orchestrator._generate_cache_key(request)
        assert cache_key.startswith("cache:")
        
        # Test cache result
        test_result = {"prediction": "benign", "confidence": 0.9}
        pipeline_orchestrator._cache_result(request, test_result)
        assert cache_key in pipeline_orchestrator.cache
        
        # Test get cached result
        cached = pipeline_orchestrator._get_cached_result(request)
        assert cached == test_result

class TestIntegration:
    """Integration tests for the real-time system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_inference(self):
        """Test end-to-end inference flow"""
        # Initialize components
        redis_manager = RedisManager()
        queue_manager = QueueManager()
        metrics_collector = MetricsCollector()
        pipeline_orchestrator = PipelineOrchestrator()
        
        # Mock Redis
        with patch('redis.redis_manager.redis.asyncio.Redis') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_redis.return_value = mock_client
            
            await redis_manager.initialize()
            await queue_manager.initialize()
            await metrics_collector.initialize()
            await pipeline_orchestrator.initialize()
            
            # Process inference request
            result = await pipeline_orchestrator.process_request(
                content="test phishing email",
                content_type="email",
                request_id="integration_test",
                priority=1,
                timeout=30
            )
            
            assert result is not None
            assert result["request_id"] == "integration_test"
            assert "prediction" in result
            assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test WebSocket integration"""
        websocket_manager = WebSocketManager()
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.receive_text = AsyncMock(return_value='{"type": "ping"}')
        mock_websocket.close = AsyncMock()
        mock_websocket.client.host = "127.0.0.1"
        mock_websocket.headers = {"user-agent": "test-client"}
        
        # Test connection handling
        await websocket_manager.start()
        
        # Simulate connection
        client_id = await websocket_manager.connection_manager.connect(mock_websocket)
        assert client_id is not None
        
        # Test message handling
        await websocket_manager.connection_manager.handle_message(
            client_id, {"type": "ping"}
        )
        
        await websocket_manager.stop()
    
    def test_configuration_consistency(self):
        """Test configuration consistency across components"""
        api_config = get_config("api")
        redis_config = get_config("redis")
        websocket_config = get_config("websocket")
        queue_config = get_config("queue")
        monitoring_config = get_config("monitoring")
        orchestration_config = get_config("orchestration")
        
        # Check that all configs are loaded
        assert api_config is not None
        assert redis_config is not None
        assert websocket_config is not None
        assert queue_config is not None
        assert monitoring_config is not None
        assert orchestration_config is not None
        
        # Check specific config values
        assert api_config.port > 0
        assert redis_config.port > 0
        assert websocket_config.port > 0
        assert queue_config.max_workers > 0
        assert monitoring_config.health_check_interval > 0
        assert orchestration_config.pipeline_timeout > 0

class TestPerformance:
    """Performance tests for the real-time system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request processing"""
        pipeline_orchestrator = PipelineOrchestrator()
        await pipeline_orchestrator.initialize()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = pipeline_orchestrator.process_request(
                content=f"test content {i}",
                content_type="text",
                request_id=f"concurrent_test_{i}",
                priority=1,
                timeout=30
            )
            tasks.append(task)
        
        # Process all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10
        
        # Check processing time (should be reasonable for concurrent processing)
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_metrics_performance(self):
        """Test metrics collection performance"""
        metrics_collector = MetricsCollector()
        await metrics_collector.initialize()
        
        # Record many requests quickly
        start_time = time.time()
        for i in range(1000):
            await metrics_collector.record_request(
                request_id=f"perf_test_{i}",
                method="POST",
                path="/infer",
                status_code=200,
                processing_time_ms=100.0
            )
        end_time = time.time()
        
        # Check performance
        total_time = end_time - start_time
        requests_per_second = 1000 / total_time
        
        assert requests_per_second > 1000  # Should handle >1000 requests/second
        assert metrics_collector.stats["total_requests"] == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])