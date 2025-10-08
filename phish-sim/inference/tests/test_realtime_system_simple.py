# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simplified comprehensive tests for real-time inference system
"""

import pytest
import asyncio
import json
import time
import random
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any

# Import components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from api.realtime_api import app, AnalysisRequest, AnalysisResponse
from redis.redis_manager import RedisManager
from websocket.websocket_manager import WebSocketManager
from queue.queue_manager import QueueManager
from monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_LATENCY
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from config import config

class TestRedisManager:
    """Test Redis manager functionality"""
    
    def test_redis_manager_init(self):
        """Test Redis manager initialization"""
        with patch('redis.StrictRedis') as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis_class.return_value = mock_client
            
            manager = RedisManager()
            assert manager.redis_client is not None
            mock_client.ping.assert_called_once()
    
    def test_cache_operations(self):
        """Test cache operations"""
        with patch('redis.StrictRedis') as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.setex.return_value = True
            mock_redis_class.return_value = mock_client
            
            manager = RedisManager()
            
            # Test cache miss
            result = manager.get_cached_result("test_key")
            assert result is None
            mock_client.get.assert_called_with("test_key")
            
            # Test cache set
            test_data = {"prediction": "benign", "confidence": 0.9}
            manager.set_cached_result("test_key", test_data, 300)
            mock_client.setex.assert_called_with("test_key", 300, json.dumps(test_data))
    
    def test_session_operations(self):
        """Test session operations"""
        with patch('redis.StrictRedis') as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.setex.return_value = True
            mock_redis_class.return_value = mock_client
            
            manager = RedisManager()
            
            # Test session data operations
            session_data = {"user_id": "test_user", "requests": 5}
            manager.set_session_data("session123", session_data, 600)
            mock_client.setex.assert_called_with("session:session123", 600, json.dumps(session_data))
            
            # Test get session data
            manager.get_session_data("session123")
            mock_client.get.assert_called_with("session:session123")
    
    def test_metrics_operations(self):
        """Test metrics operations"""
        with patch('redis.StrictRedis') as mock_redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.incr.return_value = 5
            mock_client.get.return_value = "5"
            mock_redis_class.return_value = mock_client
            
            manager = RedisManager()
            
            # Test increment metric
            manager.increment_metric("test_metric", 3)
            mock_client.incr.assert_called_with("metric:test_metric", 3)
            
            # Test get metric
            value = manager.get_metric("test_metric")
            assert value == 5
            mock_client.get.assert_called_with("metric:test_metric")

class TestWebSocketManager:
    """Test WebSocket manager functionality"""
    
    def test_websocket_manager_init(self):
        """Test WebSocket manager initialization"""
        manager = WebSocketManager()
        assert manager.active_connections == {}
    
    @pytest.mark.asyncio
    async def test_connection_management(self):
        """Test connection management"""
        manager = WebSocketManager()
        mock_websocket = AsyncMock()
        user_id = "test_user"
        
        # Test connection
        await manager.connect(mock_websocket, user_id)
        assert user_id in manager.active_connections
        assert mock_websocket in manager.active_connections[user_id]
        mock_websocket.send.assert_called_once()
        mock_websocket.wait_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test message sending"""
        manager = WebSocketManager()
        mock_websocket = AsyncMock()
        user_id = "test_user"
        
        # Connect first
        await manager.connect(mock_websocket, user_id)
        mock_websocket.send.reset_mock()
        
        # Test send message
        message = {"type": "test", "data": "hello"}
        await manager.send_message(user_id, message)
        mock_websocket.send.assert_called_once_with(json.dumps(message))
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcast message"""
        manager = WebSocketManager()
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        user_id1 = "user1"
        user_id2 = "user2"
        
        # Connect both users
        await manager.connect(mock_websocket1, user_id1)
        await manager.connect(mock_websocket2, user_id2)
        mock_websocket1.send.reset_mock()
        mock_websocket2.send.reset_mock()
        
        # Test broadcast
        message = {"type": "broadcast", "data": "everyone"}
        await manager.broadcast_message(message)
        
        mock_websocket1.send.assert_called_once_with(json.dumps(message))
        mock_websocket2.send.assert_called_once_with(json.dumps(message))

class TestQueueManager:
    """Test queue manager functionality"""
    
    def test_queue_manager_init(self):
        """Test queue manager initialization"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            
            manager = QueueManager()
            assert manager.redis_connection is not None
            assert manager.queue is not None
    
    def test_enqueue_analysis_request(self):
        """Test enqueue analysis request"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            
            with patch('rq.Queue') as mock_queue_class:
                mock_queue = MagicMock()
                mock_job = MagicMock()
                mock_job.id = "test_job_123"
                mock_queue.enqueue.return_value = mock_job
                mock_queue_class.return_value = mock_queue
                
                manager = QueueManager()
                
                def mock_func(x): return x
                job_id = manager.enqueue_analysis_request(mock_func, "test_content", content_type="url")
                
                assert job_id == "test_job_123"
                mock_queue.enqueue.assert_called_once()
    
    def test_get_job_status(self):
        """Test get job status"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            
            with patch('rq.Queue') as mock_queue_class:
                mock_queue = MagicMock()
                mock_job = MagicMock()
                mock_job.id = "test_job_123"
                mock_job.get_status.return_value = "finished"
                mock_job.result = {"prediction": "benign"}
                mock_job.exc_info = None
                mock_queue.fetch_job.return_value = mock_job
                mock_queue_class.return_value = mock_queue
                
                manager = QueueManager()
                status = manager.get_job_status("test_job_123")
                
                assert status["id"] == "test_job_123"
                assert status["status"] == "finished"
                assert status["result"] == {"prediction": "benign"}
    
    def test_get_queue_metrics(self):
        """Test get queue metrics"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = MagicMock()
            mock_redis_class.return_value = mock_redis
            
            with patch('rq.Queue') as mock_queue_class:
                mock_queue = MagicMock()
                mock_queue.count = 5
                mock_queue.failed_job_registry.count = 1
                mock_queue.deferred_job_registry.count = 0
                mock_queue.started_job_registry.count = 2
                mock_queue.finished_job_registry.count = 2
                mock_queue_class.return_value = mock_queue
                
                with patch('rq.Worker.all') as mock_worker_all:
                    mock_worker_all.return_value = [MagicMock(), MagicMock()]
                    
                    manager = QueueManager()
                    metrics = manager.get_queue_metrics()
                    
                    assert metrics["queued_jobs"] == 5
                    assert metrics["failed_jobs"] == 1
                    assert metrics["workers_count"] == 2

class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality"""
    
    def test_pipeline_orchestrator_init(self):
        """Test pipeline orchestrator initialization"""
        orchestrator = PipelineOrchestrator()
        assert orchestrator.nlp_api is not None
        assert orchestrator.visual_api is not None
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        """Test run pipeline"""
        orchestrator = PipelineOrchestrator()
        
        # Test with phishing content
        result = await orchestrator.run_pipeline(
            "http://phishing-site.com/login", "url", "test_req_001"
        )
        
        assert result is not None
        assert "prediction" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "processing_time_ms" in result
        assert result["prediction"] in ["phish", "benign", "suspicious"]
        assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_pipeline_with_different_content_types(self):
        """Test pipeline with different content types"""
        orchestrator = PipelineOrchestrator()
        
        # Test URL
        result_url = await orchestrator.run_pipeline(
            "http://suspicious-login.net", "url", "test_req_url"
        )
        assert result_url is not None
        
        # Test email
        result_email = await orchestrator.run_pipeline(
            "urgent security alert email", "email", "test_req_email"
        )
        assert result_email is not None
        
        # Test text
        result_text = await orchestrator.run_pipeline(
            "normal text content", "text", "test_req_text"
        )
        assert result_text is not None

class TestIntegration:
    """Integration tests for the real-time system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_inference(self):
        """Test end-to-end inference flow"""
        # Test the FastAPI app endpoints
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        
        # Test model info
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "nlp_model" in data
        assert "visual_model" in data
        assert "thresholds" in data
        
        # Test analysis endpoint
        response = client.post("/analyze", json={
            "content": "test phishing email",
            "content_type": "email"
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
    
    def test_configuration_consistency(self):
        """Test configuration consistency"""
        # Test that config values are reasonable
        assert config.API_PORT > 0
        assert config.REDIS_PORT > 0
        assert config.PHISHING_THRESHOLD > 0
        assert config.SUSPICIOUS_THRESHOLD > 0
        assert config.PHISHING_THRESHOLD > config.SUSPICIOUS_THRESHOLD
        assert config.REDIS_CACHE_TTL_SECONDS > 0
        assert config.QUEUE_TIMEOUT_SECONDS > 0

class TestPerformance:
    """Performance tests for the real-time system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request processing"""
        orchestrator = PipelineOrchestrator()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):  # Reduced for faster testing
            task = orchestrator.run_pipeline(
                f"test content {i}", "text", f"concurrent_test_{i}"
            )
            tasks.append(task)
        
        # Process all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 5
        
        # Check processing time (should be reasonable for concurrent processing)
        total_time = end_time - start_time
        assert total_time < 3.0  # Should complete within 3 seconds
    
    def test_metrics_performance(self):
        """Test metrics collection performance"""
        # Test that metrics can be created and updated quickly
        start_time = time.time()
        
        # Simulate many metric updates
        for i in range(100):
            REQUEST_COUNT.labels(endpoint='/analyze', method='POST', status='200').inc()
            REQUEST_LATENCY.labels(endpoint='/analyze', method='POST').observe(0.1)
            INFERENCE_LATENCY.labels(model_type='nlp', prediction='benign').observe(0.05)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be very fast
        assert total_time < 0.1  # Should complete within 100ms

if __name__ == "__main__":
    pytest.main([__file__, "-v"])