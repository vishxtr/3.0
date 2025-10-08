#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple test runner for real-time inference system
"""

import sys
import asyncio
import json
import time
import random
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any

# Add the inference directory to the path
sys.path.append('.')

from api.realtime_api import app, AnalysisRequest, AnalysisResponse
from redis.redis_manager import RedisManager
from websocket.websocket_manager import WebSocketManager
from queue.queue_manager import QueueManager
from monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_LATENCY
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from config import config

class TestRunner:
    """Simple test runner"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_test(self, name, test_func):
        """Add a test to the runner"""
        self.tests.append((name, test_func))
    
    async def run_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("Running Real-Time Inference System Tests")
        print("=" * 60)
        
        for name, test_func in self.tests:
            try:
                print(f"\nRunning: {name}")
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                print(f"✓ PASSED: {name}")
                self.passed += 1
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"  Error: {str(e)}")
                self.failed += 1
        
        print("\n" + "=" * 60)
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        print("=" * 60)
        
        return self.failed == 0

# Test functions
def test_redis_manager_init():
    """Test Redis manager initialization"""
    with patch('redis.StrictRedis') as mock_redis_class:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client
        
        manager = RedisManager()
        assert manager.redis_client is not None
        mock_client.ping.assert_called_once()

def test_redis_cache_operations():
    """Test Redis cache operations"""
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

async def test_websocket_connection_management():
    """Test WebSocket connection management"""
    manager = WebSocketManager()
    mock_websocket = AsyncMock()
    user_id = "test_user"
    
    # Test connection
    await manager.connect(mock_websocket, user_id)
    assert user_id in manager.active_connections
    assert mock_websocket in manager.active_connections[user_id]
    mock_websocket.send.assert_called_once()
    mock_websocket.wait_closed.assert_called_once()

def test_queue_manager_init():
    """Test queue manager initialization"""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis = MagicMock()
        mock_redis_class.return_value = mock_redis
        
        manager = QueueManager()
        assert manager.redis_connection is not None
        assert manager.queue is not None

def test_queue_enqueue_request():
    """Test queue enqueue request"""
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

def test_pipeline_orchestrator_init():
    """Test pipeline orchestrator initialization"""
    orchestrator = PipelineOrchestrator()
    assert orchestrator.nlp_api is not None
    assert orchestrator.visual_api is not None

async def test_pipeline_run():
    """Test pipeline run"""
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

def test_configuration_consistency():
    """Test configuration consistency"""
    # Test that config values are reasonable
    assert config.API_PORT > 0
    assert config.REDIS_PORT > 0
    assert config.PHISHING_THRESHOLD > 0
    assert config.SUSPICIOUS_THRESHOLD > 0
    assert config.PHISHING_THRESHOLD > config.SUSPICIOUS_THRESHOLD
    assert config.REDIS_CACHE_TTL_SECONDS > 0
    assert config.QUEUE_TIMEOUT_SECONDS > 0

async def test_concurrent_requests():
    """Test concurrent request processing"""
    orchestrator = PipelineOrchestrator()
    
    # Create multiple concurrent requests
    tasks = []
    for i in range(3):  # Reduced for faster testing
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
    assert len(successful_results) == 3
    
    # Check processing time (should be reasonable for concurrent processing)
    total_time = end_time - start_time
    assert total_time < 2.0  # Should complete within 2 seconds

def test_metrics_performance():
    """Test metrics collection performance"""
    # Test that metrics can be created and updated quickly
    start_time = time.time()
    
    # Simulate many metric updates
    for i in range(50):
        REQUEST_COUNT.labels(endpoint='/analyze', method='POST', status='200').inc()
        REQUEST_LATENCY.labels(endpoint='/analyze', method='POST').observe(0.1)
        INFERENCE_LATENCY.labels(model_type='nlp', prediction='benign').observe(0.05)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Should be very fast
    assert total_time < 0.1  # Should complete within 100ms

async def main():
    """Main test runner"""
    runner = TestRunner()
    
    # Add all tests
    runner.add_test("Redis Manager Initialization", test_redis_manager_init)
    runner.add_test("Redis Cache Operations", test_redis_cache_operations)
    runner.add_test("WebSocket Connection Management", test_websocket_connection_management)
    runner.add_test("Queue Manager Initialization", test_queue_manager_init)
    runner.add_test("Queue Enqueue Request", test_queue_enqueue_request)
    runner.add_test("Pipeline Orchestrator Initialization", test_pipeline_orchestrator_init)
    runner.add_test("Pipeline Run", test_pipeline_run)
    runner.add_test("Configuration Consistency", test_configuration_consistency)
    runner.add_test("Concurrent Requests", test_concurrent_requests)
    runner.add_test("Metrics Performance", test_metrics_performance)
    
    # Run tests
    success = await runner.run_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {runner.failed} tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)