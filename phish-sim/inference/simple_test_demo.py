#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple test demo for real-time inference system without external dependencies
"""

import sys
import asyncio
import json
import time
import random
from datetime import datetime
from typing import Dict, Any

# Mock classes for testing without external dependencies
class MockRedisClient:
    def __init__(self):
        self.data = {}
    
    def ping(self):
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    
    def incr(self, key, value=1):
        current = int(self.data.get(key, 0))
        self.data[key] = str(current + value)
        return current + value

class MockRedisManager:
    def __init__(self):
        self.redis_client = MockRedisClient()
    
    def get_cached_result(self, key: str):
        result = self.redis_client.get(key)
        if result:
            return json.loads(result)
        return None
    
    def set_cached_result(self, key: str, value: Dict, ttl: int = 300):
        self.redis_client.setex(key, ttl, json.dumps(value))
    
    def get_session_data(self, session_id: str):
        data = self.redis_client.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return None
    
    def set_session_data(self, session_id: str, data: Dict, ttl: int = 300):
        self.redis_client.setex(f"session:{session_id}", ttl, json.dumps(data))
    
    def increment_metric(self, metric_name: str, value: int = 1):
        self.redis_client.incr(f"metric:{metric_name}", value)
    
    def get_metric(self, metric_name: str):
        return int(self.redis_client.get(f"metric:{metric_name}") or 0)

class MockWebSocketManager:
    def __init__(self):
        self.active_connections = {}
    
    async def connect(self, websocket, user_id: str):
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        print(f"WebSocket connected: User {user_id}")
        return True
    
    async def disconnect(self, websocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        print(f"WebSocket disconnected: User {user_id}")
    
    async def send_message(self, user_id: str, message: Dict):
        if user_id in self.active_connections:
            print(f"Message sent to {user_id}: {message}")
            return True
        return False
    
    async def broadcast_message(self, message: Dict):
        for user_id in self.active_connections:
            await self.send_message(user_id, message)
        print(f"Broadcast message: {message}")

class MockQueueManager:
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
    
    def enqueue_analysis_request(self, func, *args, **kwargs):
        self.job_counter += 1
        job_id = f"job_{self.job_counter}"
        self.jobs[job_id] = {
            "status": "queued",
            "func": func,
            "args": args,
            "kwargs": kwargs
        }
        print(f"Enqueued job {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str):
        if job_id in self.jobs:
            return {
                "id": job_id,
                "status": self.jobs[job_id]["status"],
                "result": None
            }
        return None
    
    def get_queue_metrics(self):
        return {
            "queued_jobs": len([j for j in self.jobs.values() if j["status"] == "queued"]),
            "failed_jobs": len([j for j in self.jobs.values() if j["status"] == "failed"]),
            "workers_count": 1
        }

class MockPipelineOrchestrator:
    def __init__(self):
        self.nlp_api = MockNLPAPI()
        self.visual_api = MockVisualAPI()
    
    async def run_pipeline(self, content: str, content_type: str, request_id: str):
        start_time = time.perf_counter()
        
        # Run NLP analysis
        nlp_result = await self.nlp_api.analyze(content)
        
        # Run Visual analysis if URL
        visual_result = {"prediction": "benign", "confidence": 0.8, "explanation": {}}
        if content_type == "url":
            visual_result = await self.visual_api.analyze(content)
        
        # Combine results
        if nlp_result["prediction"] == "phish" or visual_result["prediction"] == "phish":
            final_prediction = "phish"
            final_confidence = max(nlp_result["confidence"], visual_result["confidence"])
        else:
            final_prediction = "benign"
            final_confidence = max(nlp_result["confidence"], visual_result["confidence"])
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        return {
            "prediction": final_prediction,
            "confidence": round(final_confidence, 4),
            "explanation": {
                "nlp": nlp_result["explanation"],
                "visual": visual_result["explanation"]
            },
            "processing_time_ms": round(processing_time_ms, 2)
        }

class MockNLPAPI:
    async def analyze(self, content: str):
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate processing
        is_phish = "phish" in content.lower() or "urgent" in content.lower()
        return {
            "prediction": "phish" if is_phish else "benign",
            "confidence": random.uniform(0.6, 0.95) if is_phish else random.uniform(0.8, 0.99),
            "explanation": {"reason": "keywords detected" if is_phish else "no suspicious keywords"}
        }

class MockVisualAPI:
    async def analyze(self, url: str):
        await asyncio.sleep(random.uniform(0.02, 0.1))  # Simulate processing
        is_phish = "login" in url.lower() and "https" not in url.lower()
        return {
            "prediction": "phish" if is_phish else "benign",
            "confidence": random.uniform(0.6, 0.95) if is_phish else random.uniform(0.8, 0.99),
            "explanation": {"reason": "suspicious login form" if is_phish else "no visual anomalies"}
        }

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
        print("Running Real-Time Inference System Tests (Mock Version)")
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
def test_redis_manager():
    """Test Redis manager functionality"""
    manager = MockRedisManager()
    
    # Test cache operations
    test_data = {"prediction": "benign", "confidence": 0.9}
    manager.set_cached_result("test_key", test_data, 300)
    
    cached_result = manager.get_cached_result("test_key")
    assert cached_result == test_data
    
    # Test session operations
    session_data = {"user_id": "test_user", "requests": 5}
    manager.set_session_data("session123", session_data, 600)
    
    retrieved_session = manager.get_session_data("session123")
    assert retrieved_session == session_data
    
    # Test metrics
    manager.increment_metric("test_metric", 3)
    value = manager.get_metric("test_metric")
    assert value == 3

async def test_websocket_manager():
    """Test WebSocket manager functionality"""
    manager = MockWebSocketManager()
    mock_websocket = "mock_websocket"
    user_id = "test_user"
    
    # Test connection
    await manager.connect(mock_websocket, user_id)
    assert user_id in manager.active_connections
    assert mock_websocket in manager.active_connections[user_id]
    
    # Test message sending
    message = {"type": "test", "data": "hello"}
    success = await manager.send_message(user_id, message)
    assert success is True
    
    # Test broadcast
    await manager.broadcast_message({"type": "broadcast", "data": "everyone"})
    
    # Test disconnection
    await manager.disconnect(mock_websocket, user_id)
    assert user_id not in manager.active_connections

def test_queue_manager():
    """Test queue manager functionality"""
    manager = MockQueueManager()
    
    # Test enqueue
    def mock_func(x): return x
    job_id = manager.enqueue_analysis_request(mock_func, "test_content", content_type="url")
    assert job_id is not None
    assert job_id in manager.jobs
    
    # Test get status
    status = manager.get_job_status(job_id)
    assert status is not None
    assert status["id"] == job_id
    
    # Test metrics
    metrics = manager.get_queue_metrics()
    assert metrics["queued_jobs"] >= 1

def test_pipeline_orchestrator():
    """Test pipeline orchestrator functionality"""
    orchestrator = MockPipelineOrchestrator()
    assert orchestrator.nlp_api is not None
    assert orchestrator.visual_api is not None

async def test_pipeline_run():
    """Test pipeline run"""
    orchestrator = MockPipelineOrchestrator()
    
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

async def test_concurrent_requests():
    """Test concurrent request processing"""
    orchestrator = MockPipelineOrchestrator()
    
    # Create multiple concurrent requests
    tasks = []
    for i in range(5):
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
    assert total_time < 1.0  # Should complete within 1 second

async def test_integration_flow():
    """Test integration flow"""
    # Test the complete flow
    redis_manager = MockRedisManager()
    websocket_manager = MockWebSocketManager()
    queue_manager = MockQueueManager()
    orchestrator = MockPipelineOrchestrator()
    
    # Simulate a request flow
    content = "suspicious phishing email"
    content_type = "email"
    request_id = "integration_test"
    
    # Check cache first
    cached_result = redis_manager.get_cached_result(f"analysis:{content_type}:{content}")
    assert cached_result is None  # Should be cache miss
    
    # Process request
    result = await orchestrator.run_pipeline(content, content_type, request_id)
    assert result is not None
    
    # Cache result
    redis_manager.set_cached_result(f"analysis:{content_type}:{content}", result, 300)
    
    # Check cache hit
    cached_result = redis_manager.get_cached_result(f"analysis:{content_type}:{content}")
    assert cached_result == result
    
    # Update metrics
    redis_manager.increment_metric("requests_total")
    redis_manager.increment_metric("requests_phishing" if result["prediction"] == "phish" else "requests_benign")
    
    # Send WebSocket notification
    await websocket_manager.broadcast_message({
        "type": "analysis_complete",
        "request_id": request_id,
        "result": result
    })

async def main():
    """Main test runner"""
    runner = TestRunner()
    
    # Add all tests
    runner.add_test("Redis Manager", test_redis_manager)
    runner.add_test("WebSocket Manager", test_websocket_manager)
    runner.add_test("Queue Manager", test_queue_manager)
    runner.add_test("Pipeline Orchestrator", test_pipeline_orchestrator)
    runner.add_test("Pipeline Run", test_pipeline_run)
    runner.add_test("Concurrent Requests", test_concurrent_requests)
    runner.add_test("Integration Flow", test_integration_flow)
    
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