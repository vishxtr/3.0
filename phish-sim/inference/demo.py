#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Real-Time Inference Pipeline Demo and Performance Benchmarks
"""

import asyncio
import json
import time
import random
import statistics
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add the inference directory to the path
sys.path.append('.')

# Mock classes for demo (same as in simple_test_demo.py)
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
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cached_result(self, key: str):
        result = self.redis_client.get(key)
        if result:
            self.cache_hits += 1
            return json.loads(result)
        self.cache_misses += 1
        return None
    
    def set_cached_result(self, key: str, value: Dict, ttl: int = 300):
        self.redis_client.setex(key, ttl, json.dumps(value))
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2)
        }

class MockWebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self.messages_sent = 0
    
    async def connect(self, websocket, user_id: str):
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        return True
    
    async def send_message(self, user_id: str, message: Dict):
        if user_id in self.active_connections:
            self.messages_sent += 1
            return True
        return False
    
    async def broadcast_message(self, message: Dict):
        for user_id in self.active_connections:
            await self.send_message(user_id, message)
        return True

class MockQueueManager:
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
        self.processed_jobs = 0
    
    def enqueue_analysis_request(self, func, *args, **kwargs):
        self.job_counter += 1
        job_id = f"job_{self.job_counter}"
        self.jobs[job_id] = {
            "status": "queued",
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "created_at": time.time()
        }
        return job_id
    
    def process_job(self, job_id: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = time.time()
            self.processed_jobs += 1
            return True
        return False
    
    def get_queue_metrics(self):
        queued = len([j for j in self.jobs.values() if j["status"] == "queued"])
        completed = len([j for j in self.jobs.values() if j["status"] == "completed"])
        return {
            "queued_jobs": queued,
            "completed_jobs": completed,
            "total_jobs": len(self.jobs),
            "processing_rate": self.processed_jobs / max(1, time.time() - min([j["created_at"] for j in self.jobs.values()], default=time.time()))
        }

class MockPipelineOrchestrator:
    def __init__(self):
        self.nlp_api = MockNLPAPI()
        self.visual_api = MockVisualAPI()
        self.requests_processed = 0
        self.total_processing_time = 0
    
    async def run_pipeline(self, content: str, content_type: str, request_id: str):
        start_time = time.perf_counter()
        self.requests_processed += 1
        
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
        self.total_processing_time += processing_time_ms
        
        return {
            "prediction": final_prediction,
            "confidence": round(final_confidence, 4),
            "explanation": {
                "nlp": nlp_result["explanation"],
                "visual": visual_result["explanation"]
            },
            "processing_time_ms": round(processing_time_ms, 2)
        }
    
    def get_performance_stats(self):
        avg_processing_time = self.total_processing_time / max(1, self.requests_processed)
        return {
            "requests_processed": self.requests_processed,
            "total_processing_time_ms": round(self.total_processing_time, 2),
            "average_processing_time_ms": round(avg_processing_time, 2)
        }

class MockNLPAPI:
    async def analyze(self, content: str):
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate processing
        is_phish = "phish" in content.lower() or "urgent" in content.lower() or "security" in content.lower()
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

class PerformanceBenchmark:
    """Performance benchmark suite"""
    
    def __init__(self):
        self.results = {}
    
    async def benchmark_single_request(self, orchestrator: MockPipelineOrchestrator, content: str, content_type: str):
        """Benchmark single request processing"""
        start_time = time.perf_counter()
        result = await orchestrator.run_pipeline(content, content_type, f"benchmark_{int(time.time())}")
        end_time = time.perf_counter()
        
        return {
            "processing_time_ms": result["processing_time_ms"],
            "total_time_ms": (end_time - start_time) * 1000,
            "prediction": result["prediction"],
            "confidence": result["confidence"]
        }
    
    async def benchmark_concurrent_requests(self, orchestrator: MockPipelineOrchestrator, num_requests: int):
        """Benchmark concurrent request processing"""
        test_contents = [
            "urgent security alert - click here immediately",
            "phishing email with malicious link",
            "normal business email content",
            "http://suspicious-login.net/login",
            "https://legitimate-bank.com/login",
            "free money offer - click now",
            "account verification required",
            "normal newsletter content"
        ]
        
        tasks = []
        for i in range(num_requests):
            content = random.choice(test_contents)
            content_type = "url" if content.startswith("http") else "email"
            task = orchestrator.run_pipeline(content, content_type, f"concurrent_{i}")
            tasks.append(task)
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        processing_times = [r["processing_time_ms"] for r in results]
        
        return {
            "num_requests": num_requests,
            "total_time_ms": round(total_time, 2),
            "requests_per_second": round(num_requests / (total_time / 1000), 2),
            "average_processing_time_ms": round(statistics.mean(processing_times), 2),
            "min_processing_time_ms": round(min(processing_times), 2),
            "max_processing_time_ms": round(max(processing_times), 2),
            "median_processing_time_ms": round(statistics.median(processing_times), 2)
        }
    
    async def benchmark_cache_performance(self, redis_manager: MockRedisManager, orchestrator: MockPipelineOrchestrator):
        """Benchmark cache performance"""
        test_content = "test phishing email content"
        content_type = "email"
        
        # First request (cache miss)
        start_time = time.perf_counter()
        result1 = await orchestrator.run_pipeline(test_content, content_type, "cache_test_1")
        redis_manager.set_cached_result(f"analysis:{content_type}:{test_content}", result1, 300)
        end_time = time.perf_counter()
        cache_miss_time = (end_time - start_time) * 1000
        
        # Second request (cache hit)
        start_time = time.perf_counter()
        cached_result = redis_manager.get_cached_result(f"analysis:{content_type}:{test_content}")
        end_time = time.perf_counter()
        cache_hit_time = (end_time - start_time) * 1000
        
        return {
            "cache_miss_time_ms": round(cache_miss_time, 2),
            "cache_hit_time_ms": round(cache_hit_time, 2),
            "speedup_factor": round(cache_miss_time / max(cache_hit_time, 0.001), 2),
            "cache_stats": redis_manager.get_cache_stats()
        }
    
    async def benchmark_websocket_performance(self, websocket_manager: MockWebSocketManager):
        """Benchmark WebSocket performance"""
        # Simulate multiple connections
        for i in range(10):
            await websocket_manager.connect(f"mock_websocket_{i}", f"user_{i}")
        
        # Benchmark message sending
        start_time = time.perf_counter()
        for i in range(100):
            await websocket_manager.send_message(f"user_{i % 10}", {"type": "test", "data": f"message_{i}"})
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        messages_per_second = 100 / (total_time / 1000)
        
        return {
            "total_messages": 100,
            "total_time_ms": round(total_time, 2),
            "messages_per_second": round(messages_per_second, 2),
            "active_connections": len(websocket_manager.active_connections)
        }
    
    async def benchmark_queue_performance(self, queue_manager: MockQueueManager):
        """Benchmark queue performance"""
        # Enqueue many jobs
        start_time = time.perf_counter()
        job_ids = []
        for i in range(100):
            job_id = queue_manager.enqueue_analysis_request(lambda x: x, f"content_{i}")
            job_ids.append(job_id)
        enqueue_time = (time.perf_counter() - start_time) * 1000
        
        # Process jobs
        start_time = time.perf_counter()
        for job_id in job_ids:
            queue_manager.process_job(job_id)
        process_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "jobs_enqueued": len(job_ids),
            "enqueue_time_ms": round(enqueue_time, 2),
            "process_time_ms": round(process_time, 2),
            "enqueue_rate_per_second": round(len(job_ids) / (enqueue_time / 1000), 2),
            "process_rate_per_second": round(len(job_ids) / (process_time / 1000), 2),
            "queue_metrics": queue_manager.get_queue_metrics()
        }

async def run_demo():
    """Run the complete demo"""
    print("=" * 80)
    print("🚀 Real-Time AI/ML-Based Phishing Detection & Prevention - Demo")
    print("=" * 80)
    
    # Initialize components
    print("\n📋 Initializing Components...")
    redis_manager = MockRedisManager()
    websocket_manager = MockWebSocketManager()
    queue_manager = MockQueueManager()
    orchestrator = MockPipelineOrchestrator()
    benchmark = PerformanceBenchmark()
    
    print("✅ All components initialized successfully!")
    
    # Demo 1: Single Request Processing
    print("\n" + "=" * 60)
    print("🔍 Demo 1: Single Request Processing")
    print("=" * 60)
    
    test_cases = [
        ("urgent security alert - click here immediately", "email"),
        ("http://suspicious-login.net/login", "url"),
        ("normal business email content", "email"),
        ("https://legitimate-bank.com/login", "url")
    ]
    
    for content, content_type in test_cases:
        print(f"\n📝 Processing: {content[:50]}...")
        result = await orchestrator.run_pipeline(content, content_type, f"demo_{int(time.time())}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Processing Time: {result['processing_time_ms']}ms")
        print(f"   Explanation: {result['explanation']}")
    
    # Demo 2: Cache Performance
    print("\n" + "=" * 60)
    print("💾 Demo 2: Cache Performance")
    print("=" * 60)
    
    cache_results = await benchmark.benchmark_cache_performance(redis_manager, orchestrator)
    print(f"Cache Miss Time: {cache_results['cache_miss_time_ms']}ms")
    print(f"Cache Hit Time: {cache_results['cache_hit_time_ms']}ms")
    print(f"Speedup Factor: {cache_results['speedup_factor']}x")
    print(f"Cache Stats: {cache_results['cache_stats']}")
    
    # Demo 3: Concurrent Processing
    print("\n" + "=" * 60)
    print("⚡ Demo 3: Concurrent Request Processing")
    print("=" * 60)
    
    concurrent_results = await benchmark.benchmark_concurrent_requests(orchestrator, 20)
    print(f"Number of Requests: {concurrent_results['num_requests']}")
    print(f"Total Time: {concurrent_results['total_time_ms']}ms")
    print(f"Requests per Second: {concurrent_results['requests_per_second']}")
    print(f"Average Processing Time: {concurrent_results['average_processing_time_ms']}ms")
    print(f"Min/Max Processing Time: {concurrent_results['min_processing_time_ms']}ms / {concurrent_results['max_processing_time_ms']}ms")
    print(f"Median Processing Time: {concurrent_results['median_processing_time_ms']}ms")
    
    # Demo 4: WebSocket Performance
    print("\n" + "=" * 60)
    print("🌐 Demo 4: WebSocket Performance")
    print("=" * 60)
    
    websocket_results = await benchmark.benchmark_websocket_performance(websocket_manager)
    print(f"Total Messages: {websocket_results['total_messages']}")
    print(f"Total Time: {websocket_results['total_time_ms']}ms")
    print(f"Messages per Second: {websocket_results['messages_per_second']}")
    print(f"Active Connections: {websocket_results['active_connections']}")
    
    # Demo 5: Queue Performance
    print("\n" + "=" * 60)
    print("📋 Demo 5: Queue Performance")
    print("=" * 60)
    
    queue_results = await benchmark.benchmark_queue_performance(queue_manager)
    print(f"Jobs Enqueued: {queue_results['jobs_enqueued']}")
    print(f"Enqueue Time: {queue_results['enqueue_time_ms']}ms")
    print(f"Process Time: {queue_results['process_time_ms']}ms")
    print(f"Enqueue Rate: {queue_results['enqueue_rate_per_second']} jobs/sec")
    print(f"Process Rate: {queue_results['process_rate_per_second']} jobs/sec")
    print(f"Queue Metrics: {queue_results['queue_metrics']}")
    
    # Demo 6: End-to-End Integration
    print("\n" + "=" * 60)
    print("🔄 Demo 6: End-to-End Integration")
    print("=" * 60)
    
    # Simulate a complete request flow
    content = "urgent security alert - verify your account now"
    content_type = "email"
    request_id = "integration_demo"
    
    print(f"📨 Processing request: {content}")
    
    # Check cache
    cache_key = f"analysis:{content_type}:{content}"
    cached_result = redis_manager.get_cached_result(cache_key)
    if cached_result:
        print("✅ Cache hit - returning cached result")
        result = cached_result
    else:
        print("❌ Cache miss - processing request")
        result = await orchestrator.run_pipeline(content, content_type, request_id)
        redis_manager.set_cached_result(cache_key, result, 300)
    
    print(f"🎯 Result: {result['prediction']} (confidence: {result['confidence']})")
    
    # Enqueue for background processing
    job_id = queue_manager.enqueue_analysis_request(
        lambda x: x, content, content_type=content_type
    )
    print(f"📋 Enqueued job: {job_id}")
    
    # Send WebSocket notification
    await websocket_manager.broadcast_message({
        "type": "analysis_complete",
        "request_id": request_id,
        "result": result
    })
    print("📡 WebSocket notification sent")
    
    # Final Performance Summary
    print("\n" + "=" * 60)
    print("📊 Performance Summary")
    print("=" * 60)
    
    orchestrator_stats = orchestrator.get_performance_stats()
    print(f"Total Requests Processed: {orchestrator_stats['requests_processed']}")
    print(f"Average Processing Time: {orchestrator_stats['average_processing_time_ms']}ms")
    print(f"Total Processing Time: {orchestrator_stats['total_processing_time_ms']}ms")
    
    cache_stats = redis_manager.get_cache_stats()
    print(f"Cache Hit Rate: {cache_stats['hit_rate_percent']}%")
    print(f"Cache Hits: {cache_stats['cache_hits']}")
    print(f"Cache Misses: {cache_stats['cache_misses']}")
    
    print(f"WebSocket Messages Sent: {websocket_manager.messages_sent}")
    print(f"Queue Jobs Processed: {queue_manager.processed_jobs}")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 80)

async def main():
    """Main function"""
    try:
        await run_demo()
        return 0
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)