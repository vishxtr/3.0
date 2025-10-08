# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Request queuing and load balancing system
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import heapq
from collections import defaultdict, deque

from config import get_config, QueueConfig

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class RequestStatus(Enum):
    """Request status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class QueueRequest:
    """Queue request structure"""
    request_id: str
    content: str
    content_type: str
    priority: Priority
    created_at: datetime
    timeout: int
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other):
        """Priority queue comparison (higher priority first)"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at

@dataclass
class Worker:
    """Worker information"""
    worker_id: str
    status: str
    current_request: Optional[str]
    started_at: datetime
    processed_count: int
    last_activity: datetime

class LoadBalancer:
    """Load balancer for distributing requests"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.worker_stats = {}
        self.round_robin_index = 0
    
    def select_worker(self, available_workers: List[str]) -> Optional[str]:
        """Select worker based on strategy"""
        if not available_workers:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(available_workers)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(available_workers)
        elif self.strategy == "weighted":
            return self._weighted_selection(available_workers)
        else:
            return available_workers[0]
    
    def _round_robin_selection(self, available_workers: List[str]) -> str:
        """Round-robin worker selection"""
        worker = available_workers[self.round_robin_index % len(available_workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_selection(self, available_workers: List[str]) -> str:
        """Select worker with least connections"""
        min_connections = float('inf')
        selected_worker = available_workers[0]
        
        for worker_id in available_workers:
            connections = self.worker_stats.get(worker_id, {}).get('connections', 0)
            if connections < min_connections:
                min_connections = connections
                selected_worker = worker_id
        
        return selected_worker
    
    def _weighted_selection(self, available_workers: List[str]) -> str:
        """Weighted worker selection based on performance"""
        total_weight = 0
        weights = {}
        
        for worker_id in available_workers:
            stats = self.worker_stats.get(worker_id, {})
            # Weight based on success rate and processing time
            success_rate = stats.get('success_rate', 0.5)
            avg_time = stats.get('avg_processing_time', 1.0)
            weight = success_rate / max(avg_time, 0.1)
            weights[worker_id] = weight
            total_weight += weight
        
        if total_weight == 0:
            return available_workers[0]
        
        # Select based on weight
        import random
        rand = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker_id, weight in weights.items():
            current_weight += weight
            if rand <= current_weight:
                return worker_id
        
        return available_workers[0]
    
    def update_worker_stats(self, worker_id: str, stats: Dict[str, Any]):
        """Update worker statistics"""
        self.worker_stats[worker_id] = stats

class QueueManager:
    """Request queue manager with load balancing"""
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or get_config("queue")
        
        # Queues
        self.priority_queue = []  # Heap for priority-based queuing
        self.batch_queue = deque()  # Queue for batch processing
        self.dead_letter_queue = deque()  # Queue for failed requests
        
        # Workers
        self.workers: Dict[str, Worker] = {}
        self.available_workers: List[str] = []
        
        # Load balancer
        self.load_balancer = LoadBalancer(self.config.load_balancer)
        
        # Processing state
        self.is_processing = False
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "processed_requests": 0,
            "failed_requests": 0,
            "timeout_requests": 0,
            "queue_depth": 0,
            "average_processing_time": 0.0,
            "throughput_per_second": 0.0
        }
        
        # Request tracking
        self.request_status: Dict[str, RequestStatus] = {}
        self.request_results: Dict[str, Any] = {}
        self.request_start_times: Dict[str, float] = {}
        
        # Batch processing
        self.batch_buffer: List[QueueRequest] = []
        self.batch_timer: Optional[asyncio.Task] = None
        
        # Circuit breaker
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half_open
        }
    
    async def initialize(self):
        """Initialize queue manager"""
        try:
            # Start processing tasks
            self.is_processing = True
            
            # Start batch processing if enabled
            if self.config.enable_batching:
                self.batch_timer = asyncio.create_task(self._batch_processing_loop())
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("Queue manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize queue manager: {e}")
            raise
    
    async def close(self):
        """Close queue manager"""
        try:
            self.is_processing = False
            
            # Cancel processing tasks
            for task in self.processing_tasks.values():
                task.cancel()
            
            # Cancel batch timer
            if self.batch_timer:
                self.batch_timer.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
            
            logger.info("Queue manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing queue manager: {e}")
    
    def is_healthy(self) -> bool:
        """Check if queue manager is healthy"""
        return (
            self.is_processing and
            len(self.available_workers) > 0 and
            self.circuit_breaker["state"] != "open"
        )
    
    async def enqueue_request(
        self,
        content: str,
        content_type: str,
        priority: Priority = Priority.NORMAL,
        timeout: int = 30,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enqueue a request for processing"""
        try:
            request_id = f"req_{uuid.uuid4().hex[:12]}_{int(time.time())}"
            
            # Create request
            request = QueueRequest(
                request_id=request_id,
                content=content,
                content_type=content_type,
                priority=priority,
                created_at=datetime.utcnow(),
                timeout=timeout,
                callback=callback,
                metadata=metadata
            )
            
            # Check circuit breaker
            if self.circuit_breaker["state"] == "open":
                if self._should_attempt_reset():
                    self.circuit_breaker["state"] = "half_open"
                else:
                    raise Exception("Circuit breaker is open")
            
            # Add to appropriate queue
            if self.config.enable_batching and priority.value < Priority.HIGH.value:
                self.batch_queue.append(request)
            else:
                heapq.heappush(self.priority_queue, request)
            
            # Update status
            self.request_status[request_id] = RequestStatus.PENDING
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["queue_depth"] = len(self.priority_queue) + len(self.batch_queue)
            
            logger.debug(f"Enqueued request {request_id} with priority {priority.name}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue request: {e}")
            raise
    
    async def dequeue_request(self) -> Optional[QueueRequest]:
        """Dequeue a request for processing"""
        try:
            # Try priority queue first
            if self.priority_queue:
                return heapq.heappop(self.priority_queue)
            
            # Then batch queue
            if self.batch_queue:
                return self.batch_queue.popleft()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue request: {e}")
            return None
    
    async def process_request(self, request: QueueRequest, worker_id: str) -> Any:
        """Process a request"""
        try:
            start_time = time.time()
            self.request_start_times[request.request_id] = start_time
            
            # Update status
            self.request_status[request.request_id] = RequestStatus.PROCESSING
            
            # Update worker status
            if worker_id in self.workers:
                self.workers[worker_id].current_request = request.request_id
                self.workers[worker_id].last_activity = datetime.utcnow()
            
            # Simulate processing (in real implementation, this would call the pipeline)
            result = await self._simulate_processing(request)
            
            # Update status
            self.request_status[request.request_id] = RequestStatus.COMPLETED
            self.request_results[request.request_id] = result
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            # Call callback if provided
            if request.callback:
                try:
                    await request.callback(request.request_id, result)
                except Exception as e:
                    logger.error(f"Callback failed for request {request.request_id}: {e}")
            
            # Reset circuit breaker on success
            if self.circuit_breaker["state"] == "half_open":
                self.circuit_breaker["state"] = "closed"
                self.circuit_breaker["failures"] = 0
            
            logger.debug(f"Processed request {request.request_id} in {processing_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            self.request_status[request.request_id] = RequestStatus.TIMEOUT
            self.stats["timeout_requests"] += 1
            logger.warning(f"Request {request.request_id} timed out")
            raise
            
        except Exception as e:
            self.request_status[request.request_id] = RequestStatus.FAILED
            self.stats["failed_requests"] += 1
            self._handle_failure()
            logger.error(f"Request {request.request_id} failed: {e}")
            raise
        
        finally:
            # Clean up worker status
            if worker_id in self.workers:
                self.workers[worker_id].current_request = None
                self.workers[worker_id].processed_count += 1
                self.workers[worker_id].last_activity = datetime.utcnow()
    
    async def _simulate_processing(self, request: QueueRequest) -> Dict[str, Any]:
        """Simulate request processing"""
        # In real implementation, this would integrate with the pipeline orchestrator
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "request_id": request.request_id,
            "prediction": "benign",
            "confidence": 0.85,
            "risk_score": 0.15,
            "risk_level": "low",
            "processing_time_ms": 100,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics"""
        self.stats["processed_requests"] += 1
        
        # Update average processing time
        total_processed = self.stats["processed_requests"]
        current_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
    
    def _handle_failure(self):
        """Handle processing failure"""
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = datetime.utcnow()
        
        if self.circuit_breaker["failures"] >= self.config.retry_attempts:
            self.circuit_breaker["state"] = "open"
            logger.warning("Circuit breaker opened due to failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.circuit_breaker["last_failure"]:
            return True
        
        time_since_failure = (
            datetime.utcnow() - self.circuit_breaker["last_failure"]
        ).total_seconds()
        
        return time_since_failure >= self.config.retry_delay
    
    # Worker management
    async def register_worker(self, worker_id: str) -> bool:
        """Register a new worker"""
        try:
            worker = Worker(
                worker_id=worker_id,
                status="available",
                current_request=None,
                started_at=datetime.utcnow(),
                processed_count=0,
                last_activity=datetime.utcnow()
            )
            
            self.workers[worker_id] = worker
            self.available_workers.append(worker_id)
            
            logger.info(f"Registered worker: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker_id}: {e}")
            return False
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker"""
        try:
            if worker_id in self.workers:
                # Cancel current request if any
                current_request = self.workers[worker_id].current_request
                if current_request:
                    self.request_status[current_request] = RequestStatus.CANCELLED
                
                # Remove from workers
                del self.workers[worker_id]
                if worker_id in self.available_workers:
                    self.available_workers.remove(worker_id)
                
                logger.info(f"Unregistered worker: {worker_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister worker {worker_id}: {e}")
            return False
    
    async def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker status"""
        if worker_id not in self.workers:
            return None
        
        worker = self.workers[worker_id]
        return {
            "worker_id": worker_id,
            "status": worker.status,
            "current_request": worker.current_request,
            "started_at": worker.started_at.isoformat(),
            "processed_count": worker.processed_count,
            "last_activity": worker.last_activity.isoformat(),
            "uptime_seconds": (datetime.utcnow() - worker.started_at).total_seconds()
        }
    
    # Batch processing
    async def _batch_processing_loop(self):
        """Batch processing loop"""
        while self.is_processing:
            try:
                await asyncio.sleep(self.config.batch_timeout)
                
                if self.batch_buffer:
                    await self._process_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
    
    async def _process_batch(self):
        """Process a batch of requests"""
        try:
            batch_size = min(len(self.batch_buffer), self.config.batch_size)
            batch = [self.batch_buffer.pop(0) for _ in range(batch_size)]
            
            logger.info(f"Processing batch of {len(batch)} requests")
            
            # Process batch concurrently
            tasks = []
            for request in batch:
                worker_id = self._select_worker_for_request(request)
                if worker_id:
                    task = asyncio.create_task(
                        self.process_request(request, worker_id)
                    )
                    tasks.append(task)
            
            # Wait for batch completion
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    def _select_worker_for_request(self, request: QueueRequest) -> Optional[str]:
        """Select worker for request"""
        if not self.available_workers:
            return None
        
        return self.load_balancer.select_worker(self.available_workers)
    
    # Cleanup and monitoring
    async def _cleanup_loop(self):
        """Cleanup loop for expired requests"""
        while self.is_processing:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up expired requests
                expired_count = await self._cleanup_expired_requests()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired requests")
                
                # Update worker statistics
                self._update_worker_statistics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_requests(self) -> int:
        """Clean up expired requests"""
        try:
            now = time.time()
            expired_requests = []
            
            for request_id, start_time in self.request_start_times.items():
                if now - start_time > 300:  # 5 minutes timeout
                    expired_requests.append(request_id)
            
            # Remove expired requests
            for request_id in expired_requests:
                self.request_status[request_id] = RequestStatus.TIMEOUT
                del self.request_start_times[request_id]
                if request_id in self.request_results:
                    del self.request_results[request_id]
            
            return len(expired_requests)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")
            return 0
    
    def _update_worker_statistics(self):
        """Update worker statistics for load balancer"""
        for worker_id, worker in self.workers.items():
            stats = {
                "connections": 1 if worker.current_request else 0,
                "success_rate": 0.95,  # Would be calculated from actual data
                "avg_processing_time": self.stats["average_processing_time"]
            }
            self.load_balancer.update_worker_stats(worker_id, stats)
    
    # Statistics and monitoring
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_depth": len(self.priority_queue) + len(self.batch_queue),
            "priority_queue_size": len(self.priority_queue),
            "batch_queue_size": len(self.batch_queue),
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "active_workers": len(self.available_workers),
            "total_workers": len(self.workers),
            "circuit_breaker_state": self.circuit_breaker["state"],
            "stats": self.stats.copy()
        }
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request status"""
        if request_id not in self.request_status:
            return None
        
        status = self.request_status[request_id]
        result = {
            "request_id": request_id,
            "status": status.value,
            "created_at": None,
            "processing_time": None
        }
        
        if request_id in self.request_start_times:
            result["processing_time"] = time.time() - self.request_start_times[request_id]
        
        if request_id in self.request_results:
            result["result"] = self.request_results[request_id]
        
        return result

def create_queue_manager(config: Optional[QueueConfig] = None) -> QueueManager:
    """Factory function to create queue manager instance"""
    return QueueManager(config)