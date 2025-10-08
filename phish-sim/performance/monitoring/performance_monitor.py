# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Performance Monitoring System
Comprehensive monitoring of system performance, resource usage, and metrics
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
except ImportError:
    # Fallback for environments without prometheus_client
    class MockMetric:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    Counter = Histogram = Gauge = Summary = MockMetric
    CollectorRegistry = MockMetric
    generate_latest = lambda: b""

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class SystemResource:
    """System resource usage"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    network_sent_mb: float
    network_recv_mb: float
    timestamp: datetime

@dataclass
class ApplicationMetric:
    """Application-specific metrics"""
    request_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    throughput_rps: float
    active_connections: int
    cache_hit_ratio: float
    timestamp: datetime

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, collection_interval: float = 1.0, 
                 history_size: int = 1000,
                 enable_prometheus: bool = True):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_prometheus = enable_prometheus
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.system_resources: deque = deque(maxlen=history_size)
        self.application_metrics: deque = deque(maxlen=history_size)
        
        # Prometheus metrics
        self.registry = CollectorRegistry() if enable_prometheus else None
        self.prometheus_metrics = {}
        
        # Performance tracking
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.active_requests: Dict[str, int] = defaultdict(int)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Custom metrics callbacks
        self.custom_metrics_callbacks: List[Callable] = []
        
        # Initialize Prometheus metrics
        if self.enable_prometheus:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            # System metrics
            self.prometheus_metrics['cpu_usage'] = Gauge(
                'system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry
            )
            self.prometheus_metrics['memory_usage'] = Gauge(
                'system_memory_usage_percent', 'Memory usage percentage', registry=self.registry
            )
            self.prometheus_metrics['disk_usage'] = Gauge(
                'system_disk_usage_percent', 'Disk usage percentage', registry=self.registry
            )
            
            # Application metrics
            self.prometheus_metrics['request_count'] = Counter(
                'application_requests_total', 'Total requests', ['method', 'endpoint'], registry=self.registry
            )
            self.prometheus_metrics['request_duration'] = Histogram(
                'application_request_duration_seconds', 'Request duration', 
                ['method', 'endpoint'], registry=self.registry
            )
            self.prometheus_metrics['active_connections'] = Gauge(
                'application_active_connections', 'Active connections', registry=self.registry
            )
            self.prometheus_metrics['error_count'] = Counter(
                'application_errors_total', 'Total errors', ['error_type'], registry=self.registry
            )
            self.prometheus_metrics['cache_hit_ratio'] = Gauge(
                'application_cache_hit_ratio', 'Cache hit ratio', registry=self.registry
            )
            self.prometheus_metrics['throughput'] = Gauge(
                'application_throughput_rps', 'Requests per second', registry=self.registry
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
            self.enable_prometheus = False
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system resources
                system_resources = await self._collect_system_resources()
                self.system_resources.append(system_resources)
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                self.application_metrics.append(app_metrics)
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self._update_prometheus_metrics(system_resources, app_metrics)
                
                # Execute custom metrics callbacks
                await self._execute_custom_callbacks()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_resources(self) -> SystemResource:
        """Collect system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            
            # Network usage
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)
            
            return SystemResource(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_used_gb=disk_used_gb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system resources: {e}")
            return SystemResource(
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                memory_available_mb=0, disk_usage_percent=0, disk_used_gb=0,
                network_sent_mb=0, network_recv_mb=0, timestamp=datetime.utcnow()
            )
    
    async def _collect_application_metrics(self) -> ApplicationMetric:
        """Collect application-specific metrics"""
        try:
            # Calculate request metrics
            total_requests = sum(len(times) for times in self.request_times.values())
            all_times = [time for times in self.request_times.values() for time in times]
            
            if all_times:
                response_time_avg = sum(all_times) / len(all_times)
                sorted_times = sorted(all_times)
                p95_index = int(len(sorted_times) * 0.95)
                p99_index = int(len(sorted_times) * 0.99)
                response_time_p95 = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
                response_time_p99 = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
            else:
                response_time_avg = response_time_p95 = response_time_p99 = 0.0
            
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
            # Calculate throughput (requests per second)
            throughput_rps = total_requests / self.collection_interval if self.collection_interval > 0 else 0.0
            
            # Active connections
            active_connections = sum(self.active_requests.values())
            
            # Cache hit ratio (placeholder - would be provided by cache system)
            cache_hit_ratio = 0.0  # This would be updated by the cache system
            
            return ApplicationMetric(
                request_count=total_requests,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                error_rate=error_rate,
                throughput_rps=throughput_rps,
                active_connections=active_connections,
                cache_hit_ratio=cache_hit_ratio,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return ApplicationMetric(
                request_count=0, response_time_avg=0, response_time_p95=0,
                response_time_p99=0, error_rate=0, throughput_rps=0,
                active_connections=0, cache_hit_ratio=0, timestamp=datetime.utcnow()
            )
    
    def _update_prometheus_metrics(self, system_resources: SystemResource, 
                                 app_metrics: ApplicationMetric):
        """Update Prometheus metrics"""
        try:
            # System metrics
            self.prometheus_metrics['cpu_usage'].set(system_resources.cpu_percent)
            self.prometheus_metrics['memory_usage'].set(system_resources.memory_percent)
            self.prometheus_metrics['disk_usage'].set(system_resources.disk_usage_percent)
            
            # Application metrics
            self.prometheus_metrics['active_connections'].set(app_metrics.active_connections)
            self.prometheus_metrics['cache_hit_ratio'].set(app_metrics.cache_hit_ratio)
            self.prometheus_metrics['throughput'].set(app_metrics.throughput_rps)
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    async def _execute_custom_callbacks(self):
        """Execute custom metrics callbacks"""
        for callback in self.custom_metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Custom metrics callback error: {e}")
    
    def record_request(self, method: str, endpoint: str, duration: float, 
                      success: bool = True):
        """Record request metrics"""
        try:
            # Record request time
            key = f"{method}:{endpoint}"
            self.request_times[key].append(duration)
            
            # Keep only recent times (last 1000 requests per endpoint)
            if len(self.request_times[key]) > 1000:
                self.request_times[key] = self.request_times[key][-1000:]
            
            # Record error if not successful
            if not success:
                self.error_counts[key] += 1
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.prometheus_metrics['request_count'].labels(
                    method=method, endpoint=endpoint
                ).inc()
                self.prometheus_metrics['request_duration'].labels(
                    method=method, endpoint=endpoint
                ).observe(duration)
                
                if not success:
                    self.prometheus_metrics['error_count'].labels(
                        error_type=key
                    ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record request: {e}")
    
    def record_active_connection(self, connection_id: str, active: bool):
        """Record active connection"""
        try:
            if active:
                self.active_requests[connection_id] = 1
            else:
                self.active_requests.pop(connection_id, None)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                active_count = sum(self.active_requests.values())
                self.prometheus_metrics['active_connections'].set(active_count)
            
        except Exception as e:
            logger.error(f"Failed to record connection: {e}")
    
    def update_cache_hit_ratio(self, hit_ratio: float):
        """Update cache hit ratio"""
        try:
            if self.enable_prometheus:
                self.prometheus_metrics['cache_hit_ratio'].set(hit_ratio)
        except Exception as e:
            logger.error(f"Failed to update cache hit ratio: {e}")
    
    def add_custom_metrics_callback(self, callback: Callable):
        """Add custom metrics collection callback"""
        self.custom_metrics_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            system_resources = self.system_resources[-1] if self.system_resources else None
            app_metrics = self.application_metrics[-1] if self.application_metrics else None
            
            return {
                'system_resources': system_resources.__dict__ if system_resources else None,
                'application_metrics': app_metrics.__dict__ if app_metrics else None,
                'monitoring_active': self.is_monitoring,
                'collection_interval': self.collection_interval,
                'history_size': len(self.metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def get_metrics_history(self, duration_minutes: int = 60) -> Dict[str, List[Dict]]:
        """Get metrics history for specified duration"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
            
            # Filter system resources
            system_history = [
                resource.__dict__ for resource in self.system_resources
                if resource.timestamp >= cutoff_time
            ]
            
            # Filter application metrics
            app_history = [
                metric.__dict__ for metric in self.application_metrics
                if metric.timestamp >= cutoff_time
            ]
            
            return {
                'system_resources': system_history,
                'application_metrics': app_history,
                'duration_minutes': duration_minutes,
                'data_points': len(system_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return {'system_resources': [], 'application_metrics': []}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            if self.enable_prometheus and self.registry:
                return generate_latest(self.registry).decode('utf-8')
            else:
                return "# Prometheus metrics not available\n"
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return f"# Error getting metrics: {e}\n"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with key metrics"""
        try:
            if not self.system_resources or not self.application_metrics:
                return {"error": "No metrics data available"}
            
            # Get recent data (last 10 minutes)
            recent_system = [r for r in self.system_resources 
                           if r.timestamp >= datetime.utcnow() - timedelta(minutes=10)]
            recent_app = [m for m in self.application_metrics 
                         if m.timestamp >= datetime.utcnow() - timedelta(minutes=10)]
            
            if not recent_system or not recent_app:
                return {"error": "No recent metrics data available"}
            
            # Calculate averages
            avg_cpu = sum(r.cpu_percent for r in recent_system) / len(recent_system)
            avg_memory = sum(r.memory_percent for r in recent_system) / len(recent_system)
            avg_response_time = sum(m.response_time_avg for m in recent_app) / len(recent_app)
            avg_throughput = sum(m.throughput_rps for m in recent_app) / len(recent_app)
            avg_error_rate = sum(m.error_rate for m in recent_app) / len(recent_app)
            
            return {
                'performance_summary': {
                    'avg_cpu_usage': round(avg_cpu, 2),
                    'avg_memory_usage': round(avg_memory, 2),
                    'avg_response_time': round(avg_response_time, 3),
                    'avg_throughput_rps': round(avg_throughput, 2),
                    'avg_error_rate': round(avg_error_rate, 2),
                    'monitoring_duration_minutes': 10,
                    'data_points': len(recent_system)
                },
                'current_status': {
                    'monitoring_active': self.is_monitoring,
                    'prometheus_enabled': self.enable_prometheus,
                    'collection_interval': self.collection_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": f"Failed to generate summary: {e}"}

# Global performance monitor instance
performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor

# Performance monitoring decorators
def monitor_performance(func_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                monitor.record_request("FUNCTION", func_name or func.__name__, duration, True)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_request("FUNCTION", func_name or func.__name__, duration, False)
                raise
        
        return wrapper
    return decorator

def track_resource_usage():
    """Decorator for tracking resource usage"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            
            # Record start resources
            start_resources = await monitor._collect_system_resources()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record end resources
                end_resources = await monitor._collect_system_resources()
                
                # Log resource usage
                cpu_delta = end_resources.cpu_percent - start_resources.cpu_percent
                memory_delta = end_resources.memory_used_mb - start_resources.memory_used_mb
                
                logger.info(f"Function {func.__name__} resource usage - CPU: {cpu_delta:.2f}%, Memory: {memory_delta:.2f}MB")
                
                return result
                
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator