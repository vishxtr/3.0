# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Real-time monitoring and metrics collection
"""

import asyncio
import logging
import time
import psutil
import platform
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from config import get_config, MonitoringConfig, PERFORMANCE_TARGETS

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None

@dataclass
class Alert:
    """Alert definition"""
    name: str
    condition: str
    threshold: float
    severity: str
    description: str
    enabled: bool = True

class MetricsCollector:
    """Real-time metrics collector"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or get_config("monitoring")
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Request tracking
        self.request_metrics: Dict[str, Dict[str, Any]] = {}
        self.response_times: deque = deque(maxlen=10000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # System metrics
        self.system_metrics = {}
        self.last_system_update = 0
        
        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, datetime] = {}
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_per_second": 0.0,
            "error_rate": 0.0,
            "uptime_seconds": 0.0
        }
        
        # Performance tracking
        self.performance_targets = PERFORMANCE_TARGETS
        self.start_time = time.time()
        
        # Initialize alerts
        self._initialize_alerts()
    
    async def initialize(self):
        """Initialize metrics collector"""
        try:
            # Start background tasks
            asyncio.create_task(self._system_metrics_loop())
            asyncio.create_task(self._metrics_aggregation_loop())
            asyncio.create_task(self._alert_monitoring_loop())
            
            logger.info("Metrics collector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            raise
    
    async def close(self):
        """Close metrics collector"""
        try:
            # Stop background tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("Metrics collector closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing metrics collector: {e}")
    
    def is_healthy(self) -> bool:
        """Check if metrics collector is healthy"""
        return len(self.metrics) > 0 and time.time() - self.start_time > 0
    
    def _initialize_alerts(self):
        """Initialize default alerts"""
        alerts_config = self.config.alert_thresholds
        
        self.alerts = {
            "high_response_time": Alert(
                name="high_response_time",
                condition="average_response_time > threshold",
                threshold=alerts_config["response_time"],
                severity="warning",
                description="Average response time exceeds threshold"
            ),
            "high_error_rate": Alert(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=alerts_config["error_rate"],
                severity="critical",
                description="Error rate exceeds threshold"
            ),
            "high_cpu_usage": Alert(
                name="high_cpu_usage",
                condition="cpu_usage > threshold",
                threshold=alerts_config["cpu_usage"],
                severity="warning",
                description="CPU usage exceeds threshold"
            ),
            "high_memory_usage": Alert(
                name="high_memory_usage",
                condition="memory_usage > threshold",
                threshold=alerts_config["memory_usage"],
                severity="warning",
                description="Memory usage exceeds threshold"
            ),
            "high_queue_size": Alert(
                name="high_queue_size",
                condition="queue_size > threshold",
                threshold=alerts_config["queue_size"],
                severity="warning",
                description="Queue size exceeds threshold"
            ),
            "high_active_connections": Alert(
                name="high_active_connections",
                condition="active_connections > threshold",
                threshold=alerts_config["active_connections"],
                severity="warning",
                description="Active connections exceed threshold"
            )
        }
    
    # Request metrics
    async def record_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        processing_time_ms: float,
        **kwargs
    ):
        """Record request metrics"""
        try:
            # Update counters
            self.counters["requests_total"] += 1
            self.counters[f"requests_{status_code}"] += 1
            self.counters[f"requests_{method.lower()}"] += 1
            
            # Record response time
            self.response_times.append(processing_time_ms)
            
            # Update request metrics
            self.request_metrics[request_id] = {
                "method": method,
                "path": path,
                "status_code": status_code,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.utcnow(),
                **kwargs
            }
            
            # Update statistics
            self._update_request_stats(status_code, processing_time_ms)
            
            # Check for alerts
            await self._check_request_alerts()
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
    
    async def record_error(self, request_id: str, error: str, processing_time_ms: float = 0):
        """Record error metrics"""
        try:
            self.counters["errors_total"] += 1
            self.error_counts[error] += 1
            
            # Update request metrics if exists
            if request_id in self.request_metrics:
                self.request_metrics[request_id]["error"] = error
                self.request_metrics[request_id]["processing_time_ms"] = processing_time_ms
            
            # Update statistics
            self._update_request_stats(500, processing_time_ms)
            
        except Exception as e:
            logger.error(f"Failed to record error metrics: {e}")
    
    def _update_request_stats(self, status_code: int, processing_time_ms: float):
        """Update request statistics"""
        self.stats["total_requests"] += 1
        
        if 200 <= status_code < 400:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average response time
        if self.response_times:
            self.stats["average_response_time"] = sum(self.response_times) / len(self.response_times)
        
        # Update error rate
        if self.stats["total_requests"] > 0:
            self.stats["error_rate"] = self.stats["failed_requests"] / self.stats["total_requests"]
        
        # Update requests per second (rolling window)
        now = time.time()
        recent_requests = sum(1 for req in self.request_metrics.values() 
                            if (now - req["timestamp"].timestamp()) < 60)
        self.stats["requests_per_second"] = recent_requests / 60.0
        
        # Update uptime
        self.stats["uptime_seconds"] = now - self.start_time
    
    # System metrics
    async def _system_metrics_loop(self):
        """Background loop for system metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                await self._collect_system_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            now = time.time()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            self.system_metrics = {
                "timestamp": datetime.utcnow(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "process_percent": process_cpu
                },
                "memory": {
                    "percent": memory_percent,
                    "used_bytes": memory_used,
                    "total_bytes": memory_total,
                    "process_bytes": process_memory.rss
                },
                "disk": {
                    "percent": disk_percent,
                    "used_bytes": disk.used,
                    "total_bytes": disk.total
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "system": {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "architecture": platform.architecture()[0],
                    "hostname": platform.node()
                }
            }
            
            # Store metrics
            self._store_metric("cpu_usage", cpu_percent)
            self._store_metric("memory_usage", memory_percent)
            self._store_metric("disk_usage", disk_percent)
            self._store_metric("process_memory", process_memory.rss)
            
            self.last_system_update = now
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _store_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Store a metric point"""
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.metrics[name].append(metric_point)
    
    # Metrics aggregation
    async def _metrics_aggregation_loop(self):
        """Background loop for metrics aggregation"""
        while True:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                await self._aggregate_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics over time windows"""
        try:
            now = datetime.utcnow()
            
            # Aggregate response times
            if self.response_times:
                self.histograms["response_time_1m"].extend(list(self.response_times)[-60:])
                self.histograms["response_time_5m"].extend(list(self.response_times)[-300:])
                self.histograms["response_time_15m"].extend(list(self.response_times)[-900:])
            
            # Clean up old metrics
            cutoff_time = now - timedelta(hours=1)
            for metric_name, metric_data in self.metrics.items():
                while metric_data and metric_data[0].timestamp < cutoff_time:
                    metric_data.popleft()
            
            # Clean up old request metrics
            old_requests = [
                req_id for req_id, req_data in self.request_metrics.items()
                if req_data["timestamp"] < cutoff_time
            ]
            for req_id in old_requests:
                del self.request_metrics[req_id]
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
    
    # Alert monitoring
    async def _alert_monitoring_loop(self):
        """Background loop for alert monitoring"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._check_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
    
    async def _check_alerts(self):
        """Check all alerts"""
        try:
            for alert_name, alert in self.alerts.items():
                if not alert.enabled:
                    continue
                
                is_triggered = await self._evaluate_alert(alert)
                
                if is_triggered:
                    if alert_name not in self.active_alerts:
                        # New alert
                        self.active_alerts[alert_name] = datetime.utcnow()
                        await self._trigger_alert(alert)
                else:
                    if alert_name in self.active_alerts:
                        # Alert resolved
                        del self.active_alerts[alert_name]
                        await self._resolve_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    async def _evaluate_alert(self, alert: Alert) -> bool:
        """Evaluate if alert condition is met"""
        try:
            if alert.name == "high_response_time":
                return self.stats["average_response_time"] > alert.threshold
            elif alert.name == "high_error_rate":
                return self.stats["error_rate"] > alert.threshold
            elif alert.name == "high_cpu_usage":
                return self.system_metrics.get("cpu", {}).get("percent", 0) > alert.threshold
            elif alert.name == "high_memory_usage":
                return self.system_metrics.get("memory", {}).get("percent", 0) > alert.threshold
            elif alert.name == "high_queue_size":
                # This would need to be provided by queue manager
                return False
            elif alert.name == "high_active_connections":
                # This would need to be provided by WebSocket manager
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert {alert.name}: {e}")
            return False
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        try:
            alert_data = {
                "name": alert.name,
                "severity": alert.severity,
                "description": alert.description,
                "threshold": alert.threshold,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "triggered"
            }
            
            logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert {alert.name}: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        try:
            alert_data = {
                "name": alert.name,
                "severity": alert.severity,
                "description": alert.description,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "resolved"
            }
            
            logger.info(f"Alert resolved: {alert.name}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert.name}: {e}")
    
    async def _check_request_alerts(self):
        """Check request-specific alerts"""
        try:
            # Check response time alert
            if self.stats["average_response_time"] > self.performance_targets["api"]["response_time_p95"]:
                await self._trigger_alert(self.alerts["high_response_time"])
            
            # Check error rate alert
            if self.stats["error_rate"] > self.performance_targets["api"]["error_rate"]:
                await self._trigger_alert(self.alerts["high_error_rate"])
            
        except Exception as e:
            logger.error(f"Failed to check request alerts: {e}")
    
    # Public API
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "stats": self.stats.copy(),
            "system_metrics": self.system_metrics,
            "active_alerts": list(self.active_alerts.keys()),
            "performance_targets": self.performance_targets
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            "response_time_p95": self._calculate_percentile(self.response_times, 95),
            "response_time_p99": self._calculate_percentile(self.response_times, 99),
            "throughput_per_second": self.stats["requests_per_second"],
            "error_rate": self.stats["error_rate"],
            "availability": self._calculate_availability(),
            "cpu_usage": self.system_metrics.get("cpu", {}).get("percent", 0),
            "memory_usage": self.system_metrics.get("memory", {}).get("percent", 0)
        }
    
    def _calculate_percentile(self, data: deque, percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_availability(self) -> float:
        """Calculate system availability"""
        if self.stats["total_requests"] == 0:
            return 1.0
        
        return self.stats["successful_requests"] / self.stats["total_requests"]
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get alert status"""
        return {
            "active_alerts": [
                {
                    "name": alert_name,
                    "triggered_at": triggered_at.isoformat(),
                    "duration_seconds": (datetime.utcnow() - triggered_at).total_seconds()
                }
                for alert_name, triggered_at in self.active_alerts.items()
            ],
            "alert_definitions": [
                {
                    "name": alert.name,
                    "condition": alert.condition,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "description": alert.description,
                    "enabled": alert.enabled
                }
                for alert in self.alerts.values()
            ]
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        health_status = "healthy"
        issues = []
        
        # Check performance targets
        if self.stats["error_rate"] > self.performance_targets["api"]["error_rate"]:
            health_status = "degraded"
            issues.append("High error rate")
        
        if self.stats["average_response_time"] > self.performance_targets["api"]["response_time_p95"]:
            health_status = "degraded"
            issues.append("High response time")
        
        if self.system_metrics.get("cpu", {}).get("percent", 0) > 80:
            health_status = "degraded"
            issues.append("High CPU usage")
        
        if self.system_metrics.get("memory", {}).get("percent", 0) > 80:
            health_status = "degraded"
            issues.append("High memory usage")
        
        if self.active_alerts:
            health_status = "unhealthy"
            issues.append(f"{len(self.active_alerts)} active alerts")
        
        return {
            "status": health_status,
            "issues": issues,
            "uptime_seconds": self.stats["uptime_seconds"],
            "last_updated": datetime.utcnow().isoformat()
        }

def create_metrics_collector(config: Optional[MonitoringConfig] = None) -> MetricsCollector:
    """Factory function to create metrics collector instance"""
    return MetricsCollector(config)