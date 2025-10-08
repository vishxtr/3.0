# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Load Balancing and Auto-Scaling System
Intelligent load balancing with auto-scaling capabilities
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import random
import threading
from datetime import datetime, timedelta

try:
    import aiohttp
    import redis.asyncio as redis
    from redis.asyncio import Redis
except ImportError:
    aiohttp = None
    redis = None
    Redis = None

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"

class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    RESPONSE_TIME_BASED = "response_time_based"
    CUSTOM = "custom"

@dataclass
class ServerInstance:
    """Server instance information"""
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    response_time_avg: float = 0.0
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    is_healthy: bool = True
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_response_time_ms: float = 1000.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period_seconds: int = 300
    scaling_policy: ScalingPolicy = ScalingPolicy.CPU_BASED

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: int = 30
    health_check_timeout: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_sticky_sessions: bool = False
    session_timeout: int = 3600

class LoadBalancer:
    """Intelligent Load Balancer with Auto-Scaling"""
    
    def __init__(self, config: LoadBalancerConfig = None, 
                 scaling_config: ScalingConfig = None,
                 redis_url: str = "redis://localhost:6379"):
        self.config = config or LoadBalancerConfig()
        self.scaling_config = scaling_config or ScalingConfig()
        self.redis_url = redis_url
        
        # Server instances
        self.servers: Dict[str, ServerInstance] = {}
        self.server_list: List[str] = []
        self.current_server_index = 0
        
        # Load balancing state
        self.request_count = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        # Auto-scaling state
        self.last_scale_time = datetime.utcnow()
        self.scaling_in_progress = False
        self.scaling_history: deque = deque(maxlen=100)
        
        # Redis connection for shared state
        self.redis_client: Optional[Redis] = None
        
        # Health check and monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Custom scaling callbacks
        self.scaling_callbacks: List[Callable] = []
        
        # Session management for sticky sessions
        self.sessions: Dict[str, str] = {}  # session_id -> server_id
        
        logger.info("Load balancer initialized")
    
    async def initialize(self):
        """Initialize load balancer"""
        try:
            # Initialize Redis connection
            if redis:
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("Redis connection established for load balancer")
            
            # Start monitoring tasks
            self.is_running = True
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Load balancer initialized successfully")
            
        except Exception as e:
            logger.error(f"Load balancer initialization failed: {e}")
    
    async def shutdown(self):
        """Shutdown load balancer"""
        try:
            self.is_running = False
            
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Load balancer shutdown completed")
            
        except Exception as e:
            logger.error(f"Load balancer shutdown error: {e}")
    
    def add_server(self, server_id: str, host: str, port: int, 
                   weight: int = 1, max_connections: int = 100,
                   metadata: Dict[str, Any] = None) -> bool:
        """Add server instance to load balancer"""
        try:
            server = ServerInstance(
                id=server_id,
                host=host,
                port=port,
                weight=weight,
                max_connections=max_connections,
                metadata=metadata or {}
            )
            
            self.servers[server_id] = server
            self.server_list.append(server_id)
            
            logger.info(f"Added server {server_id} ({host}:{port}) to load balancer")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add server {server_id}: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """Remove server instance from load balancer"""
        try:
            if server_id in self.servers:
                del self.servers[server_id]
                if server_id in self.server_list:
                    self.server_list.remove(server_id)
                
                # Remove from sessions if using sticky sessions
                if self.config.enable_sticky_sessions:
                    sessions_to_remove = [sid for sid, srv_id in self.sessions.items() 
                                        if srv_id == server_id]
                    for session_id in sessions_to_remove:
                        del self.sessions[session_id]
                
                logger.info(f"Removed server {server_id} from load balancer")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove server {server_id}: {e}")
            return False
    
    async def get_server(self, request_info: Dict[str, Any] = None) -> Optional[ServerInstance]:
        """Get server instance based on load balancing strategy"""
        try:
            if not self.servers:
                logger.warning("No servers available")
                return None
            
            # Filter healthy servers
            healthy_servers = [sid for sid in self.server_list 
                             if self.servers[sid].is_healthy and 
                             self.servers[sid].current_connections < self.servers[sid].max_connections]
            
            if not healthy_servers:
                logger.warning("No healthy servers available")
                return None
            
            # Apply load balancing strategy
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                server_id = self._round_robin_selection(healthy_servers)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                server_id = self._least_connections_selection(healthy_servers)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                server_id = self._least_response_time_selection(healthy_servers)
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                server_id = self._weighted_round_robin_selection(healthy_servers)
            elif self.config.strategy == LoadBalancingStrategy.IP_HASH:
                server_id = self._ip_hash_selection(healthy_servers, request_info)
            elif self.config.strategy == LoadBalancingStrategy.RANDOM:
                server_id = self._random_selection(healthy_servers)
            else:
                server_id = self._round_robin_selection(healthy_servers)
            
            # Handle sticky sessions
            if self.config.enable_sticky_sessions and request_info:
                session_id = request_info.get('session_id')
                if session_id and session_id in self.sessions:
                    sticky_server_id = self.sessions[session_id]
                    if sticky_server_id in healthy_servers:
                        server_id = sticky_server_id
                    else:
                        # Update session mapping
                        self.sessions[session_id] = server_id
                elif session_id:
                    self.sessions[session_id] = server_id
            
            server = self.servers[server_id]
            server.current_connections += 1
            self.request_count += 1
            self.total_requests += 1
            
            return server
            
        except Exception as e:
            logger.error(f"Failed to get server: {e}")
            return None
    
    def _round_robin_selection(self, healthy_servers: List[str]) -> str:
        """Round robin server selection"""
        if not healthy_servers:
            return None
        
        server_id = healthy_servers[self.current_server_index % len(healthy_servers)]
        self.current_server_index += 1
        return server_id
    
    def _least_connections_selection(self, healthy_servers: List[str]) -> str:
        """Least connections server selection"""
        if not healthy_servers:
            return None
        
        min_connections = min(self.servers[sid].current_connections for sid in healthy_servers)
        candidates = [sid for sid in healthy_servers 
                     if self.servers[sid].current_connections == min_connections]
        
        return random.choice(candidates)
    
    def _least_response_time_selection(self, healthy_servers: List[str]) -> str:
        """Least response time server selection"""
        if not healthy_servers:
            return None
        
        min_response_time = min(self.servers[sid].response_time_avg for sid in healthy_servers)
        candidates = [sid for sid in healthy_servers 
                     if self.servers[sid].response_time_avg == min_response_time]
        
        return random.choice(candidates)
    
    def _weighted_round_robin_selection(self, healthy_servers: List[str]) -> str:
        """Weighted round robin server selection"""
        if not healthy_servers:
            return None
        
        # Calculate total weight
        total_weight = sum(self.servers[sid].weight for sid in healthy_servers)
        
        # Weighted selection
        random_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for server_id in healthy_servers:
            current_weight += self.servers[server_id].weight
            if random_weight <= current_weight:
                return server_id
        
        return healthy_servers[0]
    
    def _ip_hash_selection(self, healthy_servers: List[str], request_info: Dict[str, Any]) -> str:
        """IP hash server selection"""
        if not healthy_servers or not request_info:
            return self._round_robin_selection(healthy_servers)
        
        client_ip = request_info.get('client_ip', '')
        hash_value = hash(client_ip) % len(healthy_servers)
        return healthy_servers[hash_value]
    
    def _random_selection(self, healthy_servers: List[str]) -> str:
        """Random server selection"""
        if not healthy_servers:
            return None
        
        return random.choice(healthy_servers)
    
    async def release_server(self, server_id: str, success: bool = True, 
                           response_time: float = 0.0):
        """Release server connection and update metrics"""
        try:
            if server_id in self.servers:
                server = self.servers[server_id]
                server.current_connections = max(0, server.current_connections - 1)
                
                # Update response time
                if response_time > 0:
                    server.response_time_history.append(response_time)
                    if server.response_time_history:
                        server.response_time_avg = statistics.mean(server.response_time_history)
                
                # Update success/error counts
                if success:
                    server.success_count += 1
                else:
                    server.error_count += 1
                    self.failed_requests += 1
                
                # Update Redis state
                if self.redis_client:
                    await self._update_server_state_redis(server)
            
        except Exception as e:
            logger.error(f"Failed to release server {server_id}: {e}")
    
    async def _update_server_state_redis(self, server: ServerInstance):
        """Update server state in Redis"""
        try:
            if self.redis_client:
                server_data = {
                    'current_connections': server.current_connections,
                    'response_time_avg': server.response_time_avg,
                    'error_count': server.error_count,
                    'success_count': server.success_count,
                    'is_healthy': server.is_healthy,
                    'cpu_usage': server.cpu_usage,
                    'memory_usage': server.memory_usage,
                    'last_health_check': server.last_health_check.isoformat()
                }
                
                await self.redis_client.hset(
                    f"server:{server.id}", 
                    mapping=server_data
                )
                
        except Exception as e:
            logger.error(f"Failed to update server state in Redis: {e}")
    
    async def _health_check_loop(self):
        """Health check loop for all servers"""
        while self.is_running:
            try:
                for server_id in list(self.servers.keys()):
                    await self._health_check_server(server_id)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _health_check_server(self, server_id: str):
        """Perform health check on a server"""
        try:
            server = self.servers[server_id]
            health_url = f"http://{server.host}:{server.port}/health"
            
            if aiohttp:
                timeout = aiohttp.ClientTimeout(total=self.config.health_check_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(health_url) as response:
                        server.is_healthy = response.status == 200
                        server.last_health_check = datetime.utcnow()
            else:
                # Fallback health check
                server.is_healthy = True
                server.last_health_check = datetime.utcnow()
            
            # Update Redis state
            if self.redis_client:
                await self._update_server_state_redis(server)
            
        except Exception as e:
            logger.error(f"Health check failed for server {server_id}: {e}")
            self.servers[server_id].is_healthy = False
            self.servers[server_id].last_health_check = datetime.utcnow()
    
    async def _monitoring_loop(self):
        """Monitoring loop for auto-scaling decisions"""
        while self.is_running:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling is needed"""
        try:
            if self.scaling_in_progress:
                return
            
            # Check cooldown period
            if datetime.utcnow() - self.last_scale_time < timedelta(seconds=self.scaling_config.cooldown_period_seconds):
                return
            
            # Get current metrics
            metrics = await self._get_scaling_metrics()
            
            # Determine scaling action
            action = self._determine_scaling_action(metrics)
            
            if action == "scale_up":
                await self._scale_up()
            elif action == "scale_down":
                await self._scale_down()
            
        except Exception as e:
            logger.error(f"Scaling evaluation error: {e}")
    
    async def _get_scaling_metrics(self) -> Dict[str, float]:
        """Get metrics for scaling decisions"""
        try:
            metrics = {
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0,
                'avg_response_time': 0.0,
                'total_requests': self.total_requests,
                'error_rate': 0.0,
                'active_servers': len([s for s in self.servers.values() if s.is_healthy])
            }
            
            if self.servers:
                # Calculate averages
                cpu_values = [s.cpu_usage for s in self.servers.values() if s.is_healthy]
                memory_values = [s.memory_usage for s in self.servers.values() if s.is_healthy]
                response_times = [s.response_time_avg for s in self.servers.values() if s.is_healthy and s.response_time_avg > 0]
                
                if cpu_values:
                    metrics['avg_cpu_usage'] = statistics.mean(cpu_values)
                if memory_values:
                    metrics['avg_memory_usage'] = statistics.mean(memory_values)
                if response_times:
                    metrics['avg_response_time'] = statistics.mean(response_times)
                
                # Calculate error rate
                total_requests = sum(s.success_count + s.error_count for s in self.servers.values())
                total_errors = sum(s.error_count for s in self.servers.values())
                if total_requests > 0:
                    metrics['error_rate'] = total_errors / total_requests
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get scaling metrics: {e}")
            return {}
    
    def _determine_scaling_action(self, metrics: Dict[str, float]) -> str:
        """Determine scaling action based on metrics"""
        try:
            current_instances = metrics.get('active_servers', 0)
            
            # Scale up conditions
            if current_instances < self.scaling_config.max_instances:
                if self.scaling_config.scaling_policy == ScalingPolicy.CPU_BASED:
                    if metrics.get('avg_cpu_usage', 0) > self.scaling_config.target_cpu_percent * self.scaling_config.scale_up_threshold:
                        return "scale_up"
                elif self.scaling_config.scaling_policy == ScalingPolicy.MEMORY_BASED:
                    if metrics.get('avg_memory_usage', 0) > self.scaling_config.target_memory_percent * self.scaling_config.scale_up_threshold:
                        return "scale_up"
                elif self.scaling_config.scaling_policy == ScalingPolicy.RESPONSE_TIME_BASED:
                    if metrics.get('avg_response_time', 0) > self.scaling_config.target_response_time_ms * self.scaling_config.scale_up_threshold:
                        return "scale_up"
            
            # Scale down conditions
            if current_instances > self.scaling_config.min_instances:
                if self.scaling_config.scaling_policy == ScalingPolicy.CPU_BASED:
                    if metrics.get('avg_cpu_usage', 0) < self.scaling_config.target_cpu_percent * self.scaling_config.scale_down_threshold:
                        return "scale_down"
                elif self.scaling_config.scaling_policy == ScalingPolicy.MEMORY_BASED:
                    if metrics.get('avg_memory_usage', 0) < self.scaling_config.target_memory_percent * self.scaling_config.scale_down_threshold:
                        return "scale_down"
                elif self.scaling_config.scaling_policy == ScalingPolicy.RESPONSE_TIME_BASED:
                    if metrics.get('avg_response_time', 0) < self.scaling_config.target_response_time_ms * self.scaling_config.scale_down_threshold:
                        return "scale_down"
            
            return "no_action"
            
        except Exception as e:
            logger.error(f"Failed to determine scaling action: {e}")
            return "no_action"
    
    async def _scale_up(self):
        """Scale up by adding new server instances"""
        try:
            self.scaling_in_progress = True
            
            # This would integrate with your infrastructure provider
            # For now, we'll simulate adding a new server
            new_server_id = f"server_{int(time.time())}"
            new_host = "localhost"  # This would be the actual new server host
            new_port = 8001 + len(self.servers)  # This would be the actual new server port
            
            success = self.add_server(new_server_id, new_host, new_port)
            
            if success:
                self.scaling_history.append({
                    'action': 'scale_up',
                    'timestamp': datetime.utcnow(),
                    'new_server_id': new_server_id,
                    'total_servers': len(self.servers)
                })
                
                logger.info(f"Scaled up: Added server {new_server_id}")
                
                # Execute scaling callbacks
                for callback in self.scaling_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback('scale_up', new_server_id)
                        else:
                            callback('scale_up', new_server_id)
                    except Exception as e:
                        logger.error(f"Scaling callback error: {e}")
            
            self.last_scale_time = datetime.utcnow()
            self.scaling_in_progress = False
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            self.scaling_in_progress = False
    
    async def _scale_down(self):
        """Scale down by removing server instances"""
        try:
            self.scaling_in_progress = True
            
            # Find least loaded server to remove
            healthy_servers = [sid for sid in self.server_list 
                             if self.servers[sid].is_healthy]
            
            if len(healthy_servers) > self.scaling_config.min_instances:
                # Remove server with least connections
                least_loaded = min(healthy_servers, 
                                 key=lambda sid: self.servers[sid].current_connections)
                
                success = self.remove_server(least_loaded)
                
                if success:
                    self.scaling_history.append({
                        'action': 'scale_down',
                        'timestamp': datetime.utcnow(),
                        'removed_server_id': least_loaded,
                        'total_servers': len(self.servers)
                    })
                    
                    logger.info(f"Scaled down: Removed server {least_loaded}")
                    
                    # Execute scaling callbacks
                    for callback in self.scaling_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback('scale_down', least_loaded)
                            else:
                                callback('scale_down', least_loaded)
                        except Exception as e:
                            logger.error(f"Scaling callback error: {e}")
            
            self.last_scale_time = datetime.utcnow()
            self.scaling_in_progress = False
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            self.scaling_in_progress = False
    
    def add_scaling_callback(self, callback: Callable):
        """Add custom scaling callback"""
        self.scaling_callbacks.append(callback)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        try:
            healthy_servers = [s for s in self.servers.values() if s.is_healthy]
            
            return {
                'total_servers': len(self.servers),
                'healthy_servers': len(healthy_servers),
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'success_rate': (self.total_requests - self.failed_requests) / self.total_requests * 100 if self.total_requests > 0 else 0,
                'load_balancing_strategy': self.config.strategy.value,
                'scaling_policy': self.scaling_config.scaling_policy.value,
                'scaling_in_progress': self.scaling_in_progress,
                'server_details': {
                    sid: {
                        'host': server.host,
                        'port': server.port,
                        'is_healthy': server.is_healthy,
                        'current_connections': server.current_connections,
                        'max_connections': server.max_connections,
                        'response_time_avg': server.response_time_avg,
                        'error_count': server.error_count,
                        'success_count': server.success_count,
                        'cpu_usage': server.cpu_usage,
                        'memory_usage': server.memory_usage
                    }
                    for sid, server in self.servers.items()
                },
                'scaling_history': list(self.scaling_history),
                'config': {
                    'strategy': self.config.strategy.value,
                    'health_check_interval': self.config.health_check_interval,
                    'enable_sticky_sessions': self.config.enable_sticky_sessions
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get load balancer stats: {e}")
            return {'error': str(e)}

# Global load balancer instance
load_balancer: Optional[LoadBalancer] = None

def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance"""
    global load_balancer
    if load_balancer is None:
        config = LoadBalancerConfig()
        scaling_config = ScalingConfig()
        load_balancer = LoadBalancer(config, scaling_config)
    return load_balancer