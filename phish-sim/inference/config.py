# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Configuration for real-time inference pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Base paths
BASE_DIR = Path(__file__).parent
API_DIR = BASE_DIR / "api"
WEBSOCKET_DIR = BASE_DIR / "websocket"
REDIS_DIR = BASE_DIR / "redis"
QUEUE_DIR = BASE_DIR / "queue"
MONITORING_DIR = BASE_DIR / "monitoring"
ORCHESTRATION_DIR = BASE_DIR / "orchestration"

# Ensure directories exist
for directory in [API_DIR, WEBSOCKET_DIR, REDIS_DIR, QUEUE_DIR, MONITORING_DIR, ORCHESTRATION_DIR]:
    directory.mkdir(exist_ok=True)

class InferenceMode(Enum):
    """Inference processing modes"""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    NONE = "none"

@dataclass
class APIConfig:
    """FastAPI configuration"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8003
    workers: int = 4
    timeout: int = 60
    
    # API settings
    title: str = "Phish-Sim Real-time Inference API"
    description: str = "Real-time phishing detection inference pipeline"
    version: str = "0.1.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    # CORS settings
    allow_origins: List[str] = None  # Will be set to ["*"]
    allow_credentials: bool = True
    allow_methods: List[str] = None  # Will be set to ["*"]
    allow_headers: List[str] = None  # Will be set to ["*"]
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    rate_limit_burst: int = 20
    
    # Request settings
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30
    max_concurrent_requests: int = 100
    
    # Security
    enable_auth: bool = False
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

@dataclass
class RedisConfig:
    """Redis configuration"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    
    # Connection pool
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = None  # Will be set
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "256mb"
    eviction_policy: str = "allkeys-lru"
    
    # Session settings
    session_ttl: int = 1800  # 30 minutes
    session_prefix: str = "session:"
    
    # Queue settings
    queue_prefix: str = "queue:"
    result_ttl: int = 3600  # 1 hour
    max_queue_size: int = 10000
    
    # Pub/Sub settings
    pubsub_prefix: str = "pubsub:"
    channel_prefix: str = "channel:"

@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    
    # Connection settings
    host: str = "0.0.0.0"
    port: int = 8004
    path: str = "/ws"
    
    # Connection limits
    max_connections: int = 1000
    max_connections_per_ip: int = 10
    connection_timeout: int = 30
    
    # Message settings
    max_message_size: int = 1024 * 1024  # 1MB
    ping_interval: int = 20
    ping_timeout: int = 10
    
    # Broadcasting
    enable_broadcast: bool = True
    broadcast_rooms: List[str] = None  # Will be set to ["inference", "monitoring"]
    
    # Authentication
    require_auth: bool = False
    auth_token_param: str = "token"

@dataclass
class QueueConfig:
    """Message queue configuration"""
    
    # Queue types
    queue_type: str = "redis"  # redis, rabbitmq, kafka
    queue_name: str = "inference_queue"
    
    # Processing settings
    max_workers: int = 10
    worker_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    
    # Batch processing
    batch_size: int = 10
    batch_timeout: int = 5  # seconds
    enable_batching: bool = True
    
    # Priority queues
    enable_priority: bool = True
    priority_levels: int = 5
    high_priority_threshold: float = 0.8
    
    # Dead letter queue
    enable_dlq: bool = True
    dlq_name: str = "inference_dlq"
    dlq_ttl: int = 86400  # 24 hours

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Prometheus settings
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    
    # Health checks
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5
    enable_health_endpoints: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    enable_structured_logging: bool = True
    
    # Performance monitoring
    enable_profiling: bool = False
    profile_interval: int = 60  # seconds
    memory_threshold: float = 0.8  # 80%
    cpu_threshold: float = 0.8  # 80%
    
    # Alerting
    enable_alerts: bool = True
    alert_webhook: Optional[str] = None
    alert_thresholds: Dict[str, float] = None  # Will be set

@dataclass
class OrchestrationConfig:
    """Pipeline orchestration configuration"""
    
    # Inference modes
    default_mode: InferenceMode = InferenceMode.REALTIME
    enable_mode_switching: bool = True
    
    # Pipeline components
    enable_ml_pipeline: bool = True
    enable_visual_pipeline: bool = True
    enable_dom_pipeline: bool = True
    
    # Load balancing
    load_balancer: str = "round_robin"  # round_robin, weighted, least_connections
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    # Timeout settings
    pipeline_timeout: int = 30  # seconds
    component_timeout: int = 10  # seconds
    retry_timeout: int = 5  # seconds
    
    # Caching strategy
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_ttl: int = 300  # 5 minutes
    enable_prediction_cache: bool = True
    enable_feature_cache: bool = True

# Default configurations
DEFAULT_API_CONFIG = APIConfig()
DEFAULT_REDIS_CONFIG = RedisConfig()
DEFAULT_WEBSOCKET_CONFIG = WebSocketConfig()
DEFAULT_QUEUE_CONFIG = QueueConfig()
DEFAULT_MONITORING_CONFIG = MonitoringConfig()
DEFAULT_ORCHESTRATION_CONFIG = OrchestrationConfig()

# Initialize config lists
DEFAULT_API_CONFIG.allow_origins = ["*"]
DEFAULT_API_CONFIG.allow_methods = ["*"]
DEFAULT_API_CONFIG.allow_headers = ["*"]

DEFAULT_REDIS_CONFIG.socket_keepalive_options = {
    "TCP_KEEPIDLE": 1,
    "TCP_KEEPINTVL": 3,
    "TCP_KEEPCNT": 5
}

DEFAULT_WEBSOCKET_CONFIG.broadcast_rooms = ["inference", "monitoring", "alerts"]

DEFAULT_MONITORING_CONFIG.alert_thresholds = {
    "response_time": 5.0,  # seconds
    "error_rate": 0.05,    # 5%
    "cpu_usage": 0.8,      # 80%
    "memory_usage": 0.8,   # 80%
    "queue_size": 1000,    # requests
    "active_connections": 800
}

# Environment-based configuration
def get_config_from_env() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    return {
        "api": {
            "host": os.getenv("API_HOST", DEFAULT_API_CONFIG.host),
            "port": int(os.getenv("API_PORT", DEFAULT_API_CONFIG.port)),
            "workers": int(os.getenv("API_WORKERS", DEFAULT_API_CONFIG.workers)),
            "secret_key": os.getenv("SECRET_KEY", DEFAULT_API_CONFIG.secret_key)
        },
        "redis": {
            "host": os.getenv("REDIS_HOST", DEFAULT_REDIS_CONFIG.host),
            "port": int(os.getenv("REDIS_PORT", DEFAULT_REDIS_CONFIG.port)),
            "db": int(os.getenv("REDIS_DB", DEFAULT_REDIS_CONFIG.db)),
            "password": os.getenv("REDIS_PASSWORD", DEFAULT_REDIS_CONFIG.password)
        },
        "websocket": {
            "host": os.getenv("WS_HOST", DEFAULT_WEBSOCKET_CONFIG.host),
            "port": int(os.getenv("WS_PORT", DEFAULT_WEBSOCKET_CONFIG.port)),
            "max_connections": int(os.getenv("WS_MAX_CONNECTIONS", DEFAULT_WEBSOCKET_CONFIG.max_connections))
        },
        "monitoring": {
            "log_level": os.getenv("LOG_LEVEL", DEFAULT_MONITORING_CONFIG.log_level),
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "metrics_port": int(os.getenv("METRICS_PORT", DEFAULT_MONITORING_CONFIG.metrics_port))
        }
    }

# Performance targets
PERFORMANCE_TARGETS = {
    "api": {
        "response_time_p95": 2.0,  # seconds
        "response_time_p99": 5.0,  # seconds
        "throughput": 1000,        # requests/second
        "error_rate": 0.01,        # 1%
        "availability": 0.999      # 99.9%
    },
    "websocket": {
        "connection_time": 0.1,    # seconds
        "message_latency": 0.05,   # seconds
        "max_connections": 1000,
        "message_throughput": 10000 # messages/second
    },
    "redis": {
        "operation_time": 0.001,   # 1ms
        "cache_hit_rate": 0.95,    # 95%
        "memory_usage": 0.8,       # 80%
        "connection_pool": 0.9     # 90% utilization
    },
    "queue": {
        "processing_time": 1.0,    # seconds
        "queue_depth": 100,        # requests
        "worker_utilization": 0.8, # 80%
        "retry_rate": 0.05         # 5%
    },
    "pipeline": {
        "end_to_end_latency": 3.0, # seconds
        "component_latency": 0.5,  # seconds
        "throughput": 500,         # requests/second
        "accuracy": 0.95           # 95%
    }
}

# Feature flags
FEATURE_FLAGS = {
    "enable_ml_pipeline": True,
    "enable_visual_pipeline": True,
    "enable_dom_pipeline": True,
    "enable_caching": True,
    "enable_websocket": True,
    "enable_monitoring": True,
    "enable_auto_scaling": True,
    "enable_circuit_breaker": True,
    "enable_rate_limiting": True,
    "enable_auth": False
}

def get_config(config_type: str = "api") -> Any:
    """Get configuration by type"""
    configs = {
        "api": DEFAULT_API_CONFIG,
        "redis": DEFAULT_REDIS_CONFIG,
        "websocket": DEFAULT_WEBSOCKET_CONFIG,
        "queue": DEFAULT_QUEUE_CONFIG,
        "monitoring": DEFAULT_MONITORING_CONFIG,
        "orchestration": DEFAULT_ORCHESTRATION_CONFIG
    }
    
    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return configs[config_type]

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config_to_file(config: Any, config_path: str):
    """Save configuration to YAML file"""
    import yaml
    
    # Convert dataclass to dict
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def validate_config(config: Any) -> bool:
    """Validate configuration values"""
    try:
        # Basic validation
        if hasattr(config, 'port') and (config.port < 1 or config.port > 65535):
            return False
        
        if hasattr(config, 'timeout') and config.timeout < 0:
            return False
        
        if hasattr(config, 'max_connections') and config.max_connections < 1:
            return False
        
        return True
    except Exception:
        return False