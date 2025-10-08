# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple demo for T007 - Backend API Integration & Testing
Demonstrates the unified API structure without external dependencies
"""

import json
import time
from typing import Dict, Any, List
from pathlib import Path

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(title: str, result: Any):
    """Print a formatted result"""
    print(f"\n{title}:")
    print("-" * 40)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)

def demo_api_structure():
    """Demo the unified API structure"""
    print_section("Unified API Structure")
    
    # API endpoints
    endpoints = {
        "GET /": "Root endpoint with API information",
        "GET /health": "Comprehensive health check for all services",
        "GET /model/info": "Model information (NLP and Visual)",
        "GET /status": "System status for all components",
        "GET /stats": "Dashboard statistics and metrics",
        "POST /analyze": "Unified content analysis endpoint",
        "POST /analyze/batch": "Batch analysis for multiple requests",
        "WebSocket /ws/{user_id}": "Real-time communication endpoint"
    }
    
    print("Available API Endpoints:")
    for endpoint, description in endpoints.items():
        print(f"  {endpoint:<25} - {description}")
    
    # Request/Response models
    print("\nRequest/Response Models:")
    models = {
        "AnalysisRequest": "Unified analysis request with all options",
        "AnalysisResponse": "Comprehensive analysis result",
        "BatchAnalysisRequest": "Batch processing request",
        "HealthResponse": "Service health information",
        "ModelInfoResponse": "Model details and configuration",
        "SystemStatus": "Component status overview",
        "DashboardStats": "Performance and usage statistics",
        "WebSocketMessage": "Real-time message format"
    }
    
    for model, description in models.items():
        print(f"  {model:<20} - {description}")

def demo_analysis_capabilities():
    """Demo analysis capabilities"""
    print_section("Analysis Capabilities")
    
    # Content types supported
    content_types = {
        "text": "Plain text content analysis",
        "url": "URL analysis with visual inspection",
        "email": "Email content and structure analysis"
    }
    
    print("Supported Content Types:")
    for content_type, description in content_types.items():
        print(f"  {content_type:<10} - {description}")
    
    # Analysis components
    components = {
        "NLP Analysis": {
            "description": "Natural Language Processing for text analysis",
            "features": ["Text classification", "Feature extraction", "Confidence scoring"],
            "models": ["PhishingClassifier", "TextPreprocessor", "Tokenizer"]
        },
        "Visual Analysis": {
            "description": "Visual inspection of web pages and images",
            "features": ["Screenshot capture", "DOM analysis", "CNN classification", "Template matching"],
            "models": ["VisualClassifier", "FeatureExtractor", "TemplateMatcher"]
        },
        "Real-time Pipeline": {
            "description": "Real-time processing with caching and queuing",
            "features": ["Redis caching", "Queue management", "WebSocket communication", "Metrics collection"],
            "models": ["PipelineOrchestrator", "RedisManager", "QueueManager", "MetricsCollector"]
        }
    }
    
    print("\nAnalysis Components:")
    for component, details in components.items():
        print(f"\n  {component}:")
        print(f"    Description: {details['description']}")
        print(f"    Features: {', '.join(details['features'])}")
        print(f"    Models/Components: {', '.join(details['models'])}")

def demo_integration_features():
    """Demo integration features"""
    print_section("Integration Features")
    
    # Service integration
    services = {
        "NLP Service": {
            "status": "Integrated",
            "endpoints": ["/analyze", "/analyze/batch"],
            "capabilities": ["Text analysis", "Feature extraction", "Confidence scoring"]
        },
        "Visual Service": {
            "status": "Integrated",
            "endpoints": ["/analyze (for URLs)", "/analyze/batch"],
            "capabilities": ["Screenshot capture", "DOM analysis", "Visual classification"]
        },
        "Real-time Service": {
            "status": "Integrated",
            "endpoints": ["WebSocket /ws/{user_id}", "/analyze", "/analyze/batch"],
            "capabilities": ["Real-time processing", "Caching", "Queue management"]
        },
        "Redis Cache": {
            "status": "Integrated",
            "endpoints": ["All analysis endpoints"],
            "capabilities": ["Result caching", "Session management", "Performance optimization"]
        }
    }
    
    print("Service Integration Status:")
    for service, details in services.items():
        print(f"\n  {service}:")
        print(f"    Status: {details['status']}")
        print(f"    Endpoints: {', '.join(details['endpoints'])}")
        print(f"    Capabilities: {', '.join(details['capabilities'])}")
    
    # Advanced features
    features = {
        "Caching": "Intelligent result caching with TTL and invalidation",
        "Rate Limiting": "Request rate limiting to prevent abuse",
        "Batch Processing": "Concurrent batch analysis with configurable limits",
        "Real-time Updates": "WebSocket-based real-time result streaming",
        "Error Handling": "Comprehensive error handling and recovery",
        "Metrics Collection": "Performance monitoring and analytics",
        "Health Monitoring": "Service health checks and status reporting",
        "Security": "Input validation, sanitization, and secure processing"
    }
    
    print("\nAdvanced Features:")
    for feature, description in features.items():
        print(f"  {feature:<20} - {description}")

def demo_sample_requests():
    """Demo sample API requests"""
    print_section("Sample API Requests")
    
    # Sample analysis request
    analysis_request = {
        "content": "URGENT: Your account will be suspended in 24 hours! Click here to verify: http://fake-bank.com/verify",
        "content_type": "email",
        "user_id": "user_123",
        "session_id": "session_456",
        "force_reanalyze": False,
        "enable_nlp": True,
        "enable_visual": False,
        "enable_realtime": True,
        "return_features": True,
        "return_explanation": True
    }
    
    print("Sample Analysis Request:")
    print_result("POST /analyze", analysis_request)
    
    # Sample analysis response
    analysis_response = {
        "request_id": "req_abc123_1704067200",
        "content": "URGENT: Your account will be suspended in 24 hours! Click here to verify: http://fake-bank.com/verify",
        "content_type": "email",
        "prediction": "phish",
        "confidence": 0.92,
        "risk_score": 0.92,
        "risk_level": "HIGH",
        "explanation": {
            "overall_assessment": "Content classified as phish with 92% confidence",
            "risk_level": "HIGH",
            "components_analyzed": ["nlp", "realtime"],
            "details": {
                "nlp": {
                    "prediction": "phish",
                    "confidence": 0.88,
                    "features": {
                        "suspicious_words": ["urgent", "suspended", "verify"],
                        "urgency_score": 0.95,
                        "threat_indicators": 3
                    }
                },
                "realtime": {
                    "prediction": "phish",
                    "confidence": 0.96,
                    "risk_score": 0.96
                }
            }
        },
        "processing_time_ms": 245.5,
        "cached": False,
        "timestamp": "2024-01-01T12:00:00Z",
        "components": {
            "nlp": {
                "prediction": "phish",
                "confidence": 0.88,
                "features": {
                    "suspicious_words": ["urgent", "suspended", "verify"],
                    "urgency_score": 0.95,
                    "threat_indicators": 3
                }
            },
            "realtime": {
                "prediction": "phish",
                "confidence": 0.96,
                "risk_score": 0.96
            }
        },
        "features": {
            "text_length": 89,
            "suspicious_words": 3,
            "url_count": 1,
            "urgency_score": 0.95,
            "threat_indicators": 3
        }
    }
    
    print("Sample Analysis Response:")
    print_result("Analysis Result", analysis_response)
    
    # Sample batch request
    batch_request = {
        "requests": [
            {
                "content": "This is a legitimate business email.",
                "content_type": "text",
                "enable_nlp": True
            },
            {
                "content": "URGENT: Verify your account now!",
                "content_type": "text",
                "enable_nlp": True
            },
            {
                "content": "https://www.microsoft.com",
                "content_type": "url",
                "enable_nlp": True,
                "enable_visual": True
            }
        ],
        "batch_id": "batch_789",
        "max_concurrent": 3
    }
    
    print("Sample Batch Request:")
    print_result("POST /analyze/batch", batch_request)

def demo_health_monitoring():
    """Demo health monitoring capabilities"""
    print_section("Health Monitoring")
    
    # Health check response
    health_response = {
        "status": "ok",
        "service": "phish-sim-unified-api",
        "version": "1.0.0",
        "uptime_seconds": 3600.0,
        "components": {
            "nlp_service": {
                "status": "healthy",
                "details": "NLP inference service"
            },
            "visual_service": {
                "status": "healthy",
                "details": "Visual analysis service"
            },
            "redis": {
                "status": "healthy",
                "details": "Redis cache and session management"
            },
            "queue": {
                "status": "healthy",
                "details": "Request queue management"
            },
            "metrics": {
                "status": "healthy",
                "details": "Metrics collection and monitoring"
            },
            "pipeline": {
                "status": "healthy",
                "details": "Pipeline orchestration"
            }
        },
        "performance": {
            "avg_response_time_ms": 150.0,
            "requests_per_second": 10.5,
            "error_rate": 0.02,
            "cache_hit_rate": 0.75
        }
    }
    
    print("Health Check Response:")
    print_result("GET /health", health_response)
    
    # System status
    system_status = {
        "backend_api": "healthy",
        "ml_pipeline": "healthy",
        "database": "healthy",
        "redis": "healthy",
        "websocket": "healthy"
    }
    
    print("System Status:")
    print_result("GET /status", system_status)
    
    # Dashboard stats
    dashboard_stats = {
        "total_scans": 1250,
        "threats_detected": 89,
        "avg_response_time_ms": 145.5,
        "cache_hit_rate": 75.2,
        "cache_hits": 940,
        "cache_misses": 310
    }
    
    print("Dashboard Statistics:")
    print_result("GET /stats", dashboard_stats)

def demo_websocket_communication():
    """Demo WebSocket communication"""
    print_section("WebSocket Communication")
    
    # WebSocket message types
    message_types = {
        "analysis": "Request real-time analysis",
        "analysis_complete": "Analysis result notification",
        "analysis_error": "Analysis error notification",
        "ping": "Connection health check",
        "pong": "Connection health response",
        "error": "General error notification"
    }
    
    print("WebSocket Message Types:")
    for message_type, description in message_types.items():
        print(f"  {message_type:<20} - {description}")
    
    # Sample WebSocket messages
    sample_messages = {
        "Analysis Request": {
            "type": "analysis",
            "payload": {
                "content": "Test message for real-time analysis",
                "content_type": "text",
                "enable_nlp": True,
                "return_explanation": True
            }
        },
        "Analysis Result": {
            "type": "analysis_complete",
            "payload": {
                "request_id": "req_ws123",
                "prediction": "benign",
                "confidence": 0.85,
                "risk_level": "LOW",
                "processing_time_ms": 120.0
            }
        },
        "Ping Message": {
            "type": "ping"
        },
        "Pong Response": {
            "type": "pong"
        }
    }
    
    print("\nSample WebSocket Messages:")
    for message_name, message in sample_messages.items():
        print(f"\n{message_name}:")
        print_result("WebSocket Message", message)

def demo_testing_strategy():
    """Demo testing strategy"""
    print_section("Testing Strategy")
    
    # Test categories
    test_categories = {
        "Unit Tests": {
            "description": "Test individual components and functions",
            "coverage": ["API models", "Validation logic", "Helper functions", "Service integration"],
            "tools": ["pytest", "unittest", "mock"]
        },
        "Integration Tests": {
            "description": "Test API endpoints and service interactions",
            "coverage": ["HTTP endpoints", "Database operations", "External service calls", "Error handling"],
            "tools": ["FastAPI TestClient", "httpx", "pytest-asyncio"]
        },
        "End-to-End Tests": {
            "description": "Test complete user workflows",
            "coverage": ["Full analysis pipeline", "Batch processing", "WebSocket communication", "Error scenarios"],
            "tools": ["Playwright", "Selenium", "WebSocket testing"]
        },
        "Performance Tests": {
            "description": "Test system performance and scalability",
            "coverage": ["Response times", "Throughput", "Memory usage", "Concurrent requests"],
            "tools": ["locust", "pytest-benchmark", "memory profiler"]
        }
    }
    
    print("Test Categories:")
    for category, details in test_categories.items():
        print(f"\n  {category}:")
        print(f"    Description: {details['description']}")
        print(f"    Coverage: {', '.join(details['coverage'])}")
        print(f"    Tools: {', '.join(details['tools'])}")
    
    # Test scenarios
    test_scenarios = [
        "Basic content analysis (text, URL, email)",
        "Batch analysis with multiple requests",
        "Real-time WebSocket communication",
        "Error handling and recovery",
        "Rate limiting and security",
        "Caching and performance optimization",
        "Service health monitoring",
        "Model integration and fallback",
        "Concurrent request handling",
        "Memory and resource management"
    ]
    
    print("\nTest Scenarios:")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"  {i:2d}. {scenario}")

def main():
    """Main demo function"""
    print("Phish-Sim Unified API - T007 Demo")
    print("Backend API Integration & Testing")
    print("=" * 60)
    
    print("\nThis demo showcases the unified backend API that integrates:")
    print("- NLP Analysis Service (T003)")
    print("- Visual Analysis Service (T004)")
    print("- Real-time Inference Pipeline (T005)")
    print("- Frontend Integration (T006)")
    
    # Run all demo sections
    demo_api_structure()
    demo_analysis_capabilities()
    demo_integration_features()
    demo_sample_requests()
    demo_health_monitoring()
    demo_websocket_communication()
    demo_testing_strategy()
    
    print_section("T007 Summary")
    print("✅ Unified Backend API Integration Complete")
    print("\nKey Achievements:")
    print("- Integrated all backend services (NLP, Visual, Real-time)")
    print("- Created unified API endpoints with comprehensive functionality")
    print("- Implemented robust error handling and validation")
    print("- Added comprehensive testing framework")
    print("- Created health monitoring and metrics collection")
    print("- Implemented WebSocket support for real-time communication")
    print("- Added caching and performance optimization")
    print("- Created comprehensive documentation and demos")
    
    print("\nNext Steps:")
    print("- T008: End-to-End System Integration")
    print("- T009: Performance Optimization & Monitoring")
    print("- T010: Security Hardening & Compliance")
    
    print("\nThe unified API is now ready for production deployment!")

if __name__ == "__main__":
    main()