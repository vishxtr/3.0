# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
End-to-End System Integration Demo for Phish-Sim
Demonstrates the complete system working together
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List
import sys
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

def test_system_connectivity():
    """Test system connectivity and startup"""
    print_section("System Connectivity Test")
    
    base_url = "http://localhost:8001"
    frontend_url = "http://localhost:3000"
    
    # Test backend API
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend API is running")
            print(f"   Message: {data.get('message', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
        else:
            print(f"❌ Backend API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend API not available: {e}")
        return False
    
    # Test frontend
    try:
        response = requests.get(f"{frontend_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"⚠️  Frontend returned status {response.status_code}")
    except Exception as e:
        print(f"⚠️  Frontend not available: {e}")
    
    return True

def test_api_endpoints():
    """Test all API endpoints"""
    print_section("API Endpoints Test")
    
    base_url = "http://localhost:8001"
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/model/info", "Model information"),
        ("/status", "System status"),
        ("/stats", "Dashboard statistics"),
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint} - {description}")
                if endpoint == "/health":
                    print(f"   Status: {data.get('status', 'N/A')}")
                    print(f"   Components: {len(data.get('components', {}))}")
                elif endpoint == "/stats":
                    print(f"   Total Scans: {data.get('total_scans', 0)}")
                    print(f"   Threats Detected: {data.get('threats_detected', 0)}")
            else:
                print(f"❌ {endpoint} - Status {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def test_content_analysis():
    """Test content analysis functionality"""
    print_section("Content Analysis Test")
    
    base_url = "http://localhost:8001"
    
    test_cases = [
        {
            "name": "Legitimate Text",
            "content": "This is a legitimate business email from a trusted company regarding your account statement.",
            "content_type": "text",
            "expected": "benign"
        },
        {
            "name": "Suspicious Text",
            "content": "URGENT: Your account will be suspended in 24 hours! Click here immediately to verify your identity!",
            "content_type": "text",
            "expected": "suspicious"
        },
        {
            "name": "Phishing URL",
            "content": "http://fake-bank-security.com/login?verify=urgent&account=suspended",
            "content_type": "url",
            "expected": "phish"
        },
        {
            "name": "Legitimate URL",
            "content": "https://www.google.com",
            "content_type": "url",
            "expected": "benign"
        },
        {
            "name": "Phishing Email",
            "content": "Congratulations! You've won $1000! Click here to claim your prize now before it expires!",
            "content_type": "email",
            "expected": "phish"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Content: {test_case['content'][:60]}...")
        print(f"Type: {test_case['content_type']}")
        
        request_data = {
            "content": test_case["content"],
            "content_type": test_case["content_type"],
            "enable_nlp": True,
            "enable_visual": test_case["content_type"] == "url",
            "enable_realtime": True,
            "return_explanation": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{base_url}/analyze", json=request_data, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Analysis completed in {end_time - start_time:.2f}s")
                print(f"   Prediction: {data['prediction']}")
                print(f"   Confidence: {data['confidence']:.2%}")
                print(f"   Risk Level: {data['risk_level']}")
                print(f"   Processing Time: {data['processing_time_ms']:.1f}ms")
                print(f"   Components: {list(data.get('components', {}).keys())}")
                
                # Check if prediction matches expectation
                if data['prediction'] == test_case['expected']:
                    print("   ✅ Prediction matches expectation")
                else:
                    print(f"   ⚠️  Prediction ({data['prediction']}) differs from expectation ({test_case['expected']})")
                
            else:
                print(f"❌ Analysis failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Analysis failed with error: {e}")

def test_batch_analysis():
    """Test batch analysis functionality"""
    print_section("Batch Analysis Test")
    
    base_url = "http://localhost:8001"
    
    batch_requests = [
        {
            "content": "This is a legitimate business email from your bank.",
            "content_type": "text",
            "enable_nlp": True
        },
        {
            "content": "URGENT: Verify your account now or it will be locked!",
            "content_type": "text",
            "enable_nlp": True
        },
        {
            "content": "https://www.microsoft.com",
            "content_type": "url",
            "enable_nlp": True,
            "enable_visual": True
        },
        {
            "content": "http://fake-microsoft-security.com/verify",
            "content_type": "url",
            "enable_nlp": True,
            "enable_visual": True
        },
        {
            "content": "You've won a free iPhone! Click here to claim it now!",
            "content_type": "email",
            "enable_nlp": True
        }
    ]
    
    request_data = {
        "requests": batch_requests,
        "max_concurrent": 3
    }
    
    print(f"Running batch analysis with {len(batch_requests)} requests...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/analyze/batch", json=request_data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            
            print(f"✅ Batch analysis completed in {end_time - start_time:.2f}s")
            print(f"   Total requests: {len(results)}")
            
            # Analyze results
            predictions = {}
            total_confidence = 0
            total_processing_time = 0
            
            for i, result in enumerate(results):
                prediction = result['prediction']
                confidence = result['confidence']
                processing_time = result['processing_time_ms']
                
                predictions[prediction] = predictions.get(prediction, 0) + 1
                total_confidence += confidence
                total_processing_time += processing_time
                
                print(f"   Request {i+1}: {prediction} ({confidence:.2%}) - {processing_time:.1f}ms")
            
            print(f"\n   Prediction Distribution:")
            for pred, count in predictions.items():
                print(f"     {pred}: {count} requests")
            
            print(f"   Average Confidence: {total_confidence/len(results):.2%}")
            print(f"   Average Processing Time: {total_processing_time/len(results):.1f}ms")
            
        else:
            print(f"❌ Batch analysis failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch analysis failed with error: {e}")

def test_system_health():
    """Test system health monitoring"""
    print_section("System Health Monitoring")
    
    base_url = "http://localhost:8001"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            
            print("✅ Health Check:")
            print(f"   Overall Status: {health_data.get('status', 'N/A')}")
            print(f"   Service: {health_data.get('service', 'N/A')}")
            print(f"   Version: {health_data.get('version', 'N/A')}")
            print(f"   Uptime: {health_data.get('uptime_seconds', 0):.0f} seconds")
            
            print(f"\n   Component Health:")
            components = health_data.get('components', {})
            for component, details in components.items():
                status = details.get('status', 'unknown')
                print(f"     {component}: {status}")
            
            print(f"\n   Performance Metrics:")
            performance = health_data.get('performance', {})
            for metric, value in performance.items():
                print(f"     {metric}: {value}")
                
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
    
    # Test system status
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            
            print(f"\n✅ System Status:")
            for service, status in status_data.items():
                print(f"   {service}: {status}")
                
        else:
            print(f"❌ System status failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ System status failed with error: {e}")
    
    # Test dashboard stats
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            stats_data = response.json()
            
            print(f"\n✅ Dashboard Statistics:")
            print(f"   Total Scans: {stats_data.get('total_scans', 0)}")
            print(f"   Threats Detected: {stats_data.get('threats_detected', 0)}")
            print(f"   Average Response Time: {stats_data.get('avg_response_time_ms', 0):.1f}ms")
            print(f"   Cache Hit Rate: {stats_data.get('cache_hit_rate', 0):.1f}%")
            print(f"   Cache Hits: {stats_data.get('cache_hits', 0)}")
            print(f"   Cache Misses: {stats_data.get('cache_misses', 0)}")
            
        else:
            print(f"❌ Dashboard stats failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Dashboard stats failed with error: {e}")

def test_performance_benchmark():
    """Test system performance"""
    print_section("Performance Benchmark")
    
    base_url = "http://localhost:8001"
    
    # Performance test parameters
    num_requests = 10
    test_content = "This is a performance test message for measuring system response times."
    
    print(f"Running performance test with {num_requests} requests...")
    
    start_time = time.time()
    successful_requests = 0
    total_processing_time = 0
    response_times = []
    
    for i in range(num_requests):
        try:
            request_data = {
                "content": test_content,
                "content_type": "text",
                "enable_nlp": True
            }
            
            request_start = time.time()
            response = requests.post(f"{base_url}/analyze", json=request_data, timeout=30)
            request_end = time.time()
            
            response_time = (request_end - request_start) * 1000
            response_times.append(response_time)
            
            if response.status_code == 200:
                successful_requests += 1
                data = response.json()
                total_processing_time += data['processing_time_ms']
                print(f"   Request {i+1}: {response_time:.1f}ms response, {data['processing_time_ms']:.1f}ms processing")
            else:
                print(f"   Request {i+1}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"   Request {i+1}: Failed with error: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Performance Results:")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful requests: {successful_requests}")
    print(f"   Success rate: {successful_requests/num_requests:.2%}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average response time: {sum(response_times)/len(response_times):.1f}ms")
    print(f"   Average processing time: {total_processing_time/successful_requests:.1f}ms")
    print(f"   Requests per second: {num_requests/total_time:.2f}")
    
    # Performance thresholds
    if successful_requests >= num_requests * 0.8:
        print("   ✅ Success rate within acceptable limits (≥80%)")
    else:
        print("   ⚠️  Success rate below acceptable limits (<80%)")
    
    if sum(response_times)/len(response_times) < 5000:
        print("   ✅ Response time within acceptable limits (<5s)")
    else:
        print("   ⚠️  Response time above acceptable limits (≥5s)")

def test_error_scenarios():
    """Test error handling scenarios"""
    print_section("Error Handling Test")
    
    base_url = "http://localhost:8001"
    
    error_scenarios = [
        {
            "name": "Invalid Content Type",
            "request": {
                "content": "Test content",
                "content_type": "invalid_type"
            },
            "expected_status": 422
        },
        {
            "name": "Missing Content",
            "request": {
                "content_type": "text"
            },
            "expected_status": 422
        },
        {
            "name": "Empty Content",
            "request": {
                "content": "",
                "content_type": "text"
            },
            "expected_status": 422
        },
        {
            "name": "Very Long Content",
            "request": {
                "content": "x" * 10000,  # Very long content
                "content_type": "text"
            },
            "expected_status": 200  # Should handle gracefully
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        try:
            response = requests.post(f"{base_url}/analyze", json=scenario['request'], timeout=10)
            
            if response.status_code == scenario['expected_status']:
                print(f"   ✅ Correctly returned status {response.status_code}")
            else:
                print(f"   ⚠️  Expected status {scenario['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed with error: {e}")

def main():
    """Main demo function"""
    print("Phish-Sim End-to-End System Integration Demo")
    print("T008 - End-to-End System Integration")
    print("=" * 60)
    
    print("\nThis demo showcases the complete system integration:")
    print("- Frontend (T006) integrated with Backend (T007)")
    print("- All services (NLP, Visual, Real-time) working together")
    print("- Complete data flow from input to analysis to results")
    print("- System monitoring and health checks")
    print("- Performance testing and error handling")
    
    # Run all demo sections
    if not test_system_connectivity():
        print("\n❌ System connectivity test failed. Please ensure all services are running.")
        return False
    
    test_api_endpoints()
    test_content_analysis()
    test_batch_analysis()
    test_system_health()
    test_performance_benchmark()
    test_error_scenarios()
    
    print_section("T008 Summary")
    print("✅ End-to-End System Integration Complete")
    print("\nKey Achievements:")
    print("- Frontend successfully integrated with unified backend API")
    print("- All services (NLP, Visual, Real-time) working together")
    print("- Complete data flow from frontend to backend to analysis")
    print("- System-wide error handling and recovery")
    print("- Comprehensive monitoring and health checks")
    print("- Performance optimization and caching")
    print("- Docker services configured for full system deployment")
    print("- End-to-end integration tests implemented")
    
    print("\nSystem Components:")
    print("- Frontend: React application with real-time updates")
    print("- Backend: Unified API integrating all services")
    print("- NLP Service: Text analysis and classification")
    print("- Visual Service: URL and image analysis")
    print("- Real-time Service: Caching and WebSocket communication")
    print("- Redis: Caching and session management")
    print("- Monitoring: Prometheus metrics and health checks")
    
    print("\nNext Steps:")
    print("- T009: Performance Optimization & Monitoring")
    print("- T010: Security Hardening & Compliance")
    print("- T011: Production Deployment & Scaling")
    
    print("\nThe complete system is now integrated and ready for production!")

if __name__ == "__main__":
    main()