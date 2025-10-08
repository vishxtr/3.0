# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
End-to-End Integration Tests for Phish-Sim
Tests the complete system integration from frontend to backend
"""

import pytest
import asyncio
import requests
import json
import time
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

class TestEndToEndIntegration:
    """End-to-end integration test suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.frontend_url = "http://localhost:3000"
        self.timeout = 30
    
    def test_system_startup(self):
        """Test that all services start up correctly"""
        print("Testing system startup...")
        
        # Test backend API
        try:
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Phish-Sim Unified API"
            print("✅ Backend API is running")
        except Exception as e:
            pytest.fail(f"Backend API not available: {e}")
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "components" in data
            print("✅ Health endpoint is working")
        except Exception as e:
            pytest.fail(f"Health endpoint failed: {e}")
        
        # Test frontend (if available)
        try:
            response = requests.get(f"{self.frontend_url}/", timeout=self.timeout)
            assert response.status_code == 200
            print("✅ Frontend is accessible")
        except Exception as e:
            print(f"⚠️  Frontend not available: {e}")
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("Testing API endpoints...")
        
        endpoints = [
            ("/", "GET"),
            ("/health", "GET"),
            ("/model/info", "GET"),
            ("/status", "GET"),
            ("/stats", "GET"),
        ]
        
        for endpoint, method in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                else:
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=self.timeout)
                
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, (dict, list))
                print(f"✅ {method} {endpoint} - OK")
                
            except Exception as e:
                pytest.fail(f"{method} {endpoint} failed: {e}")
    
    def test_content_analysis_flow(self):
        """Test complete content analysis flow"""
        print("Testing content analysis flow...")
        
        # Test cases for different content types
        test_cases = [
            {
                "name": "Legitimate Text",
                "content": "This is a legitimate business email from a trusted company.",
                "content_type": "text",
                "expected_prediction": "benign"
            },
            {
                "name": "Suspicious Text",
                "content": "URGENT: Your account will be suspended! Click here immediately!",
                "content_type": "text",
                "expected_prediction": "suspicious"
            },
            {
                "name": "Phishing URL",
                "content": "http://fake-bank-security.com/login?verify=urgent",
                "content_type": "url",
                "expected_prediction": "phish"
            },
            {
                "name": "Legitimate URL",
                "content": "https://www.google.com",
                "content_type": "url",
                "expected_prediction": "benign"
            }
        ]
        
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}")
            
            request_data = {
                "content": test_case["content"],
                "content_type": test_case["content_type"],
                "enable_nlp": True,
                "enable_visual": test_case["content_type"] == "url",
                "enable_realtime": True,
                "return_explanation": True
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/analyze",
                    json=request_data,
                    timeout=self.timeout
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Validate response structure
                assert "request_id" in data
                assert "prediction" in data
                assert "confidence" in data
                assert "risk_level" in data
                assert "processing_time_ms" in data
                assert "timestamp" in data
                assert "components" in data
                
                # Validate prediction is one of expected values
                assert data["prediction"] in ["phish", "benign", "suspicious", "unknown"]
                assert 0 <= data["confidence"] <= 1
                assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                assert data["processing_time_ms"] > 0
                
                print(f"    ✅ Prediction: {data['prediction']} (confidence: {data['confidence']:.2%})")
                print(f"    ✅ Risk Level: {data['risk_level']}")
                print(f"    ✅ Processing Time: {data['processing_time_ms']:.1f}ms")
                
            except Exception as e:
                pytest.fail(f"Analysis failed for {test_case['name']}: {e}")
    
    def test_batch_analysis_flow(self):
        """Test batch analysis flow"""
        print("Testing batch analysis flow...")
        
        batch_requests = [
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
        ]
        
        request_data = {
            "requests": batch_requests,
            "max_concurrent": 3
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/analyze/batch",
                json=request_data,
                timeout=self.timeout * 2  # Batch takes longer
            )
            
            assert response.status_code == 200
            results = response.json()
            
            assert isinstance(results, list)
            assert len(results) == len(batch_requests)
            
            for i, result in enumerate(results):
                assert "request_id" in result
                assert "prediction" in result
                assert "confidence" in result
                assert "processing_time_ms" in result
                
                print(f"  ✅ Batch item {i+1}: {result['prediction']} ({result['confidence']:.2%})")
            
            print(f"✅ Batch analysis completed: {len(results)} results")
            
        except Exception as e:
            pytest.fail(f"Batch analysis failed: {e}")
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        print("Testing error handling...")
        
        # Test invalid content type
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json={
                    "content": "Test content",
                    "content_type": "invalid_type"
                },
                timeout=self.timeout
            )
            assert response.status_code == 422  # Validation error
            print("✅ Invalid content type properly rejected")
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
        
        # Test missing content
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json={
                    "content_type": "text"
                },
                timeout=self.timeout
            )
            assert response.status_code == 422  # Validation error
            print("✅ Missing content properly rejected")
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance and metrics"""
        print("Testing performance metrics...")
        
        # Test multiple requests to measure performance
        num_requests = 5
        test_content = "This is a performance test message."
        
        start_time = time.time()
        successful_requests = 0
        total_processing_time = 0
        
        for i in range(num_requests):
            try:
                request_data = {
                    "content": test_content,
                    "content_type": "text",
                    "enable_nlp": True
                }
                
                response = requests.post(
                    f"{self.base_url}/analyze",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    successful_requests += 1
                    data = response.json()
                    total_processing_time += data["processing_time_ms"]
                
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        
        print(f"  Total requests: {num_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Success rate: {successful_requests/num_requests:.2%}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average processing time: {total_processing_time/successful_requests:.1f}ms")
        print(f"  Requests per second: {num_requests/total_time:.2f}")
        
        # Validate performance thresholds
        assert successful_requests >= num_requests * 0.8  # At least 80% success rate
        assert total_processing_time / successful_requests < 5000  # Less than 5 seconds average
        
        print("✅ Performance metrics within acceptable limits")
    
    def test_system_health_monitoring(self):
        """Test system health monitoring"""
        print("Testing system health monitoring...")
        
        # Test health endpoint
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            assert response.status_code == 200
            health_data = response.json()
            
            assert "status" in health_data
            assert "components" in health_data
            assert "performance" in health_data
            
            print(f"  System Status: {health_data['status']}")
            print(f"  Components: {len(health_data['components'])}")
            
            # Check component health
            for component, details in health_data["components"].items():
                print(f"    {component}: {details.get('status', 'unknown')}")
            
            print("✅ Health monitoring working")
            
        except Exception as e:
            pytest.fail(f"Health monitoring failed: {e}")
        
        # Test system status
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            assert response.status_code == 200
            status_data = response.json()
            
            assert "backend_api" in status_data
            assert "ml_pipeline" in status_data
            assert "database" in status_data
            assert "redis" in status_data
            assert "websocket" in status_data
            
            print("  System Status:")
            for service, status in status_data.items():
                print(f"    {service}: {status}")
            
            print("✅ System status monitoring working")
            
        except Exception as e:
            pytest.fail(f"System status monitoring failed: {e}")
        
        # Test dashboard stats
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=self.timeout)
            assert response.status_code == 200
            stats_data = response.json()
            
            assert "total_scans" in stats_data
            assert "threats_detected" in stats_data
            assert "avg_response_time_ms" in stats_data
            assert "cache_hit_rate" in stats_data
            
            print("  Dashboard Stats:")
            print(f"    Total Scans: {stats_data['total_scans']}")
            print(f"    Threats Detected: {stats_data['threats_detected']}")
            print(f"    Avg Response Time: {stats_data['avg_response_time_ms']:.1f}ms")
            print(f"    Cache Hit Rate: {stats_data['cache_hit_rate']:.1f}%")
            
            print("✅ Dashboard stats working")
            
        except Exception as e:
            pytest.fail(f"Dashboard stats failed: {e}")

def run_e2e_tests():
    """Run all end-to-end tests"""
    print("Phish-Sim End-to-End Integration Tests")
    print("=" * 50)
    
    test_suite = TestEndToEndIntegration()
    
    tests = [
        ("System Startup", test_suite.test_system_startup),
        ("API Endpoints", test_suite.test_api_endpoints),
        ("Content Analysis Flow", test_suite.test_content_analysis_flow),
        ("Batch Analysis Flow", test_suite.test_batch_analysis_flow),
        ("Error Handling", test_suite.test_error_handling),
        ("Performance Metrics", test_suite.test_performance_metrics),
        ("System Health Monitoring", test_suite.test_system_health_monitoring),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"End-to-End Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All end-to-end tests passed! System integration is working correctly.")
        return True
    else:
        print("⚠️  Some end-to-end tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_e2e_tests()
    sys.exit(0 if success else 1)