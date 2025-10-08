# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Demo script for the unified API
Demonstrates all major functionality and endpoints
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

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

def test_api_endpoint(base_url: str, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Test an API endpoint and return the result"""
    url = f"{base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "data": response.json(),
            "success": True
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": getattr(e.response, 'status_code', None),
            "error": str(e),
            "success": False
        }

def demo_basic_endpoints(base_url: str):
    """Demo basic API endpoints"""
    print_section("Basic API Endpoints")
    
    # Test root endpoint
    result = test_api_endpoint(base_url, "/")
    print_result("Root Endpoint", result)
    
    # Test health check
    result = test_api_endpoint(base_url, "/health")
    print_result("Health Check", result)
    
    # Test model info
    result = test_api_endpoint(base_url, "/model/info")
    print_result("Model Info", result)
    
    # Test system status
    result = test_api_endpoint(base_url, "/status")
    print_result("System Status", result)
    
    # Test dashboard stats
    result = test_api_endpoint(base_url, "/stats")
    print_result("Dashboard Stats", result)

def demo_content_analysis(base_url: str):
    """Demo content analysis functionality"""
    print_section("Content Analysis")
    
    # Test cases for different content types
    test_cases = [
        {
            "name": "Legitimate Text",
            "content": "This is a legitimate email from your bank regarding your account statement.",
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
            "content": "http://fake-bank-security.com/login?verify=urgent",
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
            "content": "Congratulations! You've won $1000! Click here to claim your prize now!",
            "content_type": "email",
            "expected": "phish"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Content: {test_case['content']}")
        print(f"Type: {test_case['content_type']}")
        
        request_data = {
            "content": test_case["content"],
            "content_type": test_case["content_type"],
            "enable_nlp": True,
            "enable_visual": test_case["content_type"] == "url",
            "enable_realtime": True,
            "return_explanation": True
        }
        
        result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
        print_result("Analysis Result", result)
        
        if result["success"]:
            prediction = result["data"]["prediction"]
            confidence = result["data"]["confidence"]
            risk_level = result["data"]["risk_level"]
            processing_time = result["data"]["processing_time_ms"]
            
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Risk Level: {risk_level}")
            print(f"Processing Time: {processing_time:.1f}ms")
            
            # Check if prediction matches expectation
            if prediction == test_case["expected"]:
                print("✅ Prediction matches expectation")
            else:
                print(f"⚠️  Prediction ({prediction}) differs from expectation ({test_case['expected']})")
        
        time.sleep(1)  # Rate limiting

def demo_batch_analysis(base_url: str):
    """Demo batch analysis functionality"""
    print_section("Batch Analysis")
    
    # Create batch of test cases
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
        },
        {
            "content": "http://fake-microsoft-security.com",
            "content_type": "url",
            "enable_nlp": True,
            "enable_visual": True
        }
    ]
    
    request_data = {
        "requests": batch_requests,
        "max_concurrent": 3
    }
    
    print("Running batch analysis with 4 requests...")
    result = test_api_endpoint(base_url, "/analyze/batch", "POST", request_data)
    print_result("Batch Analysis Result", result)
    
    if result["success"]:
        results = result["data"]
        print(f"\nBatch Analysis Summary:")
        print(f"Total requests: {len(results)}")
        
        predictions = {}
        total_confidence = 0
        total_processing_time = 0
        
        for i, res in enumerate(results):
            prediction = res["prediction"]
            confidence = res["confidence"]
            processing_time = res["processing_time_ms"]
            
            predictions[prediction] = predictions.get(prediction, 0) + 1
            total_confidence += confidence
            total_processing_time += processing_time
            
            print(f"  Request {i+1}: {prediction} ({confidence:.2%}) - {processing_time:.1f}ms")
        
        print(f"\nPrediction Distribution:")
        for pred, count in predictions.items():
            print(f"  {pred}: {count} requests")
        
        print(f"Average Confidence: {total_confidence/len(results):.2%}")
        print(f"Total Processing Time: {total_processing_time:.1f}ms")
        print(f"Average Processing Time: {total_processing_time/len(results):.1f}ms")

def demo_advanced_features(base_url: str):
    """Demo advanced API features"""
    print_section("Advanced Features")
    
    # Test with feature extraction
    print("\n1. Analysis with Feature Extraction")
    request_data = {
        "content": "This is a test message with some suspicious keywords like 'urgent' and 'verify'.",
        "content_type": "text",
        "return_features": True,
        "enable_nlp": True
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("Analysis with Features", result)
    
    # Test force re-analysis
    print("\n2. Force Re-analysis (bypass cache)")
    request_data = {
        "content": "This is a test message for cache bypass.",
        "content_type": "text",
        "force_reanalyze": True,
        "enable_nlp": True
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("Force Re-analysis", result)
    
    # Test with different service combinations
    print("\n3. NLP Only Analysis")
    request_data = {
        "content": "This is a test message using only NLP analysis.",
        "content_type": "text",
        "enable_nlp": True,
        "enable_visual": False,
        "enable_realtime": False
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("NLP Only Analysis", result)
    
    # Test with user and session tracking
    print("\n4. Analysis with User/Session Tracking")
    request_data = {
        "content": "This is a test message with user tracking.",
        "content_type": "text",
        "user_id": "demo_user_123",
        "session_id": "demo_session_456",
        "enable_nlp": True
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("Analysis with Tracking", result)

def demo_error_handling(base_url: str):
    """Demo error handling"""
    print_section("Error Handling")
    
    # Test invalid content type
    print("\n1. Invalid Content Type")
    request_data = {
        "content": "Test content",
        "content_type": "invalid_type",
        "enable_nlp": True
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("Invalid Content Type", result)
    
    # Test missing content
    print("\n2. Missing Content")
    request_data = {
        "content_type": "text",
        "enable_nlp": True
    }
    
    result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
    print_result("Missing Content", result)
    
    # Test empty batch
    print("\n3. Empty Batch Request")
    request_data = {
        "requests": [],
        "max_concurrent": 2
    }
    
    result = test_api_endpoint(base_url, "/analyze/batch", "POST", request_data)
    print_result("Empty Batch", result)
    
    # Test invalid batch parameters
    print("\n4. Invalid Batch Parameters")
    request_data = {
        "requests": [
            {
                "content": "Test content",
                "content_type": "text"
            }
        ],
        "max_concurrent": 0  # Invalid
    }
    
    result = test_api_endpoint(base_url, "/analyze/batch", "POST", request_data)
    print_result("Invalid Batch Parameters", result)

def demo_performance_testing(base_url: str):
    """Demo performance testing"""
    print_section("Performance Testing")
    
    # Test multiple requests to measure performance
    test_content = "This is a performance test message for measuring API response times."
    num_requests = 10
    
    print(f"\nRunning {num_requests} requests to measure performance...")
    
    request_data = {
        "content": test_content,
        "content_type": "text",
        "enable_nlp": True
    }
    
    start_time = time.time()
    successful_requests = 0
    total_processing_time = 0
    response_times = []
    
    for i in range(num_requests):
        request_start = time.time()
        result = test_api_endpoint(base_url, "/analyze", "POST", request_data)
        request_end = time.time()
        
        response_time = (request_end - request_start) * 1000
        response_times.append(response_time)
        
        if result["success"]:
            successful_requests += 1
            total_processing_time += result["data"]["processing_time_ms"]
        
        print(f"Request {i+1}: {response_time:.1f}ms response time")
    
    total_time = time.time() - start_time
    
    print(f"\nPerformance Summary:")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Success rate: {successful_requests/num_requests:.2%}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average response time: {sum(response_times)/len(response_times):.1f}ms")
    print(f"Average processing time: {total_processing_time/successful_requests:.1f}ms")
    print(f"Requests per second: {num_requests/total_time:.2f}")

def main():
    """Main demo function"""
    print("Phish-Sim Unified API Demo")
    print("=" * 60)
    
    # Configuration
    base_url = "http://localhost:8001"
    
    print(f"Testing API at: {base_url}")
    print("Make sure the unified API is running before starting the demo.")
    
    # Check if API is available
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not available at {base_url}")
            print("Please start the unified API first:")
            print("  cd /workspace/phish-sim/backend")
            print("  python unified_api.py")
            return
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to API at {base_url}")
        print("Please start the unified API first:")
        print("  cd /workspace/phish-sim/backend")
        print("  python unified_api.py")
        return
    
    print("✅ API is available, starting demo...")
    
    try:
        # Run all demo sections
        demo_basic_endpoints(base_url)
        demo_content_analysis(base_url)
        demo_batch_analysis(base_url)
        demo_advanced_features(base_url)
        demo_error_handling(base_url)
        demo_performance_testing(base_url)
        
        print_section("Demo Complete")
        print("✅ All demo sections completed successfully!")
        print("\nThe unified API provides:")
        print("- Comprehensive content analysis (NLP, Visual, Real-time)")
        print("- Batch processing capabilities")
        print("- Real-time WebSocket communication")
        print("- Advanced features (caching, feature extraction)")
        print("- Robust error handling")
        print("- Performance monitoring")
        print("- Health checks and system status")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()