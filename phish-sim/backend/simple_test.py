# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple test script for the unified API
Tests basic functionality without external dependencies
"""

import sys
import json
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "app"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from unified_api import app, AnalysisRequest, AnalysisResponse, HealthResponse
        print("✅ Core unified API imports successful")
    except ImportError as e:
        print(f"❌ Core unified API import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI import successful")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✅ Pydantic import successful")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    return True

def test_models():
    """Test Pydantic models"""
    print("\nTesting Pydantic models...")
    
    try:
        from unified_api import AnalysisRequest, AnalysisResponse, HealthResponse
        
        # Test AnalysisRequest
        request_data = {
            "content": "This is a test message",
            "content_type": "text",
            "enable_nlp": True,
            "enable_visual": False,
            "enable_realtime": True
        }
        
        request = AnalysisRequest(**request_data)
        print("✅ AnalysisRequest model works")
        print(f"   Content: {request.content}")
        print(f"   Type: {request.content_type}")
        print(f"   NLP enabled: {request.enable_nlp}")
        
        # Test AnalysisResponse
        response_data = {
            "request_id": "test_123",
            "content": "This is a test message",
            "content_type": "text",
            "prediction": "benign",
            "confidence": 0.85,
            "risk_score": 0.15,
            "risk_level": "LOW",
            "explanation": {"overall_assessment": "Content appears safe"},
            "processing_time_ms": 150.0,
            "cached": False,
            "timestamp": "2024-01-01T00:00:00",
            "components": {"nlp": {"prediction": "benign", "confidence": 0.85}}
        }
        
        response = AnalysisResponse(**response_data)
        print("✅ AnalysisResponse model works")
        print(f"   Prediction: {response.prediction}")
        print(f"   Confidence: {response.confidence}")
        print(f"   Risk Level: {response.risk_level}")
        
        # Test HealthResponse
        health_data = {
            "status": "ok",
            "service": "phish-sim-unified-api",
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
            "components": {
                "nlp_service": {"status": "healthy", "details": "NLP inference service"},
                "visual_service": {"status": "healthy", "details": "Visual analysis service"}
            },
            "performance": {"avg_response_time_ms": 150.0}
        }
        
        health = HealthResponse(**health_data)
        print("✅ HealthResponse model works")
        print(f"   Status: {health.status}")
        print(f"   Service: {health.service}")
        print(f"   Components: {len(health.components)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False

def test_validation():
    """Test model validation"""
    print("\nTesting model validation...")
    
    try:
        from unified_api import AnalysisRequest
        
        # Test valid content types
        valid_types = ["url", "email", "text"]
        for content_type in valid_types:
            request = AnalysisRequest(
                content="Test content",
                content_type=content_type
            )
            print(f"✅ Valid content type: {content_type}")
        
        # Test invalid content type
        try:
            request = AnalysisRequest(
                content="Test content",
                content_type="invalid_type"
            )
            print("❌ Should have failed for invalid content type")
            return False
        except Exception:
            print("✅ Correctly rejected invalid content type")
        
        # Test missing required fields
        try:
            request = AnalysisRequest(content_type="text")
            print("❌ Should have failed for missing content")
            return False
        except Exception:
            print("✅ Correctly rejected missing content")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation testing failed: {e}")
        return False

def test_app_creation():
    """Test FastAPI app creation"""
    print("\nTesting FastAPI app creation...")
    
    try:
        from unified_api import app
        
        print("✅ FastAPI app created successfully")
        print(f"   Title: {app.title}")
        print(f"   Version: {app.version}")
        print(f"   Description: {app.description[:50]}...")
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/model/info", "/analyze", "/analyze/batch", "/status", "/stats"]
        
        for expected_route in expected_routes:
            if expected_route in routes:
                print(f"✅ Route {expected_route} registered")
            else:
                print(f"❌ Route {expected_route} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ App creation testing failed: {e}")
        return False

def test_mock_analysis():
    """Test mock analysis function"""
    print("\nTesting mock analysis function...")
    
    try:
        from unified_api import perform_unified_analysis
        
        # Test with mock services
        import asyncio
        
        async def run_test():
            result = await perform_unified_analysis(
                content="This is a test message",
                content_type="text",
                enable_nlp=False,
                enable_visual=False,
                enable_realtime=False
            )
            
            print("✅ Mock analysis function works")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Components: {list(result['components'].keys())}")
            
            return True
        
        return asyncio.run(run_test())
        
    except Exception as e:
        print(f"❌ Mock analysis testing failed: {e}")
        return False

def main():
    """Main test function"""
    print("Phish-Sim Unified API - Simple Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Test", test_models),
        ("Validation Test", test_validation),
        ("App Creation Test", test_app_creation),
        ("Mock Analysis Test", test_mock_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified API is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)