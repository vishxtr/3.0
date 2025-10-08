# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Comprehensive tests for the unified API
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the unified API
from unified_api import app

# Create test client
client = TestClient(app)

class TestUnifiedAPI:
    """Test suite for unified API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Phish-Sim Unified API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "components" in data
        assert data["service"] == "phish-sim-unified-api"
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "nlp_model" in data
        assert "visual_model" in data
        assert "thresholds" in data
        assert data["nlp_model"]["name"] == "PhishingClassifier"
        assert data["visual_model"]["name"] == "VisualClassifier"
    
    def test_system_status(self):
        """Test system status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "backend_api" in data
        assert "ml_pipeline" in data
        assert "database" in data
        assert "redis" in data
        assert "websocket" in data
    
    def test_dashboard_stats(self):
        """Test dashboard stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_scans" in data
        assert "threats_detected" in data
        assert "avg_response_time_ms" in data
        assert "cache_hit_rate" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_content_text(self, mock_analysis):
        """Test content analysis for text"""
        # Mock analysis result
        mock_analysis.return_value = {
            "prediction": "benign",
            "confidence": 0.85,
            "risk_score": 0.15,
            "risk_level": "LOW",
            "explanation": {"overall_assessment": "Content appears safe"},
            "components": {"nlp": {"prediction": "benign", "confidence": 0.85}}
        }
        
        request_data = {
            "content": "This is a legitimate email from a trusted source.",
            "content_type": "text",
            "enable_nlp": True,
            "enable_visual": False,
            "enable_realtime": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "benign"
        assert data["confidence"] == 0.85
        assert data["risk_level"] == "LOW"
        assert data["content_type"] == "text"
        assert "request_id" in data
        assert "processing_time_ms" in data
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_content_url(self, mock_analysis):
        """Test content analysis for URL"""
        # Mock analysis result
        mock_analysis.return_value = {
            "prediction": "phish",
            "confidence": 0.92,
            "risk_score": 0.92,
            "risk_level": "HIGH",
            "explanation": {"overall_assessment": "URL appears suspicious"},
            "components": {
                "nlp": {"prediction": "phish", "confidence": 0.88},
                "visual": {"overall_risk_score": 0.95}
            }
        }
        
        request_data = {
            "content": "http://suspicious-phishing-site.com/login",
            "content_type": "url",
            "enable_nlp": True,
            "enable_visual": True,
            "enable_realtime": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "phish"
        assert data["confidence"] == 0.92
        assert data["risk_level"] == "HIGH"
        assert data["content_type"] == "url"
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_content_email(self, mock_analysis):
        """Test content analysis for email"""
        # Mock analysis result
        mock_analysis.return_value = {
            "prediction": "suspicious",
            "confidence": 0.65,
            "risk_score": 0.65,
            "risk_level": "MEDIUM",
            "explanation": {"overall_assessment": "Email shows some suspicious patterns"},
            "components": {"nlp": {"prediction": "suspicious", "confidence": 0.65}}
        }
        
        request_data = {
            "content": "Urgent: Your account will be suspended. Click here immediately!",
            "content_type": "email",
            "enable_nlp": True,
            "enable_visual": False,
            "enable_realtime": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "suspicious"
        assert data["confidence"] == 0.65
        assert data["risk_level"] == "MEDIUM"
        assert data["content_type"] == "email"
    
    def test_analyze_content_invalid_type(self):
        """Test content analysis with invalid content type"""
        request_data = {
            "content": "Some content",
            "content_type": "invalid_type",
            "enable_nlp": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_content_missing_content(self):
        """Test content analysis with missing content"""
        request_data = {
            "content_type": "text",
            "enable_nlp": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('unified_api.perform_unified_analysis')
    def test_batch_analysis(self, mock_analysis):
        """Test batch analysis"""
        # Mock analysis result
        mock_analysis.return_value = {
            "prediction": "benign",
            "confidence": 0.80,
            "risk_score": 0.20,
            "risk_level": "LOW",
            "explanation": {"overall_assessment": "Content appears safe"},
            "components": {"nlp": {"prediction": "benign", "confidence": 0.80}}
        }
        
        request_data = {
            "requests": [
                {
                    "content": "This is a legitimate message.",
                    "content_type": "text",
                    "enable_nlp": True
                },
                {
                    "content": "Another legitimate message.",
                    "content_type": "text",
                    "enable_nlp": True
                }
            ],
            "max_concurrent": 2
        }
        
        response = client.post("/analyze/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(item["prediction"] == "benign" for item in data)
        assert all(item["confidence"] == 0.80 for item in data)
    
    def test_batch_analysis_empty_requests(self):
        """Test batch analysis with empty requests"""
        request_data = {
            "requests": [],
            "max_concurrent": 2
        }
        
        response = client.post("/analyze/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
    
    def test_batch_analysis_invalid_concurrent(self):
        """Test batch analysis with invalid max_concurrent"""
        request_data = {
            "requests": [
                {
                    "content": "Test content",
                    "content_type": "text"
                }
            ],
            "max_concurrent": 0  # Invalid
        }
        
        response = client.post("/analyze/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_with_features(self, mock_analysis):
        """Test analysis with feature extraction"""
        # Mock analysis result with features
        mock_analysis.return_value = {
            "prediction": "benign",
            "confidence": 0.85,
            "risk_score": 0.15,
            "risk_level": "LOW",
            "explanation": {"overall_assessment": "Content appears safe"},
            "components": {"nlp": {"prediction": "benign", "confidence": 0.85}},
            "features": {
                "text_length": 45,
                "suspicious_words": 0,
                "url_count": 0
            }
        }
        
        request_data = {
            "content": "This is a legitimate email from a trusted source.",
            "content_type": "text",
            "return_features": True,
            "enable_nlp": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert data["features"]["text_length"] == 45
        assert data["features"]["suspicious_words"] == 0
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_with_explanation(self, mock_analysis):
        """Test analysis with detailed explanation"""
        # Mock analysis result with explanation
        mock_analysis.return_value = {
            "prediction": "phish",
            "confidence": 0.90,
            "risk_score": 0.90,
            "risk_level": "HIGH",
            "explanation": {
                "overall_assessment": "Content classified as phish with 90% confidence",
                "risk_level": "HIGH",
                "components_analyzed": ["nlp", "visual"],
                "details": {
                    "nlp": {"prediction": "phish", "confidence": 0.88},
                    "visual": {"overall_risk_score": 0.92}
                }
            },
            "components": {
                "nlp": {"prediction": "phish", "confidence": 0.88},
                "visual": {"overall_risk_score": 0.92}
            }
        }
        
        request_data = {
            "content": "http://suspicious-phishing-site.com/login",
            "content_type": "url",
            "return_explanation": True,
            "enable_nlp": True,
            "enable_visual": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert data["explanation"]["overall_assessment"] == "Content classified as phish with 90% confidence"
        assert data["explanation"]["risk_level"] == "HIGH"
        assert "components_analyzed" in data["explanation"]
    
    @patch('unified_api.perform_unified_analysis')
    def test_analyze_force_reanalyze(self, mock_analysis):
        """Test analysis with force re-analyze"""
        # Mock analysis result
        mock_analysis.return_value = {
            "prediction": "benign",
            "confidence": 0.85,
            "risk_score": 0.15,
            "risk_level": "LOW",
            "explanation": {"overall_assessment": "Content appears safe"},
            "components": {"nlp": {"prediction": "benign", "confidence": 0.85}}
        }
        
        request_data = {
            "content": "This is a legitimate message.",
            "content_type": "text",
            "force_reanalyze": True,
            "enable_nlp": True
        }
        
        response = client.post("/analyze", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["cached"] == False  # Should not be cached when force_reanalyze is True
    
    def test_analyze_service_error(self):
        """Test analysis when service fails"""
        with patch('unified_api.perform_unified_analysis', side_effect=Exception("Service error")):
            request_data = {
                "content": "Test content",
                "content_type": "text",
                "enable_nlp": True
            }
            
            response = client.post("/analyze", json=request_data)
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Service error" in data["error"]

class TestUnifiedAnalysis:
    """Test suite for unified analysis function"""
    
    @pytest.mark.asyncio
    async def test_perform_unified_analysis_nlp_only(self):
        """Test unified analysis with NLP only"""
        from unified_api import perform_unified_analysis
        
        # Mock NLP service
        mock_nlp_service = Mock()
        mock_nlp_service.analyze_text.return_value = {
            "prediction": "benign",
            "confidence": 0.85,
            "probabilities": {"benign": 0.85, "phish": 0.10, "suspicious": 0.05}
        }
        
        with patch('unified_api.nlp_service', mock_nlp_service):
            result = await perform_unified_analysis(
                content="This is a legitimate message.",
                content_type="text",
                enable_nlp=True,
                enable_visual=False,
                enable_realtime=False
            )
            
            assert result["prediction"] == "benign"
            assert result["confidence"] == 0.85
            assert result["risk_score"] == 0.15  # 1 - confidence for benign
            assert result["risk_level"] == "LOW"
            assert "nlp" in result["components"]
    
    @pytest.mark.asyncio
    async def test_perform_unified_analysis_visual_only(self):
        """Test unified analysis with visual analysis only"""
        from unified_api import perform_unified_analysis
        
        # Mock visual analysis
        mock_visual_analysis = AsyncMock(return_value={
            "overall_risk_score": 0.75,
            "risk_level": "HIGH",
            "components": {
                "screenshot": {"success": True},
                "cnn_analysis": {"prediction": "phish", "confidence": 0.80}
            }
        })
        
        with patch('unified_api.perform_visual_analysis', mock_visual_analysis):
            result = await perform_unified_analysis(
                content="http://example.com",
                content_type="url",
                enable_nlp=False,
                enable_visual=True,
                enable_realtime=False
            )
            
            assert result["prediction"] == "unknown"  # No NLP prediction
            assert result["risk_score"] == 0.75
            assert result["risk_level"] == "HIGH"
            assert "visual" in result["components"]
    
    @pytest.mark.asyncio
    async def test_perform_unified_analysis_combined(self):
        """Test unified analysis with multiple components"""
        from unified_api import perform_unified_analysis
        
        # Mock NLP service
        mock_nlp_service = Mock()
        mock_nlp_service.analyze_text.return_value = {
            "prediction": "phish",
            "confidence": 0.80,
            "probabilities": {"benign": 0.15, "phish": 0.80, "suspicious": 0.05}
        }
        
        # Mock visual analysis
        mock_visual_analysis = AsyncMock(return_value={
            "overall_risk_score": 0.70,
            "risk_level": "HIGH",
            "components": {
                "screenshot": {"success": True},
                "cnn_analysis": {"prediction": "phish", "confidence": 0.75}
            }
        })
        
        # Mock pipeline orchestrator
        mock_pipeline = AsyncMock()
        mock_pipeline.process_request.return_value = {
            "prediction": "phish",
            "confidence": 0.85,
            "risk_score": 0.85
        }
        
        with patch('unified_api.nlp_service', mock_nlp_service), \
             patch('unified_api.perform_visual_analysis', mock_visual_analysis), \
             patch('unified_api.pipeline_orchestrator', mock_pipeline):
            
            result = await perform_unified_analysis(
                content="http://suspicious-site.com",
                content_type="url",
                enable_nlp=True,
                enable_visual=True,
                enable_realtime=True
            )
            
            # Should use majority voting (all predict phish)
            assert result["prediction"] == "phish"
            # Average confidence: (0.80 + 0.85) / 2 = 0.825
            assert abs(result["confidence"] - 0.825) < 0.01
            # Average risk score: (0.80 + 0.70 + 0.85) / 3 = 0.783
            assert abs(result["risk_score"] - 0.783) < 0.01
            assert result["risk_level"] == "HIGH"
            assert "nlp" in result["components"]
            assert "visual" in result["components"]
            assert "realtime" in result["components"]
    
    @pytest.mark.asyncio
    async def test_perform_unified_analysis_no_services(self):
        """Test unified analysis with no services available"""
        from unified_api import perform_unified_analysis
        
        with patch('unified_api.nlp_service', None), \
             patch('unified_api.pipeline_orchestrator', None):
            
            result = await perform_unified_analysis(
                content="Test content",
                content_type="text",
                enable_nlp=True,
                enable_visual=False,
                enable_realtime=True
            )
            
            assert result["prediction"] == "unknown"
            assert result["confidence"] == 0.0
            assert result["risk_score"] == 0.0
            assert result["risk_level"] == "LOW"
            assert result["components"] == {}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])