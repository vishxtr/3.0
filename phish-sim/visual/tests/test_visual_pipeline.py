# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Comprehensive tests for visual analysis pipeline
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image

# Import components to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from screenshot_capture import ScreenshotCapture, create_screenshot_capture
from dom_analysis import DOMAnalyzer, create_dom_analyzer
from cnn_models.visual_classifier import SimpleCNNClassifier, create_visual_classifier
from feature_extraction.visual_features import VisualFeatureExtractor, create_visual_feature_extractor
from template_matching import TemplateMatcher, create_template_matcher
from api.visual_analysis_api import app, perform_visual_analysis
from config import get_config

class TestScreenshotCapture:
    """Test screenshot capture functionality"""
    
    @pytest.fixture
    def screenshot_capture(self):
        """Create screenshot capture instance"""
        return create_screenshot_capture()
    
    @pytest.fixture
    def mock_image(self):
        """Create mock image for testing"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_screenshot_capture_initialization(self, screenshot_capture):
        """Test screenshot capture initialization"""
        assert screenshot_capture is not None
        assert screenshot_capture.config is not None
        assert screenshot_capture.browser is None
        assert screenshot_capture.context is None
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = get_config("screenshot")
        assert config is not None
        assert config.browser_type in ["chromium", "firefox", "webkit"]
        assert config.viewport_width > 0
        assert config.viewport_height > 0
    
    @pytest.mark.asyncio
    async def test_browser_startup_shutdown(self, screenshot_capture):
        """Test browser startup and shutdown"""
        # Mock playwright
        with patch('screenshot_capture.async_playwright') as mock_playwright:
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_playwright.return_value.start.return_value = AsyncMock()
            mock_playwright.return_value.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context
            
            # Test startup
            await screenshot_capture.start_browser()
            assert screenshot_capture.browser is not None
            assert screenshot_capture.context is not None
            
            # Test shutdown
            await screenshot_capture.close_browser()
            mock_browser.close.assert_called_once()
    
    def test_image_analysis(self, screenshot_capture, mock_image):
        """Test image analysis functionality"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test analysis
            analysis = screenshot_capture.analyze_screenshot(tmp_file.name)
            
            assert "dimensions" in analysis
            assert "color_analysis" in analysis
            assert "texture_analysis" in analysis
            assert "edge_analysis" in analysis
            
            # Cleanup
            Path(tmp_file.name).unlink()

class TestDOMAnalyzer:
    """Test DOM analysis functionality"""
    
    @pytest.fixture
    def dom_analyzer(self):
        """Create DOM analyzer instance"""
        return create_dom_analyzer()
    
    @pytest.fixture
    def mock_page(self):
        """Create mock page for testing"""
        mock_page = AsyncMock()
        mock_page.content.return_value = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <form action="/login" method="post">
                    <input type="text" name="username" placeholder="Username">
                    <input type="password" name="password" placeholder="Password">
                    <input type="submit" value="Login">
                </form>
                <a href="https://external-site.com">External Link</a>
                <img src="https://external-site.com/image.jpg" alt="Test Image">
            </body>
        </html>
        """
        mock_page.url = "https://test.com"
        return mock_page
    
    def test_dom_analyzer_initialization(self, dom_analyzer):
        """Test DOM analyzer initialization"""
        assert dom_analyzer is not None
        assert dom_analyzer.config is not None
        assert hasattr(dom_analyzer, 'patterns')
    
    @pytest.mark.asyncio
    async def test_page_analysis(self, dom_analyzer, mock_page):
        """Test page analysis"""
        result = await dom_analyzer.analyze_page(mock_page, "https://test.com")
        
        assert "url" in result
        assert "timestamp" in result
        assert "basic_info" in result
        assert "form_analysis" in result
        assert "link_analysis" in result
        assert "image_analysis" in result
        assert "risk_score" in result
        
        # Check form analysis
        form_analysis = result["form_analysis"]
        assert form_analysis["total_forms"] == 1
        assert len(form_analysis["forms"]) == 1
        
        # Check link analysis
        link_analysis = result["link_analysis"]
        assert link_analysis["total_links"] == 1
        assert link_analysis["external_links"] == 1
    
    def test_phishing_detection(self, dom_analyzer):
        """Test phishing detection patterns"""
        # Test suspicious domain detection
        suspicious_url = "https://paypal-security-alert.net/login"
        assert dom_analyzer._is_suspicious_link(suspicious_url, "Click here to verify")
        
        # Test benign URL
        benign_url = "https://google.com/search"
        assert not dom_analyzer._is_suspicious_link(benign_url, "Search")
    
    def test_risk_score_calculation(self, dom_analyzer):
        """Test risk score calculation"""
        # Mock analysis results
        analysis_results = {
            "form_analysis": {
                "forms": [{"suspicious_score": 0.8}]
            },
            "link_analysis": {
                "suspicious_links": [{"href": "test"}],
                "redirect_links": [{"href": "test"}]
            },
            "content_analysis": {
                "phishing_keywords_found": 3,
                "urgency_indicators": 2
            },
            "security_analysis": {
                "mixed_content": True,
                "insecure_forms": False
            },
            "phishing_indicators": {
                "hidden_elements": [{"test": "data"}],
                "obfuscated_content": ["base64 detected"]
            }
        }
        
        risk_score = dom_analyzer._calculate_risk_score(analysis_results)
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be high risk based on mock data

class TestVisualClassifier:
    """Test visual classifier functionality"""
    
    @pytest.fixture
    def visual_classifier(self):
        """Create visual classifier instance"""
        return create_visual_classifier()
    
    @pytest.fixture
    def mock_image(self):
        """Create mock image for testing"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_classifier_initialization(self, visual_classifier):
        """Test classifier initialization"""
        assert visual_classifier is not None
        assert visual_classifier.config is not None
        assert visual_classifier.class_names == ["phish", "benign", "suspicious"]
        assert not visual_classifier.is_trained
    
    def test_model_creation(self, visual_classifier):
        """Test model creation"""
        model_info = visual_classifier.create_model()
        
        assert "architecture" in model_info
        assert "input_shape" in model_info
        assert "num_classes" in model_info
        assert "total_params" in model_info
        assert "model_size_mb" in model_info
        assert "layers" in model_info
    
    def test_image_preprocessing(self, visual_classifier, mock_image):
        """Test image preprocessing"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test preprocessing
            processed = visual_classifier.preprocess_image(tmp_file.name)
            
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float32
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_prediction(self, visual_classifier, mock_image):
        """Test prediction functionality"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test prediction
            result = visual_classifier.predict(tmp_file.name)
            
            assert "image_path" in result
            assert "prediction" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert "processing_time_ms" in result
            
            assert result["prediction"] in visual_classifier.class_names
            assert 0.0 <= result["confidence"] <= 1.0
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_training_simulation(self, visual_classifier):
        """Test training simulation"""
        # Mock training data
        training_data = [
            ("image1.jpg", "phish"),
            ("image2.jpg", "benign"),
            ("image3.jpg", "suspicious")
        ]
        
        # Test training
        training_results = visual_classifier.train(training_data)
        
        assert "training_history" in training_results
        assert "final_metrics" in training_results
        assert "model_info" in training_results
        
        # Check training history
        history = training_results["training_history"]
        assert "epochs" in history
        assert "train_loss" in history
        assert "train_accuracy" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        
        # Check final metrics
        metrics = training_results["final_metrics"]
        assert "final_train_accuracy" in metrics
        assert "final_val_accuracy" in metrics
        assert "training_time_minutes" in metrics
        
        assert visual_classifier.is_trained

class TestVisualFeatureExtractor:
    """Test visual feature extraction functionality"""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor instance"""
        return create_visual_feature_extractor()
    
    @pytest.fixture
    def mock_image(self):
        """Create mock image for testing"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_feature_extractor_initialization(self, feature_extractor):
        """Test feature extractor initialization"""
        assert feature_extractor is not None
        assert feature_extractor.config is not None
    
    def test_feature_extraction(self, feature_extractor, mock_image):
        """Test feature extraction"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test feature extraction
            features = feature_extractor.extract_features(tmp_file.name)
            
            assert "image_path" in features
            assert "basic_info" in features
            assert "color_features" in features
            assert "texture_features" in features
            assert "shape_features" in features
            assert "edge_features" in features
            assert "histogram_features" in features
            assert "extraction_time_ms" in features
            
            # Check basic info
            basic_info = features["basic_info"]
            assert "width" in basic_info
            assert "height" in basic_info
            assert "channels" in basic_info
            assert "aspect_ratio" in basic_info
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_feature_vector_conversion(self, feature_extractor, mock_image):
        """Test feature vector conversion"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Extract features
            features = feature_extractor.extract_features(tmp_file.name)
            
            # Convert to vector
            vector = feature_extractor._features_to_vector(features)
            
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0
            
            # Cleanup
            Path(tmp_file.name).unlink()

class TestTemplateMatcher:
    """Test template matching functionality"""
    
    @pytest.fixture
    def template_matcher(self):
        """Create template matcher instance"""
        return create_template_matcher()
    
    @pytest.fixture
    def mock_image(self):
        """Create mock image for testing"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_template_matcher_initialization(self, template_matcher):
        """Test template matcher initialization"""
        assert template_matcher is not None
        assert template_matcher.config is not None
        assert hasattr(template_matcher, 'template_cache')
    
    def test_template_matching(self, template_matcher, mock_image):
        """Test template matching"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test template matching
            result = template_matcher.match_template(tmp_file.name)
            
            assert "image_path" in result
            assert "processing_time_ms" in result
            assert "timestamp" in result
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_visual_similarity(self, template_matcher, mock_image):
        """Test visual similarity calculation"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file1, \
             tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file2:
            
            # Save mock images
            Image.fromarray(mock_image).save(tmp_file1.name)
            Image.fromarray(mock_image).save(tmp_file2.name)
            
            # Test similarity calculation
            result = template_matcher.calculate_visual_similarity(tmp_file1.name, tmp_file2.name)
            
            assert "image1_path" in result
            assert "image2_path" in result
            assert "similarity_metrics" in result
            assert "overall_similarity" in result
            assert "processing_time_ms" in result
            
            # Check similarity metrics
            metrics = result["similarity_metrics"]
            assert "ssim" in metrics
            assert "histogram_similarity" in metrics
            assert "feature_similarity" in metrics
            assert "color_similarity" in metrics
            assert "texture_similarity" in metrics
            
            # Cleanup
            Path(tmp_file1.name).unlink()
            Path(tmp_file2.name).unlink()
    
    def test_template_management(self, template_matcher, mock_image):
        """Test template management"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test adding template
            result = template_matcher.add_template(tmp_file.name, "test_template")
            assert result["success"] is True
            assert "test_template" in template_matcher.template_cache
            
            # Test removing template
            result = template_matcher.remove_template("test_template")
            assert result["success"] is True
            assert "test_template" not in template_matcher.template_cache
            
            # Cleanup
            Path(tmp_file.name).unlink()

class TestVisualAnalysisAPI:
    """Test visual analysis API functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "visual-analysis-api"
        assert "components" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_processing_time_ms" in data
        assert "component_status" in data
    
    @pytest.mark.asyncio
    async def test_perform_visual_analysis(self):
        """Test visual analysis function"""
        # Mock components
        with patch('api.visual_analysis_api.screenshot_capture') as mock_capture, \
             patch('api.visual_analysis_api.dom_analyzer') as mock_dom, \
             patch('api.visual_analysis_api.visual_classifier') as mock_classifier, \
             patch('api.visual_analysis_api.feature_extractor') as mock_extractor, \
             patch('api.visual_analysis_api.template_matcher') as mock_matcher:
            
            # Setup mocks
            mock_capture.context = AsyncMock()
            mock_page = AsyncMock()
            mock_capture.context.new_page.return_value = mock_page
            mock_page.goto = AsyncMock()
            mock_page.close = AsyncMock()
            
            mock_capture.capture_screenshot = AsyncMock(return_value={
                "success": True,
                "screenshot_path": "test.png"
            })
            
            mock_dom.analyze_page = AsyncMock(return_value={
                "risk_score": 0.5,
                "url": "https://test.com"
            })
            
            mock_classifier.predict.return_value = {
                "prediction": "benign",
                "confidence": 0.8
            }
            
            mock_extractor.extract_features.return_value = {
                "color_features": {"brightness": 0.5}
            }
            
            mock_matcher.match_template.return_value = {
                "best_match": None
            }
            
            # Test analysis
            result = await perform_visual_analysis("https://test.com")
            
            assert "url" in result
            assert "timestamp" in result
            assert "components" in result
            assert "overall_risk_score" in result
            assert "risk_level" in result

class TestIntegration:
    """Integration tests for the visual pipeline"""
    
    @pytest.fixture
    def mock_image(self):
        """Create mock image for testing"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_end_to_end_analysis(self, mock_image):
        """Test end-to-end analysis pipeline"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save mock image
            Image.fromarray(mock_image).save(tmp_file.name)
            
            # Test each component
            classifier = create_visual_classifier()
            feature_extractor = create_visual_feature_extractor()
            template_matcher = create_template_matcher()
            
            # Test classifier
            cnn_result = classifier.predict(tmp_file.name)
            assert cnn_result["prediction"] in ["phish", "benign", "suspicious"]
            
            # Test feature extraction
            features = feature_extractor.extract_features(tmp_file.name)
            assert "basic_info" in features
            
            # Test template matching
            template_result = template_matcher.match_template(tmp_file.name)
            assert "image_path" in template_result
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_configuration_consistency(self):
        """Test configuration consistency across components"""
        screenshot_config = get_config("screenshot")
        dom_config = get_config("dom")
        cnn_config = get_config("cnn")
        feature_config = get_config("feature")
        template_config = get_config("template")
        api_config = get_config("api")
        
        # Check that all configs are loaded
        assert screenshot_config is not None
        assert dom_config is not None
        assert cnn_config is not None
        assert feature_config is not None
        assert template_config is not None
        assert api_config is not None
        
        # Check specific config values
        assert cnn_config.num_classes == 3
        assert cnn_config.input_size == (224, 224, 3)
        assert template_config.similarity_threshold >= 0.0
        assert template_config.similarity_threshold <= 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])