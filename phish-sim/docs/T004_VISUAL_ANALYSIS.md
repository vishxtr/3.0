# T004 - Visual/DOM Structural Analyzer

## Overview

T004 implements a comprehensive visual analysis pipeline for phishing detection using Playwright screenshot capture, DOM structure analysis, CNN visual classification, feature extraction, and template matching. This component provides real-time visual analysis capabilities to detect phishing websites through multiple detection methods.

## Architecture

### Components

1. **Screenshot Capture System** (`screenshot_capture.py`)
   - Playwright-based web automation
   - Headless browser screenshot capture
   - Page information extraction
   - DOM structure analysis

2. **DOM Analysis** (`dom_analysis.py`)
   - HTML structure analysis
   - Form and input field detection
   - Link and external resource analysis
   - Security indicator detection
   - Phishing pattern recognition

3. **CNN Visual Classifier** (`cnn_models/visual_classifier.py`)
   - Simple CNN architecture for visual phishing detection
   - Real-time image classification
   - Model training and evaluation
   - Performance optimization

4. **Visual Feature Extraction** (`feature_extraction/visual_features.py`)
   - Color analysis and histogram extraction
   - Texture and edge feature detection
   - Shape and contour analysis
   - CNN-based feature extraction

5. **Template Matching** (`template_matching.py`)
   - Visual similarity analysis
   - Template database management
   - Multi-metric similarity calculation
   - Brand impersonation detection

6. **Visual Analysis API** (`api/visual_analysis_api.py`)
   - FastAPI-based REST API
   - Real-time analysis endpoints
   - Batch processing capabilities
   - Performance monitoring

## Features

### Screenshot Capture
- **Browser Automation**: Chromium, Firefox, WebKit support
- **Viewport Control**: Configurable resolution and device scale
- **Page Loading**: Network idle detection and element waiting
- **Security**: HTTPS error bypass and CSP handling
- **Performance**: Concurrent capture with rate limiting

### DOM Analysis
- **Structure Analysis**: Element counting and hierarchy analysis
- **Form Detection**: Input field analysis and suspicious pattern detection
- **Link Analysis**: External link detection and redirect analysis
- **Content Analysis**: Phishing keyword detection and urgency indicators
- **Security Analysis**: SSL certificate validation and mixed content detection
- **Risk Scoring**: Comprehensive risk assessment algorithm

### CNN Classification
- **Architecture**: Simple CNN with 3-layer convolution + dense layers
- **Input Processing**: 224x224x3 RGB image normalization
- **Classes**: Phish, Benign, Suspicious classification
- **Performance**: <100ms inference time target
- **Model Size**: <20MB optimized model

### Feature Extraction
- **Color Features**: RGB/HSV analysis, dominant color detection
- **Texture Features**: LBP, Gabor filters, GLCM analysis
- **Shape Features**: Contour detection, edge analysis
- **Edge Features**: Sobel, Laplacian, Canny edge detection
- **CNN Features**: Deep learning feature extraction
- **Histogram Analysis**: Color distribution and entropy

### Template Matching
- **Similarity Metrics**: SSIM, histogram correlation, feature matching
- **Template Database**: Known phishing template storage
- **Brand Detection**: Logo and visual element matching
- **Threshold Control**: Configurable similarity thresholds
- **Performance**: <100ms matching time

## Configuration

### Screenshot Configuration
```python
@dataclass
class ScreenshotConfig:
    browser_type: str = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    wait_timeout: int = 30000
    load_state: str = "networkidle"
```

### DOM Analysis Configuration
```python
@dataclass
class DOMAnalysisConfig:
    extract_forms: bool = True
    extract_links: bool = True
    analyze_text_content: bool = True
    check_external_resources: bool = True
    detect_suspicious_patterns: bool = True
```

### CNN Model Configuration
```python
@dataclass
class CNNModelConfig:
    input_size: tuple = (224, 224, 3)
    num_classes: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    target_accuracy: float = 0.85
    max_inference_time_ms: float = 100.0
```

## API Endpoints

### Core Analysis
- `POST /analyze` - Analyze URL for phishing indicators
- `POST /analyze/image` - Analyze uploaded image
- `POST /analyze/batch` - Batch URL analysis

### Template Management
- `GET /templates` - Get available templates
- `POST /templates` - Add new template
- `DELETE /templates/{name}` - Remove template

### System Information
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `GET /model/info` - Model information

## Performance Targets

| Component | Target Time | Memory Usage | Accuracy |
|-----------|-------------|--------------|----------|
| Screenshot Capture | < 5 seconds | 200 MB | N/A |
| DOM Analysis | < 200 ms | 50 MB | N/A |
| CNN Classification | < 100 ms | 100 MB | >85% |
| Feature Extraction | < 200 ms | 100 MB | N/A |
| Template Matching | < 100 ms | 50 MB | N/A |
| **Overall Pipeline** | **< 10 seconds** | **< 500 MB** | **>85%** |

## Detection Capabilities

### Visual Indicators
- Brand impersonation detection
- Urgency indicator recognition
- Suspicious layout patterns
- Color scheme analysis
- Logo similarity matching

### DOM Indicators
- Form field analysis
- External link detection
- Content pattern matching
- Security configuration checks
- JavaScript obfuscation detection

### Template Matching
- Known phishing template database
- Visual similarity scoring
- Brand spoofing detection
- Layout pattern matching
- Multi-scale template matching

### CNN Detection
- Visual pattern recognition
- Layout analysis
- Content classification
- Feature-based detection
- Ensemble decision making

## Usage Examples

### Basic URL Analysis
```python
from visual.api.visual_analysis_api import perform_visual_analysis

# Analyze a URL
result = await perform_visual_analysis(
    url="https://suspicious-site.com",
    enable_screenshot=True,
    enable_dom_analysis=True,
    enable_cnn_analysis=True,
    enable_template_matching=True
)

print(f"Risk Score: {result['overall_risk_score']}")
print(f"Risk Level: {result['risk_level']}")
```

### Image Analysis
```python
from visual.cnn_models.visual_classifier import create_visual_classifier

# Classify an image
classifier = create_visual_classifier()
result = classifier.predict("screenshot.png")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Template Matching
```python
from visual.template_matching import create_template_matcher

# Match against templates
matcher = create_template_matcher()
result = matcher.match_template("screenshot.png")

print(f"Best Match: {result['best_match']['template_name']}")
print(f"Similarity: {result['best_score']}")
```

## Testing

### Unit Tests
```bash
cd visual
python -m pytest tests/test_visual_pipeline.py -v
```

### Integration Tests
```bash
# Test full pipeline
python -m pytest tests/test_visual_pipeline.py::TestIntegration -v
```

### Performance Tests
```bash
# Benchmark performance
python tests/benchmark_visual_pipeline.py
```

## Dependencies

### Core Dependencies
- `playwright` - Web automation and screenshot capture
- `opencv-python` - Image processing and computer vision
- `torch` - Deep learning framework
- `scikit-learn` - Machine learning utilities
- `fastapi` - Web API framework

### Optional Dependencies
- `tensorflow` - Alternative deep learning framework
- `selenium` - Alternative web automation
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP client

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Run the API
python api/visual_analysis_api.py
```

## Monitoring and Metrics

### Performance Metrics
- Request processing time
- Component execution time
- Memory usage tracking
- Error rate monitoring
- Throughput measurement

### Health Checks
- Component availability
- Model loading status
- Template database status
- Browser automation health
- API endpoint responsiveness

## Security Considerations

### Input Validation
- URL format validation
- File type restrictions
- Size limitations
- Malicious content filtering

### Resource Protection
- Rate limiting
- Memory usage limits
- CPU usage monitoring
- Disk space management

### Privacy
- No data persistence
- Secure image handling
- Temporary file cleanup
- Log sanitization

## Troubleshooting

### Common Issues

1. **Browser Launch Failures**
   - Check Playwright installation
   - Verify browser dependencies
   - Check system permissions

2. **Memory Issues**
   - Reduce batch sizes
   - Enable garbage collection
   - Monitor memory usage

3. **Performance Degradation**
   - Check system resources
   - Optimize model parameters
   - Enable caching

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug flags
python api/visual_analysis_api.py --debug
```

## Future Enhancements

### Planned Features
- Advanced CNN architectures (ResNet, EfficientNet)
- Real-time video analysis
- Mobile app integration
- Cloud deployment support
- Advanced template learning

### Performance Improvements
- Model quantization
- GPU acceleration
- Distributed processing
- Caching optimization
- Pipeline parallelization

## Integration

### With T003 (ML Pipeline)
- Feature vector integration
- Model ensemble methods
- Cross-validation results
- Performance comparison

### With T005 (Real-time Inference)
- API endpoint integration
- Performance monitoring
- Error handling
- Load balancing

### With T006 (Evaluation Harness)
- Test dataset integration
- Performance benchmarking
- Accuracy measurement
- False positive analysis

## Conclusion

T004 provides a comprehensive visual analysis pipeline for phishing detection with multiple detection methods, real-time performance, and extensive configurability. The system achieves the target performance goals while maintaining high accuracy and reliability.

The modular architecture allows for easy integration with other components and future enhancements, making it a robust foundation for visual phishing detection in the Phish-Sim system.