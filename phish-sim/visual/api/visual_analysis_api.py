# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Visual analysis API for phishing detection
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import visual analysis components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from screenshot_capture import ScreenshotCapture, create_screenshot_capture
from dom_analysis import DOMAnalyzer, create_dom_analyzer
from cnn_models.visual_classifier import SimpleCNNClassifier, create_visual_classifier
from feature_extraction.visual_features import VisualFeatureExtractor, create_visual_feature_extractor
from template_matching import TemplateMatcher, create_template_matcher
from config import get_config, VisualAPIConfig

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phish-Sim Visual Analysis API",
    description="API for visual phishing detection and analysis",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
screenshot_capture = None
dom_analyzer = None
visual_classifier = None
feature_extractor = None
template_matcher = None

# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for visual analysis"""
    url: str = Field(..., description="URL to analyze")
    enable_screenshot: bool = Field(True, description="Enable screenshot capture")
    enable_dom_analysis: bool = Field(True, description="Enable DOM analysis")
    enable_cnn_analysis: bool = Field(True, description="Enable CNN analysis")
    enable_template_matching: bool = Field(True, description="Enable template matching")
    wait_for_element: Optional[str] = Field(None, description="Wait for specific element")
    timeout: Optional[int] = Field(30, description="Analysis timeout in seconds")

class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis"""
    image_path: str = Field(..., description="Path to image file")
    analysis_type: str = Field("all", description="Type of analysis: all, features, cnn, template")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    urls: List[str] = Field(..., description="List of URLs to analyze")
    max_concurrent: int = Field(3, description="Maximum concurrent analyses")
    analysis_options: Dict[str, bool] = Field(default_factory=dict, description="Analysis options")

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    url: str
    success: bool
    analysis_results: Dict[str, Any]
    processing_time_ms: float
    timestamp: float
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    components: Dict[str, bool]

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_processing_time_ms: float
    component_status: Dict[str, bool]

# Global metrics
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "processing_times": [],
    "component_status": {}
}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global screenshot_capture, dom_analyzer, visual_classifier, feature_extractor, template_matcher
    
    try:
        logger.info("Initializing visual analysis components...")
        
        # Initialize components
        screenshot_capture = create_screenshot_capture()
        dom_analyzer = create_dom_analyzer()
        visual_classifier = create_visual_classifier()
        feature_extractor = create_visual_feature_extractor()
        template_matcher = create_template_matcher()
        
        # Update component status
        metrics["component_status"] = {
            "screenshot_capture": screenshot_capture is not None,
            "dom_analyzer": dom_analyzer is not None,
            "visual_classifier": visual_classifier is not None,
            "feature_extractor": feature_extractor is not None,
            "template_matcher": template_matcher is not None
        }
        
        logger.info("Visual analysis components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global screenshot_capture
    
    try:
        if screenshot_capture:
            await screenshot_capture.close_browser()
        logger.info("Visual analysis API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Phish-Sim Visual Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        service="visual-analysis-api",
        version="0.1.0",
        components=metrics["component_status"]
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API metrics"""
    avg_processing_time = (
        sum(metrics["processing_times"]) / len(metrics["processing_times"])
        if metrics["processing_times"] else 0.0
    )
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        successful_requests=metrics["successful_requests"],
        failed_requests=metrics["failed_requests"],
        average_processing_time_ms=avg_processing_time,
        component_status=metrics["component_status"]
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_url(request: AnalysisRequest):
    """Analyze a URL for phishing indicators"""
    global metrics
    
    start_time = time.time()
    metrics["total_requests"] += 1
    
    try:
        logger.info(f"Starting analysis for URL: {request.url}")
        
        # Validate URL
        if not request.url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Perform analysis
        analysis_results = await perform_visual_analysis(
            url=request.url,
            enable_screenshot=request.enable_screenshot,
            enable_dom_analysis=request.enable_dom_analysis,
            enable_cnn_analysis=request.enable_cnn_analysis,
            enable_template_matching=request.enable_template_matching,
            wait_for_element=request.wait_for_element,
            timeout=request.timeout
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        metrics["processing_times"].append(processing_time)
        metrics["successful_requests"] += 1
        
        return AnalysisResponse(
            url=request.url,
            success=True,
            analysis_results=analysis_results,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.url}: {e}")
        metrics["failed_requests"] += 1
        
        return AnalysisResponse(
            url=request.url,
            success=False,
            analysis_results={},
            processing_time_ms=(time.time() - start_time) * 1000,
            timestamp=time.time(),
            error=str(e)
        )

@app.post("/analyze/image", response_model=Dict[str, Any])
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze an image for phishing indicators"""
    try:
        logger.info(f"Starting image analysis for: {request.image_path}")
        
        if not Path(request.image_path).exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        analysis_results = {}
        
        if request.analysis_type in ["all", "features"]:
            # Extract visual features
            features = feature_extractor.extract_features(request.image_path)
            analysis_results["features"] = features
        
        if request.analysis_type in ["all", "cnn"]:
            # CNN classification
            cnn_result = visual_classifier.predict(request.image_path)
            analysis_results["cnn_prediction"] = cnn_result
        
        if request.analysis_type in ["all", "template"]:
            # Template matching
            template_result = template_matcher.match_template(request.image_path)
            analysis_results["template_matching"] = template_result
        
        return {
            "image_path": request.image_path,
            "analysis_type": request.analysis_type,
            "results": analysis_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Image analysis failed for {request.image_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=List[AnalysisResponse])
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple URLs in batch"""
    try:
        logger.info(f"Starting batch analysis for {len(request.urls)} URLs")
        
        # Limit concurrent analyses
        max_concurrent = min(request.max_concurrent, 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(url):
            async with semaphore:
                return await analyze_url(AnalysisRequest(
                    url=url,
                    enable_screenshot=request.analysis_options.get("screenshot", True),
                    enable_dom_analysis=request.analysis_options.get("dom", True),
                    enable_cnn_analysis=request.analysis_options.get("cnn", True),
                    enable_template_matching=request.analysis_options.get("template", True)
                ))
        
        # Run analyses concurrently
        tasks = [analyze_with_semaphore(url) for url in request.urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AnalysisResponse(
                    url=request.urls[i],
                    success=False,
                    analysis_results={},
                    processing_time_ms=0.0,
                    timestamp=time.time(),
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/image", response_model=Dict[str, Any])
async def upload_image(file: UploadFile = File(...)):
    """Upload and analyze an image"""
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze uploaded image
        analysis_results = {}
        
        # Extract features
        features = feature_extractor.extract_features(str(file_path))
        analysis_results["features"] = features
        
        # CNN classification
        cnn_result = visual_classifier.predict(str(file_path))
        analysis_results["cnn_prediction"] = cnn_result
        
        # Template matching
        template_result = template_matcher.match_template(str(file_path))
        analysis_results["template_matching"] = template_result
        
        return {
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "analysis_results": analysis_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates", response_model=Dict[str, Any])
async def get_templates():
    """Get available templates"""
    try:
        template_info = template_matcher.get_template_info()
        return template_info
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/templates", response_model=Dict[str, Any])
async def add_template(template_name: str, file: UploadFile = File(...)):
    """Add a new template"""
    try:
        # Save template file
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        template_path = template_dir / f"{template_name}.png"
        with open(template_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add to template matcher
        result = template_matcher.add_template(str(template_path), template_name)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to add template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/templates/{template_name}", response_model=Dict[str, Any])
async def remove_template(template_name: str):
    """Remove a template"""
    try:
        result = template_matcher.remove_template(template_name)
        return result
    except Exception as e:
        logger.error(f"Failed to remove template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information"""
    try:
        model_info = visual_classifier.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train", response_model=Dict[str, Any])
async def train_model(background_tasks: BackgroundTasks, training_data: List[Dict[str, str]]):
    """Train the model (background task)"""
    try:
        # Start training in background
        background_tasks.add_task(train_model_background, training_data)
        
        return {
            "message": "Training started in background",
            "training_samples": len(training_data),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(training_data: List[Dict[str, str]]):
    """Background task for model training"""
    try:
        logger.info(f"Starting background training with {len(training_data)} samples")
        
        # Convert training data format
        formatted_data = [(item["image_path"], item["label"]) for item in training_data]
        
        # Train model
        training_results = visual_classifier.train(formatted_data)
        
        # Save model
        visual_classifier.save_model()
        
        logger.info("Background training completed successfully")
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")

async def perform_visual_analysis(
    url: str,
    enable_screenshot: bool = True,
    enable_dom_analysis: bool = True,
    enable_cnn_analysis: bool = True,
    enable_template_matching: bool = True,
    wait_for_element: Optional[str] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Perform comprehensive visual analysis"""
    analysis_results = {
        "url": url,
        "timestamp": time.time(),
        "components": {}
    }
    
    try:
        # Start browser if needed
        if not screenshot_capture.context:
            await screenshot_capture.start_browser()
        
        # Create new page
        page = await screenshot_capture.context.new_page()
        
        try:
            # Navigate to URL
            await page.goto(url, wait_until="networkidle", timeout=timeout * 1000 if timeout else 30000)
            
            # Wait for specific element if requested
            if wait_for_element:
                try:
                    await page.wait_for_selector(wait_for_element, timeout=5000)
                except:
                    logger.warning(f"Element '{wait_for_element}' not found")
            
            # Screenshot capture
            if enable_screenshot:
                screenshot_result = await screenshot_capture.capture_screenshot(url, wait_for_element)
                analysis_results["components"]["screenshot"] = screenshot_result
                
                # CNN analysis on screenshot
                if enable_cnn_analysis and screenshot_result.get("success"):
                    cnn_result = visual_classifier.predict(screenshot_result["screenshot_path"])
                    analysis_results["components"]["cnn_analysis"] = cnn_result
                
                # Feature extraction
                if screenshot_result.get("success"):
                    features = feature_extractor.extract_features(screenshot_result["screenshot_path"])
                    analysis_results["components"]["visual_features"] = features
                
                # Template matching
                if enable_template_matching and screenshot_result.get("success"):
                    template_result = template_matcher.match_template(screenshot_result["screenshot_path"])
                    analysis_results["components"]["template_matching"] = template_result
            
            # DOM analysis
            if enable_dom_analysis:
                dom_result = await dom_analyzer.analyze_page(page, url)
                analysis_results["components"]["dom_analysis"] = dom_result
            
            # Calculate overall risk score
            risk_score = calculate_overall_risk_score(analysis_results["components"])
            analysis_results["overall_risk_score"] = risk_score
            analysis_results["risk_level"] = get_risk_level(risk_score)
            
        finally:
            await page.close()
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Visual analysis failed for {url}: {e}")
        analysis_results["error"] = str(e)
        return analysis_results

def calculate_overall_risk_score(components: Dict[str, Any]) -> float:
    """Calculate overall risk score from analysis components"""
    risk_score = 0.0
    weight_sum = 0.0
    
    # DOM analysis risk
    if "dom_analysis" in components:
        dom_risk = components["dom_analysis"].get("risk_score", 0.0)
        risk_score += dom_risk * 0.4
        weight_sum += 0.4
    
    # CNN analysis risk
    if "cnn_analysis" in components:
        cnn_result = components["cnn_analysis"]
        if cnn_result.get("prediction") == "phish":
            cnn_risk = cnn_result.get("confidence", 0.0)
        else:
            cnn_risk = 1.0 - cnn_result.get("confidence", 0.0)
        risk_score += cnn_risk * 0.3
        weight_sum += 0.3
    
    # Template matching risk
    if "template_matching" in components:
        template_result = components["template_matching"]
        if "best_match" in template_result and template_result["best_match"]:
            template_risk = template_result["best_match"].get("similarity_score", 0.0)
            risk_score += template_risk * 0.2
            weight_sum += 0.2
    
    # Visual features risk
    if "visual_features" in components:
        features = components["visual_features"]
        # Simple heuristic based on features
        if "color_features" in features and "error" not in features["color_features"]:
            color_features = features["color_features"]
            if color_features.get("brightness", 0.5) < 0.3:  # Dark images might be suspicious
                risk_score += 0.1
                weight_sum += 0.1
    
    # Normalize by weight sum
    if weight_sum > 0:
        risk_score = risk_score / weight_sum
    
    return min(risk_score, 1.0)

def get_risk_level(risk_score: float) -> str:
    """Get risk level from score"""
    if risk_score >= 0.8:
        return "HIGH"
    elif risk_score >= 0.6:
        return "MEDIUM"
    elif risk_score >= 0.4:
        return "LOW"
    else:
        return "SAFE"

if __name__ == "__main__":
    # Get API configuration
    api_config = get_config("api")
    
    # Run the API
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        workers=api_config.workers
    )