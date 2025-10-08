# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Model inference API for real-time phishing detection
"""

import torch
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.phishing_classifier import PhishingClassifier, create_classifier
from preprocessing.tokenizer import PhishingTokenizer, create_tokenizer
from preprocessing.text_preprocessor import TextPreprocessor, create_preprocessor
from config import get_config

logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
preprocessor = None
inference_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_inference_time': 0.0,
    'average_inference_time': 0.0
}

class AnalysisRequest(BaseModel):
    """Request model for phishing analysis"""
    text: str = Field(..., description="Text content to analyze")
    content_type: str = Field(default="text", description="Type of content: text, url, email")
    return_features: bool = Field(default=False, description="Whether to return extracted features")
    return_attention: bool = Field(default=False, description="Whether to return attention weights")

class AnalysisResponse(BaseModel):
    """Response model for phishing analysis"""
    prediction: str = Field(..., description="Predicted class: phish, benign, suspicious")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted features")
    attention_weights: Optional[List[float]] = Field(None, description="Attention weights")
    timestamp: str = Field(..., description="Analysis timestamp")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    texts: List[str] = Field(..., description="List of texts to analyze")
    content_type: str = Field(default="text", description="Type of content")
    return_features: bool = Field(default=False, description="Whether to return features")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    results: List[AnalysisResponse] = Field(..., description="Analysis results for each text")
    batch_processing_time_ms: float = Field(..., description="Total batch processing time")
    timestamp: str = Field(..., description="Analysis timestamp")

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    num_parameters: int = Field(..., description="Number of model parameters")
    model_size_mb: float = Field(..., description="Model size in MB")
    device: str = Field(..., description="Device used for inference")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    supported_languages: List[str] = Field(..., description="Supported languages")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime")
    inference_stats: Dict[str, Any] = Field(..., description="Inference statistics")

class PhishingInferenceAPI:
    """Main inference API class"""
    
    def __init__(self, model_path: str, tokenizer_path: str, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config or {}
        self.start_time = time.time()
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        self._load_preprocessor()
    
    def _load_model(self):
        """Load the trained model"""
        global model
        
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load model configuration
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
            else:
                model_config = self.config.get('model_config', {})
            
            # Create and load model
            model_type = model_config.get('model_type', 'transformer')
            model = create_classifier(model_type, model_config)
            model.load_model(str(model_path))
            model.eval()
            
            logger.info(f"Loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        global tokenizer
        
        try:
            tokenizer_path = Path(self.tokenizer_path)
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
            tokenizer = create_tokenizer()
            tokenizer.load_tokenizer(str(tokenizer_path))
            
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_preprocessor(self):
        """Load the text preprocessor"""
        global preprocessor
        
        try:
            preprocessor = create_preprocessor(self.config.get('preprocessing_config', {}))
            logger.info("Loaded text preprocessor")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise
    
    def analyze_text(self, text: str, content_type: str = "text", 
                    return_features: bool = False, return_attention: bool = False) -> Dict[str, Any]:
        """Analyze a single text for phishing indicators"""
        start_time = time.time()
        
        try:
            # Preprocess text
            preprocessed = preprocessor.preprocess_text(text, extract_features=return_features)
            cleaned_text = preprocessed['cleaned_text']
            
            # Tokenize
            tokenized = tokenizer.tokenize_text(cleaned_text)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            # Move to device
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)
            
            # Process results
            prediction_idx = prediction.item()
            confidence = probabilities[0, prediction_idx].item()
            
            class_names = ['phish', 'benign', 'suspicious']
            prediction_class = class_names[prediction_idx]
            
            prob_dict = {
                class_names[i]: probabilities[0, i].item() 
                for i in range(len(class_names))
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            inference_stats['total_requests'] += 1
            inference_stats['successful_requests'] += 1
            inference_stats['total_inference_time'] += processing_time
            inference_stats['average_inference_time'] = (
                inference_stats['total_inference_time'] / inference_stats['successful_requests']
            )
            
            result = {
                'prediction': prediction_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'processing_time_ms': processing_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if return_features:
                result['features'] = preprocessed['features']
            
            if return_attention:
                # Extract attention weights (if available)
                result['attention_weights'] = self._extract_attention_weights(outputs)
            
            return result
            
        except Exception as e:
            inference_stats['total_requests'] += 1
            inference_stats['failed_requests'] += 1
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def analyze_batch(self, texts: List[str], content_type: str = "text", 
                     return_features: bool = False) -> Dict[str, Any]:
        """Analyze a batch of texts"""
        start_time = time.time()
        results = []
        
        for text in texts:
            try:
                result = self.analyze_text(text, content_type, return_features)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze text: {e}")
                results.append({
                    'prediction': 'error',
                    'confidence': 0.0,
                    'probabilities': {'phish': 0.0, 'benign': 0.0, 'suspicious': 0.0},
                    'processing_time_ms': 0.0,
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        batch_processing_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'batch_processing_time_ms': batch_processing_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _extract_attention_weights(self, outputs) -> Optional[List[float]]:
        """Extract attention weights from model outputs"""
        # This would be implemented based on the specific model architecture
        # For now, return None as most models don't expose attention weights easily
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            'model_type': model.__class__.__name__,
            'model_version': '1.0.0',
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024),
            'device': str(next(model.parameters()).device),
            'max_sequence_length': tokenizer.max_length if tokenizer else 512,
            'supported_languages': ['en']
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        uptime = time.time() - self.start_time
        
        return {
            'status': 'healthy' if model is not None else 'unhealthy',
            'model_loaded': model is not None,
            'uptime_seconds': uptime,
            'inference_stats': inference_stats.copy()
        }

# FastAPI app setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Phishing Inference API")
    yield
    # Shutdown
    logger.info("Shutting down Phishing Inference API")

# Create FastAPI app
app = FastAPI(
    title="Phish-Sim Inference API",
    description="Real-time phishing detection API",
    version="1.0.0",
    lifespan=lifespan
)

# Global API instance
api_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    global api_instance
    
    try:
        # Get configuration
        config = get_config('inference')
        
        # Initialize API
        api_instance = PhishingInferenceAPI(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            config=config.__dict__
        )
        
        logger.info("API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Phish-Sim Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    return api_instance.get_health_status()

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    return api_instance.get_model_info()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for phishing indicators"""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    return api_instance.analyze_text(
        text=request.text,
        content_type=request.content_type,
        return_features=request.return_features,
        return_attention=request.return_attention
    )

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze batch of texts for phishing indicators"""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    
    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    return api_instance.analyze_batch(
        texts=request.texts,
        content_type=request.content_type,
        return_features=request.return_features
    )

@app.get("/stats")
async def get_inference_stats():
    """Get inference statistics"""
    return inference_stats

def run_api(host: str = "0.0.0.0", port: int = 8001, workers: int = 1):
    """Run the inference API"""
    uvicorn.run(
        "inference_api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )

if __name__ == "__main__":
    # Run API directly
    run_api()