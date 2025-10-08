# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Configuration for visual analysis components
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Base paths
BASE_DIR = Path(__file__).parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
DOM_ANALYSIS_DIR = BASE_DIR / "dom_analysis"
CNN_MODELS_DIR = BASE_DIR / "cnn_models"
FEATURE_EXTRACTION_DIR = BASE_DIR / "feature_extraction"
TEMPLATES_DIR = BASE_DIR / "templates"
API_DIR = BASE_DIR / "api"

# Ensure directories exist
for directory in [SCREENSHOTS_DIR, DOM_ANALYSIS_DIR, CNN_MODELS_DIR, 
                 FEATURE_EXTRACTION_DIR, TEMPLATES_DIR, API_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture"""
    
    # Browser settings
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    device_scale_factor: float = 1.0
    
    # Screenshot settings
    screenshot_format: str = "png"  # png, jpeg
    screenshot_quality: int = 90  # for jpeg
    full_page: bool = True
    capture_beyond_viewport: bool = True
    
    # Timing settings
    wait_timeout: int = 30000  # milliseconds
    navigation_timeout: int = 30000
    load_state: str = "networkidle"  # load, domcontentloaded, networkidle
    
    # Security settings
    ignore_https_errors: bool = True
    bypass_csp: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Output settings
    output_dir: str = str(SCREENSHOTS_DIR)
    filename_template: str = "{timestamp}_{domain}_{hash}.{format}"
    max_screenshots: int = 1000

@dataclass
class DOMAnalysisConfig:
    """Configuration for DOM analysis"""
    
    # DOM extraction
    extract_forms: bool = True
    extract_links: bool = True
    extract_images: bool = True
    extract_scripts: bool = True
    extract_styles: bool = True
    
    # Content analysis
    analyze_text_content: bool = True
    analyze_meta_tags: bool = True
    analyze_structured_data: bool = True
    analyze_accessibility: bool = True
    
    # Security analysis
    check_external_resources: bool = True
    analyze_redirects: bool = True
    detect_suspicious_patterns: bool = True
    check_ssl_certificate: bool = True
    
    # Output settings
    output_dir: str = str(DOM_ANALYSIS_DIR)
    save_dom_tree: bool = True
    save_extracted_elements: bool = True
    max_dom_size: int = 1000000  # characters

@dataclass
class CNNModelConfig:
    """Configuration for CNN visual analysis"""
    
    # Model architecture
    model_type: str = "simple_cnn"  # simple_cnn, resnet, efficientnet
    input_size: tuple = (224, 224, 3)
    num_classes: int = 3  # phish, benign, suspicious
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    validation_split: float = 0.2
    
    # Data augmentation
    enable_augmentation: bool = True
    rotation_range: int = 20
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True
    zoom_range: float = 0.1
    
    # Model optimization
    use_pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.5
    
    # Performance targets
    target_accuracy: float = 0.85
    target_precision: float = 0.80
    target_recall: float = 0.80
    target_f1: float = 0.80
    max_inference_time_ms: float = 100.0
    
    # Model paths
    model_save_path: str = str(CNN_MODELS_DIR / "visual_classifier.h5")
    weights_save_path: str = str(CNN_MODELS_DIR / "visual_weights.h5")

@dataclass
class FeatureExtractionConfig:
    """Configuration for visual feature extraction"""
    
    # Image preprocessing
    resize_method: str = "resize"  # resize, crop, pad
    normalization: str = "imagenet"  # imagenet, custom, none
    color_space: str = "rgb"  # rgb, grayscale, hsv, lab
    
    # Feature extraction methods
    extract_histogram: bool = True
    extract_texture: bool = True
    extract_color: bool = True
    extract_shape: bool = True
    extract_edges: bool = True
    
    # Deep learning features
    use_cnn_features: bool = True
    cnn_layer: str = "fc1"  # conv1, conv2, fc1, fc2
    feature_dimension: int = 512
    
    # Traditional features
    use_sift: bool = False
    use_orb: bool = False
    use_hog: bool = True
    
    # Output settings
    output_dir: str = str(FEATURE_EXTRACTION_DIR)
    save_features: bool = True
    feature_format: str = "npy"  # npy, json, csv

@dataclass
class TemplateMatchingConfig:
    """Configuration for template matching"""
    
    # Template database
    template_dir: str = str(TEMPLATES_DIR)
    template_formats: List[str] = None  # Will be set to ["png", "jpg", "jpeg"]
    
    # Matching parameters
    similarity_threshold: float = 0.8
    matching_method: str = "cv2.TM_CCOEFF_NORMED"
    scale_range: tuple = (0.5, 2.0)
    rotation_range: tuple = (-15, 15)
    
    # Template categories
    phishing_templates: List[str] = None  # Will be populated
    legitimate_templates: List[str] = None  # Will be populated
    
    # Performance settings
    max_templates: int = 100
    parallel_processing: bool = True
    cache_templates: bool = True

@dataclass
class VisualAPIConfig:
    """Configuration for visual analysis API"""
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 1
    timeout: int = 60
    
    # Analysis settings
    enable_screenshot: bool = True
    enable_dom_analysis: bool = True
    enable_cnn_analysis: bool = True
    enable_template_matching: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Security settings
    allowed_domains: List[str] = None  # None means all domains
    blocked_domains: List[str] = None
    max_url_length: int = 2048

# Default configurations
DEFAULT_SCREENSHOT_CONFIG = ScreenshotConfig()
DEFAULT_DOM_CONFIG = DOMAnalysisConfig()
DEFAULT_CNN_CONFIG = CNNModelConfig()
DEFAULT_FEATURE_CONFIG = FeatureExtractionConfig()
DEFAULT_TEMPLATE_CONFIG = TemplateMatchingConfig()
DEFAULT_API_CONFIG = VisualAPIConfig()

# Initialize template config lists
DEFAULT_TEMPLATE_CONFIG.template_formats = ["png", "jpg", "jpeg"]
DEFAULT_TEMPLATE_CONFIG.phishing_templates = [
    "bank_login", "paypal_verify", "amazon_security", "microsoft_alert",
    "apple_id_lock", "google_security", "facebook_verify", "netflix_suspend"
]
DEFAULT_TEMPLATE_CONFIG.legitimate_templates = [
    "bank_official", "paypal_official", "amazon_official", "microsoft_official",
    "apple_official", "google_official", "facebook_official", "netflix_official"
]

# Visual analysis thresholds
VISUAL_THRESHOLDS = {
    "screenshot_quality": {
        "excellent": 0.95,
        "good": 0.85,
        "acceptable": 0.75,
        "poor": 0.65
    },
    "dom_complexity": {
        "simple": 100,
        "moderate": 500,
        "complex": 1000,
        "very_complex": 5000
    },
    "visual_similarity": {
        "identical": 0.95,
        "very_similar": 0.85,
        "similar": 0.75,
        "different": 0.50
    },
    "phishing_indicators": {
        "high_risk": 0.8,
        "medium_risk": 0.6,
        "low_risk": 0.4,
        "safe": 0.2
    }
}

# Common phishing visual patterns
PHISHING_PATTERNS = {
    "urgency_indicators": [
        "red_background", "exclamation_marks", "warning_icons",
        "countdown_timers", "urgent_text", "alert_banners"
    ],
    "brand_impersonation": [
        "logo_similarity", "color_scheme_match", "font_similarity",
        "layout_similarity", "button_style_match"
    ],
    "suspicious_elements": [
        "popup_forms", "overlay_forms", "fake_ssl_indicators",
        "suspicious_redirects", "hidden_elements", "obfuscated_content"
    ],
    "visual_deception": [
        "homograph_domains", "similar_logos", "fake_certificates",
        "spoofed_branding", "misleading_icons"
    ]
}

# Browser automation settings
BROWSER_SETTINGS = {
    "chromium": {
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor"
        ],
        "viewport": {"width": 1920, "height": 1080}
    },
    "firefox": {
        "args": [
            "--no-sandbox",
            "--disable-dev-shm-usage"
        ],
        "viewport": {"width": 1920, "height": 1080}
    },
    "webkit": {
        "args": [],
        "viewport": {"width": 1920, "height": 1080}
    }
}

def get_config(config_type: str = "screenshot") -> Any:
    """Get configuration by type"""
    configs = {
        "screenshot": DEFAULT_SCREENSHOT_CONFIG,
        "dom": DEFAULT_DOM_CONFIG,
        "cnn": DEFAULT_CNN_CONFIG,
        "feature": DEFAULT_FEATURE_CONFIG,
        "template": DEFAULT_TEMPLATE_CONFIG,
        "api": DEFAULT_API_CONFIG
    }
    
    if config_type not in configs:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return configs[config_type]

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config_to_file(config: Any, config_path: str):
    """Save configuration to YAML file"""
    import yaml
    
    # Convert dataclass to dict
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)