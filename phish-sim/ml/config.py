# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Configuration for ML pipeline components
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    
    # Model architecture
    model_name: str = "distilbert-base-uncased"
    model_type: str = "transformer"  # transformer, lstm, cnn, ensemble
    num_labels: int = 3  # phish, benign, suspicious
    
    # Training parameters
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Model optimization
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    dataloader_num_workers: int = 4
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Model paths
    model_save_path: str = str(MODELS_DIR / "phish_detector")
    tokenizer_save_path: str = str(MODELS_DIR / "tokenizer")
    
    # Performance targets
    target_accuracy: float = 0.85
    target_precision: float = 0.80
    target_recall: float = 0.80
    target_f1: float = 0.80
    max_inference_time_ms: float = 50.0

@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    
    # Text cleaning
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    
    # Text normalization
    lowercase: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
    stem: bool = False
    
    # Feature extraction
    extract_url_features: bool = True
    extract_email_features: bool = True
    extract_linguistic_features: bool = True
    extract_structural_features: bool = True
    
    # Tokenization
    tokenizer_name: str = "distilbert-base-uncased"
    max_tokens: int = 512
    padding: str = "max_length"
    truncation: bool = True
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_ratio: float = 0.1
    synonym_replacement: bool = True
    random_insertion: bool = True
    random_deletion: bool = True

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Data
    train_data_path: str = str(DATA_DIR / "processed" / "train.csv")
    val_data_path: str = str(DATA_DIR / "processed" / "val.csv")
    test_data_path: str = str(DATA_DIR / "processed" / "test.csv")
    
    # Training strategy
    train_strategy: str = "standard"  # standard, kfold, stratified
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    checkpoint_dir: str = str(MODELS_DIR / "checkpoints")
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "phish-sim"
    use_tensorboard: bool = True
    log_dir: str = str(LOGS_DIR)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Metrics
    primary_metric: str = "f1_macro"
    secondary_metrics: List[str] = None
    
    # Evaluation datasets
    eval_datasets: List[str] = None
    
    # Performance thresholds
    min_accuracy: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_f1: float = 0.75
    max_inference_time: float = 100.0  # milliseconds
    
    # Reporting
    generate_report: bool = True
    report_format: str = "json"  # json, html, pdf
    report_path: str = str(LOGS_DIR / "evaluation_report.json")
    
    # Confusion matrix
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True

@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    
    # Model loading
    model_path: str = str(MODELS_DIR / "phish_detector")
    tokenizer_path: str = str(MODELS_DIR / "tokenizer")
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Inference parameters
    batch_size: int = 1
    max_length: int = 512
    return_probabilities: bool = True
    return_attention: bool = False
    
    # Performance optimization
    use_cache: bool = True
    cache_size: int = 1000
    enable_quantization: bool = False
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_workers: int = 1
    api_timeout: int = 30

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()

# Model registry
AVAILABLE_MODELS = {
    "distilbert-base-uncased": {
        "type": "transformer",
        "size": "small",
        "max_length": 512,
        "description": "DistilBERT base model for fast inference"
    },
    "bert-base-uncased": {
        "type": "transformer", 
        "size": "base",
        "max_length": 512,
        "description": "BERT base model for high accuracy"
    },
    "roberta-base": {
        "type": "transformer",
        "size": "base", 
        "max_length": 512,
        "description": "RoBERTa base model for robust performance"
    },
    "lstm-custom": {
        "type": "lstm",
        "size": "custom",
        "max_length": 256,
        "description": "Custom LSTM model for lightweight deployment"
    }
}

# Feature extraction configurations
URL_FEATURES = [
    "length", "domain_count", "path_depth", "has_https", "has_redirect",
    "has_encoding", "has_suspicious_chars", "domain_age", "tld_type"
]

EMAIL_FEATURES = [
    "subject_length", "body_length", "has_urgency_words", "has_links",
    "has_attachments", "sender_domain_legitimacy", "spelling_errors",
    "suspicious_patterns", "greeting_type", "signature_type"
]

LINGUISTIC_FEATURES = [
    "sentiment_score", "readability_score", "word_count", "sentence_count",
    "avg_word_length", "punctuation_ratio", "uppercase_ratio", "digit_ratio"
]

STRUCTURAL_FEATURES = [
    "paragraph_count", "line_count", "indentation_consistency",
    "font_variations", "color_variations", "link_density"
]

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "inference_time_ms": {
        "excellent": 10,
        "good": 25,
        "acceptable": 50,
        "poor": 100
    },
    "accuracy": {
        "excellent": 0.95,
        "good": 0.90,
        "acceptable": 0.85,
        "poor": 0.80
    },
    "f1_score": {
        "excellent": 0.95,
        "good": 0.90,
        "acceptable": 0.85,
        "poor": 0.80
    }
}

def get_config(config_type: str = "model") -> Any:
    """Get configuration by type"""
    configs = {
        "model": DEFAULT_MODEL_CONFIG,
        "preprocessing": DEFAULT_PREPROCESSING_CONFIG,
        "training": DEFAULT_TRAINING_CONFIG,
        "evaluation": DEFAULT_EVALUATION_CONFIG,
        "inference": DEFAULT_INFERENCE_CONFIG
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