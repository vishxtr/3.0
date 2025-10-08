# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Tokenization pipeline for phishing detection models
"""

import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class PhishingTokenizer:
    """Tokenization pipeline for phishing detection"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            raise
        
        # Set default parameters
        self.max_length = self.config.get('max_length', 512)
        self.padding = self.config.get('padding', 'max_length')
        self.truncation = self.config.get('truncation', True)
        self.return_tensors = self.config.get('return_tensors', 'pt')
        
        # Special tokens
        self.special_tokens = {
            'url_token': '[URL]',
            'email_token': '[EMAIL]',
            'phone_token': '[PHONE]',
            'phish_token': '[PHISH]',
            'benign_token': '[BENIGN]'
        }
        
        # Add special tokens if not present
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to tokenizer"""
        special_tokens_dict = {
            'additional_special_tokens': list(self.special_tokens.values())
        }
        
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_tokens > 0:
            logger.info(f"Added {num_added_tokens} special tokens")
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """Tokenize a single text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids'),
            'tokens': self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]),
            'text': text
        }
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize a batch of texts"""
        if not texts:
            return {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        
        # Tokenize batch
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding.get('token_type_ids'),
            'texts': texts
        }
    
    def tokenize_with_features(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text with additional features"""
        # Basic tokenization
        tokenized = self.tokenize_text(text)
        
        # Add feature tokens
        feature_tokens = self._create_feature_tokens(features)
        
        # Combine with main tokens
        combined_tokens = {
            **tokenized,
            'feature_tokens': feature_tokens,
            'features': features
        }
        
        return combined_tokens
    
    def _create_feature_tokens(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature-based tokens"""
        feature_tokens = {}
        
        # URL features
        if 'url_count' in features:
            url_count = features['url_count']
            if url_count > 0:
                feature_tokens['has_urls'] = 1
                feature_tokens['url_count'] = min(url_count, 10)  # Cap at 10
            else:
                feature_tokens['has_urls'] = 0
                feature_tokens['url_count'] = 0
        
        # Urgency features
        if 'has_urgency_words' in features:
            feature_tokens['urgency'] = 1 if features['has_urgency_words'] else 0
        
        # Length features
        if 'word_count' in features:
            word_count = features['word_count']
            if word_count < 50:
                feature_tokens['length_category'] = 0  # short
            elif word_count < 200:
                feature_tokens['length_category'] = 1  # medium
            else:
                feature_tokens['length_category'] = 2  # long
        
        # Sentiment features
        if 'sentiment_score' in features:
            sentiment = features['sentiment_score']
            if sentiment < -0.1:
                feature_tokens['sentiment'] = 0  # negative
            elif sentiment > 0.1:
                feature_tokens['sentiment'] = 2  # positive
            else:
                feature_tokens['sentiment'] = 1  # neutral
        
        return feature_tokens
    
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        return {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in self.special_tokens.items()
        }
    
    def save_tokenizer(self, save_path: str):
        """Save tokenizer to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved tokenizer to {save_path}")
    
    def load_tokenizer(self, load_path: str):
        """Load tokenizer from disk"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {load_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        logger.info(f"Loaded tokenizer from {load_path}")

class MultiModalTokenizer:
    """Tokenization for multi-modal inputs (text + features)"""
    
    def __init__(self, text_tokenizer: PhishingTokenizer, feature_dim: int = 50):
        self.text_tokenizer = text_tokenizer
        self.feature_dim = feature_dim
    
    def tokenize_multimodal(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize text and features together"""
        # Tokenize text
        text_tokens = self.text_tokenizer.tokenize_text(text)
        
        # Process features
        feature_vector = self._features_to_vector(features)
        
        return {
            'text_tokens': text_tokens,
            'feature_vector': feature_vector,
            'combined_input': self._combine_inputs(text_tokens, feature_vector)
        }
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to vector representation"""
        vector = np.zeros(self.feature_dim)
        
        # Map features to vector positions
        feature_mapping = {
            'url_count': 0,
            'has_urgency_words': 1,
            'word_count': 2,
            'sentiment_score': 3,
            'has_links': 4,
            'suspicious_patterns': 5,
            'punctuation_ratio': 6,
            'uppercase_ratio': 7,
            'digit_ratio': 8,
            'readability_score': 9
        }
        
        for feature_name, position in feature_mapping.items():
            if feature_name in features and position < self.feature_dim:
                value = features[feature_name]
                if isinstance(value, bool):
                    vector[position] = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    # Normalize to [0, 1]
                    if feature_name == 'sentiment_score':
                        vector[position] = (value + 1) / 2  # [-1, 1] -> [0, 1]
                    elif feature_name == 'readability_score':
                        vector[position] = min(max(value / 100, 0), 1)  # [0, 100] -> [0, 1]
                    else:
                        vector[position] = min(max(value / 100, 0), 1)  # General normalization
        
        return vector
    
    def _combine_inputs(self, text_tokens: Dict[str, Any], feature_vector: np.ndarray) -> Dict[str, Any]:
        """Combine text tokens and feature vector"""
        return {
            'input_ids': text_tokens['input_ids'],
            'attention_mask': text_tokens['attention_mask'],
            'feature_vector': torch.tensor(feature_vector, dtype=torch.float32),
            'combined_embedding': self._create_combined_embedding(text_tokens, feature_vector)
        }
    
    def _create_combined_embedding(self, text_tokens: Dict[str, Any], feature_vector: np.ndarray) -> torch.Tensor:
        """Create combined embedding from text and features"""
        # This would be implemented based on the specific model architecture
        # For now, return a placeholder
        text_embedding = torch.zeros(text_tokens['input_ids'].shape[1])
        feature_embedding = torch.tensor(feature_vector)
        
        # Simple concatenation (would be more sophisticated in practice)
        combined = torch.cat([text_embedding, feature_embedding])
        return combined

def create_tokenizer(model_name: str = "distilbert-base-uncased", config: Optional[Dict[str, Any]] = None) -> PhishingTokenizer:
    """Factory function to create tokenizer"""
    return PhishingTokenizer(model_name, config)

def create_multimodal_tokenizer(text_tokenizer: PhishingTokenizer, feature_dim: int = 50) -> MultiModalTokenizer:
    """Factory function to create multimodal tokenizer"""
    return MultiModalTokenizer(text_tokenizer, feature_dim)