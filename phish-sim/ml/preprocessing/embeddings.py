# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Embedding generation pipeline for phishing detection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Base class for embedding generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        raise NotImplementedError
    
    def save_embeddings(self, embeddings: np.ndarray, save_path: str):
        """Save embeddings to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, embeddings)
        logger.info(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str) -> np.ndarray:
        """Load embeddings from disk"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {load_path}")
        
        embeddings = np.load(load_path)
        logger.info(f"Loaded embeddings from {load_path}")
        return embeddings

class TransformerEmbeddings(EmbeddingGenerator):
    """Transformer-based embeddings using pre-trained models"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = model_name
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Configuration
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 16)
        self.pooling_strategy = self.config.get('pooling_strategy', 'cls')  # cls, mean, max
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate transformer embeddings"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_batch_embeddings(batch_texts)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if self.pooling_strategy == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            elif self.pooling_strategy == 'mean':
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            elif self.pooling_strategy == 'max':
                embeddings = outputs.last_hidden_state.max(dim=1)[0]  # Max pooling
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]  # Default to CLS
        
        return embeddings.cpu().numpy()
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]

class TFIDFEmbeddings(EmbeddingGenerator):
    """TF-IDF based embeddings"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # TF-IDF parameters
        self.max_features = self.config.get('max_features', 10000)
        self.ngram_range = self.config.get('ngram_range', (1, 2))
        self.min_df = self.config.get('min_df', 2)
        self.max_df = self.config.get('max_df', 0.95)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit the TF-IDF vectorizer"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"Fitted TF-IDF vectorizer with {len(self.vectorizer.vocabulary_)} features")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before generating embeddings")
        
        embeddings = self.vectorizer.transform(texts)
        return embeddings.toarray()
    
    def save_vectorizer(self, save_path: str):
        """Save the fitted vectorizer"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Saved vectorizer to {save_path}")
    
    def load_vectorizer(self, load_path: str):
        """Load a fitted vectorizer"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {load_path}")
        
        with open(load_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"Loaded vectorizer from {load_path}")

class FeatureEmbeddings(EmbeddingGenerator):
    """Feature-based embeddings from extracted features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Feature dimensions
        self.url_feature_dim = self.config.get('url_feature_dim', 10)
        self.email_feature_dim = self.config.get('email_feature_dim', 10)
        self.linguistic_feature_dim = self.config.get('linguistic_feature_dim', 8)
        self.structural_feature_dim = self.config.get('structural_feature_dim', 6)
        
        self.total_dim = (self.url_feature_dim + self.email_feature_dim + 
                         self.linguistic_feature_dim + self.structural_feature_dim)
    
    def generate_embeddings(self, features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings from feature dictionaries"""
        embeddings = []
        
        for features in features_list:
            embedding = self._features_to_embedding(features)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _features_to_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to embedding vector"""
        embedding = np.zeros(self.total_dim)
        idx = 0
        
        # URL features
        url_features = [
            'url_count', 'avg_url_length', 'has_https', 'has_redirect',
            'has_encoding', 'suspicious_domains', 'domain_count', 'path_depth',
            'has_suspicious_chars', 'tld_type'
        ]
        
        for feature in url_features:
            if feature in features and idx < self.url_feature_dim:
                value = features[feature]
                if isinstance(value, bool):
                    embedding[idx] = 1.0 if value else 0.0
                elif isinstance(value, str):
                    # Encode categorical features
                    embedding[idx] = self._encode_categorical(value)
                else:
                    embedding[idx] = float(value)
                idx += 1
        
        # Email features
        email_features = [
            'subject_length', 'body_length', 'has_urgency_words', 'has_links',
            'sender_domain_legitimacy', 'spelling_errors', 'suspicious_patterns',
            'greeting_type', 'signature_type', 'has_attachments'
        ]
        
        for feature in email_features:
            if feature in features and idx < self.url_feature_dim + self.email_feature_dim:
                value = features[feature]
                if isinstance(value, bool):
                    embedding[idx] = 1.0 if value else 0.0
                elif isinstance(value, str):
                    embedding[idx] = self._encode_categorical(value)
                else:
                    embedding[idx] = float(value)
                idx += 1
        
        # Linguistic features
        linguistic_features = [
            'sentiment_score', 'readability_score', 'word_count', 'sentence_count',
            'avg_word_length', 'punctuation_ratio', 'uppercase_ratio', 'digit_ratio'
        ]
        
        for feature in linguistic_features:
            if feature in features and idx < (self.url_feature_dim + self.email_feature_dim + 
                                            self.linguistic_feature_dim):
                value = features[feature]
                embedding[idx] = float(value)
                idx += 1
        
        # Structural features
        structural_features = [
            'paragraph_count', 'line_count', 'indentation_consistency',
            'font_variations', 'color_variations', 'link_density'
        ]
        
        for feature in structural_features:
            if feature in features and idx < self.total_dim:
                value = features[feature]
                embedding[idx] = float(value)
                idx += 1
        
        return embedding
    
    def _encode_categorical(self, value: str) -> float:
        """Encode categorical values to numeric"""
        categorical_mapping = {
            'legitimate': 1.0,
            'suspicious': 0.5,
            'invalid': 0.0,
            'formal': 1.0,
            'informal': 0.5,
            'urgent': 0.0,
            'corporate': 1.0,
            'none': 0.0
        }
        
        return categorical_mapping.get(value.lower(), 0.0)

class HybridEmbeddings(EmbeddingGenerator):
    """Hybrid embeddings combining multiple embedding types"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize component embeddings
        self.transformer_embeddings = TransformerEmbeddings(
            model_name=config.get('transformer_model', 'distilbert-base-uncased'),
            config=config.get('transformer_config', {})
        )
        
        self.tfidf_embeddings = TFIDFEmbeddings(
            config=config.get('tfidf_config', {})
        )
        
        self.feature_embeddings = FeatureEmbeddings(
            config=config.get('feature_config', {})
        )
        
        # Dimensionality reduction
        self.use_pca = config.get('use_pca', True)
        self.pca_components = config.get('pca_components', 128)
        self.pca = None
    
    def fit(self, texts: List[str], features_list: List[Dict[str, Any]]):
        """Fit all embedding components"""
        # Fit TF-IDF
        self.tfidf_embeddings.fit(texts)
        
        # Generate all embeddings
        transformer_emb = self.transformer_embeddings.generate_embeddings(texts)
        tfidf_emb = self.tfidf_embeddings.generate_embeddings(texts)
        feature_emb = self.feature_embeddings.generate_embeddings(features_list)
        
        # Combine embeddings
        combined_embeddings = np.hstack([transformer_emb, tfidf_emb, feature_emb])
        
        # Apply PCA if configured
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(combined_embeddings)
            logger.info(f"Fitted PCA with {self.pca_components} components")
    
    def generate_embeddings(self, texts: List[str], features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Generate hybrid embeddings"""
        # Generate component embeddings
        transformer_emb = self.transformer_embeddings.generate_embeddings(texts)
        tfidf_emb = self.tfidf_embeddings.generate_embeddings(texts)
        feature_emb = self.feature_embeddings.generate_embeddings(features_list)
        
        # Combine embeddings
        combined_embeddings = np.hstack([transformer_emb, tfidf_emb, feature_emb])
        
        # Apply PCA if fitted
        if self.pca is not None:
            combined_embeddings = self.pca.transform(combined_embeddings)
        
        return combined_embeddings
    
    def save_components(self, save_dir: str):
        """Save all embedding components"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save TF-IDF vectorizer
        self.tfidf_embeddings.save_vectorizer(save_dir / "tfidf_vectorizer.pkl")
        
        # Save PCA
        if self.pca is not None:
            with open(save_dir / "pca.pkl", 'wb') as f:
                pickle.dump(self.pca, f)
        
        logger.info(f"Saved embedding components to {save_dir}")
    
    def load_components(self, load_dir: str):
        """Load all embedding components"""
        load_dir = Path(load_dir)
        
        # Load TF-IDF vectorizer
        self.tfidf_embeddings.load_vectorizer(load_dir / "tfidf_vectorizer.pkl")
        
        # Load PCA
        pca_path = load_dir / "pca.pkl"
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
        
        logger.info(f"Loaded embedding components from {load_dir}")

def create_embedding_generator(embedding_type: str = "transformer", config: Optional[Dict[str, Any]] = None) -> EmbeddingGenerator:
    """Factory function to create embedding generator"""
    generators = {
        "transformer": TransformerEmbeddings,
        "tfidf": TFIDFEmbeddings,
        "feature": FeatureEmbeddings,
        "hybrid": HybridEmbeddings
    }
    
    if embedding_type not in generators:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    return generators[embedding_type](config)