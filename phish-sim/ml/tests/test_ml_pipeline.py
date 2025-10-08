# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Comprehensive tests for ML pipeline components
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.text_preprocessor import TextPreprocessor, create_preprocessor
from preprocessing.tokenizer import PhishingTokenizer, create_tokenizer
from preprocessing.embeddings import (
    TransformerEmbeddings, TFIDFEmbeddings, FeatureEmbeddings, 
    create_embedding_generator
)
from models.phishing_classifier import (
    TransformerClassifier, LSTMClassifier, CNNClassifier,
    create_classifier, get_model_info
)
from training.trainer import PhishingTrainer, PhishingDataset, create_trainer
from inference.inference_api import PhishingInferenceAPI
from config import get_config

class TestTextPreprocessor:
    """Test text preprocessing functionality"""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = create_preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'clean_text')
        assert hasattr(preprocessor, 'extract_url_features')
    
    def test_clean_text(self):
        """Test text cleaning"""
        preprocessor = create_preprocessor()
        
        # Test basic cleaning
        text = "This is a test URL: https://example.com and email: test@example.com"
        cleaned = preprocessor.clean_text(text)
        
        assert '[URL]' in cleaned
        assert '[EMAIL]' in cleaned
        assert 'https://example.com' not in cleaned
        assert 'test@example.com' not in cleaned
    
    def test_extract_url_features(self):
        """Test URL feature extraction"""
        preprocessor = create_preprocessor()
        
        text_with_urls = "Check out https://google.com and http://suspicious-site.com"
        features = preprocessor.extract_url_features(text_with_urls)
        
        assert features['url_count'] == 2
        assert features['has_https'] is True
        assert features['avg_url_length'] > 0
    
    def test_extract_email_features(self):
        """Test email feature extraction"""
        preprocessor = create_preprocessor()
        
        email_data = {
            'subject': 'URGENT: Verify your account',
            'body': 'Click here immediately to verify your account!',
            'sender': 'security@bank.com'
        }
        
        features = preprocessor.extract_email_features(email_data)
        
        assert features['has_urgency_words'] is True
        assert features['subject_length'] > 0
        assert features['body_length'] > 0
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction"""
        preprocessor = create_preprocessor()
        
        text = "This is a test sentence. It has multiple sentences for testing."
        features = preprocessor.extract_linguistic_features(text)
        
        assert features['word_count'] > 0
        assert features['sentence_count'] > 0
        assert features['avg_word_length'] > 0
        assert 'sentiment_score' in features
    
    def test_preprocess_text(self):
        """Test complete text preprocessing"""
        preprocessor = create_preprocessor()
        
        text = "URGENT: Click here https://verify-bank.com to secure your account!"
        result = preprocessor.preprocess_text(text, extract_features=True)
        
        assert 'original_text' in result
        assert 'cleaned_text' in result
        assert 'features' in result
        assert result['features']['has_urgency_words'] is True

class TestTokenizer:
    """Test tokenization functionality"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = create_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, 'tokenize_text')
        assert hasattr(tokenizer, 'tokenize_batch')
    
    def test_tokenize_text(self):
        """Test single text tokenization"""
        tokenizer = create_tokenizer()
        
        text = "This is a test sentence for tokenization."
        result = tokenizer.tokenize_text(text)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'tokens' in result
        assert len(result['tokens']) > 0
    
    def test_tokenize_batch(self):
        """Test batch tokenization"""
        tokenizer = create_tokenizer()
        
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        result = tokenizer.tokenize_batch(texts)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert result['input_ids'].shape[0] == len(texts)
    
    def test_special_tokens(self):
        """Test special token handling"""
        tokenizer = create_tokenizer()
        
        special_tokens = tokenizer.get_special_token_ids()
        assert len(special_tokens) > 0
        assert all(isinstance(token_id, int) for token_id in special_tokens.values())
    
    def test_vocab_size(self):
        """Test vocabulary size"""
        tokenizer = create_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size > 0
        assert isinstance(vocab_size, int)

class TestEmbeddings:
    """Test embedding generation"""
    
    def test_tfidf_embeddings(self):
        """Test TF-IDF embeddings"""
        texts = [
            "This is a phishing email with urgent action required.",
            "This is a legitimate newsletter from our company.",
            "Click here to verify your account immediately."
        ]
        
        # Create and fit TF-IDF
        tfidf = TFIDFEmbeddings()
        tfidf.fit(texts)
        
        # Generate embeddings
        embeddings = tfidf.generate_embeddings(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
        assert not np.isnan(embeddings).any()
    
    def test_feature_embeddings(self):
        """Test feature-based embeddings"""
        features_list = [
            {
                'url_count': 2,
                'has_urgency_words': True,
                'word_count': 10,
                'sentiment_score': -0.5
            },
            {
                'url_count': 0,
                'has_urgency_words': False,
                'word_count': 15,
                'sentiment_score': 0.2
            }
        ]
        
        feature_emb = FeatureEmbeddings()
        embeddings = feature_emb.generate_embeddings(features_list)
        
        assert embeddings.shape[0] == len(features_list)
        assert embeddings.shape[1] > 0
        assert not np.isnan(embeddings).any()
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_transformer_embeddings(self, mock_tokenizer, mock_model):
        """Test transformer embeddings with mocked model"""
        # Mock the transformer components
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.return_value.return_value = mock_outputs
        
        texts = ["Test sentence 1", "Test sentence 2"]
        
        transformer_emb = TransformerEmbeddings()
        embeddings = transformer_emb.generate_embeddings(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 768  # BERT hidden size

class TestModels:
    """Test model architectures"""
    
    def test_transformer_classifier(self):
        """Test transformer classifier"""
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3,
            'dropout_rate': 0.1
        }
        
        model = create_classifier('transformer', config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        outputs = model(input_ids, attention_mask)
        
        assert outputs.logits.shape == (batch_size, 3)
        assert not torch.isnan(outputs.logits).any()
    
    def test_lstm_classifier(self):
        """Test LSTM classifier"""
        config = {
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_size': 64,
            'num_layers': 2,
            'num_labels': 3,
            'dropout_rate': 0.1
        }
        
        model = create_classifier('lstm', config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        outputs = model(input_ids)
        
        assert outputs.logits.shape == (batch_size, 3)
        assert not torch.isnan(outputs.logits).any()
    
    def test_cnn_classifier(self):
        """Test CNN classifier"""
        config = {
            'vocab_size': 10000,
            'embedding_dim': 128,
            'num_filters': 100,
            'filter_sizes': [3, 4, 5],
            'num_labels': 3,
            'dropout_rate': 0.1
        }
        
        model = create_classifier('cnn', config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        outputs = model(input_ids)
        
        assert outputs.logits.shape == (batch_size, 3)
        assert not torch.isnan(outputs.logits).any()
    
    def test_model_info(self):
        """Test model information extraction"""
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3
        }
        
        model = create_classifier('transformer', config)
        info = get_model_info(model)
        
        assert 'model_type' in info
        assert 'total_parameters' in info
        assert 'model_size_mb' in info
        assert info['total_parameters'] > 0

class TestTraining:
    """Test training pipeline"""
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        texts = ["Text 1", "Text 2", "Text 3"]
        labels = [0, 1, 2]
        features = [{'feature1': 1}, {'feature2': 2}, {'feature3': 3}]
        
        dataset = PhishingDataset(texts, labels, features)
        
        assert len(dataset) == 3
        
        # Test item access
        item = dataset[0]
        assert 'text' in item
        assert 'label' in item
        assert 'features' in item
        assert item['label'].item() == 0
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3
        }
        
        model = create_classifier('transformer', config)
        trainer = create_trainer(model, {'learning_rate': 1e-4, 'num_epochs': 1})
        
        assert trainer is not None
        assert trainer.model == model
        assert trainer.learning_rate == 1e-4
    
    def test_data_preparation(self):
        """Test data preparation"""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        labels = [0, 1, 0, 1, 2]
        
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3
        }
        
        model = create_classifier('transformer', config)
        trainer = create_trainer(model, {'batch_size': 2})
        
        train_loader, val_loader, test_loader = trainer.prepare_data(
            texts, labels, test_size=0.2, val_size=0.2
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check data split sizes
        total_size = len(texts)
        assert len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset) == total_size

class TestInference:
    """Test inference API"""
    
    @patch('ml.models.phishing_classifier.create_classifier')
    @patch('ml.preprocessing.tokenizer.create_tokenizer')
    @patch('ml.preprocessing.text_preprocessor.create_preprocessor')
    def test_inference_api_initialization(self, mock_preprocessor, mock_tokenizer, mock_classifier):
        """Test inference API initialization with mocked components"""
        # Mock components
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        mock_classifier.return_value = mock_model
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.tokenize_text.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.preprocess_text.return_value = {
            'cleaned_text': 'cleaned text',
            'features': {}
        }
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model"
            tokenizer_path = Path(temp_dir) / "tokenizer"
            model_path.mkdir()
            tokenizer_path.mkdir()
            
            # Create dummy files
            (model_path / "pytorch_model.bin").touch()
            (model_path / "config.json").write_text('{"model_type": "transformer"}')
            (tokenizer_path / "tokenizer.json").touch()
            
            # Initialize API
            api = PhishingInferenceAPI(str(model_path), str(tokenizer_path))
            
            assert api is not None
            assert api.model_path == str(model_path)
            assert api.tokenizer_path == str(tokenizer_path)
    
    def test_analysis_request_validation(self):
        """Test analysis request validation"""
        from inference.inference_api import AnalysisRequest
        
        # Valid request
        request = AnalysisRequest(
            text="Test text",
            content_type="text",
            return_features=False
        )
        
        assert request.text == "Test text"
        assert request.content_type == "text"
        assert request.return_features is False
    
    def test_analysis_response_validation(self):
        """Test analysis response validation"""
        from inference.inference_api import AnalysisResponse
        
        # Valid response
        response = AnalysisResponse(
            prediction="benign",
            confidence=0.95,
            probabilities={"phish": 0.02, "benign": 0.95, "suspicious": 0.03},
            processing_time_ms=25.5,
            timestamp="2024-01-01 12:00:00"
        )
        
        assert response.prediction == "benign"
        assert response.confidence == 0.95
        assert len(response.probabilities) == 3

class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_preprocessing_to_tokenization(self):
        """Test integration from preprocessing to tokenization"""
        # Preprocess text
        preprocessor = create_preprocessor()
        text = "URGENT: Click here https://verify-bank.com to secure your account!"
        preprocessed = preprocessor.preprocess_text(text, extract_features=True)
        
        # Tokenize
        tokenizer = create_tokenizer()
        tokenized = tokenizer.tokenize_text(preprocessed['cleaned_text'])
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert len(tokenized['tokens']) > 0
    
    def test_model_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        # Create model
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3
        }
        model = create_classifier('transformer', config)
        
        # Create sample input
        batch_size = 1
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        assert predictions.shape == (batch_size,)
        assert probabilities.shape == (batch_size, 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size))
    
    def test_save_load_model(self):
        """Test model saving and loading"""
        config = {
            'model_name': 'distilbert-base-uncased',
            'num_labels': 3
        }
        
        # Create and save model
        model = create_classifier('transformer', config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            model.save_model(str(save_path))
            
            # Verify files were created
            assert (save_path / "pytorch_model.bin").exists()
            assert (save_path / "config.json").exists()
            
            # Create new model and load
            new_model = create_classifier('transformer', config)
            new_model.load_model(str(save_path))
            
            # Test that models produce same output
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones(1, 10)
            
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                output1 = model(input_ids, attention_mask)
                output2 = new_model(input_ids, attention_mask)
            
            assert torch.allclose(output1.logits, output2.logits, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])