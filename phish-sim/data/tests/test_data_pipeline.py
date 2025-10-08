# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Comprehensive tests for the data pipeline
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_public_datasets import DatasetDownloader
from scripts.synthetic_url_generator import SyntheticURLGenerator
from scripts.synthetic_email_generator import SyntheticEmailGenerator
from scripts.adversarial_generator import AdversarialGenerator
from scripts.data_validator import DataValidator
from scripts.data_storage import DataStorage
from config import get_data_path, SYNTHETIC_CONFIG

class TestDatasetDownloader:
    """Test dataset downloader functionality"""
    
    def test_downloader_initialization(self):
        """Test downloader initialization"""
        downloader = DatasetDownloader()
        assert downloader.session is not None
        assert 'Phish-Sim' in downloader.session.headers['User-Agent']
    
    @patch('requests.Session.get')
    def test_generate_benign_urls(self, mock_get):
        """Test benign URL generation"""
        downloader = DatasetDownloader()
        benign_urls = downloader.generate_benign_urls(count=10)
        
        assert len(benign_urls) == 10
        assert all(url['label'] == 'benign' for url in benign_urls)
        assert all('http' in url['url'] for url in benign_urls)
    
    def test_validate_dataset(self):
        """Test dataset validation"""
        downloader = DatasetDownloader()
        
        # Create test dataset
        test_data = [
            {'url': 'https://example.com', 'label': 'benign', 'confidence': 0.9, 'source': 'test', 'timestamp': '2024-01-01'},
            {'url': 'https://phish.com', 'label': 'phish', 'confidence': 0.8, 'source': 'test', 'timestamp': '2024-01-01'}
        ]
        df = pd.DataFrame(test_data)
        
        validation_report = downloader.validate_dataset(df)
        
        assert validation_report['total_records'] == 2
        assert 'phish' in validation_report['label_distribution']
        assert 'benign' in validation_report['label_distribution']

class TestSyntheticURLGenerator:
    """Test synthetic URL generator"""
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = SyntheticURLGenerator()
        assert len(generator.legitimate_domains) > 0
        assert len(generator.suspicious_domains) > 0
        assert len(generator.obfuscation_techniques) > 0
    
    def test_generate_phishing_url(self):
        """Test phishing URL generation"""
        generator = SyntheticURLGenerator()
        
        for technique in generator.obfuscation_techniques:
            url_data = generator.generate_phishing_url(technique)
            
            assert url_data['label'] == 'phish'
            assert url_data['technique'] == technique
            assert 'url' in url_data
            assert 'confidence' in url_data
            assert 'features' in url_data
    
    def test_generate_benign_url(self):
        """Test benign URL generation"""
        generator = SyntheticURLGenerator()
        url_data = generator.generate_benign_url()
        
        assert url_data['label'] == 'benign'
        assert url_data['technique'] == 'legitimate'
        assert 'url' in url_data
        assert 'confidence' in url_data
    
    def test_generate_dataset(self):
        """Test dataset generation"""
        generator = SyntheticURLGenerator()
        dataset = generator.generate_dataset(count=20, phishing_ratio=0.3)
        
        assert len(dataset) == 20
        phishing_count = sum(1 for item in dataset if item['label'] == 'phish')
        benign_count = sum(1 for item in dataset if item['label'] == 'benign')
        
        assert phishing_count == 6  # 30% of 20
        assert benign_count == 14   # 70% of 20
    
    def test_homograph_attack(self):
        """Test homograph attack generation"""
        generator = SyntheticURLGenerator()
        url = generator.generate_homograph_url("google.com")
        
        assert "google" in url.lower()
        assert url.startswith("https://")
    
    def test_redirect_chain_attack(self):
        """Test redirect chain attack generation"""
        generator = SyntheticURLGenerator()
        url = generator.generate_redirect_chain_url("microsoft.com")
        
        assert "redirect" in url.lower() or "bit.ly" in url or "tinyurl" in url

class TestSyntheticEmailGenerator:
    """Test synthetic email generator"""
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = SyntheticEmailGenerator()
        assert len(generator.phishing_templates) > 0
        assert len(generator.benign_templates) > 0
        assert len(generator.companies) > 0
    
    def test_generate_phishing_email(self):
        """Test phishing email generation"""
        generator = SyntheticEmailGenerator()
        
        for template_type in generator.phishing_templates:
            email_data = generator.generate_phishing_email(template_type)
            
            assert email_data['label'] == 'phish'
            assert email_data['template_type'] == template_type
            assert 'subject' in email_data
            assert 'body' in email_data
            assert 'sender' in email_data
            assert 'features' in email_data
    
    def test_generate_benign_email(self):
        """Test benign email generation"""
        generator = SyntheticEmailGenerator()
        email_data = generator.generate_benign_email()
        
        assert email_data['label'] == 'benign'
        assert 'subject' in email_data
        assert 'body' in email_data
        assert 'sender' in email_data
    
    def test_generate_dataset(self):
        """Test email dataset generation"""
        generator = SyntheticEmailGenerator()
        dataset = generator.generate_dataset(count=20, phishing_ratio=0.4)
        
        assert len(dataset) == 20
        phishing_count = sum(1 for item in dataset if item['label'] == 'phish')
        benign_count = sum(1 for item in dataset if item['label'] == 'benign')
        
        assert phishing_count == 8   # 40% of 20
        assert benign_count == 12    # 60% of 20

class TestAdversarialGenerator:
    """Test adversarial generator"""
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = AdversarialGenerator()
        assert len(generator.legitimate_domains) > 0
        assert len(generator.obfuscation_methods) > 0
    
    def test_unicode_normalization_attack(self):
        """Test Unicode normalization attack"""
        generator = AdversarialGenerator()
        original = "google.com"
        adversarial = generator.unicode_normalization_attack(original)
        
        assert len(adversarial) >= len(original)
        # Should contain some non-ASCII characters
        assert any(ord(c) > 127 for c in adversarial)
    
    def test_character_substitution_attack(self):
        """Test character substitution attack"""
        generator = AdversarialGenerator()
        original = "password123"
        adversarial = generator.character_substitution_attack(original)
        
        assert len(adversarial) == len(original)
    
    def test_encoding_variation_attack(self):
        """Test encoding variation attack"""
        generator = AdversarialGenerator()
        original = "test@example.com"
        adversarial = generator.encoding_variation_attack(original)
        
        assert len(adversarial) >= len(original)
    
    def test_generate_adversarial_url(self):
        """Test adversarial URL generation"""
        generator = AdversarialGenerator()
        base_url = "https://google.com/login"
        
        for technique in generator.obfuscation_methods:
            adversarial_url = generator.generate_adversarial_url(base_url, technique)
            
            assert adversarial_url['label'] == 'phish'
            assert adversarial_url['technique'] == technique
            assert 'url' in adversarial_url
            assert 'features' in adversarial_url
    
    def test_obfuscation_score_calculation(self):
        """Test obfuscation score calculation"""
        generator = AdversarialGenerator()
        
        # Test with normal text
        normal_text = "This is a normal URL"
        normal_score = generator.calculate_obfuscation_score(normal_text)
        assert 0.0 <= normal_score <= 1.0
        
        # Test with obfuscated text
        obfuscated_text = "This%20is%20encoded%20text"
        obfuscated_score = generator.calculate_obfuscation_score(obfuscated_text)
        assert obfuscated_score > normal_score

class TestDataValidator:
    """Test data validator"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()
        assert validator.quality_thresholds is not None
        assert validator.data_schemas is not None
    
    def test_validate_url_format(self):
        """Test URL format validation"""
        validator = DataValidator()
        
        # Valid URLs
        assert validator.validate_url_format("https://example.com")
        assert validator.validate_url_format("http://test.org/path")
        
        # Invalid URLs
        assert not validator.validate_url_format("not-a-url")
        assert not validator.validate_url_format("ftp://example.com")
    
    def test_validate_email_format(self):
        """Test email format validation"""
        validator = DataValidator()
        
        # Valid emails
        assert validator.validate_email_format("test@example.com")
        assert validator.validate_email_format("user.name@domain.org")
        
        # Invalid emails
        assert not validator.validate_email_format("not-an-email")
        assert not validator.validate_email_format("test@")
    
    def test_check_url_quality(self):
        """Test URL quality checking"""
        validator = DataValidator()
        
        quality_report = validator.check_url_quality("https://example.com/path")
        
        assert quality_report['valid_format'] is True
        assert quality_report['length'] > 0
        assert quality_report['has_scheme'] is True
        assert isinstance(quality_report['suspicious_patterns'], list)
    
    def test_check_email_quality(self):
        """Test email quality checking"""
        validator = DataValidator()
        
        email_data = {
            'subject': 'Test Subject',
            'body': 'This is a test email body',
            'sender': 'test@example.com'
        }
        
        quality_report = validator.check_email_quality(email_data)
        
        assert quality_report['subject_length'] > 0
        assert quality_report['body_length'] > 0
        assert quality_report['sender_valid'] is True
        assert isinstance(quality_report['suspicious_patterns'], list)
    
    def test_detect_suspicious_url_patterns(self):
        """Test suspicious URL pattern detection"""
        validator = DataValidator()
        
        # Normal URL
        normal_patterns = validator.detect_suspicious_url_patterns("https://example.com")
        assert len(normal_patterns) == 0
        
        # Suspicious URL with IP
        suspicious_patterns = validator.detect_suspicious_url_patterns("http://192.168.1.1")
        assert 'ip_address' in suspicious_patterns
    
    def test_validate_dataset_schema(self):
        """Test dataset schema validation"""
        validator = DataValidator()
        
        # Create test dataset
        test_data = {
            'url': ['https://example.com', 'https://test.com'],
            'label': ['benign', 'phish'],
            'confidence': [0.9, 0.8],
            'timestamp': ['2024-01-01', '2024-01-01'],
            'source': ['test', 'test']
        }
        df = pd.DataFrame(test_data)
        
        validation_report = validator.validate_dataset_schema(df, 'url_record')
        
        assert validation_report['total_records'] == 2
        assert validation_report['schema_compliance'] is True
    
    def test_validate_labels(self):
        """Test label validation"""
        validator = DataValidator()
        
        # Create test dataset
        test_data = {
            'label': ['phish', 'benign', 'suspicious', 'invalid']
        }
        df = pd.DataFrame(test_data)
        
        validation_report = validator.validate_labels(df)
        
        assert validation_report['total_records'] == 4
        assert 'invalid' in validation_report['invalid_labels']
        assert validation_report['compliance'] is False

class TestDataStorage:
    """Test data storage system"""
    
    def test_storage_initialization(self):
        """Test storage initialization"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            assert storage.connection is not None
            storage.close()
    
    def test_store_and_retrieve_url(self):
        """Test URL storage and retrieval"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            
            # Store URL
            url_data = {
                'url': 'https://test.com',
                'label': 'benign',
                'confidence': 0.9,
                'features': {'length': 15},
                'timestamp': '2024-01-01T00:00:00',
                'source': 'test'
            }
            
            url_id = storage.store_url(url_data)
            assert url_id is not None
            
            # Retrieve URL
            urls = storage.get_urls()
            assert len(urls) == 1
            assert urls[0]['url'] == 'https://test.com'
            assert urls[0]['label'] == 'benign'
            
            storage.close()
    
    def test_store_and_retrieve_email(self):
        """Test email storage and retrieval"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            
            # Store email
            email_data = {
                'subject': 'Test Subject',
                'body': 'Test body content',
                'sender': 'test@example.com',
                'recipient': 'user@example.com',
                'label': 'benign',
                'confidence': 0.9,
                'features': {'length': 20},
                'timestamp': '2024-01-01T00:00:00',
                'source': 'test'
            }
            
            email_id = storage.store_email(email_data)
            assert email_id is not None
            
            # Retrieve email
            emails = storage.get_emails()
            assert len(emails) == 1
            assert emails[0]['subject'] == 'Test Subject'
            assert emails[0]['label'] == 'benign'
            
            storage.close()
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            
            # Add some test data
            url_data = {
                'url': 'https://test.com',
                'label': 'benign',
                'confidence': 0.9,
                'features': {},
                'timestamp': '2024-01-01T00:00:00',
                'source': 'test'
            }
            storage.store_url(url_data)
            
            stats = storage.get_dataset_statistics()
            assert 'urls' in stats
            assert 'emails' in stats
            assert 'analysis' in stats
            
            storage.close()

class TestDataPipelineIntegration:
    """Integration tests for the data pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end data pipeline"""
        # This test would run the complete pipeline
        # For now, we'll test the components work together
        
        # Generate synthetic data
        url_generator = SyntheticURLGenerator()
        email_generator = SyntheticEmailGenerator()
        
        url_dataset = url_generator.generate_dataset(count=10, phishing_ratio=0.3)
        email_dataset = email_generator.generate_dataset(count=10, phishing_ratio=0.4)
        
        # Validate data
        validator = DataValidator()
        
        url_df = pd.DataFrame(url_dataset)
        email_df = pd.DataFrame(email_dataset)
        
        url_validation = validator.comprehensive_validation(url_df, 'url')
        email_validation = validator.comprehensive_validation(email_df, 'email')
        
        # Check validation results
        assert url_validation['total_records'] == 10
        assert email_validation['total_records'] == 10
        
        # Test storage
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            storage = DataStorage(tmp.name)
            
            # Store data
            for url_data in url_dataset:
                storage.store_url(url_data)
            
            for email_data in email_dataset:
                storage.store_email(email_data)
            
            # Verify storage
            stored_urls = storage.get_urls()
            stored_emails = storage.get_emails()
            
            assert len(stored_urls) == 10
            assert len(stored_emails) == 10
            
            storage.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])