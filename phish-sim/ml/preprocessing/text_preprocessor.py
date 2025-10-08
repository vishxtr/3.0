# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Text preprocessing pipeline for phishing detection
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Union
import logging
from urllib.parse import urlparse
import nltk
from textblob import TextBlob
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing for phishing detection"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self.phone_pattern = re.compile(
            r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        )
        
        # Urgency words for phishing detection
        self.urgency_words = {
            'urgent', 'immediate', 'asap', 'emergency', 'critical', 'verify',
            'confirm', 'suspended', 'locked', 'expired', 'limited time',
            'act now', 'click here', 'verify now', 'security alert'
        }
        
        # Suspicious patterns
        self.suspicious_patterns = {
            'click here', 'verify now', 'act immediately', 'limited time offer',
            'congratulations', 'you have won', 'free money', 'act now'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs if configured
        if self.config.get('remove_urls', True):
            text = self.url_pattern.sub('[URL]', text)
        
        # Remove emails if configured
        if self.config.get('remove_emails', True):
            text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove phone numbers if configured
        if self.config.get('remove_phone_numbers', True):
            text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove special characters if configured
        if self.config.get('remove_special_chars', False):
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        if self.config.get('normalize_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase if configured
        if self.config.get('lowercase', True):
            text = text.lower()
        
        return text
    
    def extract_url_features(self, text: str) -> Dict[str, Any]:
        """Extract features from URLs in text"""
        urls = self.url_pattern.findall(text)
        
        if not urls:
            return {
                'url_count': 0,
                'avg_url_length': 0,
                'has_https': False,
                'has_redirect': False,
                'has_encoding': False,
                'suspicious_domains': 0
            }
        
        features = {
            'url_count': len(urls),
            'avg_url_length': sum(len(url) for url in urls) / len(urls),
            'has_https': any(url.startswith('https://') for url in urls),
            'has_redirect': any('redirect' in url.lower() for url in urls),
            'has_encoding': any('%' in url or '&' in url for url in urls),
            'suspicious_domains': 0
        }
        
        # Check for suspicious domains
        suspicious_keywords = ['secure', 'verify', 'update', 'alert', 'security']
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if any(keyword in domain for keyword in suspicious_keywords):
                    features['suspicious_domains'] += 1
            except:
                continue
        
        return features
    
    def extract_email_features(self, text: str) -> Dict[str, Any]:
        """Extract features from email content"""
        features = {
            'subject_length': 0,
            'body_length': len(text),
            'has_urgency_words': False,
            'has_links': '[URL]' in text,
            'has_attachments': False,
            'sender_domain_legitimacy': 0.5,
            'spelling_errors': 0,
            'suspicious_patterns': 0,
            'greeting_type': 'none',
            'signature_type': 'none'
        }
        
        # Check for urgency words
        text_lower = text.lower()
        urgency_count = sum(1 for word in self.urgency_words if word in text_lower)
        features['has_urgency_words'] = urgency_count > 0
        
        # Check for suspicious patterns
        suspicious_count = sum(1 for pattern in self.suspicious_patterns if pattern in text_lower)
        features['suspicious_patterns'] = suspicious_count
        
        # Detect greeting type
        greetings = {
            'formal': ['dear', 'sir', 'madam', 'valued customer'],
            'informal': ['hi', 'hello', 'hey'],
            'urgent': ['urgent', 'immediate', 'asap']
        }
        
        for greeting_type, words in greetings.items():
            if any(word in text_lower for word in words):
                features['greeting_type'] = greeting_type
                break
        
        # Detect signature type
        signatures = {
            'formal': ['sincerely', 'best regards', 'yours truly'],
            'corporate': ['customer service', 'support team', 'security team'],
            'none': []
        }
        
        for sig_type, words in signatures.items():
            if any(word in text_lower for word in words):
                features['signature_type'] = sig_type
                break
        
        # Simple spelling error detection
        if self.nlp:
            doc = self.nlp(text)
            features['spelling_errors'] = sum(1 for token in doc if not token.is_alpha and token.text.isalpha())
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        if not text.strip():
            return {
                'sentiment_score': 0,
                'readability_score': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'punctuation_ratio': 0,
                'uppercase_ratio': 0,
                'digit_ratio': 0
            }
        
        # Basic text statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        }
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_score'] = blob.sentiment.polarity
        except:
            features['sentiment_score'] = 0
        
        # Simple readability score (Flesch-like)
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            avg_sentence_length = features['word_count'] / features['sentence_count']
            avg_syllables = sum(self._count_syllables(word) for word in words) / features['word_count']
            features['readability_score'] = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        else:
            features['readability_score'] = 0
        
        return features
    
    def extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features from text"""
        features = {
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'line_count': len([l for l in text.split('\n') if l.strip()]),
            'indentation_consistency': 1.0,
            'font_variations': 0,
            'color_variations': 0,
            'link_density': 0
        }
        
        # Calculate link density
        url_count = len(self.url_pattern.findall(text))
        word_count = len(text.split())
        if word_count > 0:
            features['link_density'] = url_count / word_count
        
        # Check for HTML-like formatting (simplified)
        html_tags = re.findall(r'<[^>]+>', text)
        if html_tags:
            features['font_variations'] = len(set(tag.lower() for tag in html_tags))
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def preprocess_text(self, text: str, extract_features: bool = True) -> Dict[str, Any]:
        """Complete text preprocessing pipeline"""
        result = {
            'original_text': text,
            'cleaned_text': self.clean_text(text),
            'features': {}
        }
        
        if extract_features:
            # Extract all feature types
            result['features'].update(self.extract_url_features(text))
            result['features'].update(self.extract_email_features(text))
            result['features'].update(self.extract_linguistic_features(text))
            result['features'].update(self.extract_structural_features(text))
        
        return result
    
    def preprocess_batch(self, texts: List[str], extract_features: bool = True) -> List[Dict[str, Any]]:
        """Preprocess a batch of texts"""
        results = []
        for text in texts:
            result = self.preprocess_text(text, extract_features)
            results.append(result)
        return results

class URLPreprocessor:
    """Specialized URL preprocessing"""
    
    def __init__(self):
        self.suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.click', '.download'}
        self.legitimate_tlds = {'.com', '.org', '.net', '.edu', '.gov', '.mil'}
    
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract comprehensive URL features"""
        try:
            parsed = urlparse(url)
        except:
            return self._get_default_features()
        
        features = {
            'length': len(url),
            'domain_count': len(parsed.netloc.split('.')),
            'path_depth': len([p for p in parsed.path.split('/') if p]),
            'has_https': parsed.scheme == 'https',
            'has_redirect': 'redirect' in url.lower(),
            'has_encoding': '%' in url or '&' in url,
            'has_suspicious_chars': any(c in url for c in ['@', '!', '$', '#']),
            'domain_age': 0,  # Would need external API
            'tld_type': 'unknown'
        }
        
        # TLD analysis
        if parsed.netloc:
            tld = '.' + parsed.netloc.split('.')[-1]
            if tld in self.suspicious_tlds:
                features['tld_type'] = 'suspicious'
            elif tld in self.legitimate_tlds:
                features['tld_type'] = 'legitimate'
            else:
                features['tld_type'] = 'other'
        
        # Additional suspicious patterns
        suspicious_keywords = ['secure', 'verify', 'update', 'alert', 'security', 'bank']
        features['suspicious_keywords'] = sum(1 for keyword in suspicious_keywords if keyword in url.lower())
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features for invalid URLs"""
        return {
            'length': 0,
            'domain_count': 0,
            'path_depth': 0,
            'has_https': False,
            'has_redirect': False,
            'has_encoding': False,
            'has_suspicious_chars': False,
            'domain_age': 0,
            'tld_type': 'invalid',
            'suspicious_keywords': 0
        }

def create_preprocessor(config: Optional[Dict[str, Any]] = None) -> TextPreprocessor:
    """Factory function to create preprocessor"""
    return TextPreprocessor(config)

def create_url_preprocessor() -> URLPreprocessor:
    """Factory function to create URL preprocessor"""
    return URLPreprocessor()