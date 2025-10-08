# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Data validation and quality checks for phishing detection datasets
"""

import pandas as pd
import numpy as np
import re
import urllib.parse
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import QUALITY_THRESHOLDS, DATA_SCHEMAS, VALIDATION_RULES, get_data_path

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and quality check datasets"""
    
    def __init__(self):
        self.quality_thresholds = QUALITY_THRESHOLDS
        self.data_schemas = DATA_SCHEMAS
        self.validation_rules = VALIDATION_RULES
    
    def validate_url_format(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def check_url_quality(self, url: str) -> Dict[str, Any]:
        """Check URL quality metrics"""
        quality_report = {
            'valid_format': self.validate_url_format(url),
            'length': len(url),
            'length_valid': self.quality_thresholds['min_url_length'] <= len(url) <= self.quality_thresholds['max_url_length'],
            'has_scheme': url.startswith(('http://', 'https://')),
            'domain_count': len(url.split('.')) - 1,
            'path_depth': url.count('/') - 2,
            'has_parameters': '?' in url,
            'has_fragment': '#' in url,
            'suspicious_patterns': self.detect_suspicious_url_patterns(url)
        }
        
        return quality_report
    
    def check_email_quality(self, email_data: Dict[str, str]) -> Dict[str, Any]:
        """Check email quality metrics"""
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        sender = email_data.get('sender', '')
        
        quality_report = {
            'subject_length': len(subject),
            'body_length': len(body),
            'sender_valid': self.validate_email_format(sender),
            'length_valid': (
                self.quality_thresholds['min_email_length'] <= len(body) <= 
                self.quality_thresholds['max_email_length']
            ),
            'has_urgency_words': self.detect_urgency_words(subject + ' ' + body),
            'has_suspicious_links': self.detect_suspicious_links(body),
            'spelling_errors': self.count_spelling_errors(subject + ' ' + body),
            'suspicious_patterns': self.detect_suspicious_email_patterns(email_data)
        }
        
        return quality_report
    
    def detect_suspicious_url_patterns(self, url: str) -> List[str]:
        """Detect suspicious patterns in URLs"""
        suspicious_patterns = []
        
        # Check for common phishing patterns
        if re.search(r'\d+\.\d+\.\d+\.\d+', url):  # IP address
            suspicious_patterns.append('ip_address')
        
        if re.search(r'[^\x00-\x7F]', url):  # Non-ASCII characters
            suspicious_patterns.append('unicode_chars')
        
        if url.count('.') > 3:  # Too many dots
            suspicious_patterns.append('excessive_dots')
        
        if len(url) > 200:  # Very long URL
            suspicious_patterns.append('excessive_length')
        
        if re.search(r'[%]{2,}', url):  # Double encoding
            suspicious_patterns.append('double_encoding')
        
        if re.search(r'[a-zA-Z]{20,}', url):  # Very long subdomain
            suspicious_patterns.append('long_subdomain')
        
        return suspicious_patterns
    
    def detect_suspicious_email_patterns(self, email_data: Dict[str, str]) -> List[str]:
        """Detect suspicious patterns in emails"""
        suspicious_patterns = []
        subject = email_data.get('subject', '').lower()
        body = email_data.get('body', '').lower()
        sender = email_data.get('sender', '').lower()
        
        # Check for common phishing patterns
        if any(word in subject for word in ['urgent', 'immediate', 'verify', 'confirm', 'suspended']):
            suspicious_patterns.append('urgency_words')
        
        if any(word in body for word in ['click here', 'verify now', 'act now', 'limited time']):
            suspicious_patterns.append('action_phrases')
        
        if re.search(r'http[s]?://[^\s]+', body):  # Contains URLs
            suspicious_patterns.append('contains_urls')
        
        if sender.count('@') > 1:  # Multiple @ symbols
            suspicious_patterns.append('multiple_at_symbols')
        
        if len(sender.split('@')[0]) > 50:  # Very long username
            suspicious_patterns.append('long_username')
        
        return suspicious_patterns
    
    def detect_urgency_words(self, text: str) -> bool:
        """Detect urgency words in text"""
        urgency_words = [
            'urgent', 'immediate', 'asap', 'emergency', 'critical',
            'verify', 'confirm', 'suspended', 'locked', 'expired',
            'limited time', 'act now', 'click here', 'verify now'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in urgency_words)
    
    def detect_suspicious_links(self, text: str) -> bool:
        """Detect suspicious links in text"""
        url_pattern = r'http[s]?://[^\s]+'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            if self.detect_suspicious_url_patterns(url):
                return True
        
        return False
    
    def count_spelling_errors(self, text: str) -> int:
        """Count potential spelling errors (simple heuristic)"""
        # Simple heuristic: count words with repeated characters or unusual patterns
        words = re.findall(r'\b\w+\b', text)
        error_count = 0
        
        for word in words:
            if len(word) > 3:
                # Check for repeated characters
                if re.search(r'(.)\1{2,}', word):
                    error_count += 1
                # Check for unusual character combinations
                if re.search(r'[bcdfghjklmnpqrstvwxyz]{3,}', word.lower()):
                    error_count += 1
        
        return error_count
    
    def validate_dataset_schema(self, df: pd.DataFrame, schema_type: str) -> Dict[str, Any]:
        """Validate dataset against schema"""
        if schema_type not in self.data_schemas:
            raise ValueError(f"Unknown schema type: {schema_type}")
        
        schema = self.data_schemas[schema_type]
        validation_report = {
            'schema_type': schema_type,
            'total_records': len(df),
            'missing_columns': [],
            'invalid_types': [],
            'missing_values': {},
            'invalid_values': {},
            'schema_compliance': True
        }
        
        # Check for required columns
        for column, expected_type in schema.items():
            if column not in df.columns:
                validation_report['missing_columns'].append(column)
                validation_report['schema_compliance'] = False
            else:
                # Check for missing values
                missing_count = df[column].isnull().sum()
                validation_report['missing_values'][column] = missing_count
                
                # Check data types (simplified)
                if expected_type == 'string':
                    non_string_count = df[~df[column].astype(str).str.match(r'^.*$')].shape[0]
                    if non_string_count > 0:
                        validation_report['invalid_types'].append(column)
                
                elif expected_type == 'float':
                    try:
                        pd.to_numeric(df[column], errors='coerce')
                    except:
                        validation_report['invalid_types'].append(column)
        
        return validation_report
    
    def validate_labels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate label distribution and values"""
        if 'label' not in df.columns:
            return {'error': 'No label column found'}
        
        valid_labels = self.quality_thresholds['label_values']
        label_counts = df['label'].value_counts()
        invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()
        
        validation_report = {
            'total_records': len(df),
            'label_distribution': label_counts.to_dict(),
            'invalid_labels': invalid_labels.tolist(),
            'label_balance': self.calculate_label_balance(label_counts),
            'compliance': len(invalid_labels) == 0
        }
        
        return validation_report
    
    def calculate_label_balance(self, label_counts: pd.Series) -> Dict[str, float]:
        """Calculate label balance metrics"""
        total = label_counts.sum()
        balance_report = {}
        
        for label, count in label_counts.items():
            balance_report[label] = count / total
        
        # Calculate balance score (closer to 0.5 is more balanced)
        if len(label_counts) == 2:
            values = list(balance_report.values())
            balance_score = 1 - abs(values[0] - values[1])
        else:
            # For multiple labels, calculate entropy
            import math
            entropy = -sum(p * math.log2(p) for p in balance_report.values() if p > 0)
            max_entropy = math.log2(len(balance_report))
            balance_score = entropy / max_entropy if max_entropy > 0 else 0
        
        balance_report['balance_score'] = balance_score
        return balance_report
    
    def comprehensive_validation(self, df: pd.DataFrame, dataset_type: str = 'url') -> Dict[str, Any]:
        """Perform comprehensive dataset validation"""
        logger.info(f"Starting comprehensive validation for {dataset_type} dataset")
        
        validation_report = {
            'dataset_type': dataset_type,
            'validation_timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'schema_validation': {},
            'label_validation': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Schema validation
        schema_type = 'url_record' if dataset_type == 'url' else 'email_record'
        validation_report['schema_validation'] = self.validate_dataset_schema(df, schema_type)
        
        # Label validation
        validation_report['label_validation'] = self.validate_labels(df)
        
        # Quality metrics
        if dataset_type == 'url' and 'url' in df.columns:
            quality_reports = []
            for url in df['url'].head(100):  # Sample first 100 URLs
                quality_reports.append(self.check_url_quality(url))
            
            # Aggregate quality metrics
            validation_report['quality_metrics'] = {
                'valid_format_ratio': sum(r['valid_format'] for r in quality_reports) / len(quality_reports),
                'average_length': np.mean([r['length'] for r in quality_reports]),
                'suspicious_patterns_found': sum(len(r['suspicious_patterns']) for r in quality_reports),
                'has_scheme_ratio': sum(r['has_scheme'] for r in quality_reports) / len(quality_reports)
            }
        
        elif dataset_type == 'email':
            quality_reports = []
            for _, row in df.head(100).iterrows():  # Sample first 100 emails
                email_data = {
                    'subject': row.get('subject', ''),
                    'body': row.get('body', ''),
                    'sender': row.get('sender', '')
                }
                quality_reports.append(self.check_email_quality(email_data))
            
            # Aggregate quality metrics
            validation_report['quality_metrics'] = {
                'sender_valid_ratio': sum(r['sender_valid'] for r in quality_reports) / len(quality_reports),
                'average_subject_length': np.mean([r['subject_length'] for r in quality_reports]),
                'average_body_length': np.mean([r['body_length'] for r in quality_reports]),
                'urgency_words_ratio': sum(r['has_urgency_words'] for r in quality_reports) / len(quality_reports),
                'suspicious_links_ratio': sum(r['has_suspicious_links'] for r in quality_reports) / len(quality_reports)
            }
        
        # Generate recommendations
        validation_report['recommendations'] = self.generate_recommendations(validation_report)
        
        logger.info("Comprehensive validation completed")
        return validation_report
    
    def generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Schema recommendations
        if not validation_report['schema_validation']['schema_compliance']:
            recommendations.append("Fix schema compliance issues - missing columns or invalid types")
        
        # Label recommendations
        if not validation_report['label_validation']['compliance']:
            recommendations.append("Remove or correct invalid labels")
        
        balance_score = validation_report['label_validation']['label_balance'].get('balance_score', 0)
        if balance_score < 0.3:
            recommendations.append("Consider balancing the dataset - labels are highly imbalanced")
        
        # Quality recommendations
        quality_metrics = validation_report['quality_metrics']
        
        if 'valid_format_ratio' in quality_metrics and quality_metrics['valid_format_ratio'] < 0.9:
            recommendations.append("Improve URL format validation - many invalid URLs detected")
        
        if 'sender_valid_ratio' in quality_metrics and quality_metrics['sender_valid_ratio'] < 0.9:
            recommendations.append("Improve email format validation - many invalid sender addresses")
        
        if 'suspicious_patterns_found' in quality_metrics and quality_metrics['suspicious_patterns_found'] > 50:
            recommendations.append("Review suspicious patterns - high number of suspicious URLs detected")
        
        return recommendations

def main():
    """Main function to validate datasets"""
    validator = DataValidator()
    
    # Example validation of synthetic datasets
    try:
        # Load synthetic URL dataset
        url_df = pd.read_csv(get_data_path('synthetic', 'synthetic_urls.csv'))
        url_validation = validator.comprehensive_validation(url_df, 'url')
        
        # Save validation report
        import json
        report_path = get_data_path('validation', 'url_validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(url_validation, f, indent=2)
        
        print("URL Dataset Validation Report:")
        print(f"Total Records: {url_validation['total_records']}")
        print(f"Schema Compliance: {url_validation['schema_validation']['schema_compliance']}")
        print(f"Label Compliance: {url_validation['label_validation']['compliance']}")
        print(f"Valid Format Ratio: {url_validation['quality_metrics'].get('valid_format_ratio', 'N/A')}")
        print(f"Recommendations: {len(url_validation['recommendations'])}")
        
    except FileNotFoundError:
        print("Synthetic URL dataset not found. Run synthetic_url_generator.py first.")
    
    try:
        # Load synthetic email dataset
        email_df = pd.read_csv(get_data_path('synthetic', 'synthetic_emails.csv'))
        email_validation = validator.comprehensive_validation(email_df, 'email')
        
        # Save validation report
        report_path = get_data_path('validation', 'email_validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(email_validation, f, indent=2)
        
        print("\nEmail Dataset Validation Report:")
        print(f"Total Records: {email_validation['total_records']}")
        print(f"Schema Compliance: {email_validation['schema_validation']['schema_compliance']}")
        print(f"Label Compliance: {email_validation['label_validation']['compliance']}")
        print(f"Sender Valid Ratio: {email_validation['quality_metrics'].get('sender_valid_ratio', 'N/A')}")
        print(f"Recommendations: {len(email_validation['recommendations'])}")
        
    except FileNotFoundError:
        print("Synthetic email dataset not found. Run synthetic_email_generator.py first.")

if __name__ == "__main__":
    main()