#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Demo script for T002 - Data ingestion & synthetic dataset generator
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from scripts.synthetic_url_generator import SyntheticURLGenerator
from scripts.synthetic_email_generator import SyntheticEmailGenerator
from scripts.adversarial_generator import AdversarialGenerator
from scripts.data_validator import DataValidator
from scripts.data_storage import DataStorage
from config import get_data_path, SYNTHETIC_CONFIG

def demo_synthetic_url_generation():
    """Demo synthetic URL generation"""
    print("=" * 60)
    print("DEMO: Synthetic URL Generation")
    print("=" * 60)
    
    generator = SyntheticURLGenerator()
    
    # Generate a small sample
    print("Generating 10 synthetic URLs (30% phishing ratio)...")
    dataset = generator.generate_dataset(count=10, phishing_ratio=0.3)
    
    print(f"\nGenerated {len(dataset)} URLs:")
    for i, item in enumerate(dataset, 1):
        print(f"{i:2d}. [{item['label'].upper():8s}] {item['url']}")
        if item['label'] == 'phish':
            print(f"     Technique: {item['technique']}")
            print(f"     Confidence: {item['confidence']:.2f}")
    
    # Show technique distribution
    techniques = {}
    for item in dataset:
        if item['label'] == 'phish':
            tech = item['technique']
            techniques[tech] = techniques.get(tech, 0) + 1
    
    print(f"\nPhishing Techniques Used:")
    for tech, count in techniques.items():
        print(f"  - {tech}: {count}")
    
    return dataset

def demo_synthetic_email_generation():
    """Demo synthetic email generation"""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Email Generation")
    print("=" * 60)
    
    generator = SyntheticEmailGenerator()
    
    # Generate a small sample
    print("Generating 8 synthetic emails (40% phishing ratio)...")
    dataset = generator.generate_dataset(count=8, phishing_ratio=0.4)
    
    print(f"\nGenerated {len(dataset)} emails:")
    for i, item in enumerate(dataset, 1):
        print(f"{i:2d}. [{item['label'].upper():8s}] {item['subject']}")
        print(f"     From: {item['sender']}")
        if item['label'] == 'phish':
            print(f"     Template: {item['template_type']}")
            print(f"     Company: {item['company']}")
    
    # Show template distribution
    templates = {}
    for item in dataset:
        template = item['template_type']
        templates[template] = templates.get(template, 0) + 1
    
    print(f"\nTemplate Types Used:")
    for template, count in templates.items():
        print(f"  - {template}: {count}")
    
    return dataset

def demo_adversarial_generation():
    """Demo adversarial generation"""
    print("\n" + "=" * 60)
    print("DEMO: Adversarial Content Generation")
    print("=" * 60)
    
    generator = AdversarialGenerator()
    
    # Generate adversarial URLs
    base_url = "https://google.com/login"
    print(f"Base URL: {base_url}")
    print("\nGenerating adversarial versions...")
    
    adversarial_urls = []
    for technique in generator.obfuscation_methods[:5]:  # Show first 5 techniques
        adversarial_url = generator.generate_adversarial_url(base_url, technique)
        adversarial_urls.append(adversarial_url)
    
    print(f"\nGenerated {len(adversarial_urls)} adversarial URLs:")
    for i, item in enumerate(adversarial_urls, 1):
        print(f"{i:2d}. [{item['technique']:20s}] {item['url']}")
        print(f"     Obfuscation Score: {item['features']['obfuscation_score']:.3f}")
        print(f"     Unicode Ratio: {item['features']['unicode_ratio']:.3f}")
    
    return adversarial_urls

def demo_data_validation():
    """Demo data validation"""
    print("\n" + "=" * 60)
    print("DEMO: Data Validation")
    print("=" * 60)
    
    validator = DataValidator()
    
    # Test URL validation
    test_urls = [
        "https://google.com/login",
        "http://192.168.1.1",  # IP address
        "https://example.com/very/long/path/with/many/segments",
        "not-a-url",
        "https://test.com"
    ]
    
    print("URL Validation Tests:")
    for url in test_urls:
        is_valid = validator.validate_url_format(url)
        quality = validator.check_url_quality(url)
        suspicious = validator.detect_suspicious_url_patterns(url)
        
        print(f"  {url}")
        print(f"    Valid: {is_valid}")
        print(f"    Length: {quality['length']}")
        print(f"    Suspicious Patterns: {suspicious}")
        print()
    
    # Test email validation
    test_emails = [
        "test@example.com",
        "user.name@domain.org",
        "invalid-email",
        "test@",
        "user@domain"
    ]
    
    print("Email Validation Tests:")
    for email in test_emails:
        is_valid = validator.validate_email_format(email)
        print(f"  {email}: {'✓' if is_valid else '✗'}")
    
    return True

def demo_data_storage():
    """Demo data storage system"""
    print("\n" + "=" * 60)
    print("DEMO: Data Storage System")
    print("=" * 60)
    
    # Create temporary storage
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        storage = DataStorage(tmp.name)
        
        # Store sample data
        print("Storing sample data...")
        
        # Sample URL
        url_data = {
            'url': 'https://demo.example.com',
            'label': 'phish',
            'confidence': 0.85,
            'features': {'length': 25, 'has_redirect': False},
            'timestamp': '2024-01-01T00:00:00',
            'source': 'demo',
            'technique': 'homograph'
        }
        url_id = storage.store_url(url_data)
        print(f"  Stored URL with ID: {url_id}")
        
        # Sample email
        email_data = {
            'subject': 'Demo Email Subject',
            'body': 'This is a demo email body for testing purposes.',
            'sender': 'demo@example.com',
            'recipient': 'user@example.com',
            'label': 'benign',
            'confidence': 0.95,
            'features': {'length': 50, 'has_urgency_words': False},
            'timestamp': '2024-01-01T00:00:00',
            'source': 'demo',
            'template_type': 'newsletter'
        }
        email_id = storage.store_email(email_data)
        print(f"  Stored Email with ID: {email_id}")
        
        # Retrieve data
        print("\nRetrieving stored data...")
        urls = storage.get_urls()
        emails = storage.get_emails()
        
        print(f"  Retrieved {len(urls)} URLs")
        print(f"  Retrieved {len(emails)} emails")
        
        # Get statistics
        stats = storage.get_dataset_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  URLs: {stats.get('urls', {})}")
        print(f"  Emails: {stats.get('emails', {})}")
        
        storage.close()
        os.unlink(tmp.name)  # Clean up temp file
    
    return True

def main():
    """Main demo function"""
    print("Phish-Sim T002 Demo - Data Ingestion & Synthetic Dataset Generator")
    print("=" * 80)
    
    try:
        # Run all demos
        url_dataset = demo_synthetic_url_generation()
        email_dataset = demo_synthetic_email_generation()
        adversarial_dataset = demo_adversarial_generation()
        demo_data_validation()
        demo_data_storage()
        
        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print(f"✓ Generated {len(url_dataset)} synthetic URLs")
        print(f"✓ Generated {len(email_dataset)} synthetic emails")
        print(f"✓ Generated {len(adversarial_dataset)} adversarial URLs")
        print("✓ Validated data quality and formats")
        print("✓ Tested data storage and retrieval")
        
        print(f"\nData Pipeline Features Demonstrated:")
        print(f"  - Synthetic URL generation with 7 obfuscation techniques")
        print(f"  - Synthetic email generation with phishing/benign templates")
        print(f"  - Adversarial content generation with Unicode attacks")
        print(f"  - Comprehensive data validation and quality checks")
        print(f"  - SQLite-based data storage with full CRUD operations")
        print(f"  - Automated testing and validation reporting")
        
        print(f"\nNext Steps:")
        print(f"  - Run full pipeline: python scripts/run_data_pipeline.py --step all")
        print(f"  - Generate larger datasets: python scripts/run_data_pipeline.py --step urls --count 1000")
        print(f"  - Run tests: python -m pytest data/tests/ -v")
        
        print("\n✅ T002 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)