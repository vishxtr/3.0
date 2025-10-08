#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple demo script for T002 - Data ingestion & synthetic dataset generator
(No external dependencies required)
"""

import sys
import json
import random
import base64
import urllib.parse
from pathlib import Path
from datetime import datetime

def demo_synthetic_url_generation():
    """Demo synthetic URL generation (simplified)"""
    print("=" * 60)
    print("DEMO: Synthetic URL Generation")
    print("=" * 60)
    
    # Simulate URL generation
    legitimate_domains = ["google.com", "microsoft.com", "apple.com", "amazon.com"]
    suspicious_domains = ["secure-bank-update.com", "paypal-security-alert.net"]
    
    techniques = ["homograph", "redirect_chain", "base64_encoding", "subdomain_spoofing"]
    
    print("Generating 10 synthetic URLs (30% phishing ratio)...")
    
    dataset = []
    for i in range(10):
        if i < 3:  # 30% phishing
            domain = random.choice(suspicious_domains)
            technique = random.choice(techniques)
            label = "phish"
            confidence = random.uniform(0.7, 1.0)
        else:  # 70% benign
            domain = random.choice(legitimate_domains)
            technique = "legitimate"
            label = "benign"
            confidence = random.uniform(0.8, 1.0)
        
        url = f"https://{domain}/login"
        
        dataset.append({
            'url': url,
            'label': label,
            'confidence': confidence,
            'technique': technique,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"\nGenerated {len(dataset)} URLs:")
    for i, item in enumerate(dataset, 1):
        print(f"{i:2d}. [{item['label'].upper():8s}] {item['url']}")
        if item['label'] == 'phish':
            print(f"     Technique: {item['technique']}")
            print(f"     Confidence: {item['confidence']:.2f}")
    
    return dataset

def demo_synthetic_email_generation():
    """Demo synthetic email generation (simplified)"""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Email Generation")
    print("=" * 60)
    
    phishing_templates = [
        "URGENT: Security Alert - Account Compromise Detected",
        "Account Verification Required - Action Needed",
        "Payment Confirmation - Please Review"
    ]
    
    benign_templates = [
        "Weekly Newsletter - Latest Updates",
        "Order Confirmation - #12345",
        "Meeting Invitation - Project Review"
    ]
    
    companies = ["Microsoft", "Google", "Apple", "Amazon"]
    
    print("Generating 8 synthetic emails (40% phishing ratio)...")
    
    dataset = []
    for i in range(8):
        company = random.choice(companies)
        
        if i < 3:  # 40% phishing
            subject = random.choice(phishing_templates)
            body = f"Dear Customer,\n\nWe have detected suspicious activity on your {company} account.\n\nClick here to verify: http://{company.lower()}-security-verify.net\n\nBest regards,\n{company} Security Team"
            label = "phish"
            confidence = random.uniform(0.8, 1.0)
            template_type = "security_alert"
        else:  # 60% benign
            subject = random.choice(benign_templates)
            body = f"Hello,\n\nThank you for using {company} services.\n\nThis is a legitimate communication from {company}.\n\nBest regards,\n{company} Team"
            label = "benign"
            confidence = random.uniform(0.9, 1.0)
            template_type = "newsletter"
        
        dataset.append({
            'subject': subject,
            'body': body,
            'sender': f"noreply@{company.lower()}.com",
            'label': label,
            'confidence': confidence,
            'template_type': template_type,
            'company': company,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"\nGenerated {len(dataset)} emails:")
    for i, item in enumerate(dataset, 1):
        print(f"{i:2d}. [{item['label'].upper():8s}] {item['subject']}")
        print(f"     From: {item['sender']}")
        if item['label'] == 'phish':
            print(f"     Template: {item['template_type']}")
    
    return dataset

def demo_adversarial_generation():
    """Demo adversarial generation (simplified)"""
    print("\n" + "=" * 60)
    print("DEMO: Adversarial Content Generation")
    print("=" * 60)
    
    base_url = "https://google.com/login"
    print(f"Base URL: {base_url}")
    print("\nGenerating adversarial versions...")
    
    # Simulate different obfuscation techniques
    techniques = [
        ("unicode_normalization", "https://gооgle.com/login"),  # Cyrillic 'о'
        ("character_substitution", "https://g00gle.com/login"),  # 0 instead of o
        ("url_encoding", "https://google.com/%6C%6F%67%69%6E"),  # URL encoded
        ("base64_encoding", "https://google.com/redirect?data=aHR0cHM6Ly9nb29nbGUuY29tL2xvZ2lu"),  # Base64
        ("subdomain_spoofing", "https://google-security.legitimate-site.com/login")
    ]
    
    adversarial_urls = []
    for technique, adversarial_url in techniques:
        obfuscation_score = random.uniform(0.3, 0.9)
        unicode_ratio = 0.1 if technique == "unicode_normalization" else 0.0
        
        adversarial_urls.append({
            'url': adversarial_url,
            'technique': technique,
            'obfuscation_score': obfuscation_score,
            'unicode_ratio': unicode_ratio,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"\nGenerated {len(adversarial_urls)} adversarial URLs:")
    for i, item in enumerate(adversarial_urls, 1):
        print(f"{i:2d}. [{item['technique']:20s}] {item['url']}")
        print(f"     Obfuscation Score: {item['obfuscation_score']:.3f}")
        print(f"     Unicode Ratio: {item['unicode_ratio']:.3f}")
    
    return adversarial_urls

def demo_data_validation():
    """Demo data validation (simplified)"""
    print("\n" + "=" * 60)
    print("DEMO: Data Validation")
    print("=" * 60)
    
    def validate_url_format(url):
        """Simple URL validation"""
        return url.startswith(('http://', 'https://')) and '.' in url
    
    def validate_email_format(email):
        """Simple email validation"""
        return '@' in email and '.' in email.split('@')[-1]
    
    # Test URL validation
    test_urls = [
        "https://google.com/login",
        "http://192.168.1.1",
        "https://example.com/very/long/path",
        "not-a-url",
        "https://test.com"
    ]
    
    print("URL Validation Tests:")
    for url in test_urls:
        is_valid = validate_url_format(url)
        length = len(url)
        has_ip = any(c.isdigit() for c in url.split('.')[-1].split('/')[0])
        
        print(f"  {url}")
        print(f"    Valid: {'✓' if is_valid else '✗'}")
        print(f"    Length: {length}")
        print(f"    Contains IP: {'✓' if has_ip else '✗'}")
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
        is_valid = validate_email_format(email)
        print(f"  {email}: {'✓' if is_valid else '✗'}")
    
    return True

def save_demo_data(url_dataset, email_dataset, adversarial_dataset):
    """Save demo data to files"""
    print("\n" + "=" * 60)
    print("DEMO: Data Storage")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)
    
    # Save URL dataset
    url_file = data_dir / "demo_urls.json"
    with open(url_file, 'w') as f:
        json.dump(url_dataset, f, indent=2)
    print(f"✓ Saved {len(url_dataset)} URLs to {url_file}")
    
    # Save email dataset
    email_file = data_dir / "demo_emails.json"
    with open(email_file, 'w') as f:
        json.dump(email_dataset, f, indent=2)
    print(f"✓ Saved {len(email_dataset)} emails to {email_file}")
    
    # Save adversarial dataset
    adversarial_file = data_dir / "demo_adversarial.json"
    with open(adversarial_file, 'w') as f:
        json.dump(adversarial_dataset, f, indent=2)
    print(f"✓ Saved {len(adversarial_dataset)} adversarial URLs to {adversarial_file}")
    
    # Create summary report
    summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'datasets_generated': {
            'urls': len(url_dataset),
            'emails': len(email_dataset),
            'adversarial_urls': len(adversarial_dataset)
        },
        'url_labels': {
            'phish': len([u for u in url_dataset if u['label'] == 'phish']),
            'benign': len([u for u in url_dataset if u['label'] == 'benign'])
        },
        'email_labels': {
            'phish': len([e for e in email_dataset if e['label'] == 'phish']),
            'benign': len([e for e in email_dataset if e['label'] == 'benign'])
        },
        'techniques_used': list(set([u['technique'] for u in url_dataset if u['label'] == 'phish']))
    }
    
    summary_file = data_dir / "demo_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved demo summary to {summary_file}")
    
    return summary

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
        summary = save_demo_data(url_dataset, email_dataset, adversarial_dataset)
        
        print("\n" + "=" * 80)
        print("DEMO SUMMARY")
        print("=" * 80)
        print(f"✓ Generated {summary['datasets_generated']['urls']} synthetic URLs")
        print(f"✓ Generated {summary['datasets_generated']['emails']} synthetic emails")
        print(f"✓ Generated {summary['datasets_generated']['adversarial_urls']} adversarial URLs")
        print("✓ Validated data quality and formats")
        print("✓ Saved all datasets to JSON files")
        
        print(f"\nData Pipeline Features Demonstrated:")
        print(f"  - Synthetic URL generation with obfuscation techniques")
        print(f"  - Synthetic email generation with phishing/benign templates")
        print(f"  - Adversarial content generation with Unicode attacks")
        print(f"  - Data validation and quality checks")
        print(f"  - JSON-based data storage")
        
        print(f"\nGenerated Data Summary:")
        print(f"  - URLs: {summary['url_labels']['phish']} phishing, {summary['url_labels']['benign']} benign")
        print(f"  - Emails: {summary['email_labels']['phish']} phishing, {summary['email_labels']['benign']} benign")
        print(f"  - Techniques: {', '.join(summary['techniques_used'])}")
        
        print(f"\nNext Steps:")
        print(f"  - Install dependencies: pip install -r requirements.txt")
        print(f"  - Run full pipeline: python scripts/run_data_pipeline.py --step all")
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