# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Data configuration and constants for Phish-Sim
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Base paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "processed"
SYNTHETIC_DATA_DIR = BASE_DIR / "synthetic"
PUBLIC_DATA_DIR = BASE_DIR / "public"
VALIDATION_DATA_DIR = BASE_DIR / "validation"

# Public dataset URLs and sources
PUBLIC_DATASETS = {
    "phish_tank": {
        "name": "PhishTank Public Feed",
        "url": "http://data.phishtank.com/data/online-valid.csv",
        "description": "Public phishing URLs from PhishTank",
        "format": "csv",
        "update_frequency": "daily"
    },
    "open_phish": {
        "name": "OpenPhish Public Feed",
        "url": "https://openphish.com/feed.txt",
        "description": "Public phishing URLs from OpenPhish",
        "format": "txt",
        "update_frequency": "hourly"
    },
    "enron_emails": {
        "name": "Enron Email Dataset",
        "url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
        "description": "Legitimate emails from Enron corpus",
        "format": "tar.gz",
        "update_frequency": "static"
    },
    "common_crawl": {
        "name": "Common Crawl Sample",
        "url": "https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2023-23/segments/1685224646237.23/wet/CC-MAIN-2023-05-28-20230528195412-00000.warc.wet.gz",
        "description": "Sample of legitimate web content",
        "format": "warc.wet.gz",
        "update_frequency": "monthly"
    }
}

# Synthetic data generation parameters
SYNTHETIC_CONFIG = {
    "urls": {
        "count": 10000,
        "phishing_ratio": 0.3,  # 30% phishing, 70% benign
        "domains": {
            "legitimate": [
                "google.com", "microsoft.com", "apple.com", "amazon.com",
                "facebook.com", "twitter.com", "linkedin.com", "github.com",
                "stackoverflow.com", "reddit.com", "wikipedia.org", "youtube.com"
            ],
            "suspicious": [
                "secure-bank-update.com", "paypal-security-alert.net",
                "microsoft-account-verify.org", "amazon-payment-confirm.com",
                "apple-id-locked.info", "facebook-security-check.net"
            ]
        },
        "obfuscation_techniques": [
            "homograph", "redirect_chain", "base64_encoding", "url_shortening",
            "subdomain_spoofing", "path_traversal", "parameter_pollution"
        ]
    },
    "emails": {
        "count": 5000,
        "phishing_ratio": 0.4,  # 40% phishing, 60% benign
        "templates": {
            "phishing": [
                "urgent_security_alert", "account_verification", "payment_confirmation",
                "password_reset", "suspicious_activity", "account_locked",
                "billing_dispute", "refund_notification"
            ],
            "benign": [
                "newsletter", "order_confirmation", "meeting_invitation",
                "system_notification", "marketing_promotion", "account_statement"
            ]
        }
    }
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    "min_url_length": 10,
    "max_url_length": 2048,
    "min_email_length": 50,
    "max_email_length": 10000,
    "required_fields": ["content", "label", "timestamp", "source"],
    "label_values": ["phish", "benign", "suspicious"]
}

# File formats and schemas
DATA_SCHEMAS = {
    "url_record": {
        "url": "string",
        "label": "string",  # phish, benign, suspicious
        "confidence": "float",  # 0.0 to 1.0
        "features": "dict",
        "timestamp": "string",
        "source": "string"
    },
    "email_record": {
        "subject": "string",
        "body": "string",
        "sender": "string",
        "recipient": "string",
        "label": "string",
        "confidence": "float",
        "features": "dict",
        "timestamp": "string",
        "source": "string"
    }
}

# Validation rules
VALIDATION_RULES = {
    "url_validation": [
        "valid_url_format",
        "accessible_domain",
        "not_malware_blacklisted",
        "reasonable_response_time"
    ],
    "email_validation": [
        "valid_email_format",
        "reasonable_length",
        "not_spam_blacklisted",
        "contains_required_fields"
    ]
}

def get_data_path(data_type: str, filename: str = None) -> Path:
    """Get the full path for a data file"""
    base_paths = {
        "raw": RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "synthetic": SYNTHETIC_DATA_DIR,
        "public": PUBLIC_DATA_DIR,
        "validation": VALIDATION_DATA_DIR
    }
    
    if data_type not in base_paths:
        raise ValueError(f"Unknown data type: {data_type}")
    
    base_path = base_paths[data_type]
    base_path.mkdir(parents=True, exist_ok=True)
    
    if filename:
        return base_path / filename
    return base_path

def ensure_data_directories():
    """Ensure all data directories exist"""
    directories = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
        PUBLIC_DATA_DIR, VALIDATION_DATA_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return True