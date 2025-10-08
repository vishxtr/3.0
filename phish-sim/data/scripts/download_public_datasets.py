# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Download and preprocess public datasets for phishing detection
"""

import os
import sys
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import PUBLIC_DATASETS, get_data_path, ensure_data_directories

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and preprocess public datasets"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Phish-Sim Data Pipeline/1.0 (Educational Use Only)'
        })
        ensure_data_directories()
    
    def download_phish_tank(self) -> pd.DataFrame:
        """Download PhishTank public feed"""
        logger.info("Downloading PhishTank dataset...")
        
        try:
            url = PUBLIC_DATASETS["phish_tank"]["url"]
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            lines = response.text.strip().split('\n')
            data = []
            
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        data.append({
                            'url': parts[1].strip('"'),
                            'label': 'phish',
                            'confidence': 1.0,
                            'source': 'phish_tank',
                            'timestamp': datetime.now().isoformat(),
                            'verified': parts[0].strip('"') == '1'
                        })
            
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} URLs from PhishTank")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download PhishTank data: {e}")
            return pd.DataFrame()
    
    def download_open_phish(self) -> pd.DataFrame:
        """Download OpenPhish public feed"""
        logger.info("Downloading OpenPhish dataset...")
        
        try:
            url = PUBLIC_DATASETS["open_phish"]["url"]
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse text data (one URL per line)
            urls = response.text.strip().split('\n')
            data = []
            
            for url in urls:
                if url.strip() and url.startswith('http'):
                    data.append({
                        'url': url.strip(),
                        'label': 'phish',
                        'confidence': 1.0,
                        'source': 'open_phish',
                        'timestamp': datetime.now().isoformat(),
                        'verified': True
                    })
            
            df = pd.DataFrame(data)
            logger.info(f"Downloaded {len(df)} URLs from OpenPhish")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download OpenPhish data: {e}")
            return pd.DataFrame()
    
    def generate_benign_urls(self, count: int = 1000) -> pd.DataFrame:
        """Generate benign URLs for training"""
        logger.info(f"Generating {count} benign URLs...")
        
        # Common legitimate domains and paths
        domains = [
            "google.com", "microsoft.com", "apple.com", "amazon.com",
            "facebook.com", "twitter.com", "linkedin.com", "github.com",
            "stackoverflow.com", "reddit.com", "wikipedia.org", "youtube.com",
            "netflix.com", "spotify.com", "dropbox.com", "slack.com"
        ]
        
        paths = [
            "/", "/about", "/contact", "/help", "/support", "/privacy",
            "/terms", "/login", "/signup", "/search", "/news", "/blog",
            "/products", "/services", "/pricing", "/features"
        ]
        
        import random
        data = []
        
        for _ in range(count):
            domain = random.choice(domains)
            path = random.choice(paths)
            url = f"https://{domain}{path}"
            
            data.append({
                'url': url,
                'label': 'benign',
                'confidence': 0.9,
                'source': 'synthetic_benign',
                'timestamp': datetime.now().isoformat(),
                'verified': True
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} benign URLs")
        return df
    
    def download_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Download all available datasets"""
        logger.info("Starting dataset download process...")
        
        datasets = {}
        
        # Download PhishTank
        phish_tank_df = self.download_phish_tank()
        if not phish_tank_df.empty:
            datasets['phish_tank'] = phish_tank_df
            self.save_dataset(phish_tank_df, 'phish_tank.csv')
        
        # Download OpenPhish
        open_phish_df = self.download_open_phish()
        if not open_phish_df.empty:
            datasets['open_phish'] = open_phish_df
            self.save_dataset(open_phish_df, 'open_phish.csv')
        
        # Generate benign URLs
        benign_df = self.generate_benign_urls()
        datasets['benign_urls'] = benign_df
        self.save_dataset(benign_df, 'benign_urls.csv')
        
        # Combine all datasets
        if datasets:
            combined_df = pd.concat(datasets.values(), ignore_index=True)
            datasets['combined'] = combined_df
            self.save_dataset(combined_df, 'combined_urls.csv')
            
            logger.info(f"Total datasets downloaded: {len(datasets)}")
            logger.info(f"Total URLs: {len(combined_df)}")
            logger.info(f"Phishing URLs: {len(combined_df[combined_df['label'] == 'phish'])}")
            logger.info(f"Benign URLs: {len(combined_df[combined_df['label'] == 'benign'])}")
        
        return datasets
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset to file"""
        filepath = get_data_path('public', filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset quality"""
        validation_report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'label_distribution': df['label'].value_counts().to_dict(),
            'source_distribution': df['source'].value_counts().to_dict(),
            'duplicate_urls': df['url'].duplicated().sum(),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Check for required fields
        required_fields = ['url', 'label', 'confidence', 'source', 'timestamp']
        missing_fields = [field for field in required_fields if field not in df.columns]
        validation_report['missing_required_fields'] = missing_fields
        
        # Check label values
        valid_labels = ['phish', 'benign', 'suspicious']
        invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique().tolist()
        validation_report['invalid_labels'] = invalid_labels
        
        return validation_report

def main():
    """Main function to download and process datasets"""
    downloader = DatasetDownloader()
    
    # Download all datasets
    datasets = downloader.download_all_datasets()
    
    # Validate and report
    if 'combined' in datasets:
        validation_report = downloader.validate_dataset(datasets['combined'])
        
        # Save validation report
        report_path = get_data_path('validation', 'dataset_validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info("Dataset download and validation complete!")
        logger.info(f"Validation report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Total URLs: {validation_report['total_records']}")
        print(f"Phishing URLs: {validation_report['label_distribution'].get('phish', 0)}")
        print(f"Benign URLs: {validation_report['label_distribution'].get('benign', 0)}")
        print(f"Duplicate URLs: {validation_report['duplicate_urls']}")
        print(f"Missing required fields: {validation_report['missing_required_fields']}")
        print("="*50)

if __name__ == "__main__":
    main()