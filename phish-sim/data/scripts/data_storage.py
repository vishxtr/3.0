# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Data storage and retrieval system for phishing detection datasets
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_data_path, DATA_SCHEMAS

logger = logging.getLogger(__name__)

class DataStorage:
    """SQLite-based data storage and retrieval system"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_data_path('processed', 'phish_sim.db')
        
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        
        # Create tables
        self._create_urls_table()
        self._create_emails_table()
        self._create_analysis_results_table()
        self._create_metadata_table()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _create_urls_table(self):
        """Create URLs table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL CHECK (label IN ('phish', 'benign', 'suspicious')),
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            features TEXT,  -- JSON string
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            technique TEXT,
            target_domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_urls_label ON urls(label)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_urls_source ON urls(source)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_urls_timestamp ON urls(timestamp)")
    
    def _create_emails_table(self):
        """Create emails table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            sender TEXT NOT NULL,
            recipient TEXT NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('phish', 'benign', 'suspicious')),
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            features TEXT,  -- JSON string
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            template_type TEXT,
            company TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_emails_label ON emails(label)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_emails_source ON emails(source)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_emails_timestamp ON emails(timestamp)")
    
    def _create_analysis_results_table(self):
        """Create analysis results table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id INTEGER NOT NULL,
            content_type TEXT NOT NULL CHECK (content_type IN ('url', 'email')),
            analysis_type TEXT NOT NULL,
            score REAL NOT NULL,
            decision TEXT NOT NULL,
            confidence REAL NOT NULL,
            reasons TEXT,  -- JSON string
            processing_time_ms REAL,
            model_version TEXT,
            timestamp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.connection.execute(create_table_sql)
        
        # Create indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_analysis_content_id ON analysis_results(content_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(analysis_type)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(timestamp)")
    
    def _create_metadata_table(self):
        """Create metadata table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            total_records INTEGER NOT NULL,
            label_distribution TEXT,  -- JSON string
            quality_metrics TEXT,  -- JSON string
            validation_report TEXT,  -- JSON string
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.connection.execute(create_table_sql)
    
    def store_url(self, url_data: Dict[str, Any]) -> int:
        """Store a URL record"""
        insert_sql = """
        INSERT OR REPLACE INTO urls 
        (url, label, confidence, features, timestamp, source, technique, target_domain)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        features_json = json.dumps(url_data.get('features', {}))
        
        cursor = self.connection.execute(insert_sql, (
            url_data['url'],
            url_data['label'],
            url_data['confidence'],
            features_json,
            url_data['timestamp'],
            url_data['source'],
            url_data.get('technique'),
            url_data.get('target_domain')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def store_email(self, email_data: Dict[str, Any]) -> int:
        """Store an email record"""
        insert_sql = """
        INSERT OR REPLACE INTO emails 
        (subject, body, sender, recipient, label, confidence, features, timestamp, source, template_type, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        features_json = json.dumps(email_data.get('features', {}))
        
        cursor = self.connection.execute(insert_sql, (
            email_data['subject'],
            email_data['body'],
            email_data['sender'],
            email_data['recipient'],
            email_data['label'],
            email_data['confidence'],
            features_json,
            email_data['timestamp'],
            email_data['source'],
            email_data.get('template_type'),
            email_data.get('company')
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def store_analysis_result(self, analysis_data: Dict[str, Any]) -> int:
        """Store analysis result"""
        insert_sql = """
        INSERT INTO analysis_results 
        (content_id, content_type, analysis_type, score, decision, confidence, reasons, processing_time_ms, model_version, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        reasons_json = json.dumps(analysis_data.get('reasons', []))
        
        cursor = self.connection.execute(insert_sql, (
            analysis_data['content_id'],
            analysis_data['content_type'],
            analysis_data['analysis_type'],
            analysis_data['score'],
            analysis_data['decision'],
            analysis_data['confidence'],
            reasons_json,
            analysis_data.get('processing_time_ms'),
            analysis_data.get('model_version'),
            analysis_data['timestamp']
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def get_urls(self, label: str = None, source: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve URLs with optional filtering"""
        query = "SELECT * FROM urls WHERE 1=1"
        params = []
        
        if label:
            query += " AND label = ?"
            params.append(label)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = dict(row)
            result['features'] = json.loads(result['features']) if result['features'] else {}
            results.append(result)
        
        return results
    
    def get_emails(self, label: str = None, source: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve emails with optional filtering"""
        query = "SELECT * FROM emails WHERE 1=1"
        params = []
        
        if label:
            query += " AND label = ?"
            params.append(label)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = dict(row)
            result['features'] = json.loads(result['features']) if result['features'] else {}
            results.append(result)
        
        return results
    
    def get_analysis_results(self, content_id: int = None, analysis_type: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve analysis results with optional filtering"""
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params = []
        
        if content_id:
            query += " AND content_id = ?"
            params.append(content_id)
        
        if analysis_type:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.connection.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result = dict(row)
            result['reasons'] = json.loads(result['reasons']) if result['reasons'] else []
            results.append(result)
        
        return results
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {}
        
        # URL statistics
        url_stats = self.connection.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN label = 'phish' THEN 1 END) as phishing,
                COUNT(CASE WHEN label = 'benign' THEN 1 END) as benign,
                COUNT(CASE WHEN label = 'suspicious' THEN 1 END) as suspicious,
                AVG(confidence) as avg_confidence
            FROM urls
        """).fetchone()
        
        stats['urls'] = dict(url_stats) if url_stats else {}
        
        # Email statistics
        email_stats = self.connection.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN label = 'phish' THEN 1 END) as phishing,
                COUNT(CASE WHEN label = 'benign' THEN 1 END) as benign,
                COUNT(CASE WHEN label = 'suspicious' THEN 1 END) as suspicious,
                AVG(confidence) as avg_confidence
            FROM emails
        """).fetchone()
        
        stats['emails'] = dict(email_stats) if email_stats else {}
        
        # Analysis statistics
        analysis_stats = self.connection.execute("""
            SELECT 
                COUNT(*) as total_analyses,
                AVG(processing_time_ms) as avg_processing_time,
                COUNT(DISTINCT model_version) as model_versions
            FROM analysis_results
        """).fetchone()
        
        stats['analysis'] = dict(analysis_stats) if analysis_stats else {}
        
        return stats
    
    def store_dataset_metadata(self, dataset_name: str, dataset_type: str, 
                             total_records: int, label_distribution: Dict[str, int],
                             quality_metrics: Dict[str, Any], validation_report: Dict[str, Any]):
        """Store dataset metadata"""
        insert_sql = """
        INSERT OR REPLACE INTO metadata 
        (dataset_name, dataset_type, total_records, label_distribution, quality_metrics, validation_report)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.connection.execute(insert_sql, (
            dataset_name,
            dataset_type,
            total_records,
            json.dumps(label_distribution),
            json.dumps(quality_metrics),
            json.dumps(validation_report)
        ))
        
        self.connection.commit()
        return cursor.lastrowid
    
    def load_dataset_from_csv(self, csv_path: str, dataset_type: str) -> int:
        """Load dataset from CSV file"""
        df = pd.read_csv(csv_path)
        records_stored = 0
        
        if dataset_type == 'url':
            for _, row in df.iterrows():
                url_data = {
                    'url': row['url'],
                    'label': row['label'],
                    'confidence': row['confidence'],
                    'features': row.get('features', {}),
                    'timestamp': row['timestamp'],
                    'source': row['source'],
                    'technique': row.get('technique'),
                    'target_domain': row.get('target_domain')
                }
                self.store_url(url_data)
                records_stored += 1
        
        elif dataset_type == 'email':
            for _, row in df.iterrows():
                email_data = {
                    'subject': row['subject'],
                    'body': row['body'],
                    'sender': row['sender'],
                    'recipient': row['recipient'],
                    'label': row['label'],
                    'confidence': row['confidence'],
                    'features': row.get('features', {}),
                    'timestamp': row['timestamp'],
                    'source': row['source'],
                    'template_type': row.get('template_type'),
                    'company': row.get('company')
                }
                self.store_email(email_data)
                records_stored += 1
        
        logger.info(f"Loaded {records_stored} {dataset_type} records from {csv_path}")
        return records_stored
    
    def export_to_csv(self, dataset_type: str, output_path: str, 
                     label: str = None, source: str = None) -> int:
        """Export dataset to CSV"""
        if dataset_type == 'url':
            data = self.get_urls(label=label, source=source)
        elif dataset_type == 'email':
            data = self.get_emails(label=label, source=source)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(data)} {dataset_type} records to {output_path}")
            return len(data)
        else:
            logger.warning(f"No {dataset_type} data found for export")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def main():
    """Main function to demonstrate data storage functionality"""
    storage = DataStorage()
    
    try:
        # Load synthetic datasets if they exist
        synthetic_urls_path = get_data_path('synthetic', 'synthetic_urls.csv')
        synthetic_emails_path = get_data_path('synthetic', 'synthetic_emails.csv')
        
        if Path(synthetic_urls_path).exists():
            print("Loading synthetic URL dataset...")
            records_loaded = storage.load_dataset_from_csv(synthetic_urls_path, 'url')
            print(f"Loaded {records_loaded} URL records")
        
        if Path(synthetic_emails_path).exists():
            print("Loading synthetic email dataset...")
            records_loaded = storage.load_dataset_from_csv(synthetic_emails_path, 'email')
            print(f"Loaded {records_loaded} email records")
        
        # Get statistics
        stats = storage.get_dataset_statistics()
        print("\nDataset Statistics:")
        print(f"URLs: {stats.get('urls', {})}")
        print(f"Emails: {stats.get('emails', {})}")
        print(f"Analysis: {stats.get('analysis', {})}")
        
        # Export sample data
        sample_urls_path = get_data_path('processed', 'sample_urls.csv')
        sample_emails_path = get_data_path('processed', 'sample_emails.csv')
        
        url_count = storage.export_to_csv('url', sample_urls_path, limit=100)
        email_count = storage.export_to_csv('email', sample_emails_path, limit=100)
        
        print(f"\nExported {url_count} sample URLs and {email_count} sample emails")
        
    finally:
        storage.close()

if __name__ == "__main__":
    main()