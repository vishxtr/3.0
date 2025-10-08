# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Main data pipeline script to orchestrate all data generation and processing
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import get_data_path, SYNTHETIC_CONFIG, ensure_data_directories
from scripts.download_public_datasets import DatasetDownloader
from scripts.synthetic_url_generator import SyntheticURLGenerator
from scripts.synthetic_email_generator import SyntheticEmailGenerator
from scripts.adversarial_generator import AdversarialGenerator
from scripts.data_validator import DataValidator
from scripts.data_storage import DataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(get_data_path('validation', 'pipeline.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self):
        ensure_data_directories()
        self.downloader = DatasetDownloader()
        self.url_generator = SyntheticURLGenerator()
        self.email_generator = SyntheticEmailGenerator()
        self.adversarial_generator = AdversarialGenerator()
        self.validator = DataValidator()
        self.storage = DataStorage()
        
        self.pipeline_start_time = datetime.now()
        self.pipeline_report = {
            'start_time': self.pipeline_start_time.isoformat(),
            'steps_completed': [],
            'datasets_generated': {},
            'validation_reports': {},
            'errors': [],
            'end_time': None,
            'total_duration_minutes': 0
        }
    
    def run_public_dataset_download(self) -> bool:
        """Download public datasets"""
        logger.info("Starting public dataset download...")
        
        try:
            datasets = self.downloader.download_all_datasets()
            
            if datasets:
                self.pipeline_report['steps_completed'].append('public_dataset_download')
                self.pipeline_report['datasets_generated']['public'] = {
                    'count': len(datasets),
                    'types': list(datasets.keys())
                }
                
                # Store in database
                for dataset_name, df in datasets.items():
                    if isinstance(df, list):
                        for item in df:
                            if 'url' in item:
                                self.storage.store_url(item)
                            elif 'subject' in item:
                                self.storage.store_email(item)
                
                logger.info(f"Successfully downloaded {len(datasets)} public datasets")
                return True
            else:
                logger.warning("No public datasets were downloaded")
                return False
                
        except Exception as e:
            error_msg = f"Public dataset download failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_report['errors'].append(error_msg)
            return False
    
    def run_synthetic_url_generation(self) -> bool:
        """Generate synthetic URL dataset"""
        logger.info("Starting synthetic URL generation...")
        
        try:
            config = SYNTHETIC_CONFIG["urls"]
            dataset = self.url_generator.generate_dataset(
                count=config["count"],
                phishing_ratio=config["phishing_ratio"]
            )
            
            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(dataset)
            output_path = get_data_path('synthetic', 'synthetic_urls.csv')
            df.to_csv(output_path, index=False)
            
            # Store in database
            for url_data in dataset:
                self.storage.store_url(url_data)
            
            self.pipeline_report['steps_completed'].append('synthetic_url_generation')
            self.pipeline_report['datasets_generated']['synthetic_urls'] = {
                'count': len(dataset),
                'phishing_ratio': config["phishing_ratio"]
            }
            
            logger.info(f"Generated {len(dataset)} synthetic URLs")
            return True
            
        except Exception as e:
            error_msg = f"Synthetic URL generation failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_report['errors'].append(error_msg)
            return False
    
    def run_synthetic_email_generation(self) -> bool:
        """Generate synthetic email dataset"""
        logger.info("Starting synthetic email generation...")
        
        try:
            config = SYNTHETIC_CONFIG["emails"]
            dataset = self.email_generator.generate_dataset(
                count=config["count"],
                phishing_ratio=config["phishing_ratio"]
            )
            
            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(dataset)
            output_path = get_data_path('synthetic', 'synthetic_emails.csv')
            df.to_csv(output_path, index=False)
            
            # Store in database
            for email_data in dataset:
                self.storage.store_email(email_data)
            
            self.pipeline_report['steps_completed'].append('synthetic_email_generation')
            self.pipeline_report['datasets_generated']['synthetic_emails'] = {
                'count': len(dataset),
                'phishing_ratio': config["phishing_ratio"]
            }
            
            logger.info(f"Generated {len(dataset)} synthetic emails")
            return True
            
        except Exception as e:
            error_msg = f"Synthetic email generation failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_report['errors'].append(error_msg)
            return False
    
    def run_adversarial_generation(self) -> bool:
        """Generate adversarial dataset"""
        logger.info("Starting adversarial generation...")
        
        try:
            # Generate adversarial URLs
            base_urls = [
                "https://google.com/login",
                "https://microsoft.com/verify",
                "https://apple.com/account",
                "https://amazon.com/signin",
                "https://paypal.com/login"
            ]
            
            adversarial_urls = []
            for url in base_urls:
                for technique in self.adversarial_generator.obfuscation_methods:
                    adversarial_url = self.adversarial_generator.generate_adversarial_url(url, technique)
                    adversarial_urls.append(adversarial_url)
            
            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(adversarial_urls)
            output_path = get_data_path('synthetic', 'adversarial_urls.csv')
            df.to_csv(output_path, index=False)
            
            # Store in database
            for url_data in adversarial_urls:
                self.storage.store_url(url_data)
            
            self.pipeline_report['steps_completed'].append('adversarial_generation')
            self.pipeline_report['datasets_generated']['adversarial_urls'] = {
                'count': len(adversarial_urls),
                'techniques_used': len(self.adversarial_generator.obfuscation_methods)
            }
            
            logger.info(f"Generated {len(adversarial_urls)} adversarial URLs")
            return True
            
        except Exception as e:
            error_msg = f"Adversarial generation failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_report['errors'].append(error_msg)
            return False
    
    def run_data_validation(self) -> bool:
        """Run data validation on all datasets"""
        logger.info("Starting data validation...")
        
        try:
            validation_reports = {}
            
            # Validate synthetic URLs
            synthetic_urls_path = get_data_path('synthetic', 'synthetic_urls.csv')
            if Path(synthetic_urls_path).exists():
                import pandas as pd
                df = pd.read_csv(synthetic_urls_path)
                url_validation = self.validator.comprehensive_validation(df, 'url')
                validation_reports['synthetic_urls'] = url_validation
            
            # Validate synthetic emails
            synthetic_emails_path = get_data_path('synthetic', 'synthetic_emails.csv')
            if Path(synthetic_emails_path).exists():
                import pandas as pd
                df = pd.read_csv(synthetic_emails_path)
                email_validation = self.validator.comprehensive_validation(df, 'email')
                validation_reports['synthetic_emails'] = email_validation
            
            # Validate adversarial URLs
            adversarial_urls_path = get_data_path('synthetic', 'adversarial_urls.csv')
            if Path(adversarial_urls_path).exists():
                import pandas as pd
                df = pd.read_csv(adversarial_urls_path)
                adversarial_validation = self.validator.comprehensive_validation(df, 'url')
                validation_reports['adversarial_urls'] = adversarial_validation
            
            # Save validation reports
            for dataset_name, report in validation_reports.items():
                report_path = get_data_path('validation', f'{dataset_name}_validation_report.json')
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
            
            self.pipeline_report['steps_completed'].append('data_validation')
            self.pipeline_report['validation_reports'] = validation_reports
            
            logger.info(f"Completed validation for {len(validation_reports)} datasets")
            return True
            
        except Exception as e:
            error_msg = f"Data validation failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_report['errors'].append(error_msg)
            return False
    
    def generate_pipeline_report(self) -> dict:
        """Generate final pipeline report"""
        end_time = datetime.now()
        duration = (end_time - self.pipeline_start_time).total_seconds() / 60
        
        self.pipeline_report['end_time'] = end_time.isoformat()
        self.pipeline_report['total_duration_minutes'] = round(duration, 2)
        
        # Get database statistics
        try:
            stats = self.storage.get_dataset_statistics()
            self.pipeline_report['database_statistics'] = stats
        except Exception as e:
            logger.warning(f"Could not retrieve database statistics: {e}")
        
        # Save pipeline report
        report_path = get_data_path('validation', 'pipeline_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.pipeline_report, f, indent=2)
        
        return self.pipeline_report
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline"""
        logger.info("Starting full data pipeline...")
        
        steps = [
            ('Public Dataset Download', self.run_public_dataset_download),
            ('Synthetic URL Generation', self.run_synthetic_url_generation),
            ('Synthetic Email Generation', self.run_synthetic_email_generation),
            ('Adversarial Generation', self.run_adversarial_generation),
            ('Data Validation', self.run_data_validation)
        ]
        
        success_count = 0
        for step_name, step_function in steps:
            logger.info(f"Running step: {step_name}")
            if step_function():
                success_count += 1
                logger.info(f"✓ {step_name} completed successfully")
            else:
                logger.error(f"✗ {step_name} failed")
        
        # Generate final report
        report = self.generate_pipeline_report()
        
        logger.info(f"Pipeline completed: {success_count}/{len(steps)} steps successful")
        logger.info(f"Total duration: {report['total_duration_minutes']} minutes")
        
        if report['errors']:
            logger.warning(f"Pipeline completed with {len(report['errors'])} errors")
            for error in report['errors']:
                logger.warning(f"  - {error}")
        
        return success_count == len(steps)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'storage'):
            self.storage.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Phish-Sim data pipeline')
    parser.add_argument('--step', choices=[
        'download', 'urls', 'emails', 'adversarial', 'validate', 'all'
    ], default='all', help='Pipeline step to run')
    parser.add_argument('--count', type=int, help='Number of records to generate')
    parser.add_argument('--phishing-ratio', type=float, help='Phishing ratio for synthetic data')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline()
    
    try:
        if args.step == 'download':
            success = pipeline.run_public_dataset_download()
        elif args.step == 'urls':
            success = pipeline.run_synthetic_url_generation()
        elif args.step == 'emails':
            success = pipeline.run_synthetic_email_generation()
        elif args.step == 'adversarial':
            success = pipeline.run_adversarial_generation()
        elif args.step == 'validate':
            success = pipeline.run_data_validation()
        elif args.step == 'all':
            success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("Pipeline step completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline step failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()