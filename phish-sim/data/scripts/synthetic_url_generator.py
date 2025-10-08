# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Synthetic phishing URL generator with various obfuscation techniques
"""

import random
import string
import base64
import urllib.parse
from typing import List, Dict, Any
from datetime import datetime
import logging

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import SYNTHETIC_CONFIG, get_data_path

logger = logging.getLogger(__name__)

class SyntheticURLGenerator:
    """Generate synthetic phishing URLs with various obfuscation techniques"""
    
    def __init__(self):
        self.legitimate_domains = SYNTHETIC_CONFIG["urls"]["domains"]["legitimate"]
        self.suspicious_domains = SYNTHETIC_CONFIG["urls"]["domains"]["suspicious"]
        self.obfuscation_techniques = SYNTHETIC_CONFIG["urls"]["obfuscation_techniques"]
    
    def generate_homograph_url(self, target_domain: str) -> str:
        """Generate homograph attack URL (Unicode character substitution)"""
        # Common homograph substitutions
        homographs = {
            'a': ['а', 'ɑ'],  # Cyrillic 'а', Latin 'ɑ'
            'e': ['е', 'ε'],  # Cyrillic 'е', Greek 'ε'
            'o': ['о', 'ο'],  # Cyrillic 'о', Greek 'ο'
            'p': ['р'],       # Cyrillic 'р'
            'c': ['с'],       # Cyrillic 'с'
            'x': ['х'],       # Cyrillic 'х'
            'y': ['у'],       # Cyrillic 'у'
            'i': ['і', 'ι'],  # Cyrillic 'і', Greek 'ι'
        }
        
        # Replace some characters with homographs
        modified_domain = target_domain
        for original, substitutes in homographs.items():
            if original in modified_domain and random.random() < 0.3:
                modified_domain = modified_domain.replace(
                    original, random.choice(substitutes), 1
                )
        
        return f"https://{modified_domain}/login"
    
    def generate_redirect_chain_url(self, target_domain: str) -> str:
        """Generate URL with multiple redirects"""
        redirect_domains = [
            "bit.ly", "tinyurl.com", "short.link", "t.co",
            "goo.gl", "ow.ly", "is.gd", "v.gd"
        ]
        
        # Create a chain of redirects
        redirect_url = f"https://{target_domain}/secure-login"
        for _ in range(random.randint(2, 4)):
            shortener = random.choice(redirect_domains)
            redirect_url = f"https://{shortener}/redirect?url={urllib.parse.quote(redirect_url)}"
        
        return redirect_url
    
    def generate_base64_encoded_url(self, target_domain: str) -> str:
        """Generate URL with base64 encoded parameters"""
        target_path = f"https://{target_domain}/login"
        encoded_path = base64.b64encode(target_path.encode()).decode()
        
        # Use a legitimate-looking domain with encoded payload
        fake_domain = random.choice(self.legitimate_domains)
        return f"https://{fake_domain}/redirect?data={encoded_path}"
    
    def generate_subdomain_spoofing_url(self, target_domain: str) -> str:
        """Generate URL with subdomain spoofing"""
        # Create subdomain that looks like the target
        subdomain_parts = target_domain.split('.')
        if len(subdomain_parts) >= 2:
            main_domain = subdomain_parts[-2]
            spoofed_subdomain = f"{main_domain}-security.{random.choice(self.legitimate_domains)}"
        else:
            spoofed_subdomain = f"secure-{target_domain}.{random.choice(self.legitimate_domains)}"
        
        return f"https://{spoofed_subdomain}/verify-account"
    
    def generate_path_traversal_url(self, target_domain: str) -> str:
        """Generate URL with path traversal attempts"""
        traversal_paths = [
            "../../../login",
            "..%2F..%2F..%2Flogin",
            "....//....//....//login",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2flogin"
        ]
        
        fake_domain = random.choice(self.legitimate_domains)
        traversal_path = random.choice(traversal_paths)
        return f"https://{fake_domain}/files/{traversal_path}"
    
    def generate_parameter_pollution_url(self, target_domain: str) -> str:
        """Generate URL with parameter pollution"""
        fake_domain = random.choice(self.legitimate_domains)
        params = [
            "redirect=https://{target_domain}/login",
            "url=https://{target_domain}/login",
            "next=https://{target_domain}/login",
            "return=https://{target_domain}/login"
        ]
        
        param = random.choice(params).format(target_domain=target_domain)
        return f"https://{fake_domain}/auth?{param}"
    
    def generate_typosquatting_url(self, target_domain: str) -> str:
        """Generate typosquatting URL"""
        # Common typosquatting techniques
        typos = {
            'google.com': ['gooogle.com', 'gogle.com', 'google.co', 'googl.com'],
            'microsoft.com': ['microsft.com', 'microsoft.co', 'microsof.com'],
            'apple.com': ['aple.com', 'apple.co', 'appl.com'],
            'amazon.com': ['amazom.com', 'amazon.co', 'amazn.com'],
            'facebook.com': ['facebok.com', 'facebook.co', 'faceboo.com']
        }
        
        if target_domain in typos:
            typo_domain = random.choice(typos[target_domain])
        else:
            # Generate random typo
            typo_domain = target_domain
            if len(typo_domain) > 5:
                pos = random.randint(0, len(typo_domain) - 1)
                typo_domain = typo_domain[:pos] + typo_domain[pos+1:]
        
        return f"https://{typo_domain}/login"
    
    def generate_phishing_url(self, technique: str = None) -> Dict[str, Any]:
        """Generate a single phishing URL using specified technique"""
        if technique is None:
            technique = random.choice(self.obfuscation_techniques)
        
        # Choose a target domain
        target_domain = random.choice(self.legitimate_domains)
        
        # Generate URL based on technique
        if technique == "homograph":
            url = self.generate_homograph_url(target_domain)
        elif technique == "redirect_chain":
            url = self.generate_redirect_chain_url(target_domain)
        elif technique == "base64_encoding":
            url = self.generate_base64_encoded_url(target_domain)
        elif technique == "subdomain_spoofing":
            url = self.generate_subdomain_spoofing_url(target_domain)
        elif technique == "path_traversal":
            url = self.generate_path_traversal_url(target_domain)
        elif technique == "parameter_pollution":
            url = self.generate_parameter_pollution_url(target_domain)
        elif technique == "typosquatting":
            url = self.generate_typosquatting_url(target_domain)
        else:
            # Default to suspicious domain
            url = f"https://{random.choice(self.suspicious_domains)}/login"
        
        return {
            'url': url,
            'label': 'phish',
            'confidence': random.uniform(0.7, 1.0),
            'technique': technique,
            'target_domain': target_domain,
            'source': 'synthetic_phishing',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'length': len(url),
                'has_redirect': 'redirect' in url.lower(),
                'has_encoding': any(enc in url for enc in ['%', 'base64', '=']),
                'subdomain_count': url.count('.') - 1,
                'path_depth': url.count('/') - 2
            }
        }
    
    def generate_benign_url(self) -> Dict[str, Any]:
        """Generate a benign URL"""
        domain = random.choice(self.legitimate_domains)
        paths = ['/', '/about', '/contact', '/help', '/support', '/privacy', '/terms']
        path = random.choice(paths)
        
        url = f"https://{domain}{path}"
        
        return {
            'url': url,
            'label': 'benign',
            'confidence': random.uniform(0.8, 1.0),
            'technique': 'legitimate',
            'target_domain': domain,
            'source': 'synthetic_benign',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'length': len(url),
                'has_redirect': False,
                'has_encoding': False,
                'subdomain_count': 0,
                'path_depth': url.count('/') - 2
            }
        }
    
    def generate_dataset(self, count: int, phishing_ratio: float = 0.3) -> List[Dict[str, Any]]:
        """Generate a dataset of synthetic URLs"""
        logger.info(f"Generating {count} synthetic URLs with {phishing_ratio*100}% phishing ratio")
        
        phishing_count = int(count * phishing_ratio)
        benign_count = count - phishing_count
        
        dataset = []
        
        # Generate phishing URLs
        for _ in range(phishing_count):
            technique = random.choice(self.obfuscation_techniques)
            url_data = self.generate_phishing_url(technique)
            dataset.append(url_data)
        
        # Generate benign URLs
        for _ in range(benign_count):
            url_data = self.generate_benign_url()
            dataset.append(url_data)
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        logger.info(f"Generated {len(dataset)} URLs: {phishing_count} phishing, {benign_count} benign")
        return dataset

def main():
    """Main function to generate synthetic URL dataset"""
    generator = SyntheticURLGenerator()
    
    # Generate dataset
    config = SYNTHETIC_CONFIG["urls"]
    dataset = generator.generate_dataset(
        count=config["count"],
        phishing_ratio=config["phishing_ratio"]
    )
    
    # Save dataset
    import pandas as pd
    df = pd.DataFrame(dataset)
    
    output_path = get_data_path('synthetic', 'synthetic_urls.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic URL dataset saved to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SYNTHETIC URL GENERATION SUMMARY")
    print("="*50)
    print(f"Total URLs: {len(dataset)}")
    print(f"Phishing URLs: {len([d for d in dataset if d['label'] == 'phish'])}")
    print(f"Benign URLs: {len([d for d in dataset if d['label'] == 'benign'])}")
    
    # Technique distribution
    techniques = {}
    for item in dataset:
        if item['label'] == 'phish':
            tech = item['technique']
            techniques[tech] = techniques.get(tech, 0) + 1
    
    print("\nPhishing Techniques Used:")
    for tech, count in techniques.items():
        print(f"  {tech}: {count}")
    print("="*50)

if __name__ == "__main__":
    main()