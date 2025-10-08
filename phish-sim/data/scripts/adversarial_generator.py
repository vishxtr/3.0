# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Adversarial content generator for creating challenging phishing cases
"""

import random
import string
import re
import base64
import urllib.parse
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_data_path

logger = logging.getLogger(__name__)

class AdversarialGenerator:
    """Generate adversarial content to challenge detection systems"""
    
    def __init__(self):
        self.legitimate_domains = [
            "google.com", "microsoft.com", "apple.com", "amazon.com",
            "facebook.com", "twitter.com", "linkedin.com", "paypal.com"
        ]
        
        # Common obfuscation techniques
        self.obfuscation_methods = [
            "unicode_normalization", "character_substitution", "encoding_variations",
            "whitespace_manipulation", "case_variations", "punctuation_insertion",
            "url_encoding", "base64_encoding", "hex_encoding", "html_entities"
        ]
    
    def unicode_normalization_attack(self, text: str) -> str:
        """Apply Unicode normalization attacks"""
        # Replace ASCII characters with visually similar Unicode characters
        unicode_map = {
            'a': 'а',  # Cyrillic
            'e': 'е',  # Cyrillic
            'o': 'о',  # Cyrillic
            'p': 'р',  # Cyrillic
            'c': 'с',  # Cyrillic
            'x': 'х',  # Cyrillic
            'y': 'у',  # Cyrillic
            'i': 'і',  # Cyrillic
            'A': 'А',  # Cyrillic
            'E': 'Е',  # Cyrillic
            'O': 'О',  # Cyrillic
            'P': 'Р',  # Cyrillic
            'C': 'С',  # Cyrillic
            'X': 'Х',  # Cyrillic
            'Y': 'У',  # Cyrillic
            'I': 'І',  # Cyrillic
        }
        
        result = text
        for ascii_char, unicode_char in unicode_map.items():
            if random.random() < 0.3:  # 30% chance to replace
                result = result.replace(ascii_char, unicode_char)
        
        return result
    
    def character_substitution_attack(self, text: str) -> str:
        """Apply character substitution attacks"""
        # Common character substitutions
        substitutions = {
            '0': 'O', '1': 'l', '3': 'E', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B', '9': 'g',
            'O': '0', 'l': '1', 'E': '3', 'A': '4', 'S': '5',
            'G': '6', 'T': '7', 'B': '8', 'g': '9'
        }
        
        result = text
        for original, substitute in substitutions.items():
            if random.random() < 0.2:  # 20% chance to replace
                result = result.replace(original, substitute)
        
        return result
    
    def encoding_variation_attack(self, text: str) -> str:
        """Apply various encoding attacks"""
        encoding_type = random.choice(['url', 'base64', 'hex', 'html'])
        
        if encoding_type == 'url':
            return urllib.parse.quote(text)
        elif encoding_type == 'base64':
            encoded = base64.b64encode(text.encode()).decode()
            return f"data:text/plain;base64,{encoded}"
        elif encoding_type == 'hex':
            return text.encode().hex()
        elif encoding_type == 'html':
            # HTML entity encoding
            html_entities = {
                '<': '&lt;', '>': '&gt;', '&': '&amp;',
                '"': '&quot;', "'": '&#39;'
            }
            result = text
            for char, entity in html_entities.items():
                result = result.replace(char, entity)
            return result
        
        return text
    
    def whitespace_manipulation_attack(self, text: str) -> str:
        """Apply whitespace manipulation attacks"""
        # Insert invisible characters
        invisible_chars = ['\u200B', '\u200C', '\u200D', '\uFEFF']  # Zero-width characters
        
        result = text
        for i in range(len(text)):
            if random.random() < 0.1:  # 10% chance to insert invisible char
                char = random.choice(invisible_chars)
                result = result[:i] + char + result[i:]
        
        return result
    
    def case_variation_attack(self, text: str) -> str:
        """Apply case variation attacks"""
        # Randomly change case of characters
        result = ""
        for char in text:
            if char.isalpha():
                if random.random() < 0.3:  # 30% chance to change case
                    result += char.swapcase()
                else:
                    result += char
            else:
                result += char
        
        return result
    
    def punctuation_insertion_attack(self, text: str) -> str:
        """Insert punctuation to break patterns"""
        punctuation = ['.', ',', ';', ':', '!', '?', '-', '_']
        
        result = text
        for i in range(len(text)):
            if random.random() < 0.05:  # 5% chance to insert punctuation
                punct = random.choice(punctuation)
                result = result[:i] + punct + result[i:]
        
        return result
    
    def generate_adversarial_url(self, base_url: str, technique: str = None) -> Dict[str, Any]:
        """Generate an adversarial URL"""
        if technique is None:
            technique = random.choice(self.obfuscation_methods)
        
        original_url = base_url
        
        # Apply obfuscation technique
        if technique == "unicode_normalization":
            adversarial_url = self.unicode_normalization_attack(base_url)
        elif technique == "character_substitution":
            adversarial_url = self.character_substitution_attack(base_url)
        elif technique == "encoding_variations":
            adversarial_url = self.encoding_variation_attack(base_url)
        elif technique == "whitespace_manipulation":
            adversarial_url = self.whitespace_manipulation_attack(base_url)
        elif technique == "case_variations":
            adversarial_url = self.case_variation_attack(base_url)
        elif technique == "punctuation_insertion":
            adversarial_url = self.punctuation_insertion_attack(base_url)
        else:
            adversarial_url = base_url
        
        return {
            'url': adversarial_url,
            'original_url': original_url,
            'label': 'phish',
            'confidence': random.uniform(0.6, 0.9),  # Lower confidence due to obfuscation
            'technique': technique,
            'source': 'adversarial',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'length': len(adversarial_url),
                'obfuscation_score': self.calculate_obfuscation_score(adversarial_url),
                'unicode_ratio': self.calculate_unicode_ratio(adversarial_url),
                'encoding_ratio': self.calculate_encoding_ratio(adversarial_url),
                'invisible_chars': self.count_invisible_chars(adversarial_url)
            }
        }
    
    def generate_adversarial_email(self, base_email: Dict[str, Any], technique: str = None) -> Dict[str, Any]:
        """Generate an adversarial email"""
        if technique is None:
            technique = random.choice(self.obfuscation_methods)
        
        adversarial_email = base_email.copy()
        
        # Apply obfuscation to subject and body
        if technique == "unicode_normalization":
            adversarial_email['subject'] = self.unicode_normalization_attack(base_email['subject'])
            adversarial_email['body'] = self.unicode_normalization_attack(base_email['body'])
        elif technique == "character_substitution":
            adversarial_email['subject'] = self.character_substitution_attack(base_email['subject'])
            adversarial_email['body'] = self.character_substitution_attack(base_email['body'])
        elif technique == "whitespace_manipulation":
            adversarial_email['subject'] = self.whitespace_manipulation_attack(base_email['subject'])
            adversarial_email['body'] = self.whitespace_manipulation_attack(base_email['body'])
        elif technique == "case_variations":
            adversarial_email['subject'] = self.case_variation_attack(base_email['subject'])
            adversarial_email['body'] = self.case_variation_attack(base_email['body'])
        elif technique == "punctuation_insertion":
            adversarial_email['subject'] = self.punctuation_insertion_attack(base_email['subject'])
            adversarial_email['body'] = self.punctuation_insertion_attack(base_email['body'])
        
        # Update features
        adversarial_email['technique'] = technique
        adversarial_email['source'] = 'adversarial'
        adversarial_email['confidence'] = random.uniform(0.6, 0.9)
        adversarial_email['features'].update({
            'obfuscation_score': self.calculate_obfuscation_score(
                adversarial_email['subject'] + ' ' + adversarial_email['body']
            ),
            'unicode_ratio': self.calculate_unicode_ratio(
                adversarial_email['subject'] + ' ' + adversarial_email['body']
            ),
            'invisible_chars': self.count_invisible_chars(
                adversarial_email['subject'] + ' ' + adversarial_email['body']
            )
        })
        
        return adversarial_email
    
    def calculate_obfuscation_score(self, text: str) -> float:
        """Calculate obfuscation score (0.0 to 1.0)"""
        score = 0.0
        
        # Unicode characters
        unicode_chars = sum(1 for c in text if ord(c) > 127)
        score += min(unicode_chars / len(text), 0.3)
        
        # Encoding patterns
        if '%' in text:
            score += 0.2
        if '=' in text and len(text) > 20:
            score += 0.2
        if '&' in text and ';' in text:
            score += 0.1
        
        # Invisible characters
        invisible_chars = self.count_invisible_chars(text)
        score += min(invisible_chars / len(text), 0.2)
        
        return min(score, 1.0)
    
    def calculate_unicode_ratio(self, text: str) -> float:
        """Calculate ratio of Unicode characters"""
        unicode_chars = sum(1 for c in text if ord(c) > 127)
        return unicode_chars / len(text) if text else 0.0
    
    def calculate_encoding_ratio(self, text: str) -> float:
        """Calculate ratio of encoded content"""
        encoded_chars = sum(1 for c in text if c in '%=&;')
        return encoded_chars / len(text) if text else 0.0
    
    def count_invisible_chars(self, text: str) -> int:
        """Count invisible characters"""
        invisible_chars = ['\u200B', '\u200C', '\u200D', '\uFEFF']
        return sum(text.count(char) for char in invisible_chars)
    
    def generate_adversarial_dataset(self, base_dataset: List[Dict[str, Any]], 
                                   adversarial_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """Generate adversarial versions of existing dataset"""
        logger.info(f"Generating adversarial dataset with {adversarial_ratio*100}% adversarial ratio")
        
        adversarial_count = int(len(base_dataset) * adversarial_ratio)
        adversarial_indices = random.sample(range(len(base_dataset)), adversarial_count)
        
        adversarial_dataset = []
        
        for i, item in enumerate(base_dataset):
            if i in adversarial_indices:
                # Generate adversarial version
                if 'url' in item:
                    # URL adversarial
                    technique = random.choice(self.obfuscation_methods)
                    adversarial_item = self.generate_adversarial_url(item['url'], technique)
                else:
                    # Email adversarial
                    technique = random.choice(self.obfuscation_methods)
                    adversarial_item = self.generate_adversarial_email(item, technique)
                
                adversarial_dataset.append(adversarial_item)
            else:
                # Keep original
                adversarial_dataset.append(item)
        
        # Shuffle the dataset
        random.shuffle(adversarial_dataset)
        
        logger.info(f"Generated {len(adversarial_dataset)} items: {adversarial_count} adversarial")
        return adversarial_dataset

def main():
    """Main function to generate adversarial dataset"""
    generator = AdversarialGenerator()
    
    # Generate some base URLs for adversarial generation
    base_urls = [
        "https://google.com/login",
        "https://microsoft.com/verify",
        "https://apple.com/account",
        "https://amazon.com/signin",
        "https://paypal.com/login"
    ]
    
    # Generate adversarial URLs
    adversarial_urls = []
    for url in base_urls:
        for technique in generator.obfuscation_methods:
            adversarial_url = generator.generate_adversarial_url(url, technique)
            adversarial_urls.append(adversarial_url)
    
    # Save adversarial dataset
    import pandas as pd
    df = pd.DataFrame(adversarial_urls)
    
    output_path = get_data_path('synthetic', 'adversarial_urls.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Adversarial URL dataset saved to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("ADVERSARIAL GENERATION SUMMARY")
    print("="*50)
    print(f"Total Adversarial URLs: {len(adversarial_urls)}")
    
    # Technique distribution
    techniques = {}
    for item in adversarial_urls:
        tech = item['technique']
        techniques[tech] = techniques.get(tech, 0) + 1
    
    print("\nObfuscation Techniques Used:")
    for tech, count in techniques.items():
        print(f"  {tech}: {count}")
    
    # Average obfuscation scores
    avg_obfuscation = sum(item['features']['obfuscation_score'] for item in adversarial_urls) / len(adversarial_urls)
    print(f"\nAverage Obfuscation Score: {avg_obfuscation:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()