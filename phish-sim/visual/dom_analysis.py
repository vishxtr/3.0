# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
DOM structure analysis and extraction for phishing detection
"""

import asyncio
import json
import logging
import re
import ssl
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin
import hashlib

from playwright.async_api import Page, BrowserContext
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import get_config, DOMAnalysisConfig

logger = logging.getLogger(__name__)

class DOMAnalyzer:
    """DOM structure analysis for phishing detection"""
    
    def __init__(self, config: Optional[DOMAnalysisConfig] = None):
        self.config = config or get_config("dom")
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize patterns for analysis
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for analysis"""
        self.patterns = {
            "phishing_keywords": [
                r"verify\s+your\s+account",
                r"urgent\s+action\s+required",
                r"account\s+suspended",
                r"security\s+alert",
                r"click\s+here\s+immediately",
                r"confirm\s+your\s+identity",
                r"update\s+your\s+information",
                r"unusual\s+activity"
            ],
            "suspicious_domains": [
                r"paypal-[a-z0-9]+\.com",
                r"amazon-[a-z0-9]+\.net",
                r"microsoft-[a-z0-9]+\.org",
                r"apple-[a-z0-9]+\.info",
                r"google-[a-z0-9]+\.co",
                r"facebook-[a-z0-9]+\.tk"
            ],
            "form_fields": [
                r"password", r"username", r"email", r"phone",
                r"ssn", r"credit\s+card", r"cvv", r"pin"
            ],
            "external_resources": [
                r"https?://[^/]+\.(?:com|net|org|info|tk|ml|ga)",
                r"//[^/]+\.(?:com|net|org|info|tk|ml|ga)"
            ]
        }
        
        # Compile patterns
        for category, patterns in self.patterns.items():
            self.patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    async def analyze_page(self, page: Page, url: str) -> Dict[str, Any]:
        """Analyze a page for phishing indicators"""
        try:
            # Get page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Perform various analyses
            analysis_results = {
                "url": url,
                "timestamp": time.time(),
                "basic_info": await self._analyze_basic_info(page, soup),
                "form_analysis": await self._analyze_forms(soup) if self.config.extract_forms else {},
                "link_analysis": await self._analyze_links(soup) if self.config.extract_links else {},
                "image_analysis": await self._analyze_images(soup) if self.config.extract_images else {},
                "script_analysis": await self._analyze_scripts(soup) if self.config.extract_scripts else {},
                "style_analysis": await self._analyze_styles(soup) if self.config.extract_styles else {},
                "content_analysis": await self._analyze_content(soup) if self.config.analyze_text_content else {},
                "meta_analysis": await self._analyze_meta_tags(soup) if self.config.analyze_meta_tags else {},
                "security_analysis": await self._analyze_security(page, soup) if self.config.check_external_resources else {},
                "phishing_indicators": await self._detect_phishing_indicators(soup),
                "risk_score": 0.0
            }
            
            # Calculate overall risk score
            analysis_results["risk_score"] = self._calculate_risk_score(analysis_results)
            
            # Save results if configured
            if self.config.save_dom_tree:
                await self._save_dom_tree(soup, url)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze page {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": time.time(),
                "risk_score": 0.5  # Default risk score on error
            }
    
    async def _analyze_basic_info(self, page: Page, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze basic page information"""
        try:
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Get page dimensions
            dimensions = await page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight,
                        viewport_width: window.innerWidth,
                        viewport_height: window.innerHeight
                    }
                }
            """)
            
            # Count elements
            element_counts = {
                "total_elements": len(soup.find_all()),
                "divs": len(soup.find_all('div')),
                "spans": len(soup.find_all('span')),
                "paragraphs": len(soup.find_all('p')),
                "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "forms": len(soup.find_all('form')),
                "inputs": len(soup.find_all('input')),
                "links": len(soup.find_all('a')),
                "images": len(soup.find_all('img')),
                "scripts": len(soup.find_all('script')),
                "stylesheets": len(soup.find_all('link', rel='stylesheet'))
            }
            
            return {
                "title": title_text,
                "dimensions": dimensions,
                "element_counts": element_counts,
                "has_favicon": bool(soup.find('link', rel='icon')),
                "has_robots_meta": bool(soup.find('meta', attrs={'name': 'robots'}))
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze basic info: {e}")
            return {"error": str(e)}
    
    async def _analyze_forms(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze forms for phishing indicators"""
        forms = soup.find_all('form')
        form_analysis = {
            "total_forms": len(forms),
            "forms": [],
            "suspicious_indicators": []
        }
        
        for i, form in enumerate(forms):
            form_info = {
                "index": i,
                "action": form.get('action', ''),
                "method": form.get('method', 'get').lower(),
                "target": form.get('target', ''),
                "inputs": [],
                "suspicious_score": 0.0
            }
            
            # Analyze inputs
            inputs = form.find_all('input')
            for input_elem in inputs:
                input_info = {
                    "type": input_elem.get('type', 'text'),
                    "name": input_elem.get('name', ''),
                    "id": input_elem.get('id', ''),
                    "placeholder": input_elem.get('placeholder', ''),
                    "required": input_elem.has_attr('required'),
                    "autocomplete": input_elem.get('autocomplete', '')
                }
                form_info["inputs"].append(input_info)
            
            # Check for suspicious patterns
            suspicious_score = 0.0
            
            # Check for password fields
            password_inputs = [inp for inp in form_info["inputs"] if inp["type"] == "password"]
            if password_inputs:
                suspicious_score += 0.3
            
            # Check for sensitive field names
            sensitive_fields = ['ssn', 'credit', 'card', 'cvv', 'pin', 'social']
            for input_info in form_info["inputs"]:
                field_text = f"{input_info['name']} {input_info['placeholder']}".lower()
                if any(field in field_text for field in sensitive_fields):
                    suspicious_score += 0.2
            
            # Check for external action URLs
            if form_info["action"] and not self._is_same_domain(form_info["action"]):
                suspicious_score += 0.4
            
            form_info["suspicious_score"] = suspicious_score
            form_analysis["forms"].append(form_info)
            
            if suspicious_score > 0.5:
                form_analysis["suspicious_indicators"].append(f"Form {i}: High suspicious score ({suspicious_score:.2f})")
        
        return form_analysis
    
    async def _analyze_links(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze links for phishing indicators"""
        links = soup.find_all('a', href=True)
        link_analysis = {
            "total_links": len(links),
            "external_links": 0,
            "suspicious_links": [],
            "redirect_links": [],
            "javascript_links": []
        }
        
        for link in links:
            href = link.get('href', '')
            link_text = link.get_text().strip()
            
            # Check for external links
            if href.startswith('http') and not self._is_same_domain(href):
                link_analysis["external_links"] += 1
                
                # Check for suspicious patterns
                if self._is_suspicious_link(href, link_text):
                    link_analysis["suspicious_links"].append({
                        "href": href,
                        "text": link_text,
                        "reason": "Suspicious domain or text"
                    })
            
            # Check for redirect links
            if 'redirect' in href.lower() or 'url=' in href:
                link_analysis["redirect_links"].append({
                    "href": href,
                    "text": link_text
                })
            
            # Check for JavaScript links
            if href.startswith('javascript:'):
                link_analysis["javascript_links"].append({
                    "href": href,
                    "text": link_text
                })
        
        return link_analysis
    
    async def _analyze_images(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze images for phishing indicators"""
        images = soup.find_all('img')
        image_analysis = {
            "total_images": len(images),
            "external_images": 0,
            "missing_alt": 0,
            "suspicious_images": []
        }
        
        for img in images:
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            # Check for external images
            if src.startswith('http') and not self._is_same_domain(src):
                image_analysis["external_images"] += 1
            
            # Check for missing alt text
            if not alt:
                image_analysis["missing_alt"] += 1
            
            # Check for suspicious image sources
            if self._is_suspicious_image_src(src):
                image_analysis["suspicious_images"].append({
                    "src": src,
                    "alt": alt
                })
        
        return image_analysis
    
    async def _analyze_scripts(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze scripts for phishing indicators"""
        scripts = soup.find_all('script')
        script_analysis = {
            "total_scripts": len(scripts),
            "external_scripts": 0,
            "inline_scripts": 0,
            "suspicious_scripts": []
        }
        
        for script in scripts:
            src = script.get('src', '')
            
            if src:
                script_analysis["external_scripts"] += 1
                if not self._is_same_domain(src):
                    script_analysis["suspicious_scripts"].append({
                        "type": "external",
                        "src": src
                    })
            else:
                script_analysis["inline_scripts"] += 1
                script_content = script.get_text()
                
                # Check for suspicious patterns in inline scripts
                if self._is_suspicious_script(script_content):
                    script_analysis["suspicious_scripts"].append({
                        "type": "inline",
                        "content_preview": script_content[:200] + "..." if len(script_content) > 200 else script_content
                    })
        
        return script_analysis
    
    async def _analyze_styles(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze stylesheets for phishing indicators"""
        stylesheets = soup.find_all('link', rel='stylesheet')
        style_analysis = {
            "total_stylesheets": len(stylesheets),
            "external_stylesheets": 0,
            "inline_styles": len(soup.find_all('style'))
        }
        
        for stylesheet in stylesheets:
            href = stylesheet.get('href', '')
            if href.startswith('http') and not self._is_same_domain(href):
                style_analysis["external_stylesheets"] += 1
        
        return style_analysis
    
    async def _analyze_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze text content for phishing indicators"""
        # Get all text content
        text_content = soup.get_text().lower()
        
        # Check for phishing keywords
        phishing_matches = []
        for pattern in self.patterns["phishing_keywords"]:
            matches = pattern.findall(text_content)
            if matches:
                phishing_matches.extend(matches)
        
        # Analyze urgency indicators
        urgency_indicators = [
            'urgent', 'immediately', 'asap', 'expires', 'limited time',
            'act now', 'click here', 'verify now', 'confirm now'
        ]
        
        urgency_count = sum(text_content.count(indicator) for indicator in urgency_indicators)
        
        # Check for brand names
        brand_names = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'netflix']
        brand_mentions = {brand: text_content.count(brand) for brand in brand_names}
        
        return {
            "total_text_length": len(text_content),
            "phishing_keywords_found": len(set(phishing_matches)),
            "phishing_matches": list(set(phishing_matches)),
            "urgency_indicators": urgency_count,
            "brand_mentions": brand_mentions,
            "has_contact_info": bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_content)),
            "has_email_patterns": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content))
        }
    
    async def _analyze_meta_tags(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze meta tags for phishing indicators"""
        meta_tags = soup.find_all('meta')
        meta_analysis = {
            "total_meta_tags": len(meta_tags),
            "description": "",
            "keywords": "",
            "author": "",
            "viewport": "",
            "robots": "",
            "og_tags": {},
            "twitter_tags": {}
        }
        
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                meta_analysis["description"] = content
            elif name == 'keywords':
                meta_analysis["keywords"] = content
            elif name == 'author':
                meta_analysis["author"] = content
            elif name == 'viewport':
                meta_analysis["viewport"] = content
            elif name == 'robots':
                meta_analysis["robots"] = content
            elif property_attr.startswith('og:'):
                meta_analysis["og_tags"][property_attr] = content
            elif name.startswith('twitter:'):
                meta_analysis["twitter_tags"][name] = content
        
        return meta_analysis
    
    async def _analyze_security(self, page: Page, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze security aspects"""
        security_analysis = {
            "ssl_info": {},
            "external_resources": [],
            "mixed_content": False,
            "insecure_forms": False
        }
        
        try:
            # Check SSL certificate
            url = page.url
            if url.startswith('https:'):
                security_analysis["ssl_info"] = await self._check_ssl_certificate(url)
            
            # Check for mixed content
            insecure_resources = soup.find_all(['img', 'script', 'link'], src=True)
            for resource in insecure_resources:
                src = resource.get('src', '')
                if src.startswith('http:') and url.startswith('https:'):
                    security_analysis["mixed_content"] = True
                    break
            
            # Check for insecure forms
            forms = soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                if action.startswith('http:') and url.startswith('https:'):
                    security_analysis["insecure_forms"] = True
                    break
            
        except Exception as e:
            logger.warning(f"Failed to analyze security: {e}")
            security_analysis["error"] = str(e)
        
        return security_analysis
    
    async def _detect_phishing_indicators(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Detect specific phishing indicators"""
        indicators = {
            "homograph_domains": [],
            "suspicious_redirects": [],
            "hidden_elements": [],
            "obfuscated_content": [],
            "fake_ssl_indicators": []
        }
        
        # Check for hidden elements
        hidden_elements = soup.find_all(attrs={'style': re.compile(r'display\s*:\s*none|visibility\s*:\s*hidden')})
        indicators["hidden_elements"] = [str(elem)[:100] for elem in hidden_elements[:5]]
        
        # Check for obfuscated content (base64, hex, etc.)
        scripts = soup.find_all('script')
        for script in scripts:
            content = script.get_text()
            if 'base64' in content.lower() or 'atob(' in content.lower():
                indicators["obfuscated_content"].append("Base64 encoding detected")
        
        # Check for fake SSL indicators in text
        text_content = soup.get_text().lower()
        if 'secure' in text_content and 'https' not in text_content:
            indicators["fake_ssl_indicators"].append("Claims security without HTTPS")
        
        return indicators
    
    def _calculate_risk_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall risk score based on analysis results"""
        risk_score = 0.0
        
        # Form analysis risk
        if "form_analysis" in analysis_results:
            forms = analysis_results["form_analysis"].get("forms", [])
            for form in forms:
                risk_score += form.get("suspicious_score", 0.0) * 0.3
        
        # Link analysis risk
        if "link_analysis" in analysis_results:
            link_analysis = analysis_results["link_analysis"]
            risk_score += len(link_analysis.get("suspicious_links", [])) * 0.1
            risk_score += len(link_analysis.get("redirect_links", [])) * 0.05
        
        # Content analysis risk
        if "content_analysis" in analysis_results:
            content_analysis = analysis_results["content_analysis"]
            risk_score += content_analysis.get("phishing_keywords_found", 0) * 0.1
            risk_score += content_analysis.get("urgency_indicators", 0) * 0.05
        
        # Security analysis risk
        if "security_analysis" in analysis_results:
            security_analysis = analysis_results["security_analysis"]
            if security_analysis.get("mixed_content", False):
                risk_score += 0.2
            if security_analysis.get("insecure_forms", False):
                risk_score += 0.3
        
        # Phishing indicators risk
        if "phishing_indicators" in analysis_results:
            indicators = analysis_results["phishing_indicators"]
            risk_score += len(indicators.get("hidden_elements", [])) * 0.1
            risk_score += len(indicators.get("obfuscated_content", [])) * 0.2
            risk_score += len(indicators.get("fake_ssl_indicators", [])) * 0.15
        
        # Normalize to 0-1 range
        return min(risk_score, 1.0)
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL is from the same domain"""
        try:
            parsed_url = urlparse(url)
            return bool(parsed_url.netloc)
        except:
            return False
    
    def _is_suspicious_link(self, href: str, text: str) -> bool:
        """Check if link is suspicious"""
        # Check for suspicious domain patterns
        for pattern in self.patterns["suspicious_domains"]:
            if pattern.search(href):
                return True
        
        # Check for suspicious text patterns
        suspicious_texts = ['click here', 'verify now', 'urgent', 'security alert']
        if any(suspicious in text.lower() for suspicious in suspicious_texts):
            return True
        
        return False
    
    def _is_suspicious_image_src(self, src: str) -> bool:
        """Check if image source is suspicious"""
        # Check for external domains with suspicious patterns
        for pattern in self.patterns["suspicious_domains"]:
            if pattern.search(src):
                return True
        
        return False
    
    def _is_suspicious_script(self, content: str) -> bool:
        """Check if script content is suspicious"""
        suspicious_patterns = [
            r'eval\s*\(',
            r'document\.write\s*\(',
            r'innerHTML\s*=',
            r'window\.location\s*=',
            r'atob\s*\(',
            r'btoa\s*\('
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    async def _check_ssl_certificate(self, url: str) -> Dict[str, Any]:
        """Check SSL certificate information"""
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            port = parsed_url.port or 443
            
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            with ssl.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
            
            return {
                "valid": True,
                "subject": dict(x[0] for x in cert.get('subject', [])),
                "issuer": dict(x[0] for x in cert.get('issuer', [])),
                "version": cert.get('version'),
                "serial_number": cert.get('serialNumber'),
                "not_before": cert.get('notBefore'),
                "not_after": cert.get('notAfter')
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def _save_dom_tree(self, soup: BeautifulSoup, url: str):
        """Save DOM tree to file"""
        try:
            # Generate filename
            domain = urlparse(url).netloc.replace('.', '_')
            timestamp = int(time.time())
            filename = f"dom_{timestamp}_{domain}.html"
            
            filepath = Path(self.config.output_dir) / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            logger.info(f"DOM tree saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save DOM tree: {e}")
    
    async def analyze_multiple_pages(self, pages: List[Tuple[Page, str]]) -> List[Dict[str, Any]]:
        """Analyze multiple pages"""
        results = []
        
        for page, url in pages:
            result = await self.analyze_page(page, url)
            results.append(result)
        
        return results
    
    def save_analysis_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save analysis results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")

def create_dom_analyzer(config: Optional[DOMAnalysisConfig] = None) -> DOMAnalyzer:
    """Factory function to create DOM analyzer instance"""
    return DOMAnalyzer(config)