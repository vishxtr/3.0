# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Screenshot capture system using Playwright for visual analysis
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import json

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from PIL import Image
import numpy as np

from config import get_config, ScreenshotConfig

logger = logging.getLogger(__name__)

class ScreenshotCapture:
    """Screenshot capture system using Playwright"""
    
    def __init__(self, config: Optional[ScreenshotConfig] = None):
        self.config = config or get_config("screenshot")
        self.browser = None
        self.context = None
        self.playwright = None
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_browser()
    
    async def start_browser(self):
        """Start browser instance"""
        try:
            self.playwright = await async_playwright().start()
            
            # Get browser settings
            browser_settings = {
                "chromium": self.playwright.chromium,
                "firefox": self.playwright.firefox,
                "webkit": self.playwright.webkit
            }
            
            browser_launcher = browser_settings.get(self.config.browser_type, self.playwright.chromium)
            
            # Launch browser
            self.browser = await browser_launcher.launch(
                headless=self.config.headless,
                args=self._get_browser_args()
            )
            
            # Create context
            self.context = await self.browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                },
                device_scale_factor=self.config.device_scale_factor,
                user_agent=self.config.user_agent,
                ignore_https_errors=self.config.ignore_https_errors,
                bypass_csp=self.config.bypass_csp
            )
            
            logger.info(f"Started {self.config.browser_type} browser")
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise
    
    async def close_browser(self):
        """Close browser instance"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("Browser closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    def _get_browser_args(self) -> List[str]:
        """Get browser arguments based on browser type"""
        from config import BROWSER_SETTINGS
        
        browser_settings = BROWSER_SETTINGS.get(self.config.browser_type, BROWSER_SETTINGS["chromium"])
        return browser_settings.get("args", [])
    
    async def capture_screenshot(self, url: str, 
                               wait_for: Optional[str] = None,
                               timeout: Optional[int] = None) -> Dict[str, Any]:
        """Capture screenshot of a URL"""
        if not self.context:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        timeout = timeout or self.config.wait_timeout
        
        try:
            # Create new page
            page = await self.context.new_page()
            
            # Set timeouts
            page.set_default_timeout(timeout)
            page.set_default_navigation_timeout(self.config.navigation_timeout)
            
            # Navigate to URL
            logger.info(f"Navigating to: {url}")
            response = await page.goto(url, wait_until=self.config.load_state)
            
            # Wait for additional elements if specified
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=5000)
                except:
                    logger.warning(f"Element '{wait_for}' not found, continuing...")
            
            # Get page information
            page_info = await self._get_page_info(page)
            
            # Capture screenshot
            screenshot_path = await self._capture_page_screenshot(page, url)
            
            # Get DOM information
            dom_info = await self._get_dom_info(page)
            
            # Close page
            await page.close()
            
            result = {
                "url": url,
                "screenshot_path": screenshot_path,
                "page_info": page_info,
                "dom_info": dom_info,
                "response_status": response.status if response else None,
                "timestamp": time.time(),
                "success": True
            }
            
            logger.info(f"Screenshot captured successfully: {screenshot_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot for {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    async def _capture_page_screenshot(self, page: Page, url: str) -> str:
        """Capture screenshot of the page"""
        # Generate filename
        domain = urlparse(url).netloc
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        
        filename = self.config.filename_template.format(
            timestamp=timestamp,
            domain=domain.replace(".", "_"),
            hash=url_hash,
            format=self.config.screenshot_format
        )
        
        screenshot_path = Path(self.config.output_dir) / filename
        
        # Capture screenshot
        await page.screenshot(
            path=str(screenshot_path),
            full_page=self.config.full_page,
            type=self.config.screenshot_format
        )
        
        return str(screenshot_path)
    
    async def _get_page_info(self, page: Page) -> Dict[str, Any]:
        """Get page information"""
        try:
            title = await page.title()
            url = page.url
            
            # Get viewport size
            viewport_size = page.viewport_size
            
            # Get page dimensions
            content_size = await page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.scrollWidth,
                        height: document.documentElement.scrollHeight
                    }
                }
            """)
            
            # Get page load metrics
            metrics = await page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    return {
                        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
                        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
                    }
                }
            """)
            
            return {
                "title": title,
                "url": url,
                "viewport_size": viewport_size,
                "content_size": content_size,
                "load_metrics": metrics
            }
            
        except Exception as e:
            logger.warning(f"Failed to get page info: {e}")
            return {"error": str(e)}
    
    async def _get_dom_info(self, page: Page) -> Dict[str, Any]:
        """Get DOM information"""
        try:
            # Get basic DOM statistics
            dom_stats = await page.evaluate("""
                () => {
                    return {
                        totalElements: document.querySelectorAll('*').length,
                        forms: document.querySelectorAll('form').length,
                        inputs: document.querySelectorAll('input').length,
                        links: document.querySelectorAll('a').length,
                        images: document.querySelectorAll('img').length,
                        scripts: document.querySelectorAll('script').length,
                        stylesheets: document.querySelectorAll('link[rel="stylesheet"]').length,
                        iframes: document.querySelectorAll('iframe').length
                    }
                }
            """)
            
            # Get form information
            forms_info = await page.evaluate("""
                () => {
                    const forms = Array.from(document.querySelectorAll('form'));
                    return forms.map(form => ({
                        action: form.action,
                        method: form.method,
                        inputs: Array.from(form.querySelectorAll('input')).map(input => ({
                            type: input.type,
                            name: input.name,
                            placeholder: input.placeholder
                        }))
                    }));
                }
            """)
            
            # Get link information
            links_info = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a'));
                    return links.map(link => ({
                        href: link.href,
                        text: link.textContent?.trim(),
                        target: link.target
                    }));
                }
            """)
            
            return {
                "dom_stats": dom_stats,
                "forms": forms_info,
                "links": links_info[:50]  # Limit to first 50 links
            }
            
        except Exception as e:
            logger.warning(f"Failed to get DOM info: {e}")
            return {"error": str(e)}
    
    async def capture_multiple_screenshots(self, urls: List[str], 
                                         max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Capture screenshots for multiple URLs"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def capture_with_semaphore(url):
            async with semaphore:
                return await self.capture_screenshot(url)
        
        tasks = [capture_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i],
                    "error": str(result),
                    "timestamp": time.time(),
                    "success": False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def analyze_screenshot(self, screenshot_path: str) -> Dict[str, Any]:
        """Analyze captured screenshot"""
        try:
            # Load image
            image = Image.open(screenshot_path)
            img_array = np.array(image)
            
            # Basic image analysis
            analysis = {
                "dimensions": image.size,
                "mode": image.mode,
                "file_size": Path(screenshot_path).stat().st_size,
                "aspect_ratio": image.size[0] / image.size[1],
                "color_analysis": self._analyze_colors(img_array),
                "texture_analysis": self._analyze_texture(img_array),
                "edge_analysis": self._analyze_edges(img_array)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze screenshot {screenshot_path}: {e}")
            return {"error": str(e)}
    
    def _analyze_colors(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution"""
        try:
            # Convert to RGB if needed
            if len(img_array.shape) == 3:
                # Calculate color statistics
                mean_colors = np.mean(img_array, axis=(0, 1))
                std_colors = np.std(img_array, axis=(0, 1))
                
                # Dominant colors (simplified)
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                
                return {
                    "mean_rgb": mean_colors.tolist(),
                    "std_rgb": std_colors.tolist(),
                    "unique_colors": int(unique_colors),
                    "brightness": float(np.mean(img_array))
                }
            else:
                return {"error": "Invalid image format"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_texture(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze texture features"""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Calculate texture features
            texture_variance = np.var(gray)
            texture_std = np.std(gray)
            
            # Edge density (simplified)
            edges = np.abs(np.diff(gray, axis=1))
            edge_density = np.mean(edges > np.std(edges))
            
            return {
                "variance": float(texture_variance),
                "std": float(texture_std),
                "edge_density": float(edge_density)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_edges(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze edge features"""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Simple edge detection
            edges_x = np.abs(np.diff(gray, axis=1))
            edges_y = np.abs(np.diff(gray, axis=0))
            
            edge_strength = np.mean(edges_x) + np.mean(edges_y)
            edge_density = np.mean((edges_x > np.std(edges_x)) | (edges_y > np.std(edges_y)))
            
            return {
                "strength": float(edge_strength),
                "density": float(edge_density)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def save_analysis_results(self, results: List[Dict[str, Any]], 
                            output_path: str):
        """Save analysis results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_file}")

async def capture_screenshot(url: str, config: Optional[ScreenshotConfig] = None) -> Dict[str, Any]:
    """Convenience function to capture a single screenshot"""
    async with ScreenshotCapture(config) as capture:
        return await capture.capture_screenshot(url)

async def capture_multiple_screenshots(urls: List[str], 
                                     config: Optional[ScreenshotConfig] = None,
                                     max_concurrent: int = 3) -> List[Dict[str, Any]]:
    """Convenience function to capture multiple screenshots"""
    async with ScreenshotCapture(config) as capture:
        return await capture.capture_multiple_screenshots(urls, max_concurrent)

def create_screenshot_capture(config: Optional[ScreenshotConfig] = None) -> ScreenshotCapture:
    """Factory function to create screenshot capture instance"""
    return ScreenshotCapture(config)