# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Visual similarity and template matching for phishing detection
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import hashlib

import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.image import extract_patches_2d

from config import get_config, TemplateMatchingConfig

logger = logging.getLogger(__name__)

class TemplateMatcher:
    """Template matching and visual similarity analysis"""
    
    def __init__(self, config: Optional[TemplateMatchingConfig] = None):
        self.config = config or get_config("template")
        
        # Ensure template directory exists
        Path(self.config.template_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize template cache
        self.template_cache = {}
        self.template_features = {}
        
        # Load templates if cache is enabled
        if self.config.cache_templates:
            self._load_template_cache()
    
    def _load_template_cache(self):
        """Load templates into cache"""
        try:
            template_dir = Path(self.config.template_dir)
            
            for template_file in template_dir.glob("*"):
                if template_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    template_name = template_file.stem
                    self.template_cache[template_name] = str(template_file)
            
            logger.info(f"Loaded {len(self.template_cache)} templates into cache")
            
        except Exception as e:
            logger.warning(f"Failed to load template cache: {e}")
    
    def match_template(self, image_path: str, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Match image against templates"""
        try:
            start_time = time.time()
            
            # Load image
            image = self._load_image(image_path)
            
            if template_name:
                # Match against specific template
                result = self._match_single_template(image, template_name)
            else:
                # Match against all templates
                result = self._match_all_templates(image)
            
            # Calculate processing time
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            result["image_path"] = image_path
            result["timestamp"] = time.time()
            
            logger.info(f"Template matching completed in {result['processing_time_ms']:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to match template for {image_path}: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _match_single_template(self, image: np.ndarray, template_name: str) -> Dict[str, Any]:
        """Match image against a single template"""
        try:
            # Get template path
            template_path = self.template_cache.get(template_name)
            if not template_path:
                return {"error": f"Template {template_name} not found"}
            
            # Load template
            template = self._load_image(template_path)
            
            # Perform template matching
            match_result = self._perform_template_matching(image, template)
            
            return {
                "template_name": template_name,
                "template_path": template_path,
                "match_result": match_result,
                "similarity_score": match_result["max_similarity"],
                "is_match": match_result["max_similarity"] >= self.config.similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to match single template {template_name}: {e}")
            return {"error": str(e)}
    
    def _match_all_templates(self, image: np.ndarray) -> Dict[str, Any]:
        """Match image against all templates"""
        try:
            matches = []
            best_match = None
            best_score = 0.0
            
            for template_name, template_path in self.template_cache.items():
                try:
                    # Load template
                    template = self._load_image(template_path)
                    
                    # Perform template matching
                    match_result = self._perform_template_matching(image, template)
                    
                    match_info = {
                        "template_name": template_name,
                        "template_path": template_path,
                        "similarity_score": match_result["max_similarity"],
                        "match_location": match_result["match_location"],
                        "is_match": match_result["max_similarity"] >= self.config.similarity_threshold
                    }
                    
                    matches.append(match_info)
                    
                    # Track best match
                    if match_result["max_similarity"] > best_score:
                        best_score = match_result["max_similarity"]
                        best_match = match_info
                
                except Exception as e:
                    logger.warning(f"Failed to match template {template_name}: {e}")
                    continue
            
            # Sort matches by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return {
                "total_templates": len(self.template_cache),
                "matches": matches,
                "best_match": best_match,
                "best_score": best_score,
                "matches_above_threshold": len([m for m in matches if m["is_match"]])
            }
            
        except Exception as e:
            logger.error(f"Failed to match all templates: {e}")
            return {"error": str(e)}
    
    def _perform_template_matching(self, image: np.ndarray, template: np.ndarray) -> Dict[str, Any]:
        """Perform template matching between image and template"""
        try:
            # Convert to grayscale for template matching
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            
            # Resize template if it's larger than image
            if template_gray.shape[0] > image_gray.shape[0] or template_gray.shape[1] > image_gray.shape[1]:
                scale = min(image_gray.shape[0] / template_gray.shape[0], 
                           image_gray.shape[1] / template_gray.shape[1])
                new_size = (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale))
                template_gray = cv2.resize(template_gray, new_size)
            
            # Perform template matching
            result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Get match location and size
            match_location = {
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "width": template_gray.shape[1],
                "height": template_gray.shape[0]
            }
            
            return {
                "max_similarity": float(max_val),
                "min_similarity": float(min_val),
                "match_location": match_location,
                "template_size": template_gray.shape,
                "result_shape": result.shape
            }
            
        except Exception as e:
            logger.error(f"Failed to perform template matching: {e}")
            return {"error": str(e)}
    
    def calculate_visual_similarity(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Calculate visual similarity between two images"""
        try:
            start_time = time.time()
            
            # Load images
            image1 = self._load_image(image1_path)
            image2 = self._load_image(image2_path)
            
            # Resize images to same size for comparison
            target_size = (224, 224)
            image1_resized = cv2.resize(image1, target_size)
            image2_resized = cv2.resize(image2, target_size)
            
            # Calculate different similarity metrics
            similarity_metrics = {
                "ssim": self._calculate_ssim(image1_resized, image2_resized),
                "histogram_similarity": self._calculate_histogram_similarity(image1_resized, image2_resized),
                "feature_similarity": self._calculate_feature_similarity(image1_resized, image2_resized),
                "color_similarity": self._calculate_color_similarity(image1_resized, image2_resized),
                "texture_similarity": self._calculate_texture_similarity(image1_resized, image2_resized)
            }
            
            # Calculate overall similarity score
            overall_similarity = np.mean(list(similarity_metrics.values()))
            
            result = {
                "image1_path": image1_path,
                "image2_path": image2_path,
                "similarity_metrics": similarity_metrics,
                "overall_similarity": float(overall_similarity),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }
            
            logger.info(f"Visual similarity calculated: {overall_similarity:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate visual similarity: {e}")
            return {
                "image1_path": image1_path,
                "image2_path": image2_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _calculate_ssim(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM)"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            # Calculate SSIM
            similarity = ssim(gray1, gray2)
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate SSIM: {e}")
            return 0.0
    
    def _calculate_histogram_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate histogram similarity"""
        try:
            # Calculate histograms for each channel
            hist1 = [cv2.calcHist([image1], [i], None, [256], [0, 256]) for i in range(3)]
            hist2 = [cv2.calcHist([image2], [i], None, [256], [0, 256]) for i in range(3)]
            
            # Calculate correlation for each channel
            correlations = []
            for h1, h2 in zip(hist1, hist2):
                correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                correlations.append(correlation)
            
            return float(np.mean(correlations))
            
        except Exception as e:
            logger.warning(f"Failed to calculate histogram similarity: {e}")
            return 0.0
    
    def _calculate_feature_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate feature-based similarity"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            # Extract features using ORB
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0.0
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Calculate similarity based on number of matches
            if len(matches) == 0:
                return 0.0
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity score
            good_matches = len([m for m in matches if m.distance < 50])
            similarity = good_matches / max(len(kp1), len(kp2))
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate feature similarity: {e}")
            return 0.0
    
    def _calculate_color_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate color similarity"""
        try:
            # Calculate mean colors
            mean1 = np.mean(image1, axis=(0, 1))
            mean2 = np.mean(image2, axis=(0, 1))
            
            # Calculate color distance
            color_distance = np.linalg.norm(mean1 - mean2)
            
            # Convert to similarity (0-1)
            similarity = 1.0 / (1.0 + color_distance / 255.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate color similarity: {e}")
            return 0.0
    
    def _calculate_texture_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate texture similarity"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            # Calculate Local Binary Pattern
            from skimage.feature import local_binary_pattern
            
            lbp1 = local_binary_pattern(gray1, P=8, R=1, method='uniform')
            lbp2 = local_binary_pattern(gray2, P=8, R=1, method='uniform')
            
            # Calculate LBP histograms
            hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 10))
            hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 10))
            
            # Normalize histograms
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Calculate correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate texture similarity: {e}")
            return 0.0
    
    def find_similar_images(self, query_image_path: str, image_database: List[str], 
                          top_k: int = 5) -> Dict[str, Any]:
        """Find similar images in a database"""
        try:
            start_time = time.time()
            
            similarities = []
            
            for image_path in image_database:
                similarity_result = self.calculate_visual_similarity(query_image_path, image_path)
                
                if "error" not in similarity_result:
                    similarities.append({
                        "image_path": image_path,
                        "similarity": similarity_result["overall_similarity"],
                        "metrics": similarity_result["similarity_metrics"]
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Get top k results
            top_results = similarities[:top_k]
            
            result = {
                "query_image": query_image_path,
                "database_size": len(image_database),
                "top_results": top_results,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }
            
            logger.info(f"Found {len(top_results)} similar images")
            return result
            
        except Exception as e:
            logger.error(f"Failed to find similar images: {e}")
            return {
                "query_image": query_image_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def add_template(self, template_path: str, template_name: Optional[str] = None) -> Dict[str, Any]:
        """Add a new template to the database"""
        try:
            if not template_name:
                template_name = Path(template_path).stem
            
            # Copy template to template directory
            template_file = Path(self.config.template_dir) / f"{template_name}.png"
            
            # Load and save template
            template_image = self._load_image(template_path)
            template_pil = Image.fromarray(template_image)
            template_pil.save(template_file)
            
            # Add to cache
            self.template_cache[template_name] = str(template_file)
            
            logger.info(f"Added template: {template_name}")
            return {
                "template_name": template_name,
                "template_path": str(template_file),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to add template: {e}")
            return {
                "template_name": template_name,
                "error": str(e),
                "success": False
            }
    
    def remove_template(self, template_name: str) -> Dict[str, Any]:
        """Remove a template from the database"""
        try:
            if template_name not in self.template_cache:
                return {
                    "template_name": template_name,
                    "error": "Template not found",
                    "success": False
                }
            
            # Remove from cache
            template_path = self.template_cache.pop(template_name)
            
            # Delete file
            Path(template_path).unlink(missing_ok=True)
            
            logger.info(f"Removed template: {template_name}")
            return {
                "template_name": template_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to remove template: {e}")
            return {
                "template_name": template_name,
                "error": str(e),
                "success": False
            }
    
    def get_template_info(self) -> Dict[str, Any]:
        """Get information about available templates"""
        try:
            template_info = {
                "total_templates": len(self.template_cache),
                "template_names": list(self.template_cache.keys()),
                "template_paths": list(self.template_cache.values()),
                "config": {
                    "similarity_threshold": self.config.similarity_threshold,
                    "template_dir": self.config.template_dir,
                    "cache_enabled": self.config.cache_templates
                }
            }
            
            # Get template sizes
            template_sizes = {}
            for name, path in self.template_cache.items():
                try:
                    image = self._load_image(path)
                    template_sizes[name] = {
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "channels": image.shape[2] if len(image.shape) == 3 else 1
                    }
                except Exception as e:
                    template_sizes[name] = {"error": str(e)}
            
            template_info["template_sizes"] = template_sizes
            
            return template_info
            
        except Exception as e:
            logger.error(f"Failed to get template info: {e}")
            return {"error": str(e)}
    
    def save_template_database(self, output_path: str):
        """Save template database information"""
        try:
            template_info = self.get_template_info()
            
            with open(output_path, 'w') as f:
                json.dump(template_info, f, indent=2)
            
            logger.info(f"Template database saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save template database: {e}")

def create_template_matcher(config: Optional[TemplateMatchingConfig] = None) -> TemplateMatcher:
    """Factory function to create template matcher instance"""
    return TemplateMatcher(config)