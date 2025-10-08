# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Visual feature extraction pipeline for phishing detection
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json

import numpy as np
from PIL import Image, ImageStat
import cv2
from skimage import feature, filters, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import gabor
import matplotlib.pyplot as plt

from config import get_config, FeatureExtractionConfig

logger = logging.getLogger(__name__)

class VisualFeatureExtractor:
    """Visual feature extraction for phishing detection"""
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        self.config = config or get_config("feature")
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive visual features from image"""
        try:
            start_time = time.time()
            
            # Load and preprocess image
            image = self._load_image(image_path)
            processed_image = self._preprocess_image(image)
            
            # Extract different types of features
            features = {
                "image_path": image_path,
                "basic_info": self._extract_basic_info(image),
                "color_features": self._extract_color_features(processed_image) if self.config.extract_color else {},
                "texture_features": self._extract_texture_features(processed_image) if self.config.extract_texture else {},
                "shape_features": self._extract_shape_features(processed_image) if self.config.extract_shape else {},
                "edge_features": self._extract_edge_features(processed_image) if self.config.extract_edges else {},
                "histogram_features": self._extract_histogram_features(processed_image) if self.config.extract_histogram else {},
                "cnn_features": self._extract_cnn_features(processed_image) if self.config.use_cnn_features else {},
                "extraction_time_ms": 0.0,
                "timestamp": time.time()
            }
            
            # Calculate extraction time
            features["extraction_time_ms"] = (time.time() - start_time) * 1000
            
            logger.info(f"Features extracted in {features['extraction_time_ms']:.2f}ms for {image_path}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
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
    
    def _preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image for feature extraction"""
        processed = {}
        
        # Original image
        processed["original"] = image
        
        # Resize if needed
        if self.config.resize_method == "resize":
            target_size = (224, 224)  # Standard size for feature extraction
            processed["resized"] = cv2.resize(image, target_size)
        else:
            processed["resized"] = image
        
        # Convert to different color spaces
        if self.config.color_space == "grayscale":
            processed["grayscale"] = rgb2gray(processed["resized"])
        elif self.config.color_space == "hsv":
            processed["hsv"] = rgb2hsv(processed["resized"])
        else:
            processed["rgb"] = processed["resized"]
        
        # Normalize
        if self.config.normalization == "imagenet":
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            processed["normalized"] = (processed["resized"] / 255.0 - mean) / std
        elif self.config.normalization == "custom":
            processed["normalized"] = processed["resized"] / 255.0
        else:
            processed["normalized"] = processed["resized"]
        
        return processed
    
    def _extract_basic_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract basic image information"""
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "aspect_ratio": image.shape[1] / image.shape[0],
            "total_pixels": image.size,
            "dtype": str(image.dtype)
        }
    
    def _extract_color_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract color-based features"""
        try:
            image = processed["rgb"]
            
            # Convert to different color spaces
            hsv = rgb2hsv(image)
            gray = rgb2gray(image)
            
            # Color statistics
            color_stats = {
                "mean_rgb": np.mean(image, axis=(0, 1)).tolist(),
                "std_rgb": np.std(image, axis=(0, 1)).tolist(),
                "mean_hsv": np.mean(hsv, axis=(0, 1)).tolist(),
                "std_hsv": np.std(hsv, axis=(0, 1)).tolist(),
                "brightness": np.mean(gray),
                "contrast": np.std(gray)
            }
            
            # Dominant colors (simplified)
            dominant_colors = self._get_dominant_colors(image)
            color_stats["dominant_colors"] = dominant_colors
            
            # Color distribution
            color_distribution = self._analyze_color_distribution(image)
            color_stats["color_distribution"] = color_distribution
            
            return color_stats
            
        except Exception as e:
            logger.warning(f"Failed to extract color features: {e}")
            return {"error": str(e)}
    
    def _get_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Dict[str, Any]]:
        """Get dominant colors in image"""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Simple k-means clustering for dominant colors
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = []
            for i, color in enumerate(kmeans.cluster_centers_):
                dominant_colors.append({
                    "color": color.astype(int).tolist(),
                    "percentage": float(np.sum(kmeans.labels_ == i) / len(pixels)),
                    "cluster_id": i
                })
            
            return dominant_colors
            
        except Exception as e:
            logger.warning(f"Failed to get dominant colors: {e}")
            return []
    
    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution patterns"""
        try:
            # Calculate color histograms
            hist_r = np.histogram(image[:, :, 0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(image[:, :, 1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(image[:, :, 2], bins=32, range=(0, 256))[0]
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            return {
                "histogram_r": hist_r.tolist(),
                "histogram_g": hist_g.tolist(),
                "histogram_b": hist_b.tolist(),
                "color_entropy": {
                    "r": float(-np.sum(hist_r * np.log2(hist_r + 1e-10))),
                    "g": float(-np.sum(hist_g * np.log2(hist_g + 1e-10))),
                    "b": float(-np.sum(hist_b * np.log2(hist_b + 1e-10)))
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze color distribution: {e}")
            return {"error": str(e)}
    
    def _extract_texture_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract texture-based features"""
        try:
            gray = processed["grayscale"]
            
            # Local Binary Pattern (LBP)
            lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            
            # Gabor filters
            gabor_features = []
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3]:
                    filtered = gabor(gray, frequency=frequency, theta=np.deg2rad(theta))
                    gabor_features.append(np.mean(filtered[0]))
            
            # Gray-level co-occurrence matrix (GLCM)
            glcm = feature.graycomatrix(
                (gray * 255).astype(np.uint8),
                distances=[1, 2],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # GLCM properties
            contrast = feature.graycoprops(glcm, 'contrast')
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
            homogeneity = feature.graycoprops(glcm, 'homogeneity')
            energy = feature.graycoprops(glcm, 'energy')
            
            return {
                "lbp_histogram": lbp_hist.tolist(),
                "gabor_features": gabor_features,
                "glcm_contrast": float(np.mean(contrast)),
                "glcm_dissimilarity": float(np.mean(dissimilarity)),
                "glcm_homogeneity": float(np.mean(homogeneity)),
                "glcm_energy": float(np.mean(energy)),
                "texture_variance": float(np.var(gray)),
                "texture_std": float(np.std(gray))
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract texture features: {e}")
            return {"error": str(e)}
    
    def _extract_shape_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract shape-based features"""
        try:
            gray = processed["grayscale"]
            
            # Edge detection
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Shape features
            shape_features = {
                "num_contours": len(contours),
                "total_edge_pixels": int(np.sum(edges > 0)),
                "edge_density": float(np.sum(edges > 0) / edges.size)
            }
            
            if contours:
                # Analyze largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Shape descriptors
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    shape_features["largest_contour_area"] = float(area)
                    shape_features["largest_contour_perimeter"] = float(perimeter)
                    shape_features["circularity"] = float(circularity)
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                shape_features["bounding_box"] = {
                    "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h),
                    "aspect_ratio": float(w / h) if h > 0 else 0
                }
            
            return shape_features
            
        except Exception as e:
            logger.warning(f"Failed to extract shape features: {e}")
            return {"error": str(e)}
    
    def _extract_edge_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract edge-based features"""
        try:
            gray = processed["grayscale"]
            
            # Different edge detection methods
            sobel_x = filters.sobel_h(gray)
            sobel_y = filters.sobel_v(gray)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Laplacian
            laplacian = filters.laplace(gray)
            
            # Canny edges
            canny_edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            
            # Edge statistics
            edge_features = {
                "sobel_magnitude_mean": float(np.mean(sobel_magnitude)),
                "sobel_magnitude_std": float(np.std(sobel_magnitude)),
                "laplacian_mean": float(np.mean(laplacian)),
                "laplacian_std": float(np.std(laplacian)),
                "canny_edge_density": float(np.sum(canny_edges > 0) / canny_edges.size),
                "edge_orientation_histogram": self._calculate_edge_orientation_histogram(sobel_x, sobel_y)
            }
            
            return edge_features
            
        except Exception as e:
            logger.warning(f"Failed to extract edge features: {e}")
            return {"error": str(e)}
    
    def _calculate_edge_orientation_histogram(self, sobel_x: np.ndarray, sobel_y: np.ndarray) -> List[float]:
        """Calculate edge orientation histogram"""
        try:
            # Calculate orientations
            orientations = np.arctan2(sobel_y, sobel_x)
            
            # Create histogram
            hist, _ = np.histogram(orientations, bins=8, range=(-np.pi, np.pi))
            hist = hist / np.sum(hist)
            
            return hist.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to calculate edge orientation histogram: {e}")
            return []
    
    def _extract_histogram_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract histogram-based features"""
        try:
            image = processed["rgb"]
            
            # Color histograms
            hist_r = np.histogram(image[:, :, 0], bins=64, range=(0, 256))[0]
            hist_g = np.histogram(image[:, :, 1], bins=64, range=(0, 256))[0]
            hist_b = np.histogram(image[:, :, 2], bins=64, range=(0, 256))[0]
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            # Histogram statistics
            histogram_features = {
                "histogram_r": hist_r.tolist(),
                "histogram_g": hist_g.tolist(),
                "histogram_b": hist_b.tolist(),
                "histogram_entropy": {
                    "r": float(-np.sum(hist_r * np.log2(hist_r + 1e-10))),
                    "g": float(-np.sum(hist_g * np.log2(hist_g + 1e-10))),
                    "b": float(-np.sum(hist_b * np.log2(hist_b + 1e-10)))
                },
                "histogram_skewness": {
                    "r": float(self._calculate_skewness(hist_r)),
                    "g": float(self._calculate_skewness(hist_g)),
                    "b": float(self._calculate_skewness(hist_b))
                }
            }
            
            return histogram_features
            
        except Exception as e:
            logger.warning(f"Failed to extract histogram features: {e}")
            return {"error": str(e)}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return float(np.mean(((data - mean) / std) ** 3))
        except:
            return 0.0
    
    def _extract_cnn_features(self, processed: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract CNN-based features (simulated)"""
        try:
            # In real implementation, would use pre-trained CNN
            # For demo, simulate CNN feature extraction
            
            image = processed["normalized"]
            
            # Simulate CNN feature extraction
            # This would typically use a pre-trained model like VGG, ResNet, etc.
            feature_dim = self.config.feature_dimension
            
            # Simulate feature extraction based on image characteristics
            cnn_features = np.random.normal(0, 1, feature_dim)
            
            # Add some structure based on image content
            image_mean = np.mean(image)
            image_std = np.std(image)
            
            # Modify features based on image characteristics
            cnn_features[:10] += image_mean * 0.1
            cnn_features[10:20] += image_std * 0.1
            
            return {
                "cnn_features": cnn_features.tolist(),
                "feature_dimension": feature_dim,
                "layer": self.config.cnn_layer,
                "model_type": "simulated_cnn"
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract CNN features: {e}")
            return {"error": str(e)}
    
    def extract_features_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract features for multiple images"""
        results = []
        
        for image_path in image_paths:
            result = self.extract_features(image_path)
            results.append(result)
        
        return results
    
    def save_features(self, features: Dict[str, Any], output_path: str):
        """Save extracted features to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.feature_format == "json":
                with open(output_file, 'w') as f:
                    json.dump(features, f, indent=2)
            elif self.config.feature_format == "npy":
                # Convert to numpy array and save
                feature_vector = self._features_to_vector(features)
                np.save(output_file, feature_vector)
            else:
                logger.warning(f"Unknown feature format: {self.config.feature_format}")
                return
            
            logger.info(f"Features saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to feature vector"""
        try:
            vector = []
            
            # Add basic info
            if "basic_info" in features:
                basic = features["basic_info"]
                vector.extend([basic.get("width", 0), basic.get("height", 0), 
                             basic.get("aspect_ratio", 0)])
            
            # Add color features
            if "color_features" in features and "error" not in features["color_features"]:
                color = features["color_features"]
                if "mean_rgb" in color:
                    vector.extend(color["mean_rgb"])
                if "brightness" in color:
                    vector.append(color["brightness"])
                if "contrast" in color:
                    vector.append(color["contrast"])
            
            # Add texture features
            if "texture_features" in features and "error" not in features["texture_features"]:
                texture = features["texture_features"]
                if "texture_variance" in texture:
                    vector.append(texture["texture_variance"])
                if "texture_std" in texture:
                    vector.append(texture["texture_std"])
            
            # Add CNN features
            if "cnn_features" in features and "error" not in features["cnn_features"]:
                cnn = features["cnn_features"]
                if "cnn_features" in cnn:
                    vector.extend(cnn["cnn_features"])
            
            return np.array(vector)
            
        except Exception as e:
            logger.warning(f"Failed to convert features to vector: {e}")
            return np.array([])
    
    def load_features(self, features_path: str) -> Dict[str, Any]:
        """Load features from file"""
        try:
            features_file = Path(features_path)
            
            if not features_file.exists():
                logger.warning(f"Features file not found: {features_path}")
                return {}
            
            if self.config.feature_format == "json":
                with open(features_file, 'r') as f:
                    features = json.load(f)
            elif self.config.feature_format == "npy":
                features = {"feature_vector": np.load(features_file).tolist()}
            else:
                logger.warning(f"Unknown feature format: {self.config.feature_format}")
                return {}
            
            logger.info(f"Features loaded from {features_path}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return {}
    
    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of extracted features"""
        summary = {
            "total_features": 0,
            "feature_types": [],
            "extraction_time_ms": features.get("extraction_time_ms", 0),
            "image_info": features.get("basic_info", {})
        }
        
        # Count features by type
        for feature_type in ["color_features", "texture_features", "shape_features", 
                           "edge_features", "histogram_features", "cnn_features"]:
            if feature_type in features and "error" not in features[feature_type]:
                summary["feature_types"].append(feature_type)
                summary["total_features"] += 1
        
        return summary

def create_visual_feature_extractor(config: Optional[FeatureExtractionConfig] = None) -> VisualFeatureExtractor:
    """Factory function to create visual feature extractor instance"""
    return VisualFeatureExtractor(config)