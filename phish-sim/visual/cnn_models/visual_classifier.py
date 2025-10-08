# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Simple CNN model for visual phishing detection
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
from PIL import Image
import cv2

from config import get_config, CNNModelConfig

logger = logging.getLogger(__name__)

class SimpleCNNClassifier:
    """Simple CNN classifier for visual phishing detection"""
    
    def __init__(self, config: Optional[CNNModelConfig] = None):
        self.config = config or get_config("cnn")
        self.model = None
        self.is_trained = False
        self.class_names = ["phish", "benign", "suspicious"]
        
        # Ensure model directory exists
        Path(self.config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def create_model(self) -> Dict[str, Any]:
        """Create simple CNN model architecture"""
        try:
            # Simulate model creation (in real implementation, would use TensorFlow/Keras)
            model_info = {
                "architecture": "SimpleCNN",
                "input_shape": self.config.input_size,
                "num_classes": self.config.num_classes,
                "total_params": self._calculate_model_params(),
                "model_size_mb": self._calculate_model_size(),
                "layers": self._get_model_layers(),
                "created_at": time.time()
            }
            
            logger.info(f"Created CNN model: {model_info['total_params']} parameters, {model_info['model_size_mb']:.2f} MB")
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def _calculate_model_params(self) -> int:
        """Calculate approximate number of model parameters"""
        # Simplified calculation for demo
        input_size = np.prod(self.config.input_size)
        
        # Conv layers
        conv_params = 32 * 3 * 3 * 3 + 32  # First conv layer
        conv_params += 64 * 32 * 3 * 3 + 64  # Second conv layer
        conv_params += 128 * 64 * 3 * 3 + 128  # Third conv layer
        
        # Dense layers
        dense_params = 128 * 512 + 512  # First dense layer
        dense_params += 512 * 256 + 256  # Second dense layer
        dense_params += 256 * self.config.num_classes + self.config.num_classes  # Output layer
        
        return conv_params + dense_params
    
    def _calculate_model_size(self) -> float:
        """Calculate approximate model size in MB"""
        params = self._calculate_model_params()
        # Assuming 4 bytes per parameter (float32)
        size_bytes = params * 4
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_model_layers(self) -> List[Dict[str, Any]]:
        """Get model layer information"""
        return [
            {
                "type": "Conv2D",
                "filters": 32,
                "kernel_size": (3, 3),
                "activation": "relu",
                "input_shape": self.config.input_size
            },
            {
                "type": "MaxPooling2D",
                "pool_size": (2, 2)
            },
            {
                "type": "Conv2D",
                "filters": 64,
                "kernel_size": (3, 3),
                "activation": "relu"
            },
            {
                "type": "MaxPooling2D",
                "pool_size": (2, 2)
            },
            {
                "type": "Conv2D",
                "filters": 128,
                "kernel_size": (3, 3),
                "activation": "relu"
            },
            {
                "type": "GlobalAveragePooling2D"
            },
            {
                "type": "Dense",
                "units": 512,
                "activation": "relu",
                "dropout": self.config.dropout_rate
            },
            {
                "type": "Dense",
                "units": 256,
                "activation": "relu",
                "dropout": self.config.dropout_rate
            },
            {
                "type": "Dense",
                "units": self.config.num_classes,
                "activation": "softmax"
            }
        ]
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.config.input_size[0], self.config.input_size[1]))
            
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            # Normalize
            if self.config.normalization == "imagenet":
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array / 255.0 - mean) / std
            elif self.config.normalization == "custom":
                # Custom normalization
                img_array = img_array / 255.0
            # else: no normalization
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Predict phishing probability for an image"""
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Simulate model prediction (in real implementation, would use actual model)
            prediction_result = self._simulate_prediction(processed_image)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = {
                "image_path": image_path,
                "prediction": prediction_result["class"],
                "confidence": prediction_result["confidence"],
                "probabilities": prediction_result["probabilities"],
                "processing_time_ms": processing_time,
                "model_info": {
                    "architecture": "SimpleCNN",
                    "input_size": self.config.input_size,
                    "num_classes": self.config.num_classes
                },
                "timestamp": time.time()
            }
            
            logger.info(f"Prediction completed in {processing_time:.2f}ms: {result['prediction']} ({result['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict for image {image_path}: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _simulate_prediction(self, processed_image: np.ndarray) -> Dict[str, Any]:
        """Simulate model prediction (placeholder for actual model)"""
        # Simulate prediction based on image characteristics
        img_mean = np.mean(processed_image)
        img_std = np.std(processed_image)
        
        # Simple heuristic-based prediction
        if img_std > 0.3:  # High variation might indicate phishing
            probabilities = [0.7, 0.2, 0.1]  # phish, benign, suspicious
        elif img_mean < 0.3:  # Dark images might be suspicious
            probabilities = [0.3, 0.4, 0.3]  # phish, benign, suspicious
        else:  # Normal images likely benign
            probabilities = [0.1, 0.8, 0.1]  # phish, benign, suspicious
        
        # Add some randomness for demo
        probabilities = np.array(probabilities) + np.random.normal(0, 0.05, 3)
        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        # Get prediction
        class_idx = np.argmax(probabilities)
        confidence = probabilities[class_idx]
        
        return {
            "class": self.class_names[class_idx],
            "confidence": float(confidence),
            "probabilities": {
                self.class_names[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict phishing probability for multiple images"""
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        
        return results
    
    def train(self, training_data: List[Tuple[str, str]], 
              validation_data: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """Train the model (simulated)"""
        try:
            logger.info(f"Starting training with {len(training_data)} samples")
            
            # Simulate training process
            training_history = {
                "epochs": [],
                "train_loss": [],
                "train_accuracy": [],
                "val_loss": [],
                "val_accuracy": []
            }
            
            for epoch in range(1, self.config.num_epochs + 1):
                # Simulate training metrics
                train_loss = max(0.1, 1.0 - (epoch / self.config.num_epochs) * 0.8 + np.random.normal(0, 0.05))
                train_acc = min(0.95, 0.5 + (epoch / self.config.num_epochs) * 0.4 + np.random.normal(0, 0.02))
                
                val_loss = train_loss + np.random.normal(0, 0.1)
                val_acc = train_acc - np.random.normal(0, 0.05)
                
                training_history["epochs"].append(epoch)
                training_history["train_loss"].append(float(train_loss))
                training_history["train_accuracy"].append(float(train_acc))
                training_history["val_loss"].append(float(val_loss))
                training_history["val_accuracy"].append(float(val_acc))
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.config.num_epochs}: "
                              f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
                              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
            # Calculate final metrics
            final_metrics = {
                "final_train_loss": training_history["train_loss"][-1],
                "final_train_accuracy": training_history["train_accuracy"][-1],
                "final_val_loss": training_history["val_loss"][-1],
                "final_val_accuracy": training_history["val_accuracy"][-1],
                "training_time_minutes": self.config.num_epochs * 0.1,  # Simulated
                "convergence_epoch": self.config.num_epochs // 2,
                "best_val_accuracy": max(training_history["val_accuracy"])
            }
            
            self.is_trained = True
            
            # Save training results
            self._save_training_results(training_history, final_metrics)
            
            logger.info(f"Training completed: {final_metrics['final_val_accuracy']:.4f} validation accuracy")
            return {
                "training_history": training_history,
                "final_metrics": final_metrics,
                "model_info": self.create_model()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_training_results(self, training_history: Dict[str, List], 
                              final_metrics: Dict[str, Any]):
        """Save training results to file"""
        try:
            results = {
                "training_history": training_history,
                "final_metrics": final_metrics,
                "config": {
                    "num_epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "input_size": self.config.input_size,
                    "num_classes": self.config.num_classes
                },
                "timestamp": time.time()
            }
            
            results_path = Path(self.config.model_save_path).parent / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Training results saved to {results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training results: {e}")
    
    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            logger.info(f"Evaluating model with {len(test_data)} test samples")
            
            # Simulate evaluation
            predictions = []
            true_labels = []
            
            for image_path, true_label in test_data:
                result = self.predict(image_path)
                predictions.append(result["prediction"])
                true_labels.append(true_label)
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_labels, predictions)
            
            logger.info(f"Evaluation completed: {metrics['accuracy']:.4f} accuracy")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _calculate_metrics(self, true_labels: List[str], predictions: List[str]) -> Dict[str, Any]:
        """Calculate classification metrics"""
        from collections import Counter
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_name in self.class_names:
            true_positives = sum(1 for true, pred in zip(true_labels, predictions) 
                               if true == class_name and pred == class_name)
            false_positives = sum(1 for true, pred in zip(true_labels, predictions) 
                                if true != class_name and pred == class_name)
            false_negatives = sum(1 for true, pred in zip(true_labels, predictions) 
                                if true == class_name and pred != class_name)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": true_labels.count(class_name)
            }
        
        # Calculate macro averages
        macro_precision = np.mean([metrics["precision"] for metrics in class_metrics.values()])
        macro_recall = np.mean([metrics["recall"] for metrics in class_metrics.values()])
        macro_f1 = np.mean([metrics["f1_score"] for metrics in class_metrics.values()])
        
        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "class_metrics": class_metrics,
            "confusion_matrix": self._create_confusion_matrix(true_labels, predictions),
            "total_samples": len(true_labels)
        }
    
    def _create_confusion_matrix(self, true_labels: List[str], predictions: List[str]) -> Dict[str, Dict[str, int]]:
        """Create confusion matrix"""
        matrix = {}
        
        for true_class in self.class_names:
            matrix[true_class] = {}
            for pred_class in self.class_names:
                count = sum(1 for true, pred in zip(true_labels, predictions) 
                           if true == true_class and pred == pred_class)
                matrix[true_class][pred_class] = count
        
        return matrix
    
    def save_model(self, model_path: Optional[str] = None):
        """Save model to file"""
        try:
            save_path = model_path or self.config.model_save_path
            
            # In real implementation, would save actual model
            model_info = {
                "model_type": "SimpleCNN",
                "config": {
                    "input_size": self.config.input_size,
                    "num_classes": self.config.num_classes,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate
                },
                "is_trained": self.is_trained,
                "class_names": self.class_names,
                "saved_at": time.time()
            }
            
            with open(save_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model from file"""
        try:
            load_path = model_path or self.config.model_save_path
            
            if not Path(load_path).exists():
                logger.warning(f"Model file not found: {load_path}")
                return False
            
            with open(load_path, 'r') as f:
                model_info = json.load(f)
            
            self.is_trained = model_info.get("is_trained", False)
            self.class_names = model_info.get("class_names", self.class_names)
            
            logger.info(f"Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "SimpleCNN",
            "is_trained": self.is_trained,
            "config": {
                "input_size": self.config.input_size,
                "num_classes": self.config.num_classes,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs
            },
            "performance_targets": {
                "target_accuracy": self.config.target_accuracy,
                "target_precision": self.config.target_precision,
                "target_recall": self.config.target_recall,
                "target_f1": self.config.target_f1,
                "max_inference_time_ms": self.config.max_inference_time_ms
            },
            "model_size_mb": self._calculate_model_size(),
            "total_params": self._calculate_model_params()
        }

def create_visual_classifier(config: Optional[CNNModelConfig] = None) -> SimpleCNNClassifier:
    """Factory function to create visual classifier instance"""
    return SimpleCNNClassifier(config)