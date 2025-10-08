# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Model training and evaluation pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PhishingDataset(Dataset):
    """PyTorch dataset for phishing detection"""
    
    def __init__(self, texts: List[str], labels: List[int], features: Optional[List[Dict[str, Any]]] = None):
        self.texts = texts
        self.labels = labels
        self.features = features or [{}] * len(texts)
        
        # Validate data
        assert len(self.texts) == len(self.labels), "Texts and labels must have same length"
        assert len(self.texts) == len(self.features), "Texts and features must have same length"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'features': self.features[idx]
        }

class PhishingTrainer:
    """Training pipeline for phishing detection models"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 3)
        self.batch_size = config.get('batch_size', 16)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 100)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Early stopping
        self.early_stopping = config.get('early_stopping', True)
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.early_stopping_threshold = config.get('early_stopping_threshold', 0.001)
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    features: Optional[List[Dict[str, Any]]] = None,
                    test_size: float = 0.2, val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders for training, validation, and testing"""
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Split features if provided
        if features:
            f_temp, f_test = train_test_split(
                features, test_size=test_size, random_state=42
            )
            f_train, f_val = train_test_split(
                f_temp, test_size=val_size/(1-test_size), random_state=42
            )
        else:
            f_train = f_val = f_test = None
        
        # Create datasets
        train_dataset = PhishingDataset(X_train, y_train, f_train)
        val_dataset = PhishingDataset(X_val, y_val, f_val)
        test_dataset = PhishingDataset(X_test, y_test, f_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader, tokenizer) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Tokenize texts
            texts = batch['text']
            labels = batch['label'].to(self.device)
            
            # Tokenize
            tokenized = tokenizer.tokenize_batch(texts)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Logging
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self, val_loader: DataLoader, tokenizer) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Tokenize texts
                texts = batch['text']
                labels = batch['label'].to(self.device)
                
                # Tokenize
                tokenized = tokenizer.tokenize_batch(texts)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, tokenizer) -> Dict[str, Any]:
        """Complete training loop"""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, tokenizer)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, tokenizer)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Early stopping check
            if self.early_stopping:
                if val_metrics['f1'] > self.best_val_f1 + self.early_stopping_threshold:
                    self.best_val_f1 = val_metrics['f1']
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'best_val_f1': self.best_val_f1,
            'final_epoch': epoch + 1,
            'training_history': self.training_history
        }
    
    def evaluate(self, test_loader: DataLoader, tokenizer) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Tokenize texts
                texts = batch['text']
                labels = batch['label'].to(self.device)
                
                # Tokenize
                tokenized = tokenizer.tokenize_batch(texts)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        class_names = ['phish', 'benign', 'suspicious']
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                per_class_metrics[class_name] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i]
                }
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        logger.info(f"Test Results - Accuracy: {accuracy:.4f}, "
                   f"F1 Macro: {f1_macro:.4f}")
        
        return evaluation_results
    
    def save_training_results(self, save_path: str, evaluation_results: Dict[str, Any]):
        """Save training results and plots"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation results
        with open(save_path / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves(save_path)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(save_path, evaluation_results['confusion_matrix'])
        
        logger.info(f"Saved training results to {save_path}")
    
    def _plot_training_curves(self, save_path: Path):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.training_history['train_f1'], label='Train')
        axes[1, 0].plot(self.training_history['val_f1'], label='Validation')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, f'Learning Rate: {self.learning_rate}', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Configuration')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, save_path: Path, confusion_matrix: List[List[int]]):
        """Plot confusion matrix"""
        class_names = ['Phish', 'Benign', 'Suspicious']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_trainer(model: nn.Module, config: Dict[str, Any]) -> PhishingTrainer:
    """Factory function to create trainer"""
    return PhishingTrainer(model, config)