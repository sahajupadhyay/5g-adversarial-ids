"""
Industrial-Grade Baseline Deep Learning Model Implementation
==========================================================

This module implements a robust, production-ready Multi-Layer Perceptron (MLP) 
for intrusion detection with comprehensive error handling, monitoring, and 
security features.

Security Classification: CONFIDENTIAL
Performance Target: ≥94% accuracy with <100ms inference time
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import joblib

# Industrial-grade imports
# from config.data_config import DataConfig  # Optional configuration integration


class SecurityException(Exception):
    """Custom exception for security-related issues"""
    pass


class ModelException(Exception):
    """Custom exception for model-related issues"""
    pass


class DataValidationException(Exception):
    """Custom exception for data validation issues"""
    pass


class IDSDataset(Dataset):
    """
    Industrial-grade PyTorch Dataset for IDS with comprehensive validation
    and security checks.
    """
    
    def __init__(self, data_path: str, columns_path: str, validate_data: bool = True):
        """
        Initialize IDS Dataset with robust validation.
        
        Args:
            data_path: Path to processed CSV data
            columns_path: Path to feature columns JSON
            validate_data: Whether to perform comprehensive data validation
            
        Raises:
            DataValidationException: If data validation fails
            SecurityException: If security checks fail
        """
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        try:
            # Security: Validate paths
            self._validate_file_paths(data_path, columns_path)
            
            # Load and validate feature schema
            self.feature_columns = self._load_feature_schema(columns_path)
            
            # Load and validate dataset
            self.data_df = self._load_dataset(data_path)
            
            # Extract features and labels with validation
            self.X, self.y = self._extract_features_labels()
            
            if validate_data:
                self._perform_data_validation()
                
            self.logger.info(f"Dataset initialized: {len(self)} samples, {self.X.shape[1]} features")
            
        except Exception as e:
            self.logger.error(f"Dataset initialization failed: {e}")
            raise DataValidationException(f"Dataset initialization failed: {e}")
    
    def _validate_file_paths(self, data_path: str, columns_path: str) -> None:
        """Validate file paths for security and existence"""
        for path in [data_path, columns_path]:
            path_obj = Path(path)
            if not path_obj.exists():
                raise SecurityException(f"File does not exist: {path}")
            if not path_obj.is_file():
                raise SecurityException(f"Path is not a file: {path}")
            # Security: Check for path traversal attacks
            if ".." in str(path_obj):
                raise SecurityException(f"Path traversal detected: {path}")
    
    def _load_feature_schema(self, columns_path: str) -> List[str]:
        """Load and validate feature schema"""
        try:
            with open(columns_path, 'r') as f:
                columns = json.load(f)
            
            if not isinstance(columns, list) or not columns:
                raise DataValidationException("Invalid feature schema format")
                
            # Validate column names
            for col in columns:
                if not isinstance(col, str) or not col.strip():
                    raise DataValidationException(f"Invalid column name: {col}")
            
            return columns
            
        except json.JSONDecodeError as e:
            raise DataValidationException(f"Invalid JSON in feature schema: {e}")
        except Exception as e:
            raise DataValidationException(f"Failed to load feature schema: {e}")
    
    def _load_dataset(self, data_path: str) -> pd.DataFrame:
        """Load and validate dataset"""
        try:
            df = pd.read_csv(data_path)
            
            # Basic validation
            if df.empty:
                raise DataValidationException("Dataset is empty")
            
            # Validate required columns
            required_cols = set(self.feature_columns + ['Label'])
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise DataValidationException(f"Missing columns: {missing_cols}")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise DataValidationException("Dataset file is empty")
        except pd.errors.ParserError as e:
            raise DataValidationException(f"Failed to parse dataset: {e}")
        except Exception as e:
            raise DataValidationException(f"Failed to load dataset: {e}")
    
    def _extract_features_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and convert features and labels to tensors"""
        try:
            # Extract features in correct order
            features_df = self.data_df[self.feature_columns]
            X = torch.tensor(features_df.values, dtype=torch.float32)
            
            # Extract labels and convert to binary classification
            labels = self.data_df['Label'].values
            
            # Convert multi-class to binary (Normal vs Any Attack)
            binary_labels = np.where(labels == 'Normal', 0, 1)
            y = torch.tensor(binary_labels, dtype=torch.float32)
            
            return X, y
            
        except Exception as e:
            raise DataValidationException(f"Feature/label extraction failed: {e}")
    
    def _perform_data_validation(self) -> None:
        """Comprehensive data validation"""
        # Check for NaN/Inf values
        if torch.isnan(self.X).any() or torch.isinf(self.X).any():
            raise DataValidationException("Dataset contains NaN or Inf values")
        
        # Validate label distribution
        unique_labels = torch.unique(self.y)
        if not torch.equal(unique_labels, torch.tensor([0., 1.])):
            self.logger.warning(f"Unexpected labels found: {unique_labels}")
        
        # Check class balance
        class_counts = torch.bincount(self.y.long())
        minority_ratio = class_counts.min().item() / class_counts.sum().item()
        if minority_ratio < 0.01:  # Less than 1%
            self.logger.warning(f"Severe class imbalance detected: {minority_ratio:.3f}")
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        class_counts = torch.bincount(self.y.long())
        total_samples = class_counts.sum().float()
        
        # Inverse frequency weighting
        weights = total_samples / (2.0 * class_counts.float())
        return weights[1]  # Return weight for positive class


class BaselineIDS(nn.Module):
    """
    Industrial-Grade Multi-Layer Perceptron for Intrusion Detection
    
    Architecture optimized for adversarial robustness and performance.
    """
    
    def __init__(self, 
                 input_features: int,
                 hidden_layers: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.5,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        """
        Initialize baseline IDS model.
        
        Args:
            input_features: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(BaselineIDS, self).__init__()
        
        self.input_features = input_features
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Activation function mapping
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU()
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        
        # Build network layers
        self.layers = self._build_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        # Performance tracking
        self.training_start_time = None
        self.inference_times = []
    
    def _build_layers(self) -> nn.ModuleList:
        """Build network layers dynamically"""
        layers = nn.ModuleList()
        
        prev_size = self.input_features
        
        # Hidden layers
        for hidden_size in self.hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_size, 1))
        
        return layers
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU-based activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with performance monitoring"""
        start_time = time.time()
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Maintain only recent inference times (last 1000)
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        
        return x
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000  # Convert to ms
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'Multi-Layer Perceptron',
            'input_features': self.input_features,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'avg_inference_time_ms': self.get_average_inference_time()
        }


class BaselineTrainer:
    """
    Industrial-grade trainer for baseline IDS model with comprehensive
    monitoring, validation, and security features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize trainer with configuration"""
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        # Default training configuration
        default_config = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'min_delta': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip_value': 1.0,
            'num_workers': min(4, os.cpu_count()),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = {**default_config, **(config or {})}
        self.device = torch.device(self.config['device'])
        
        # Training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.best_metrics = {}
        self.training_history = []
        
        self.logger.info(f"Trainer initialized with device: {self.device}")
    
    @contextmanager
    def _training_context(self):
        """Context manager for training with proper cleanup"""
        try:
            self.logger.info("Starting training context")
            yield
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            # Cleanup GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.logger.info("Training context closed")
    
    def prepare_data(self, data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders with validation"""
        try:
            # File paths
            train_path = os.path.join(data_dir, 'train.csv')
            val_path = os.path.join(data_dir, 'val.csv')
            test_path = os.path.join(data_dir, 'test.csv')
            columns_path = os.path.join(data_dir, 'feature_columns.json')
            
            # Create datasets
            train_dataset = IDSDataset(train_path, columns_path, validate_data=True)
            val_dataset = IDSDataset(val_path, columns_path, validate_data=True)
            test_dataset = IDSDataset(test_path, columns_path, validate_data=False)
            
            # Calculate class weights from training data
            self.class_weight = train_dataset.get_class_weights().to(self.device)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Store feature count for model initialization
            self.num_features = len(train_dataset.feature_columns)
            
            self.logger.info(f"Data loaders prepared:")
            self.logger.info(f"  Train: {len(train_dataset)} samples")
            self.logger.info(f"  Val: {len(val_dataset)} samples") 
            self.logger.info(f"  Test: {len(test_dataset)} samples")
            self.logger.info(f"  Features: {self.num_features}")
            self.logger.info(f"  Class weight: {self.class_weight.item():.4f}")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise DataValidationException(f"Data preparation failed: {e}")
    
    def initialize_model(self) -> None:
        """Initialize model, optimizer, and loss function"""
        try:
            # Initialize model
            self.model = BaselineIDS(
                input_features=self.num_features,
                hidden_layers=[256, 128, 64],
                dropout_rate=0.5,
                use_batch_norm=True,
                activation='relu'
            ).to(self.device)
            
            # Initialize optimizer with weight decay
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # Initialize loss function with class weights
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
            
            self.logger.info("Model initialized:")
            model_info = self.model.get_model_info()
            for key, value in model_info.items():
                self.logger.info(f"  {key}: {value}")
                
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise ModelException(f"Model initialization failed: {e}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train single epoch with comprehensive monitoring"""
        self.model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Training progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['grad_clip_value']
            )
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(outputs).round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': epoch_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0)
        }
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate single epoch"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs).round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Calculate validation metrics
        val_metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0)
        }
        
        return val_metrics
    
    def train(self, data_dir: str, model_dir: str, results_dir: str) -> Dict[str, Any]:
        """Main training loop with comprehensive monitoring"""
        with self._training_context():
            try:
                # Prepare data
                train_loader, val_loader, test_loader = self.prepare_data(data_dir)
                
                # Initialize model
                self.initialize_model()
                
                # Training variables
                best_val_f1 = -1.0
                epochs_no_improve = 0
                
                # Create directories
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)
                
                # Training loop
                for epoch in range(self.config['epochs']):
                    self.logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
                    
                    # Train epoch
                    train_metrics = self.train_epoch(train_loader)
                    
                    # Validate epoch
                    val_metrics = self.validate_epoch(val_loader)
                    
                    # Log metrics
                    self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                                   f"Acc: {train_metrics['accuracy']:.4f}, "
                                   f"F1: {train_metrics['f1']:.4f}")
                    self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                                   f"Acc: {val_metrics['accuracy']:.4f}, "
                                   f"F1: {val_metrics['f1']:.4f}")
                    
                    # Save training history
                    epoch_record = {
                        'epoch': epoch + 1,
                        'train': train_metrics,
                        'validation': val_metrics
                    }
                    self.training_history.append(epoch_record)
                    
                    # Early stopping and model checkpointing
                    improvement = val_metrics['f1'] - best_val_f1
                    if improvement > self.config['min_delta']:
                        best_val_f1 = val_metrics['f1']
                        epochs_no_improve = 0
                        self.best_metrics = val_metrics.copy()
                        
                        # Save best model
                        model_path = os.path.join(model_dir, "baseline_model.pt")
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'metrics': val_metrics,
                            'config': self.config,
                            'model_info': self.model.get_model_info()
                        }, model_path)
                        
                        self.logger.info(f"✅ New best F1: {best_val_f1:.4f} - Model saved!")
                        
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.config['patience']:
                            self.logger.info(f"Early stopping after {self.config['patience']} epochs without improvement")
                            break
                
                # Final evaluation on test set
                self.logger.info("\n" + "="*80)
                self.logger.info("🧪 FINAL EVALUATION ON TEST SET")
                self.logger.info("="*80)
                
                test_metrics = self.evaluate_test_set(test_loader, model_dir)
                
                # Save comprehensive results
                final_results = {
                    'training_config': self.config,
                    'model_info': self.model.get_model_info(),
                    'best_validation_metrics': self.best_metrics,
                    'final_test_metrics': test_metrics,
                    'training_history': self.training_history,
                    'total_epochs': epoch + 1
                }
                
                results_path = os.path.join(results_dir, "training_results.json")
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2)
                
                self.logger.info(f"📊 Results saved to: {results_path}")
                
                return final_results
                
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                raise ModelException(f"Training failed: {e}")
    
    def evaluate_test_set(self, test_loader: DataLoader, model_dir: str) -> Dict[str, Any]:
        """Comprehensive test set evaluation"""
        # Load best model (trusted source - internal training)
        model_path = os.path.join(model_dir, "baseline_model.pt")
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test Evaluation")
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(inputs)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                preds = torch.sigmoid(outputs).squeeze().round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'avg_inference_time_ms': np.mean(inference_times),
            'classification_report': classification_report(all_labels, all_preds, output_dict=True),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        # Log final results
        self.logger.info(f"🎯 FINAL TEST RESULTS:")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
        self.logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        self.logger.info(f"  Avg Inference: {test_metrics['avg_inference_time_ms']:.2f}ms")
        
        # Performance validation
        if test_metrics['accuracy'] >= 0.94:
            self.logger.info("🎉 TARGET ACCURACY ACHIEVED! (≥94%)")
        else:
            self.logger.warning(f"❌ Target accuracy not met. Current: {test_metrics['accuracy']:.4f}")
        
        if test_metrics['avg_inference_time_ms'] <= 100:
            self.logger.info("⚡ PERFORMANCE TARGET MET! (≤100ms)")
        else:
            self.logger.warning(f"⚠️ Inference time exceeded target: {test_metrics['avg_inference_time_ms']:.2f}ms")
        
        return test_metrics


if __name__ == "__main__":
    """Direct execution for testing"""
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Train baseline IDS model")
    parser.add_argument('--data_dir', type=str, required=True, help="Processed data directory")
    parser.add_argument('--model_dir', type=str, required=True, help="Model output directory")
    parser.add_argument('--results_dir', type=str, required=True, help="Results output directory")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BaselineTrainer()
    
    # Train model
    results = trainer.train(args.data_dir, args.model_dir, args.results_dir)
    
    print("\n🎉 Training completed successfully!")
    print(f"Final accuracy: {results['final_test_metrics']['accuracy']:.4f}")