"""
Enhanced Baseline IDS - Industrial Grade Implementation
Built upon friend's proven deep learning architecture with config management and robust error handling.
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import joblib
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse

# Import friend's proven components
from friend_baseline import BaselineIDS, PFCPDataset


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate required configuration sections"""
        required_sections = ['model', 'data', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class EnhancedBaselineIDS(BaselineIDS):
    """
    Enhanced version of friend's BaselineIDS with configuration support
    Preserves the proven 256->128->64->1 architecture
    """
    
    def __init__(self, input_features: int, config: Dict):
        # Initialize with friend's proven architecture
        super().__init__(input_features)
        self.config = config
        
        # Log the architecture for verification
        arch = config.get('architecture', {})
        logging.info(f"BaselineIDS initialized: {input_features} -> {arch.get('hidden_layers', [256, 128, 64])} -> 1")
    
    @classmethod
    def from_config(cls, input_features: int, config_manager: ConfigManager):
        """Factory method to create model from configuration"""
        model_config = config_manager.get('model', {})
        return cls(input_features, model_config)


class EnhancedTrainer:
    """Enhanced trainer with friend's proven training logic + industrial features"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        # Create log directory if saving to file
        if log_config.get('save_to_file', False):
            log_dir = Path(log_config.get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup file handler
            log_file = log_dir / 'baseline_training.log'
            logging.basicConfig(
                level=getattr(logging, log_config.get('level', 'INFO')),
                format=log_config.get('format', '%(asctime)s [%(levelname)s] - %(message)s'),
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=getattr(logging, log_config.get('level', 'INFO')),
                format=log_config.get('format', '%(asctime)s [%(levelname)s] - %(message)s')
            )
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data using friend's proven data pipeline"""
        try:
            # Load friend's processed data and artifacts
            data_config = self.config.get('data', {})
            splits = data_config.get('splits', {})
            paths = data_config.get('paths', {})
            
            train_df = pd.read_csv(splits['train'])
            val_df = pd.read_csv(splits['val']) 
            test_df = pd.read_csv(splits['test'])
            
            # Load friend's feature columns and scaler
            with open(paths['processed_features'], 'r') as f:
                feature_columns = json.load(f)
            
            scaler = joblib.load(paths['scaler'])
            
            logging.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            logging.info(f"Using {len(feature_columns)} features")
            
            # Create datasets using friend's PFCPDataset interface
            # Note: Friend's dataset expects file paths, not dataframes
            train_dataset = PFCPDataset(splits['train'], paths['processed_features'])
            val_dataset = PFCPDataset(splits['val'], paths['processed_features'])
            test_dataset = PFCPDataset(splits['test'], paths['processed_features'])
            
            # Create data loaders
            batch_size = self.config.get('model.training.batch_size', 512)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise
    
    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Calculate class weights using friend's approach"""
        try:
            # Count classes in training data
            total_samples = 0
            positive_samples = 0
            
            for _, labels in train_loader:
                total_samples += len(labels)
                positive_samples += labels.sum().item()
            
            negative_samples = total_samples - positive_samples
            
            # Calculate weights (inverse frequency)
            pos_weight = negative_samples / positive_samples if positive_samples > 0 else 1.0
            
            logging.info(f"Class distribution - Negative: {negative_samples}, Positive: {positive_samples}")
            logging.info(f"Calculated pos_weight: {pos_weight:.4f}")
            
            return torch.tensor([pos_weight], device=self.device)
            
        except Exception as e:
            logging.error(f"Failed to calculate class weights: {str(e)}")
            return torch.tensor([1.0], device=self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, save_path: str) -> Dict:
        """
        Train model using friend's proven training loop with enhancements
        """
        try:
            model.to(self.device)
            
            # Setup training parameters from config
            training_config = self.config.get('model.training', {})
            epochs = training_config.get('epochs', 50)
            lr = training_config.get('learning_rate', 0.001)
            
            # Setup optimizer (friend used Adam)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Setup loss with class weights (friend's approach)
            class_weights = self.calculate_class_weights(train_loader)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            
            # Early stopping configuration
            early_stop_config = training_config.get('early_stopping', {})
            patience = early_stop_config.get('patience', 5)
            min_delta = early_stop_config.get('min_delta', 0.0001)
            
            # Training state
            best_val_f1 = 0.0
            epochs_no_improve = 0
            training_history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
            
            logging.info(f"Starting training - {epochs} epochs, lr={lr}, patience={patience}")
            
            for epoch in range(epochs):
                # Training phase (friend's proven loop)
                model.train()
                train_loss = 0.0
                
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
                for inputs, labels in train_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Validation phase (friend's proven evaluation)
                model.eval()
                val_preds, val_labels = [], []
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for inputs, labels in val_pbar:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        preds = torch.sigmoid(outputs).squeeze().round()
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                val_f1 = f1_score(val_labels, val_preds, zero_division=0)
                val_precision = precision_score(val_labels, val_preds, zero_division=0)
                val_recall = recall_score(val_labels, val_preds, zero_division=0)
                
                # Store history
                training_history['train_loss'].append(train_loss / len(train_loader))
                training_history['val_f1'].append(val_f1)
                training_history['val_precision'].append(val_precision)
                training_history['val_recall'].append(val_recall)
                
                logging.info(
                    f"Epoch {epoch+1} | Val F1: {val_f1:.4f} | "
                    f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
                )
                
                # Early stopping and checkpointing (friend's approach)
                if val_f1 > best_val_f1 + min_delta:
                    best_val_f1 = val_f1
                    epochs_no_improve = 0
                    
                    # Save best model
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_f1': val_f1,
                        'config': self.config.config
                    }, save_path)
                    logging.info(f"New best model saved (F1: {val_f1:.4f})")
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            return {
                'best_val_f1': best_val_f1,
                'final_epoch': epoch + 1,
                'training_history': training_history
            }
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model using friend's proven evaluation logic"""
        try:
            model.eval()
            test_preds, test_labels = [], []
            
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc="Testing")
                for inputs, labels in test_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs).squeeze().round()
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            # Calculate comprehensive metrics
            metrics = {
                'f1': f1_score(test_labels, test_preds, zero_division=0),
                'precision': precision_score(test_labels, test_preds, zero_division=0),
                'recall': recall_score(test_labels, test_preds, zero_division=0),
                'accuracy': accuracy_score(test_labels, test_preds)
            }
            
            logging.info(
                f"Test Results - F1: {metrics['f1']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Enhanced Baseline IDS Training")
    parser.add_argument('--config', type=str, 
                       default='configs/baseline_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['train', 'evaluate'], 
                       default='train',
                       help='Mode: train new model or evaluate existing')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        
        # Initialize trainer
        trainer = EnhancedTrainer(config_manager)
        
        # Load data
        train_loader, val_loader, test_loader = trainer.load_data()
        
        # Determine input features from data
        sample_batch = next(iter(train_loader))
        input_features = sample_batch[0].shape[1]
        
        if args.mode == 'train':
            # Create model
            model = EnhancedBaselineIDS.from_config(input_features, config_manager)
            
            # Train model
            save_path = config_manager.get('paths.models.enhanced_baseline')
            training_results = trainer.train_model(model, train_loader, val_loader, save_path)
            
            # Load best model for final evaluation
            checkpoint = torch.load(save_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Final evaluation
            test_metrics = trainer.evaluate_model(model, test_loader)
            
            # Save results
            results_dir = config_manager.get('paths.results.base_dir')
            os.makedirs(results_dir, exist_ok=True)
            
            final_results = {
                'training': training_results,
                'test_metrics': test_metrics,
                'config': config_manager.config
            }
            
            results_path = os.path.join(results_dir, 'training_results.json')
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=4, default=str)
            
            logging.info(f"Training complete! Results saved to {results_path}")
            
        elif args.mode == 'evaluate':
            # Load existing model for evaluation - try enhanced first, fall back to friend's
            enhanced_model_path = config_manager.get('paths.models.enhanced_baseline')
            friend_model_path = config_manager.get('paths.models.friend_baseline')
            
            if os.path.exists(enhanced_model_path):
                logging.info(f"Loading enhanced model from {enhanced_model_path}")
                model = EnhancedBaselineIDS.from_config(input_features, config_manager)
                checkpoint = torch.load(enhanced_model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            elif os.path.exists(friend_model_path):
                logging.info(f"Loading friend's baseline model from {friend_model_path}")
                model = EnhancedBaselineIDS.from_config(input_features, config_manager)
                # Friend's model is just state_dict, not checkpoint
                model.load_state_dict(torch.load(friend_model_path, weights_only=False))
            else:
                raise FileNotFoundError(f"No model found at {enhanced_model_path} or {friend_model_path}")
            
            test_metrics = trainer.evaluate_model(model, test_loader)
            
            logging.info("Evaluation complete!")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()