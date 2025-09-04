"""
Baseline IDS CLI Module

Handles baseline Random Forest model training, evaluation, and optimization
for the 5G adversarial IDS system.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from src.cli.utils import (
    load_processed_data, save_model, save_results, 
    create_output_directory, print_experiment_header, 
    print_experiment_footer, format_performance_metrics,
    get_class_names
)
# Import the fixed baseline training functions
from src.models.baseline_rf_tuned import train_baseline_rf_tuned, evaluate_model

class BaselineCLI:
    """CLI interface for baseline IDS training and evaluation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.baseline_config = config.get('baseline', {})
        self.data_config = config.get('data', {})
        self.models_config = config.get('models', {})
        
    def validate_baseline_config(self) -> bool:
        """Validate baseline-specific configuration."""
        required_keys = ['model_type', 'hyperparameters']
        
        for key in required_keys:
            if key not in self.baseline_config:
                self.logger.error(f"❌ Missing baseline config key: {key}")
                return False
        
        # Validate hyperparameters
        hyperparams = self.baseline_config.get('hyperparameters', {})
        expected_params = ['n_estimators', 'max_depth', 'random_state']
        
        for param in expected_params:
            if param not in hyperparams:
                self.logger.warning(f"⚠️  Missing hyperparameter: {param}, using default")
        
        return True
    
    def load_training_data(self) -> Dict[str, np.ndarray]:
        """Load and validate training data."""
        data_dir = self.data_config.get('processed_dir', 'data/processed')
        
        try:
            data = load_processed_data(data_dir, self.logger)
            return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load training data: {e}")
            raise
    
    def train_baseline_model(self, data: Dict[str, np.ndarray]) -> Any:
        """Train baseline Random Forest model."""
        self.logger.info("🏗️  Training baseline Random Forest model...")
        
        # Extract training data
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Get hyperparameters
        hyperparams = self.baseline_config.get('hyperparameters', {})
        
        # Set defaults if not provided
        default_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'max_features': 'sqrt',
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        for key, default_value in default_params.items():
            if key not in hyperparams:
                hyperparams[key] = default_value
        
        self.logger.info(f"🔧 Hyperparameters: {hyperparams}")
        
        try:
            # Use the proper training function from baseline_rf_tuned
            model, scaler, results = train_baseline_rf_tuned(
                X_train, y_train, X_test, y_test, **hyperparams
            )
            
            self.logger.info("✅ Baseline model training completed")
            self.logger.info(f"📊 Performance: {format_performance_metrics(results)}")
            
            return model, scaler, results
            
        except Exception as e:
            self.logger.error(f"❌ Baseline training failed: {e}")
            raise
    
    def evaluate_baseline_model(self, model, scaler, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive evaluation of baseline model."""
        self.logger.info("📊 Evaluating baseline model...")
        
        try:
            # Use the proper evaluation function from baseline_rf_tuned
            results = evaluate_model(model, scaler, data)
            
            self.logger.info("✅ Baseline model evaluation completed")
            self.logger.info(f"📈 Results: {format_performance_metrics(results)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Baseline evaluation failed: {e}")
            raise
    
    def save_baseline_artifacts(self, model, scaler, results: Dict[str, Any], 
                              evaluation_results: Dict[str, Any]) -> bool:
        """Save trained model, scaler, and results."""
        try:
            # Create output directory
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'baseline_training',
                self.logger
            )
            
            # Save model
            model_path = output_dir / 'rf_baseline_tuned.joblib'
            model_metadata = {
                'model_type': 'RandomForest',
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': self.baseline_config.get('hyperparameters', {}),
                'training_results': results,
                'evaluation_results': {k: v for k, v in evaluation_results.items() 
                                     if k not in ['predictions', 'prediction_probabilities']}
            }
            
            save_model(model, str(model_path), model_metadata, self.logger)
            
            # Save scaler
            scaler_path = output_dir / 'scaler.joblib'
            save_model(scaler, str(scaler_path), None, self.logger)
            
            # Save detailed results
            results_path = output_dir / 'baseline_results.json'
            combined_results = {
                'training_results': results,
                'evaluation_results': evaluation_results,
                'configuration': self.baseline_config
            }
            save_results(combined_results, str(results_path), self.logger)
            
            # Update models directory if specified
            models_dir = self.models_config.get('save_dir')
            if models_dir:
                models_path = Path(models_dir)
                models_path.mkdir(parents=True, exist_ok=True)
                
                # Copy to models directory
                import shutil
                shutil.copy2(model_path, models_path / 'rf_baseline_tuned.joblib')
                shutil.copy2(scaler_path, models_path / 'scaler.joblib')
                
                self.logger.info(f"✅ Models copied to: {models_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save baseline artifacts: {e}")
            return False
    
    def execute(self) -> bool:
        """Execute complete baseline training pipeline."""
        print_experiment_header("BASELINE IDS TRAINING", self.config, self.logger)
        
        try:
            # Validate configuration
            if not self.validate_baseline_config():
                return False
            
            # Load training data
            self.logger.info("📂 Loading training data...")
            data = self.load_training_data()
            
            # Train baseline model
            model, scaler, training_results = self.train_baseline_model(data)
            
            # Evaluate model
            evaluation_results = self.evaluate_baseline_model(model, scaler, data)
            
            # Save artifacts
            if not self.save_baseline_artifacts(model, scaler, training_results, evaluation_results):
                return False
            
            # Log final summary
            self.logger.info("📈 BASELINE TRAINING SUMMARY:")
            self.logger.info(f"   Accuracy: {evaluation_results.get('accuracy', 'N/A'):.4f}")
            self.logger.info(f"   Macro-F1: {evaluation_results.get('macro_f1', 'N/A'):.4f}")
            self.logger.info(f"   Training samples: {data['X_train'].shape[0]}")
            self.logger.info(f"   Test samples: {data['X_test'].shape[0]}")
            self.logger.info(f"   Features: {data['X_train'].shape[1]}")
            
            print_experiment_footer("BASELINE IDS TRAINING", True, self.logger)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Baseline training pipeline failed: {e}")
            print_experiment_footer("BASELINE IDS TRAINING", False, self.logger)
            return False
