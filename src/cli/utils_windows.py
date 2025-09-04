"""
Windows-Compatible CLI Utility Functions for 5G Adversarial IDS

Provides common functionality for all CLI modules without emoji characters
that cause encoding issues in Windows terminals.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import os
import subprocess
import importlib
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def check_dependencies(logger: logging.Logger) -> bool:
    """Check if all required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'joblib', 'yaml', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('pyyaml')
            else:
                importlib.import_module(package)
            logger.debug(f"[OK] {package} - Available")
        except ImportError:
            logger.error(f"[MISSING] {package} - Not found")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("[OK] All dependencies satisfied")
    return True


def validate_config(config: dict, logger: logging.Logger) -> bool:
    """Validate configuration dictionary."""
    required_sections = ['data', 'models', 'output_dir']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"[ERROR] Missing required config section: {section}")
            return False
    
    # Validate data paths
    if 'data' in config:
        data_config = config['data']
        for key, path_value in data_config.items():
            if isinstance(path_value, str) and not os.path.exists(path_value):
                # Only warn for paths, not for other string values
                if any(x in key.lower() for x in ['dir', 'path', 'file']):
                    logger.warning(f"[WARNING] Data path may not exist: {path_value}")
    
    # Validate model save directory
    if 'models' in config:
        save_dir = config['models'].get('save_dir', 'models')
        os.makedirs(save_dir, exist_ok=True)
        logger.debug(f"[OK] Model save directory: {save_dir}")
    
    logger.info("[OK] Configuration validation passed")
    return True


def load_processed_data(data_dir: str, logger: logging.Logger) -> dict:
    """Load processed training and test data."""
    data_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    data = {}
    
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            logger.error(f"[ERROR] Required data file missing: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            data[file_name.replace('.npy', '')] = np.load(file_path)
            logger.debug(f"[OK] Loaded {file_name}: shape {data[file_name.replace('.npy', '')].shape}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {file_name}: {e}")
            raise
    
    # Load metadata if available
    metadata_path = os.path.join(data_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"[OK] Loaded metadata: {metadata}")
    
    logger.info(f"[OK] Data loaded successfully from {data_dir}")
    return data


def save_results(results: dict, output_file: str, logger: logging.Logger) -> bool:
    """Save results to JSON file."""
    try:
        import json
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        clean_results = clean_dict(results)
        
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"[OK] Results saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to save results: {e}")
        return False


def load_model(model_path: str, logger: logging.Logger):
    """Load a saved model."""
    try:
        import joblib
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"[OK] Model loaded from: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        raise


def save_model(model, model_file: str, metadata: dict = None, logger: logging.Logger = None) -> bool:
    """Save model and optionally metadata."""
    try:
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_file)
        
        # Save metadata if provided
        if metadata:
            metadata_file = model_file.replace('.joblib', '.json')
            save_results(metadata, metadata_file, logger)
            if logger:
                logger.info(f"[OK] Model and metadata saved to: {model_file}")
        else:
            if logger:
                logger.info(f"[OK] Model saved to: {model_file}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"[ERROR] Failed to save model: {e}")
        return False


def create_output_directory(base_dir: str, prefix: str, logger: logging.Logger) -> str:
    """Create timestamped output directory."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"[OK] Output directory created: {output_dir}")
    return output_dir


def verify_model_exists(model_path: str, logger: logging.Logger) -> bool:
    """Verify that model file exists."""
    if not os.path.exists(model_path):
        logger.error(f"[ERROR] Model file not found: {model_path}")
        return False
    return True


def load_model_safely(model_path: str, logger: logging.Logger):
    """Safely load model with comprehensive error handling."""
    try:
        import joblib
        
        if not verify_model_exists(model_path, logger):
            return None
        
        logger.debug(f"[INFO] Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Basic validation
        if not hasattr(model, 'predict'):
            logger.error(f"[ERROR] Loaded object is not a valid model (no predict method)")
            return None
        
        logger.info(f"[OK] Model loaded successfully: {type(model).__name__}")
        return model
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model from {model_path}: {e}")
        return None


def format_performance_metrics(results: dict) -> str:
    """Format performance metrics for display."""
    metrics = []
    
    # Core metrics
    if 'accuracy' in results:
        metrics.append(f"accuracy: {results['accuracy']:.4f}")
    if 'macro_f1' in results:
        metrics.append(f"macro_f1: {results['macro_f1']:.4f}")
    if 'weighted_f1' in results:
        metrics.append(f"weighted_f1: {results['weighted_f1']:.4f}")
    if 'precision' in results:
        metrics.append(f"precision: {results['precision']:.4f}")
    if 'recall' in results:
        metrics.append(f"recall: {results['recall']:.4f}")
    
    # Additional details
    if 'classification_report' in results:
        metrics.append(f"classification_report: {results['classification_report']}")
    if 'confusion_matrix' in results:
        metrics.append(f"confusion_matrix: {results['confusion_matrix']}")
    if 'hyperparameters' in results:
        metrics.append(f"hyperparameters: {results['hyperparameters']}")
    
    return " | ".join(metrics)


def print_experiment_header(experiment_name: str, config: dict, logger: logging.Logger):
    """Print standardized experiment header."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 80)
    
    # Print key configuration parameters
    if 'data' in config:
        logger.info(f"Data: {config['data'].get('processed_dir', 'Not specified')}")
    
    if 'models' in config:
        logger.info(f"Models: {config['models'].get('save_dir', 'Not specified')}")
    
    if 'output_dir' in config:
        logger.info(f"Output: {config['output_dir']}")
    
    logger.info("-" * 80)


def print_experiment_footer(experiment_name: str, success: bool, logger: logging.Logger):
    """Print standardized experiment footer."""
    logger.info("-" * 80)
    if success:
        logger.info(f"[SUCCESS] EXPERIMENT COMPLETED: {experiment_name}")
    else:
        logger.error(f"[FAILED] EXPERIMENT FAILED: {experiment_name}")
    logger.info("=" * 80)
