"""
CLI Utility Functions for 5G Adversarial IDS

Provides common functionality for all CLI modules including validation,
dependency checking, and configuration management.

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
import logging
import numpy as np
import pandas as pd

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup standardized logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Adv5G-Utils")

def check_dependencies(logger: logging.Logger) -> bool:
    """Check all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'joblib', 'yaml',
        'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.debug(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - MISSING")
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All dependencies satisfied")
    return True

def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validate configuration file structure and required fields."""
    required_sections = ['data', 'models', 'output']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"❌ Missing required config section: {section}")
            return False
    
    # Validate data paths
    data_config = config.get('data', {})
    required_data_paths = ['processed_dir', 'raw_dir']
    
    for path_key in required_data_paths:
        if path_key in data_config:
            path_value = data_config[path_key]
            if not Path(path_value).exists():
                logger.warning(f"⚠️  Data path may not exist: {path_value}")
    
    # Validate model paths
    models_config = config.get('models', {})
    if 'save_dir' in models_config:
        save_dir = Path(models_config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"✅ Model save directory: {save_dir}")
    
    logger.info("✅ Configuration validation passed")
    return True

def load_processed_data(data_dir: str, logger: logging.Logger) -> Dict[str, np.ndarray]:
    """Load preprocessed training and testing data."""
    data_path = Path(data_dir)
    
    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    data = {}
    
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            logger.error(f"❌ Required data file missing: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            data[file_name.replace('.npy', '')] = np.load(file_path)
            logger.debug(f"✅ Loaded {file_name}: shape {data[file_name.replace('.npy', '')].shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load {file_name}: {e}")
            raise
    
    # Load metadata if available
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        data['metadata'] = metadata
        logger.info(f"✅ Loaded metadata: {metadata}")
    
    logger.info(f"✅ Data loaded successfully from {data_dir}")
    logger.info(f"   Training samples: {data['X_train'].shape[0]}")
    logger.info(f"   Testing samples: {data['X_test'].shape[0]}")
    logger.info(f"   Features: {data['X_train'].shape[1]}")
    
    return data

def save_results(results: Dict[str, Any], output_path: str, logger: logging.Logger) -> bool:
    """Save results to JSON file with proper formatting."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert objects (recursively) to JSON-serializable forms
        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [make_json_serializable(v) for v in obj]
            # Fallback for objects with __dict__ (avoid large model objects)
            if hasattr(obj, '__dict__') and not isinstance(obj, (str, bytes)):
                try:
                    return make_json_serializable({k: v for k, v in obj.__dict__.items() if not k.startswith('_')})
                except Exception:
                    return str(obj)
            return obj

        serializable_results = make_json_serializable(results)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✅ Results saved to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        return False

def load_model(model_path: str, logger: logging.Logger):
    """Load a trained model from file."""
    try:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        import joblib
        model = joblib.load(model_file)
        logger.info(f"✅ Model loaded from: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def save_model(model, model_path: str, metadata: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
    """Save a trained model with optional metadata."""
    try:
        model_file = Path(model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(model, model_file)
        
        # Save metadata if provided (fully JSON-safe conversion)
        if metadata:
            metadata_path = model_file.with_suffix('.json')

            def make_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [make_json_serializable(v) for v in obj]
                if hasattr(obj, '__dict__') and not isinstance(obj, (str, bytes)):
                    try:
                        return make_json_serializable({k: v for k, v in obj.__dict__.items() if not k.startswith('_')})
                    except Exception:
                        return str(obj)
                return obj

            serializable_metadata = make_json_serializable(metadata)

            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
            
            if logger:
                logger.info(f"✅ Model and metadata saved to: {model_file}")
        elif logger:
            logger.info(f"✅ Model saved to: {model_file}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to save model: {e}")
        raise

def create_output_directory(base_dir: str, experiment_name: str, logger: logging.Logger) -> Path:
    """Create timestamped output directory for experiment results."""
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Output directory created: {output_dir}")
    return output_dir

def validate_model_file(model_path: str, logger: logging.Logger) -> bool:
    """Validate that a model file exists and can be loaded."""
    try:
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Try to load the model
        import joblib
        model = joblib.load(model_file)
        
        # Basic validation - check if it has predict method
        if not hasattr(model, 'predict'):
            logger.error(f"❌ Model does not have predict method: {model_path}")
            return False
        
        logger.info(f"✅ Model validation passed: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        return False

def get_class_names() -> List[str]:
    """Get standard class names for 5G IDS dataset."""
    return ['Mal_Del', 'Mal_Estab', 'Mal_Mod', 'Mal_Mod2', 'Normal']

def format_performance_metrics(metrics: Dict[str, float]) -> str:
    """Format performance metrics for display."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return " | ".join(formatted)

def safe_log(logger: logging.Logger, level: str, message: str):
    """Safely log messages with fallback for Unicode issues."""
    try:
        if level.lower() == 'info':
            logger.info(message)
        elif level.lower() == 'error':
            logger.error(message)
        elif level.lower() == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    except UnicodeEncodeError:
        # Fallback: replace problematic Unicode characters
        safe_message = message.encode('ascii', errors='replace').decode('ascii')
        if level.lower() == 'info':
            logger.info(safe_message)
        elif level.lower() == 'error':
            logger.error(safe_message)
        elif level.lower() == 'warning':
            logger.warning(safe_message)
        else:
            logger.info(safe_message)

def print_experiment_header(experiment_name: str, config: Dict[str, Any], logger: logging.Logger):
    """Print standardized experiment header."""
    safe_log(logger, 'info', "=" * 80)
    safe_log(logger, 'info', f"[EXPERIMENT] {experiment_name}")
    safe_log(logger, 'info', "=" * 80)
    
    # Print key configuration parameters
    if 'data' in config:
        safe_log(logger, 'info', f"[DATA] {config['data'].get('processed_dir', 'Not specified')}")
    
    if 'models' in config:
        safe_log(logger, 'info', f"[MODELS] {config['models'].get('save_dir', 'Not specified')}")
    
    if 'output_dir' in config:
        safe_log(logger, 'info', f"[OUTPUT] {config['output_dir']}")
    
    safe_log(logger, 'info', "-" * 80)

def print_experiment_footer(experiment_name: str, success: bool, logger: logging.Logger):
    """Print standardized experiment footer."""
    safe_log(logger, 'info', "-" * 80)
    if success:
        safe_log(logger, 'info', f"[SUCCESS] EXPERIMENT COMPLETED: {experiment_name}")
    else:
        safe_log(logger, 'info', f"[FAILED] EXPERIMENT FAILED: {experiment_name}")
    safe_log(logger, 'info', "=" * 80)
