"""
CLI Configuration management for the Adversarial 5G IDS system.

Handles configuration loading, validation, and default settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class CLIConfig:
    """Manages CLI configuration and settings."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings."""
        return {
            # File paths
            'paths': {
                'data_dir': str(self.project_root / 'data'),
                'models_dir': str(self.project_root / 'models'),
                'reports_dir': str(self.project_root / 'reports'),
                'src_dir': str(self.project_root / 'src'),
                
                # Model files
                'baseline_model': str(self.project_root / 'models' / 'baseline_random_forest.joblib'),
                'robust_model': str(self.project_root / 'models' / 'simple_robust_rf.joblib'),
                'transformers': str(self.project_root / 'models' / 'feature_transformers.joblib'),
                'scaler': str(self.project_root / 'models' / 'feature_scaler.joblib'),
                
                # Data files
                'test_data': str(self.project_root / 'data' / 'processed' / 'X_test.npy'),
                'test_labels': str(self.project_root / 'data' / 'processed' / 'y_test.npy'),
                'sample_data': str(self.project_root / 'data' / 'sample_traffic.csv'),
            },
            
            # Model settings
            'models': {
                'baseline': {
                    'name': 'Random Forest Baseline',
                    'accuracy': 0.608,
                    'features': 7,
                    'type': 'baseline'
                },
                'robust': {
                    'name': 'Adversarially Trained Random Forest',
                    'accuracy': 0.665,
                    'features': 43,
                    'type': 'robust'
                }
            },
            
            # Attack settings
            'attacks': {
                'fgsm': {
                    'name': 'Fast Gradient Sign Method',
                    'default_epsilon': 0.1,
                    'max_epsilon': 1.0
                },
                'pgd': {
                    'name': 'Projected Gradient Descent',
                    'default_epsilon': 0.3,
                    'max_epsilon': 1.0,
                    'steps': 10,
                    'step_size': 0.01
                },
                'enhanced_pgd': {
                    'name': 'Enhanced Constraint-Aware PGD',
                    'default_epsilon': 0.3,
                    'max_epsilon': 1.0,
                    'steps': 20,
                    'step_size': 0.01
                }
            },
            
            # Defense settings
            'defenses': {
                'adversarial_training': {
                    'noise_levels': [0.1, 0.2, 0.3],
                    'adversarial_ratio': 0.4,
                    'progressive': True
                },
                'evaluation': {
                    'noise_levels': [0.05, 0.1, 0.2, 0.3, 0.5],
                    'sample_size': 100
                }
            },
            
            # Output settings
            'output': {
                'colored': True,
                'verbose': False,
                'report_format': 'json',
                'max_table_width': 80
            },
            
            # PFCP protocol constraints
            'constraints': {
                'feature_bounds': [-3.0, 3.0],
                'categorical_features': [],
                'protected_features': []
            },
            
            # Class labels
            'classes': {
                0: 'Mal_Del',
                1: 'Mal_Estab', 
                2: 'Mal_Mod',
                3: 'Mal_Mod2',
                4: 'Normal'
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with default config
            self._deep_merge(self.config, user_config)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that required files and directories exist."""
        validation = {}
        paths = self.get('paths', {})
        
        for name, path in paths.items():
            if name.endswith('_dir'):
                validation[name] = os.path.isdir(path)
            else:
                validation[name] = os.path.isfile(path)
        
        return validation
    
    def get_model_path(self, model_type: str) -> str:
        """Get path to model file."""
        if model_type == 'baseline':
            return self.get('paths.baseline_model')
        elif model_type == 'robust':
            return self.get('paths.robust_model')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get model information."""
        return self.get(f'models.{model_type}', {})
    
    def get_attack_info(self, attack_type: str) -> Dict[str, Any]:
        """Get attack configuration."""
        return self.get(f'attacks.{attack_type}', {})
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        return self.get(f'classes.{class_id}', f'Unknown_{class_id}')
    
    def get_all_classes(self) -> Dict[int, str]:
        """Get all class mappings."""
        return self.get('classes', {})
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def create_sample_config(self, output_path: str):
        """Create a sample configuration file."""
        sample_config = {
            "output": {
                "colored": True,
                "verbose": False,
                "report_format": "json"
            },
            "attacks": {
                "pgd": {
                    "default_epsilon": 0.2,
                    "steps": 15
                }
            },
            "defenses": {
                "evaluation": {
                    "sample_size": 200
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'path_validation': self.validate_paths()
        }
        
        # Check required paths
        critical_paths = ['baseline_model', 'robust_model', 'test_data', 'test_labels']
        for path_name in critical_paths:
            if not validation_results['path_validation'].get(path_name, False):
                validation_results['errors'].append(f"Missing required file: {path_name}")
                validation_results['valid'] = False
        
        # Check model configurations
        for model_type in ['baseline', 'robust']:
            model_info = self.get_model_info(model_type)
            if not model_info:
                validation_results['warnings'].append(f"No configuration found for {model_type} model")
        
        # Check attack configurations
        for attack_type in ['fgsm', 'pgd', 'enhanced_pgd']:
            attack_info = self.get_attack_info(attack_type)
            if not attack_info:
                validation_results['warnings'].append(f"No configuration found for {attack_type} attack")
        
        return validation_results
