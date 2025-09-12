"""
Phase 4: Enhanced Adversarial Attack System
Built upon friend's proven attack implementations with comprehensive evaluation framework.
"""

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import joblib
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse

# Import torchattacks and friend's components
import torchattacks
from friend_baseline import BaselineIDS, PFCPDataset
from enhanced_baseline import ConfigManager, EnhancedBaselineIDS


class ModelWrapper(nn.Module):
    """
    Friend's proven model wrapper for torchattacks compatibility
    Converts single-output binary model to multi-class format
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Get single logit output [batch_size, 1]
        output = self.model(x)
        # Transform to 2-class format [batch_size, 2]: [-logit, logit]
        return torch.cat([-output, output], dim=1)


class AdversarialAttackEvaluator:
    """Enhanced adversarial attack evaluation system"""
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
        # Results storage
        self.attack_results = {}
        self.robustness_metrics = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config_manager.get('logging', {})
        log_dir = Path(log_config.get('log_dir'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s [%(levelname)s] - %(message)s'),
            handlers=[
                logging.FileHandler(log_dir / 'adversarial_evaluation.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_model(self, model_path: str) -> nn.Module:
        """Load model for attack evaluation"""
        try:
            # Load test data to determine input features
            test_data_path = self.config_manager.get('paths.data.test_data')
            feature_cols_path = self.config_manager.get('paths.data.feature_columns')
            
            with open(feature_cols_path, 'r') as f:
                feature_columns = json.load(f)
            input_features = len(feature_columns)
            
            # Create model instance
            model = EnhancedBaselineIDS(input_features, {})
            
            # Load weights
            if 'friend' in model_path:
                # Friend's model is just state_dict
                model.load_state_dict(torch.load(model_path, weights_only=False))
            else:
                # Our enhanced model has full checkpoint
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.to(self.device)
            model.eval()
            
            logging.info(f"Model loaded from {model_path}")
            logging.info(f"Model architecture: {input_features} -> [256, 128, 64] -> 1")
            
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def load_test_data(self) -> torch.utils.data.DataLoader:
        """Load test dataset for attack evaluation"""
        try:
            test_data_path = self.config_manager.get('paths.data.test_data')
            feature_cols_path = self.config_manager.get('paths.data.feature_columns')
            
            # Use friend's proven dataset class
            test_dataset = PFCPDataset(test_data_path, feature_cols_path)
            
            batch_size = self.config_manager.get('attack_config.evaluation.batch_size', 256)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            
            logging.info(f"Test data loaded: {len(test_dataset)} samples")
            return test_loader
            
        except Exception as e:
            logging.error(f"Failed to load test data: {str(e)}")
            raise
    
    def create_attack(self, attack_type: str, model_wrapper: nn.Module, **kwargs):
        """Create torchattacks instance based on type and parameters"""
        try:
            if attack_type == 'fgsm':
                epsilon = kwargs.get('epsilon', 0.01)
                return torchattacks.FGSM(model_wrapper, eps=epsilon)
                
            elif attack_type.startswith('pgd'):
                epsilon = kwargs.get('epsilon', 0.01)
                alpha = kwargs.get('alpha', 0.00784)  # 2/255
                steps = kwargs.get('steps', 10)
                return torchattacks.PGD(model_wrapper, eps=epsilon, alpha=alpha, steps=steps)
                
            elif attack_type == 'c_w':
                c = kwargs.get('c', 1.0)
                kappa = kwargs.get('kappa', 0)
                steps = kwargs.get('max_iterations', 1000)
                lr = kwargs.get('learning_rate', 0.01)
                return torchattacks.CW(model_wrapper, c=c, kappa=kappa, steps=steps, lr=lr)
                
            else:
                raise ValueError(f"Unsupported attack type: {attack_type}")
                
        except Exception as e:
            logging.error(f"Failed to create {attack_type} attack: {str(e)}")
            raise
    
    def evaluate_clean_performance(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model performance on clean (non-adversarial) data"""
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Clean Evaluation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).squeeze().round()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0)
        }
        
        logging.info(f"Clean Performance - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def evaluate_attack_robustness(self, model: nn.Module, test_loader: torch.utils.data.DataLoader,
                                 attack_type: str, **attack_params) -> Dict:
        """Evaluate model robustness against specific attack"""
        try:
            # Wrap model for torchattacks
            model_wrapper = ModelWrapper(model)
            
            # Create attack
            attack = self.create_attack(attack_type, model_wrapper, **attack_params)
            
            all_clean_preds, all_adv_preds, all_labels = [], [], []
            all_perturbations = []
            
            model.eval()
            
            for inputs, labels in tqdm(test_loader, desc=f"{attack_type.upper()} Attack"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Generate adversarial examples
                adv_inputs = attack(inputs, labels.long())  # torchattacks expects long labels
                
                # Calculate perturbation magnitude
                perturbation = torch.norm((adv_inputs - inputs).view(inputs.size(0), -1), p=2, dim=1)
                all_perturbations.extend(perturbation.cpu().numpy())
                
                with torch.no_grad():
                    # Clean predictions
                    clean_outputs = model(inputs)
                    clean_preds = torch.sigmoid(clean_outputs).squeeze().round()
                    
                    # Adversarial predictions
                    adv_outputs = model(adv_inputs)
                    adv_preds = torch.sigmoid(adv_outputs).squeeze().round()
                    
                    all_clean_preds.extend(clean_preds.cpu().numpy())
                    all_adv_preds.extend(adv_preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            clean_metrics = {
                'accuracy': accuracy_score(all_labels, all_clean_preds),
                'f1': f1_score(all_labels, all_clean_preds, zero_division=0),
                'precision': precision_score(all_labels, all_clean_preds, zero_division=0),
                'recall': recall_score(all_labels, all_clean_preds, zero_division=0)
            }
            
            robust_metrics = {
                'accuracy': accuracy_score(all_labels, all_adv_preds),
                'f1': f1_score(all_labels, all_adv_preds, zero_division=0),
                'precision': precision_score(all_labels, all_adv_preds, zero_division=0),
                'recall': recall_score(all_labels, all_adv_preds, zero_division=0)
            }
            
            # Attack success rate (percentage of examples where prediction changed)
            attack_success = np.mean(np.array(all_clean_preds) != np.array(all_adv_preds)) * 100
            
            # Average perturbation magnitude
            avg_perturbation = np.mean(all_perturbations)
            
            results = {
                'attack_type': attack_type,
                'attack_params': attack_params,
                'clean_performance': clean_metrics,
                'robust_performance': robust_metrics,
                'attack_success_rate': attack_success,
                'avg_perturbation_magnitude': avg_perturbation,
                'robustness_drop': {
                    'accuracy': clean_metrics['accuracy'] - robust_metrics['accuracy'],
                    'f1': clean_metrics['f1'] - robust_metrics['f1'],
                    'precision': clean_metrics['precision'] - robust_metrics['precision'],
                    'recall': clean_metrics['recall'] - robust_metrics['recall']
                }
            }
            
            logging.info(
                f"{attack_type.upper()} Results - "
                f"Attack Success: {attack_success:.2f}%, "
                f"Robust F1: {robust_metrics['f1']:.4f}, "
                f"F1 Drop: {results['robustness_drop']['f1']:.4f}"
            )
            
            return results
            
        except Exception as e:
            logging.error(f"Attack evaluation failed for {attack_type}: {str(e)}")
            raise
    
    def comprehensive_attack_evaluation(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Run comprehensive attack evaluation across multiple attack types and parameters"""
        logging.info("Starting comprehensive adversarial attack evaluation...")
        
        results = {
            'model_info': {
                'architecture': 'BaselineIDS: 80 -> [256, 128, 64] -> 1',
                'device': str(self.device)
            },
            'clean_performance': self.evaluate_clean_performance(model, test_loader),
            'attack_results': {}
        }
        
        # FGSM attacks with multiple epsilons
        fgsm_config = self.config_manager.get('attack_config.fgsm', {})
        epsilons = fgsm_config.get('epsilons', [0.01, 0.05, 0.1])
        
        logging.info(f"Evaluating FGSM attacks with epsilons: {epsilons}")
        for epsilon in epsilons:
            attack_key = f"fgsm_eps_{epsilon}"
            results['attack_results'][attack_key] = self.evaluate_attack_robustness(
                model, test_loader, 'fgsm', epsilon=epsilon
            )
        
        # PGD attacks with multiple configurations
        pgd_config = self.config_manager.get('attack_config.pgd', {})
        pgd_epsilons = pgd_config.get('epsilons', [0.01, 0.05])
        pgd_steps = pgd_config.get('steps', [10, 20])
        alpha = pgd_config.get('alpha', 0.00784)
        
        logging.info(f"Evaluating PGD attacks - epsilons: {pgd_epsilons}, steps: {pgd_steps}")
        for epsilon in pgd_epsilons:
            for steps in pgd_steps:
                attack_key = f"pgd_eps_{epsilon}_steps_{steps}"
                results['attack_results'][attack_key] = self.evaluate_attack_robustness(
                    model, test_loader, 'pgd', epsilon=epsilon, alpha=alpha, steps=steps
                )
        
        logging.info("Comprehensive attack evaluation completed!")
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save attack evaluation results"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            logging.info(f"Attack evaluation results saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise


def main():
    """Main adversarial attack evaluation pipeline"""
    parser = argparse.ArgumentParser(description="Enhanced Adversarial Attack Evaluation")
    parser.add_argument('--config', type=str,
                       default='/Users/sahajupadhyay/Desktop/Capstone/ADVERSARIAL_IDS_DEEP_LEARNING/configs/adversarial_config.yaml',
                       help='Path to adversarial configuration file')
    parser.add_argument('--model', type=str,
                       help='Path to model file (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output path for results (overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = AdversarialAttackEvaluator(args.config)
        
        # Load model
        model_path = args.model or evaluator.config_manager.get('paths.models.target_model')
        model = evaluator.load_model(model_path)
        
        # Load test data
        test_loader = evaluator.load_test_data()
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_attack_evaluation(model, test_loader)
        
        # Save results
        output_path = args.output or Path(evaluator.config_manager.get('paths.results.attack_results')) / 'comprehensive_attack_results.json'
        evaluator.save_results(results, output_path)
        
        # Print summary
        clean_f1 = results['clean_performance']['f1']
        print(f"\n🎯 ADVERSARIAL ATTACK EVALUATION COMPLETE!")
        print(f"📊 Clean Performance: F1 = {clean_f1:.4f}")
        print(f"🔍 Attack Results:")
        
        for attack_name, attack_result in results['attack_results'].items():
            robust_f1 = attack_result['robust_performance']['f1']
            success_rate = attack_result['attack_success_rate']
            f1_drop = attack_result['robustness_drop']['f1']
            
            print(f"   {attack_name:20s}: F1 = {robust_f1:.4f} (-{f1_drop:.4f}), Success Rate = {success_rate:.1f}%")
        
        logging.info("Adversarial attack evaluation pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Adversarial attack evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()