#!/usr/bin/env python3
"""
Standalone Adversarial Robustness Evaluation
Evaluates the trained adversarial model's robustness against various attacks
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import sys
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from capstone_presentation.src.baseline import BaselineIDS, PFCPDataset  
import torchattacks
from torch.utils.data import DataLoader

class SimpleConfig:
    """Simple configuration class for loading YAML configs."""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

class AdversarialRobustnessEvaluator:
    """Comprehensive adversarial robustness evaluation."""
    
    def __init__(self, config_path: str, model_path: str):
        self.config = SimpleConfig(config_path)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Load test dataset
        self.load_test_dataset()
        
        # Load model
        self.model = self.load_model()
        
        # Setup attacks
        self.setup_attacks()
        
    def setup_logging(self):
        """Configure logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"robustness_evaluation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Robustness evaluation initialized. Logs: {log_file}")
        
    def load_test_dataset(self):
        """Load test dataset."""
        test_dataset = PFCPDataset(
            self.config.get('data.test_file'),
            self.config.get('data.feature_columns_file')
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=512, 
            shuffle=False,
            num_workers=2
        )
        
        self.num_features = test_dataset.X.shape[1]
        self.logger.info(f"Loaded test dataset: {len(test_dataset)} samples, {self.num_features} features")
        
    def load_model(self):
        """Load the trained adversarial model."""
        model = BaselineIDS(input_features=self.num_features).to(self.device)
        
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Handle different save formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Our adversarial training format
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded adversarial model from {self.model_path}")
                
                # Log training metrics if available
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    self.logger.info(f"Model training metrics - F1: {metrics.get('f1', 'N/A'):.4f}")
            else:
                # Friend's model format (direct state dict)
                model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded model (direct state dict) from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        return model
        
    def setup_attacks(self):
        """Initialize adversarial attacks."""
        # Wrapper for torchattacks compatibility
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                return torch.stack([1-probs.squeeze(), probs.squeeze()], dim=1)
        
        wrapped_model = ModelWrapper(self.model)
        
        # Multiple epsilon values for comprehensive evaluation
        epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        self.attacks = {}
        
        # FGSM attacks with different epsilons
        for eps in epsilons:
            self.attacks[f'fgsm_eps_{eps}'] = torchattacks.FGSM(wrapped_model, eps=eps)
            
        # PGD attacks with different epsilons
        for eps in epsilons:
            self.attacks[f'pgd_eps_{eps}'] = torchattacks.PGD(
                wrapped_model,
                eps=eps,
                alpha=eps/4,  # alpha = eps/4 is a common choice
                steps=10
            )
            
        # C&W attack (if available)
        try:
            self.attacks['cw'] = torchattacks.CW(wrapped_model, c=1, lr=0.01, steps=100)
        except:
            self.logger.warning("C&W attack not available")
            
        self.logger.info(f"Initialized {len(self.attacks)} attack variants")
        
    def evaluate_clean_performance(self):
        """Evaluate clean (non-adversarial) performance."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Clean Evaluation"):
                x, y = x.to(self.device), y.to(self.device).float()
                
                outputs = self.model(x)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
        
        predictions_np = np.array(all_predictions)
        labels_np = np.array(all_labels)
        
        metrics = {
            'f1': f1_score(labels_np, predictions_np),
            'precision': precision_score(labels_np, predictions_np, zero_division=0),
            'recall': recall_score(labels_np, predictions_np, zero_division=0),
            'accuracy': accuracy_score(labels_np, predictions_np)
        }
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, predictions_np)
        metrics['confusion_matrix'] = cm.tolist()
        
        self.logger.info(f"Clean Performance - F1: {metrics['f1']:.4f}, "
                        f"Precision: {metrics['precision']:.4f}, "
                        f"Recall: {metrics['recall']:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
        
    def evaluate_attack_robustness(self, attack_name, attack):
        """Evaluate robustness against a specific attack."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        successful_attacks = 0
        total_samples = 0
        
        for x, y in tqdm(self.test_loader, desc=f"{attack_name} Attack"):
            x, y = x.to(self.device), y.to(self.device).float()
            
            # Store original predictions
            with torch.no_grad():
                orig_outputs = self.model(x)
                orig_predictions = (torch.sigmoid(orig_outputs) > 0.5).float()
            
            # Enable gradients for adversarial attack generation
            x.requires_grad_(True)
            y_attack = y.long()
            
            try:
                # Generate adversarial examples
                with torch.enable_grad():
                    x_adv = attack(x, y_attack)
                
                # Get predictions on adversarial examples
                with torch.no_grad():
                    adv_outputs = self.model(x_adv.detach())
                    adv_predictions = (torch.sigmoid(adv_outputs) > 0.5).float()
                    
                    # Count successful attacks (prediction changed)
                    successful_attacks += torch.sum(orig_predictions != adv_predictions).item()
                    total_samples += x.size(0)
                    
                    all_predictions.extend(adv_predictions.squeeze().cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    
            except Exception as e:
                self.logger.warning(f"Attack {attack_name} failed on batch: {str(e)}")
                # Use original predictions if attack fails
                all_predictions.extend(orig_predictions.squeeze().cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total_samples += x.size(0)
                
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        predictions_np = np.array(all_predictions)
        labels_np = np.array(all_labels)
        
        attack_success_rate = successful_attacks / total_samples if total_samples > 0 else 0
        
        metrics = {
            'f1': f1_score(labels_np, predictions_np, zero_division=0),
            'precision': precision_score(labels_np, predictions_np, zero_division=0),
            'recall': recall_score(labels_np, predictions_np, zero_division=0),
            'accuracy': accuracy_score(labels_np, predictions_np),
            'attack_success_rate': attack_success_rate
        }
        
        self.logger.info(f"{attack_name} - F1: {metrics['f1']:.4f}, "
                        f"Success Rate: {attack_success_rate:.4f}")
        
        return metrics
        
    def run_comprehensive_evaluation(self):
        """Run comprehensive robustness evaluation."""
        self.logger.info("Starting comprehensive adversarial robustness evaluation...")
        
        results = {}
        
        # Clean performance
        self.logger.info("Evaluating clean performance...")
        results['clean'] = self.evaluate_clean_performance()
        
        # Adversarial robustness
        self.logger.info("Evaluating adversarial robustness...")
        results['adversarial'] = {}
        
        for attack_name, attack in self.attacks.items():
            self.logger.info(f"Testing {attack_name}...")
            results['adversarial'][attack_name] = self.evaluate_attack_robustness(attack_name, attack)
            
        # Summary statistics
        self.generate_summary(results)
        
        # Save results
        self.save_results(results)
        
        return results
        
    def generate_summary(self, results):
        """Generate and log summary statistics."""
        self.logger.info("\n" + "="*50)
        self.logger.info("ADVERSARIAL ROBUSTNESS EVALUATION SUMMARY")
        self.logger.info("="*50)
        
        # Clean performance
        clean_f1 = results['clean']['f1']
        self.logger.info(f"Clean Performance F1: {clean_f1:.4f}")
        
        # Attack performance summary
        self.logger.info("\nAdversarial Attack Results:")
        self.logger.info("-" * 30)
        
        # Group by attack type
        fgsm_results = []
        pgd_results = []
        other_results = []
        
        for attack_name, metrics in results['adversarial'].items():
            f1_drop = clean_f1 - metrics['f1']
            success_rate = metrics['attack_success_rate']
            
            if 'fgsm' in attack_name:
                epsilon = attack_name.split('_')[-1]
                fgsm_results.append((epsilon, metrics['f1'], f1_drop, success_rate))
            elif 'pgd' in attack_name:
                epsilon = attack_name.split('_')[-1]
                pgd_results.append((epsilon, metrics['f1'], f1_drop, success_rate))
            else:
                other_results.append((attack_name, metrics['f1'], f1_drop, success_rate))
        
        # FGSM summary
        if fgsm_results:
            self.logger.info("\nFGSM Attack Results:")
            for eps, f1, drop, success in sorted(fgsm_results, key=lambda x: float(x[0])):
                self.logger.info(f"  ε={eps}: F1={f1:.4f} (↓{drop:.4f}), Success={success:.4f}")
        
        # PGD summary
        if pgd_results:
            self.logger.info("\nPGD Attack Results:")
            for eps, f1, drop, success in sorted(pgd_results, key=lambda x: float(x[0])):
                self.logger.info(f"  ε={eps}: F1={f1:.4f} (↓{drop:.4f}), Success={success:.4f}")
        
        # Other attacks
        if other_results:
            self.logger.info("\nOther Attack Results:")
            for name, f1, drop, success in other_results:
                self.logger.info(f"  {name}: F1={f1:.4f} (↓{drop:.4f}), Success={success:.4f}")
        
        # Overall robustness metrics
        all_f1_drops = [clean_f1 - metrics['f1'] for metrics in results['adversarial'].values()]
        avg_f1_drop = np.mean(all_f1_drops)
        max_f1_drop = np.max(all_f1_drops)
        
        self.logger.info(f"\nOverall Robustness:")
        self.logger.info(f"  Average F1 Drop: {avg_f1_drop:.4f}")
        self.logger.info(f"  Maximum F1 Drop: {max_f1_drop:.4f}")
        self.logger.info(f"  Robustness Score: {1 - avg_f1_drop:.4f}")
        
    def save_results(self, results):
        """Save evaluation results."""
        results_dir = Path('results/robustness_evaluation')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"adversarial_robustness_{timestamp}.json"
        
        # Add metadata
        results['metadata'] = {
            'model_path': str(self.model_path),
            'evaluation_time': timestamp,
            'device': str(self.device),
            'num_attacks': len(self.attacks)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained adversarial model')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = AdversarialRobustnessEvaluator(args.config, args.model)
    results = evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()