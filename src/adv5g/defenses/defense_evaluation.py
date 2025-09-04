"""
Defense Evaluation Framework for 5G PFCP IDS
Phase 2B: Defense Development

This module provides comprehensive evaluation of defense mechanisms against
various adversarial attacks with detailed metrics and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import joblib
import os
import sys
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main')
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/src')

from src.attacks.enhanced_attacks import EnhancedAdversarialAttacks
from src.attacks.pfcp_constraints import PFCPConstraints


class DefenseEvaluator:
    """
    Comprehensive evaluation framework for adversarial defenses
    
    This class provides systematic testing of defense mechanisms against
    various attacks with detailed performance analysis.
    """
    
    def __init__(self):
        """Initialize the defense evaluator"""
        
        self.class_names = [
            'Normal',
            'Mal_Estab', 
            'Mal_Del',
            'Mal_Mod',
            'Mal_Mod2'
        ]
        
        self.feature_names = [
            'Flow Duration',
            'Total Fwd Packets',
            'Total Backward Packets', 
            'Total Length of Fwd Packets',
            'Total Length of Bwd Packets',
            'Fwd Packet Length Mean',
            'Flow Bytes/s'
        ]
        
        # Initialize attack components
        self.constraints = PFCPConstraints()
        
        # Results storage
        self.evaluation_results = {}
        
    def load_models(self, model_configs):
        """
        Load multiple models for comparison
        
        Args:
            model_configs: List of dicts with 'name', 'model_path', 'scaler_path'
        """
        self.models = {}
        
        for config in model_configs:
            try:
                model = joblib.load(config['model_path'])
                scaler = joblib.load(config['scaler_path']) if config.get('scaler_path') else None
                
                self.models[config['name']] = {
                    'model': model,
                    'scaler': scaler,
                    'config': config
                }
                print(f"Loaded model: {config['name']}")
                
            except Exception as e:
                print(f"Error loading {config['name']}: {e}")
    
    def load_data(self):
        """Load test data for evaluation"""
        data_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/data/processed'
        
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        print(f"Loaded test data: {X_test.shape} samples, {len(np.unique(y_test))} classes")
        # Data is already preprocessed and scaled, so return as-is
        return X_test, y_test
    
    def evaluate_clean_performance(self, X_test, y_test):
        """
        Evaluate all models on clean (non-adversarial) data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            clean_results: Performance on clean data
        """
        print("Evaluating clean performance...")
        
        clean_results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Apply scaling if available and data dimensions match
            if scaler and hasattr(scaler, 'n_features_in_'):
                if X_test.shape[1] == scaler.n_features_in_:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    print(f"  Warning: Scaler expects {scaler.n_features_in_} features, data has {X_test.shape[1]}. Using data as-is.")
                    X_test_scaled = X_test
            else:
                X_test_scaled = X_test
            
            # Get predictions
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro')
            
            # Per-class metrics
            class_report = classification_report(y_test, predictions, output_dict=True)
            confusion_mat = confusion_matrix(y_test, predictions)
            
            clean_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': confusion_mat,
                'predictions': predictions
            }
            
            print(f"{model_name} - Clean Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return clean_results
    
    def evaluate_adversarial_robustness(self, X_test, y_test, attack_configs):
        """
        Evaluate model robustness against various adversarial attacks
        
        Args:
            X_test: Test features
            y_test: Test labels  
            attack_configs: List of attack configurations
            
        Returns:
            robustness_results: Robustness evaluation results
        """
        print("Evaluating adversarial robustness...")
        
        robustness_results = {}
        
        for model_name, model_info in self.models.items():
            print(f"\nTesting {model_name}...")
            
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Apply scaling
            X_test_scaled = scaler.transform(X_test) if scaler else X_test
            
            # Initialize attack engine for this model
            attack_engine = EnhancedAdversarialAttacks(
                model=model,
                constraints=self.constraints
            )
            
            model_results = {}
            
            for attack_config in attack_configs:
                attack_name = attack_config['name']
                print(f"  Testing {attack_name}...")
                
                try:
                    # Generate adversarial examples
                    if attack_config['method'] == 'enhanced_pgd':
                        X_adv = attack_engine.enhanced_pgd_attack(
                            X=X_test_scaled,
                            y=y_test,
                            epsilon=attack_config['epsilon'],
                            alpha=attack_config.get('alpha', attack_config['epsilon']/4),
                            num_iter=attack_config.get('num_iter', 10),
                            momentum=attack_config.get('momentum', 0.9)
                        )
                    elif attack_config['method'] == 'enhanced_fgsm':
                        X_adv = attack_engine.enhanced_fgsm_attack(
                            X=X_test_scaled,
                            y=y_test,
                            epsilon=attack_config['epsilon']
                        )
                    elif attack_config['method'] in ['fgsm', 'constraint_fgsm']:
                        X_adv = attack_engine.enhanced_fgsm_attack(
                            X=X_test_scaled,
                            y=y_test,
                            epsilon=attack_config['epsilon']
                        )
                    elif attack_config['method'] in ['pgd', 'enhanced_pgd']:
                        X_adv = attack_engine.enhanced_pgd_attack(
                            X=X_test_scaled,
                            y=y_test,
                            epsilon=attack_config['epsilon'],
                            alpha=attack_config.get('alpha', attack_config['epsilon']/4),
                            num_iter=attack_config.get('num_iter', 10),
                            momentum=attack_config.get('momentum', 0.9)
                        )
                    else:
                        print(f"    Unknown attack method: {attack_config['method']}")
                        continue
                    
                    # Evaluate on adversarial examples
                    adv_predictions = model.predict(X_adv)
                    
                    # Calculate robustness metrics
                    robust_accuracy = accuracy_score(y_test, adv_predictions)
                    
                    # Calculate evasion rate (how often attacks succeed)
                    clean_predictions = model.predict(X_test_scaled)
                    clean_correct = (clean_predictions == y_test)
                    adv_incorrect = (adv_predictions != y_test)
                    successful_attacks = clean_correct & adv_incorrect
                    evasion_rate = np.sum(successful_attacks) / np.sum(clean_correct)
                    
                    # Per-class analysis
                    class_robustness = self._analyze_class_robustness(
                        y_test, clean_predictions, adv_predictions
                    )
                    
                    # Perturbation analysis
                    perturbation_stats = self._analyze_perturbations(X_test_scaled, X_adv)
                    
                    model_results[attack_name] = {
                        'robust_accuracy': robust_accuracy,
                        'evasion_rate': evasion_rate,
                        'successful_attacks': np.sum(successful_attacks),
                        'total_clean_correct': np.sum(clean_correct),
                        'class_robustness': class_robustness,
                        'perturbation_stats': perturbation_stats,
                        'attack_config': attack_config
                    }
                    
                    print(f"    Robust Accuracy: {robust_accuracy:.3f}, Evasion Rate: {evasion_rate:.3f}")
                    
                except Exception as e:
                    print(f"    Error in {attack_name}: {e}")
                    model_results[attack_name] = {'error': str(e)}
            
            robustness_results[model_name] = model_results
        
        return robustness_results
    
    def _analyze_class_robustness(self, y_true, clean_pred, adv_pred):
        """Analyze robustness per class"""
        
        class_robustness = {}
        
        for class_idx in range(len(self.class_names)):
            class_mask = (y_true == class_idx)
            
            if np.sum(class_mask) == 0:
                continue
                
            clean_correct = (clean_pred[class_mask] == y_true[class_mask])
            adv_correct = (adv_pred[class_mask] == y_true[class_mask])
            
            clean_accuracy = np.mean(clean_correct)
            robust_accuracy = np.mean(adv_correct)
            
            # Evasion rate for this class
            if np.sum(clean_correct) > 0:
                successful_attacks = clean_correct & ~adv_correct
                evasion_rate = np.sum(successful_attacks) / np.sum(clean_correct)
            else:
                evasion_rate = 0.0
            
            class_robustness[self.class_names[class_idx]] = {
                'clean_accuracy': clean_accuracy,
                'robust_accuracy': robust_accuracy,
                'evasion_rate': evasion_rate,
                'sample_count': np.sum(class_mask)
            }
        
        return class_robustness
    
    def _analyze_perturbations(self, X_clean, X_adv):
        """Analyze perturbation statistics"""
        
        perturbations = X_adv - X_clean
        
        stats = {
            'l_inf_norm': {
                'mean': np.mean(np.max(np.abs(perturbations), axis=1)),
                'max': np.max(np.abs(perturbations)),
                'std': np.std(np.max(np.abs(perturbations), axis=1))
            },
            'l2_norm': {
                'mean': np.mean(np.linalg.norm(perturbations, axis=1)),
                'max': np.max(np.linalg.norm(perturbations, axis=1)),
                'std': np.std(np.linalg.norm(perturbations, axis=1))
            },
            'feature_perturbations': {
                feature: {
                    'mean': np.mean(np.abs(perturbations[:, i])),
                    'max': np.max(np.abs(perturbations[:, i])),
                    'std': np.std(perturbations[:, i])
                }
                for i, feature in enumerate(self.feature_names)
            }
        }
        
        return stats
    
    def compare_defenses(self, clean_results, robustness_results):
        """
        Compare different defense mechanisms
        
        Args:
            clean_results: Clean performance results
            robustness_results: Adversarial robustness results
            
        Returns:
            comparison: Defense comparison analysis
        """
        print("Comparing defense mechanisms...")
        
        comparison = {
            'summary': {},
            'detailed': {},
            'rankings': {}
        }
        
        # Extract key metrics for each model
        for model_name in self.models.keys():
            clean_acc = clean_results[model_name]['accuracy']
            clean_f1 = clean_results[model_name]['f1_score']
            
            # Average robustness across attacks
            robust_accuracies = []
            evasion_rates = []
            
            for attack_name, attack_results in robustness_results[model_name].items():
                if 'error' not in attack_results:
                    robust_accuracies.append(attack_results['robust_accuracy'])
                    evasion_rates.append(attack_results['evasion_rate'])
            
            avg_robust_acc = np.mean(robust_accuracies) if robust_accuracies else 0
            avg_evasion_rate = np.mean(evasion_rates) if evasion_rates else 1
            
            # Calculate robustness improvement (lower evasion = better)
            robustness_score = (1 - avg_evasion_rate) * 100  # Convert to percentage
            
            comparison['summary'][model_name] = {
                'clean_accuracy': clean_acc,
                'clean_f1': clean_f1,
                'avg_robust_accuracy': avg_robust_acc,
                'avg_evasion_rate': avg_evasion_rate,
                'robustness_score': robustness_score,
                'robustness_improvement': max(0, robustness_score - 43)  # vs baseline ~57% evasion
            }
        
        # Rank models by different criteria
        models_by_clean_acc = sorted(
            comparison['summary'].items(),
            key=lambda x: x[1]['clean_accuracy'],
            reverse=True
        )
        
        models_by_robustness = sorted(
            comparison['summary'].items(), 
            key=lambda x: x[1]['robustness_score'],
            reverse=True
        )
        
        comparison['rankings'] = {
            'by_clean_accuracy': [name for name, _ in models_by_clean_acc],
            'by_robustness': [name for name, _ in models_by_robustness]
        }
        
        return comparison
    
    def generate_evaluation_report(self, clean_results, robustness_results, comparison):
        """Generate comprehensive evaluation report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'models_evaluated': list(self.models.keys()),
                'attacks_tested': self._extract_attack_names(robustness_results),
                'classes': self.class_names,
                'features': self.feature_names
            },
            'clean_performance': clean_results,
            'adversarial_robustness': robustness_results,
            'defense_comparison': comparison,
            'recommendations': self._generate_recommendations(comparison)
        }
        
        return report
    
    def _extract_attack_names(self, robustness_results):
        """Extract unique attack names from results"""
        attack_names = set()
        for model_results in robustness_results.values():
            attack_names.update(model_results.keys())
        return list(attack_names)
    
    def _generate_recommendations(self, comparison):
        """Generate recommendations based on evaluation results"""
        
        recommendations = []
        
        # Find best performing models
        best_clean = comparison['rankings']['by_clean_accuracy'][0]
        best_robust = comparison['rankings']['by_robustness'][0]
        
        if best_clean == best_robust:
            recommendations.append(f"✅ {best_clean} provides the best balance of clean accuracy and robustness.")
        else:
            recommendations.append(f"🎯 {best_clean} has highest clean accuracy, {best_robust} is most robust.")
        
        # Check if target robustness achieved
        best_evasion = min(comparison['summary'][model]['avg_evasion_rate'] 
                          for model in comparison['summary'])
        
        if best_evasion < 0.30:  # 30% target
            recommendations.append(f"🎉 Target robustness achieved! Best evasion rate: {best_evasion:.1%}")
        else:
            recommendations.append(f"⚠️ Target not met. Best evasion rate: {best_evasion:.1%} (target: <30%)")
        
        # Specific improvement suggestions
        for model_name, metrics in comparison['summary'].items():
            if metrics['avg_evasion_rate'] > 0.50:
                recommendations.append(f"🔧 {model_name} needs improvement (evasion: {metrics['avg_evasion_rate']:.1%})")
        
        return recommendations
    
    def visualize_evaluation_results(self, comparison, save_dir='reports'):
        """Create visualizations for evaluation results"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Clean vs Robust Accuracy
        models = list(comparison['summary'].keys())
        clean_acc = [comparison['summary'][m]['clean_accuracy'] for m in models]
        robust_acc = [comparison['summary'][m]['avg_robust_accuracy'] for m in models]
        
        axes[0, 0].scatter(clean_acc, robust_acc, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 0].annotate(model, (clean_acc[i], robust_acc[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Clean Accuracy')
        axes[0, 0].set_ylabel('Robust Accuracy') 
        axes[0, 0].set_title('Clean vs Robust Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Evasion Rates
        evasion_rates = [comparison['summary'][m]['avg_evasion_rate'] for m in models]
        colors = ['green' if rate < 0.3 else 'orange' if rate < 0.5 else 'red' for rate in evasion_rates]
        
        axes[0, 1].bar(models, evasion_rates, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0.3, color='red', linestyle='--', label='Target (<30%)')
        axes[0, 1].set_ylabel('Evasion Rate')
        axes[0, 1].set_title('Average Evasion Rates')
        axes[0, 1].legend()
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Robustness Improvement
        improvements = [comparison['summary'][m]['robustness_improvement'] for m in models]
        
        axes[1, 0].bar(models, improvements, alpha=0.7, color='blue')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_ylabel('Robustness Improvement (%)')
        axes[1, 0].set_title('Robustness Improvement vs Baseline')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Performance Trade-off
        robustness_scores = [comparison['summary'][m]['robustness_score'] for m in models]
        
        axes[1, 1].scatter(clean_acc, robustness_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (clean_acc[i], robustness_scores[i]),
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Clean Accuracy')
        axes[1, 1].set_ylabel('Robustness Score (%)')
        axes[1, 1].set_title('Accuracy vs Robustness Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'defense_evaluation_results.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation visualization saved to {save_dir}/defense_evaluation_results.png")
    
    def save_evaluation_report(self, report, save_dir='reports'):
        """Save evaluation report to file"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full report as JSON
        report_path = os.path.join(save_dir, 'defense_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary as Markdown
        summary_path = os.path.join(save_dir, 'defense_evaluation_summary.md')
        self._save_markdown_summary(report, summary_path)
        
        print(f"Evaluation report saved to {report_path}")
        print(f"Summary saved to {summary_path}")
    
    def _save_markdown_summary(self, report, file_path):
        """Save evaluation summary as Markdown"""
        
        with open(file_path, 'w') as f:
            f.write("# Defense Evaluation Summary\n\n")
            f.write(f"**Date**: {report['timestamp']}\n\n")
            
            f.write("## Models Evaluated\n")
            for model in report['summary']['models_evaluated']:
                f.write(f"- {model}\n")
            f.write("\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Model | Clean Acc | Robust Acc | Evasion Rate | Robustness Score |\n")
            f.write("|-------|-----------|------------|--------------|------------------|\n")
            
            for model, metrics in report['defense_comparison']['summary'].items():
                f.write(f"| {model} | {metrics['clean_accuracy']:.3f} | "
                       f"{metrics['avg_robust_accuracy']:.3f} | "
                       f"{metrics['avg_evasion_rate']:.3f} | "
                       f"{metrics['robustness_score']:.1f}% |\n")
            
            f.write("\n## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")


def main():
    """Main evaluation pipeline"""
    print("=== Defense Evaluation Framework ===")
    
    # Initialize evaluator
    evaluator = DefenseEvaluator()
    
    # Define models to evaluate
    model_configs = [
        {
            'name': 'Baseline_RF',
            'model_path': '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/rf_advanced.joblib',
            'scaler_path': '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/scaler_advanced.joblib'
        }
    ]
    
    # Check for robust model
    robust_model_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/simple_robust_rf.joblib'
    if os.path.exists(robust_model_path):
        model_configs.append({
            'name': 'Simple_Robust_RF',
            'model_path': robust_model_path,
            'scaler_path': None  # Simple robust model doesn't use separate scaler
        })
    
    # Load models
    evaluator.load_models(model_configs)
    
    # Load data
    X_test, y_test = evaluator.load_data()
    
    # Define attack configurations
    attack_configs = [
        {'name': 'FGSM_0.1', 'method': 'fgsm', 'epsilon': 0.1},
        {'name': 'FGSM_0.3', 'method': 'fgsm', 'epsilon': 0.3},
        {'name': 'PGD_0.1', 'method': 'pgd', 'epsilon': 0.1, 'num_iter': 10},
        {'name': 'PGD_0.3', 'method': 'pgd', 'epsilon': 0.3, 'num_iter': 10},
    ]
    
    # Evaluate clean performance
    clean_results = evaluator.evaluate_clean_performance(X_test, y_test)
    
    # Evaluate adversarial robustness
    robustness_results = evaluator.evaluate_adversarial_robustness(X_test, y_test, attack_configs)
    
    # Compare defenses
    comparison = evaluator.compare_defenses(clean_results, robustness_results)
    
    # Generate report
    report = evaluator.generate_evaluation_report(clean_results, robustness_results, comparison)
    
    # Create visualizations
    evaluator.visualize_evaluation_results(comparison)
    
    # Save results
    evaluator.save_evaluation_report(report)
    
    print("\n=== Defense Evaluation Complete ===")


if __name__ == "__main__":
    main()
