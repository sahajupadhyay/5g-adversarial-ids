"""
Evaluation CLI Module

Handles comprehensive evaluation of the complete 5G adversarial IDS system,
comparing baseline, attack, and defense performance.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.cli.utils import (
    load_processed_data, load_model, save_results, 
    create_output_directory, print_experiment_header, 
    print_experiment_footer, format_performance_metrics,
    get_class_names, validate_model_file
)

class EvaluationCLI:
    """CLI interface for comprehensive system evaluation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.eval_config = config.get('evaluation', {})
        self.data_config = config.get('data', {})
        self.models_config = config.get('models', {})
        
    def validate_evaluation_config(self) -> bool:
        """Validate evaluation-specific configuration."""
        # Check what models to evaluate
        models_to_eval = self.eval_config.get('models', ['baseline', 'robust'])
        valid_models = ['baseline', 'robust']
        
        for model in models_to_eval:
            if model not in valid_models:
                self.logger.error(f"❌ Invalid model type for evaluation: {model}")
                return False
        
        # Check if model files exist
        model_paths = self.eval_config.get('model_paths', {})
        
        # Set default paths if not specified
        if 'baseline' not in model_paths:
            model_paths['baseline'] = 'models/rf_baseline_tuned.joblib'
        if 'robust' not in model_paths:
            model_paths['robust'] = 'models/simple_robust_rf.joblib'
        
        self.eval_config['model_paths'] = model_paths
        
        # Validate that required models exist
        for model_type in models_to_eval:
            model_path = model_paths.get(model_type)
            if model_path and not validate_model_file(model_path, self.logger):
                self.logger.warning(f"⚠️  {model_type} model not found: {model_path}")
        
        return True
    
    def load_models_and_data(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Load all models and test data for evaluation."""
        models = {}
        model_paths = self.eval_config.get('model_paths', {})
        models_to_eval = self.eval_config.get('models', ['baseline', 'robust'])
        
        # Load models
        for model_type in models_to_eval:
            model_path = model_paths.get(model_type)
            if model_path and Path(model_path).exists():
                try:
                    models[model_type] = load_model(model_path, self.logger)
                    self.logger.info(f"✅ Loaded {model_type} model from: {model_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to load {model_type} model: {e}")
        
        # Load scaler
        scaler_path = self.eval_config.get('scaler_path', 'models/scaler.joblib')
        if Path(scaler_path).exists():
            models['scaler'] = load_model(scaler_path, self.logger)
        
        # Load test data
        data_dir = self.data_config.get('processed_dir', 'data/processed')
        data = load_processed_data(data_dir, self.logger)
        
        return models, data
    
    def evaluate_model_performance(self, model: Any, scaler: Any, 
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 model_name: str) -> Dict[str, Any]:
        """Evaluate basic model performance metrics."""
        self.logger.info(f"📊 Evaluating {model_name} model performance...")
        
        try:
            # Scale test data
            X_test_scaled = scaler.transform(X_test) if scaler else X_test
            
            # Generate predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_recall_fscore_support,
                classification_report, confusion_matrix
            )
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0
            )
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            
            # Per-class results
            class_names = get_class_names()
            per_class_results = {}
            for i, class_name in enumerate(class_names):
                per_class_results[class_name] = {
                    'precision': precision[i] if i < len(precision) else 0,
                    'recall': recall[i] if i < len(recall) else 0,
                    'f1': f1[i] if i < len(f1) else 0,
                    'support': support[i] if i < len(support) else 0
                }
            
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'per_class_results': per_class_results,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, 
                                                             target_names=class_names)
            }
            
            self.logger.info(f"✅ {model_name} evaluation completed:")
            self.logger.info(f"   Accuracy: {accuracy:.4f}")
            self.logger.info(f"   Macro-F1: {macro_f1:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ {model_name} evaluation failed: {e}")
            raise
    
    def evaluate_adversarial_robustness(self, models: Dict[str, Any], 
                                       data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate adversarial robustness of models."""
        self.logger.info("🛡️  Evaluating adversarial robustness...")
        
        robustness_results = {}
        
        try:
            # Import attack modules
            from src.attacks.enhanced_attacks import EnhancedPGDAttack, EnhancedFGSMAttack
            from src.attacks.attack_utils import evaluate_attack_success
            
            X_test = data['X_test']
            y_test = data['y_test']
            scaler = models.get('scaler')
            
            if scaler:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Test attacks
            attack_configs = self.eval_config.get('adversarial_attacks', {
                'enhanced_pgd': {'epsilon': 0.3, 'alpha': 0.01, 'num_iter': 40},
                'enhanced_fgsm': {'epsilon': 0.3}
            })
            
            # Evaluate each model against attacks
            models_to_test = ['baseline', 'robust']
            for model_name in models_to_test:
                if model_name not in models:
                    continue
                
                model = models[model_name]
                model_robustness = {}
                
                for attack_name, attack_params in attack_configs.items():
                    self.logger.info(f"  Testing {model_name} against {attack_name}...")
                    
                    try:
                        # Initialize attack
                        if attack_name == 'enhanced_pgd':
                            attack = EnhancedPGDAttack(**attack_params)
                        elif attack_name == 'enhanced_fgsm':
                            attack = EnhancedFGSMAttack(**attack_params)
                        else:
                            continue
                        
                        # Generate adversarial examples
                        X_adv = attack.generate(model, X_test_scaled, y_test)
                        
                        # Evaluate attack success
                        attack_results = evaluate_attack_success(
                            model, X_test_scaled, X_adv, y_test,
                            class_names=get_class_names()
                        )
                        
                        model_robustness[attack_name] = {
                            'evasion_rate': attack_results.get('overall_success_rate', 0),
                            'robustness_score': 1 - attack_results.get('overall_success_rate', 0),
                            'per_class_robustness': {}
                        }
                        
                        # Per-class robustness
                        per_class_results = attack_results.get('per_class_results', {})
                        for class_idx, class_result in per_class_results.items():
                            class_name = get_class_names()[class_idx]
                            evasion_rate = class_result.get('success_rate', 0)
                            model_robustness[attack_name]['per_class_robustness'][class_name] = {
                                'evasion_rate': evasion_rate,
                                'robustness_score': 1 - evasion_rate
                            }
                        
                        self.logger.info(f"    {attack_name}: {attack_results.get('overall_success_rate', 0):.2%} evasion")
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️  {attack_name} attack on {model_name} failed: {e}")
                        model_robustness[attack_name] = {'error': str(e)}
                
                robustness_results[model_name] = model_robustness
            
            return robustness_results
            
        except Exception as e:
            self.logger.warning(f"⚠️  Adversarial robustness evaluation failed: {e}")
            return {}
    
    def compare_models(self, performance_results: Dict[str, Dict],
                      robustness_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare performance and robustness across models."""
        self.logger.info("📈 Comparing model performance...")
        
        comparison = {
            'performance_comparison': {},
            'robustness_comparison': {},
            'overall_ranking': {},
            'improvement_analysis': {}
        }
        
        # Performance comparison
        performance_metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        
        for metric in performance_metrics:
            comparison['performance_comparison'][metric] = {}
            for model_name, results in performance_results.items():
                comparison['performance_comparison'][metric][model_name] = \
                    results.get(metric, 0)
        
        # Robustness comparison
        if robustness_results:
            for model_name, robustness in robustness_results.items():
                model_avg_robustness = []
                for attack_name, attack_results in robustness.items():
                    if 'robustness_score' in attack_results:
                        model_avg_robustness.append(attack_results['robustness_score'])
                
                avg_robustness = np.mean(model_avg_robustness) if model_avg_robustness else 0
                comparison['robustness_comparison'][model_name] = avg_robustness
        
        # Calculate improvement (robust vs baseline)
        if 'baseline' in performance_results and 'robust' in performance_results:
            baseline_perf = performance_results['baseline']
            robust_perf = performance_results['robust']
            
            comparison['improvement_analysis'] = {
                'accuracy_improvement': robust_perf.get('accuracy', 0) - baseline_perf.get('accuracy', 0),
                'f1_improvement': robust_perf.get('macro_f1', 0) - baseline_perf.get('macro_f1', 0),
                'robustness_gain': comparison['robustness_comparison'].get('robust', 0) - 
                                 comparison['robustness_comparison'].get('baseline', 0)
            }
        
        # Overall ranking (simple scoring)
        for model_name in performance_results.keys():
            performance_score = performance_results[model_name].get('accuracy', 0)
            robustness_score = comparison['robustness_comparison'].get(model_name, 0)
            
            # Weighted average: 70% performance, 30% robustness
            overall_score = 0.7 * performance_score + 0.3 * robustness_score
            comparison['overall_ranking'][model_name] = overall_score
        
        return comparison
    
    def generate_evaluation_report(self, performance_results: Dict[str, Dict],
                                 robustness_results: Dict[str, Dict],
                                 comparison: Dict[str, Any], 
                                 output_dir: Path) -> bool:
        """Generate comprehensive evaluation report."""
        report_path = output_dir / 'comprehensive_evaluation_report.md'
        
        try:
            with open(report_path, 'w') as f:
                f.write("# 5G Adversarial IDS - Comprehensive Evaluation Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive summary
                f.write("## Executive Summary\n\n")
                if 'improvement_analysis' in comparison:
                    improvements = comparison['improvement_analysis']
                    acc_imp = improvements.get('accuracy_improvement', 0)
                    f1_imp = improvements.get('f1_improvement', 0)
                    rob_gain = improvements.get('robustness_gain', 0)
                    
                    f.write(f"- **Accuracy Improvement**: {acc_imp:+.4f}\n")
                    f.write(f"- **F1-Score Improvement**: {f1_imp:+.4f}\n")
                    f.write(f"- **Robustness Gain**: {rob_gain:+.4f}\n\n")
                
                # Performance comparison
                f.write("## Performance Comparison\n\n")
                f.write("| Model | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |\n")
                f.write("|-------|----------|----------|-----------------|---------------|\n")
                
                for model_name, results in performance_results.items():
                    acc = results.get('accuracy', 0)
                    f1 = results.get('macro_f1', 0)
                    prec = results.get('macro_precision', 0)
                    rec = results.get('macro_recall', 0)
                    
                    f.write(f"| {model_name} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |\n")
                
                f.write("\n")
                
                # Robustness comparison
                if robustness_results:
                    f.write("## Robustness Analysis\n\n")
                    f.write("| Model | Average Robustness | PGD Robustness | FGSM Robustness |\n")
                    f.write("|-------|-------------------|----------------|------------------|\n")
                    
                    for model_name in robustness_results.keys():
                        avg_rob = comparison['robustness_comparison'].get(model_name, 0)
                        pgd_rob = robustness_results[model_name].get('enhanced_pgd', {}).get('robustness_score', 0)
                        fgsm_rob = robustness_results[model_name].get('enhanced_fgsm', {}).get('robustness_score', 0)
                        
                        f.write(f"| {model_name} | {avg_rob:.4f} | {pgd_rob:.4f} | {fgsm_rob:.4f} |\n")
                    
                    f.write("\n")
                
                # Overall ranking
                f.write("## Overall Model Ranking\n\n")
                ranking = comparison.get('overall_ranking', {})
                sorted_models = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
                
                f.write("| Rank | Model | Overall Score | Notes |\n")
                f.write("|------|-------|---------------|-------|\n")
                
                for i, (model_name, score) in enumerate(sorted_models, 1):
                    f.write(f"| {i} | {model_name} | {score:.4f} | - |\n")
                
                f.write("\n")
                
                # Recommendations
                f.write("## Recommendations\n\n")
                if sorted_models:
                    best_model = sorted_models[0][0]
                    f.write(f"1. **Best Overall Model**: {best_model}\n")
                
                if 'improvement_analysis' in comparison:
                    improvements = comparison['improvement_analysis']
                    if improvements.get('accuracy_improvement', 0) > 0:
                        f.write("2. ✅ Adversarial training improved clean accuracy\n")
                    if improvements.get('robustness_gain', 0) > 0:
                        f.write("3. ✅ Adversarial training improved robustness\n")
                
                f.write("4. Deploy the robust model for production use\n")
                f.write("5. Monitor performance in real-world scenarios\n")
            
            self.logger.info(f"✅ Comprehensive evaluation report generated: {report_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to generate evaluation report: {e}")
            return False
    
    def save_evaluation_artifacts(self, performance_results: Dict[str, Dict],
                                robustness_results: Dict[str, Dict],
                                comparison: Dict[str, Any]) -> bool:
        """Save all evaluation results and artifacts."""
        try:
            # Create output directory
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'comprehensive_evaluation',
                self.logger
            )
            
            # Save detailed results
            evaluation_results = {
                'performance_results': performance_results,
                'robustness_results': robustness_results,
                'comparison_analysis': comparison,
                'configuration': self.eval_config,
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results_path = output_dir / 'evaluation_results.json'
            save_results(evaluation_results, str(results_path), self.logger)
            
            # Generate comprehensive report
            self.generate_evaluation_report(performance_results, robustness_results,
                                          comparison, output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save evaluation artifacts: {e}")
            return False
    
    def execute(self) -> bool:
        """Execute comprehensive evaluation pipeline."""
        print_experiment_header("COMPREHENSIVE SYSTEM EVALUATION", self.config, self.logger)
        
        try:
            # Validate configuration
            if not self.validate_evaluation_config():
                return False
            
            # Load models and data
            self.logger.info("📂 Loading models and test data...")
            models, data = self.load_models_and_data()
            
            if not models:
                self.logger.error("❌ No models available for evaluation")
                return False
            
            # Evaluate model performance
            performance_results = {}
            models_to_eval = self.eval_config.get('models', ['baseline', 'robust'])
            scaler = models.get('scaler')
            
            X_test = data['X_test']
            y_test = data['y_test']
            
            for model_name in models_to_eval:
                if model_name in models:
                    model_results = self.evaluate_model_performance(
                        models[model_name], scaler, X_test, y_test, model_name
                    )
                    performance_results[model_name] = model_results
            
            # Evaluate adversarial robustness
            robustness_results = {}
            if self.eval_config.get('evaluate_robustness', True):
                robustness_results = self.evaluate_adversarial_robustness(models, data)
            
            # Compare models
            comparison = self.compare_models(performance_results, robustness_results)
            
            # Save artifacts
            if not self.save_evaluation_artifacts(performance_results, robustness_results, comparison):
                return False
            
            # Log final summary
            self.logger.info("📊 COMPREHENSIVE EVALUATION SUMMARY:")
            
            for model_name, results in performance_results.items():
                acc = results.get('accuracy', 0)
                f1 = results.get('macro_f1', 0)
                self.logger.info(f"   {model_name}: {acc:.4f} accuracy, {f1:.4f} macro-F1")
            
            if 'improvement_analysis' in comparison:
                improvements = comparison['improvement_analysis']
                acc_imp = improvements.get('accuracy_improvement', 0)
                rob_gain = improvements.get('robustness_gain', 0)
                self.logger.info(f"   Accuracy improvement: {acc_imp:+.4f}")
                self.logger.info(f"   Robustness gain: {rob_gain:+.4f}")
            
            # Best model
            ranking = comparison.get('overall_ranking', {})
            if ranking:
                best_model = max(ranking, key=ranking.get)
                best_score = ranking[best_model]
                self.logger.info(f"   Best overall model: {best_model} (score: {best_score:.4f})")
            
            print_experiment_footer("COMPREHENSIVE SYSTEM EVALUATION", True, self.logger)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive evaluation failed: {e}")
            print_experiment_footer("COMPREHENSIVE SYSTEM EVALUATION", False, self.logger)
            return False
