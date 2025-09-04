"""
Defense CLI Module

Handles adversarial defense training, hardening, and evaluation
for the 5G adversarial IDS system.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import time
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.cli.utils import (
    load_processed_data, load_model, save_model, save_results, 
    create_output_directory, print_experiment_header, 
    print_experiment_footer, format_performance_metrics,
    get_class_names, validate_model_file
)
from src.adv5g.defenses.simple_adversarial_trainer import SimpleAdversarialTrainer
from src.adv5g.defenses.defense_evaluation import DefenseEvaluator

class DefenseCLI:
    """CLI interface for adversarial defense training and evaluation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.defense_config = config.get('defense', {})
        self.data_config = config.get('data', {})
        self.models_config = config.get('models', {})
        
    def validate_defense_config(self) -> bool:
        """Validate defense-specific configuration."""
        # Check if base model is specified
        base_model = self.defense_config.get('base_model')
        if base_model and not validate_model_file(base_model, self.logger):
            self.logger.warning("⚠️  Base model validation failed, will train from scratch")
        
        # Check defense strategy
        strategy = self.defense_config.get('strategy', 'adversarial_training')
        if strategy not in ['adversarial_training']:
            self.logger.warning(f"⚠️  Unknown defense strategy: {strategy}, using adversarial_training")
            self.defense_config['strategy'] = 'adversarial_training'
        
        return True
    
    def load_base_model_and_data(self) -> Tuple[Any, Any, Dict[str, np.ndarray]]:
        """Load base model, scaler, and training data."""
        # Load base model if specified
        base_model_path = self.defense_config.get('base_model')
        base_model = None
        base_scaler = None
        
        if base_model_path and Path(base_model_path).exists():
            try:
                base_model = load_model(base_model_path, self.logger)
                
                # Load corresponding scaler
                scaler_path = self.defense_config.get('base_scaler', 'models/scaler.joblib')
                if Path(scaler_path).exists():
                    base_scaler = load_model(scaler_path, self.logger)
                    
                self.logger.info("✅ Base model and scaler loaded successfully")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to load base model: {e}, training from scratch")
                base_model = None
                base_scaler = None
        
        # Load training data
        data_dir = self.data_config.get('processed_dir', 'data/processed')
        data = load_processed_data(data_dir, self.logger)
        
        return base_model, base_scaler, data
    
    def setup_adversarial_trainer(self, data: Dict[str, np.ndarray]) -> SimpleAdversarialTrainer:
        """Initialize adversarial trainer with configuration."""
        training_config = self.defense_config.get('adversarial_training', {})
        
        # Get training parameters
        noise_levels = training_config.get('noise_levels', [0.1, 0.2, 0.3])
        training_rounds = training_config.get('training_rounds', 3)
        model_params = training_config.get('model_params', {})
        
        # Set default model parameters
        default_model_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'random_state': 42,
            'n_jobs': -1
        }
        
        for key, value in default_model_params.items():
            if key not in model_params:
                model_params[key] = value
        
        self.logger.info(f"🔧 Defense training configuration:")
        self.logger.info(f"   Noise levels: {noise_levels}")
        self.logger.info(f"   Training rounds: {training_rounds}")
        self.logger.info(f"   Model params: {model_params}")
        
        # Initialize trainer with just base_model_params
        trainer = SimpleAdversarialTrainer(base_model_params=model_params)
        
        # Override the noise levels if specified
        if noise_levels:
            trainer.noise_levels = noise_levels
        
        return trainer
    
    def train_robust_model(self, trainer: SimpleAdversarialTrainer, 
                          data: Dict[str, np.ndarray]) -> Tuple[Any, Dict[str, Any]]:
        """Train robust model using adversarial training."""
        self.logger.info("🛡️  Starting adversarial defense training...")
        
        try:
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            
            # Check if we should use all features instead of PCA
            feature_config = self.defense_config.get('feature_robustness', {})
            use_all_features = feature_config.get('use_all_features', True)
            
            if use_all_features:
                self.logger.info("🔧 Using all 43 features for enhanced robustness")
                # We'll assume the data is already in the right format
                # In a real scenario, you might need to reload raw features
            
            # Start training
            start_time = time.time()
            
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Train using the correct method name
            trainer.train_progressive_robust_model(X_train_split, y_train_split, X_val, y_val)
            
            # Get the trained model and training history
            robust_model = trainer.model
            training_history = trainer.training_history
            
            training_time = time.time() - start_time
            
            # Add timing information
            if not isinstance(training_history, dict):
                training_history = {'epochs': training_history} if training_history else {}
            training_history['training_time'] = training_time
            training_history['features_used'] = X_train.shape[1]
            training_history['training_samples'] = X_train.shape[0]
            
            self.logger.info(f"✅ Adversarial training completed in {training_time:.2f}s")
            return robust_model, training_history
            
        except Exception as e:
            self.logger.error(f"❌ Adversarial training failed: {e}")
            raise
    
    def evaluate_robust_model(self, robust_model: Any, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Comprehensive evaluation of the robust model."""
        self.logger.info("📊 Evaluating robust model...")
        
        try:
            # Initialize defense evaluator
            evaluation_config = self.defense_config.get('evaluation', {})
            
            evaluator = DefenseEvaluator()
            
            # Load the robust model for evaluation
            model_configs = [{
                'name': 'robust_model',
                'model_path': None,  # We'll pass the model directly
                'scaler_path': None
            }]
            
            # Since DefenseEvaluator expects to load models from files,
            # we need to temporarily save our robust model
            import tempfile
            import os
            temp_model_path = os.path.join('models', 'temp_robust_model.joblib')
            joblib.dump(robust_model, temp_model_path)
            
            try:
                # Update model config with temp path
                model_configs[0]['model_path'] = temp_model_path
                evaluator.load_models(model_configs)
                
                X_test = data['X_test']
                y_test = data['y_test']
                
                # First evaluate clean performance
                self.logger.info("📊 Evaluating clean performance...")
                clean_results = evaluator.evaluate_clean_performance(X_test, y_test)
                
                # Then evaluate adversarial robustness
                attack_types = evaluation_config.get('attack_types', ['enhanced_pgd', 'enhanced_fgsm'])
                noise_levels = evaluation_config.get('noise_levels', [0.1, 0.2, 0.3])
                
                # Create attack configs for robustness evaluation
                attack_configs = []
                for attack_type in attack_types:
                    for noise_level in noise_levels:
                        attack_configs.append({
                            'name': f"{attack_type}_eps_{noise_level}",
                            'method': attack_type,
                            'type': attack_type,
                            'epsilon': noise_level,
                            'num_steps': 20 if 'pgd' in attack_type else None,
                            'num_iter': 20 if 'pgd' in attack_type else None
                        })
                
                self.logger.info("🛡️  Evaluating adversarial robustness...")
                robustness_results = evaluator.evaluate_adversarial_robustness(X_test, y_test, attack_configs)
                
                # Compare defense performance
                self.logger.info("📈 Comparing defense performance...")
                comparison = evaluator.compare_defenses(clean_results, robustness_results)
                
                # Calculate overall defense score from comparison results
                if evaluation_config.get('calculate_defense_score', True):
                    # Extract key metrics for defense score calculation
                    clean_accuracy = clean_results.get('robust_model', {}).get('accuracy', 0.0)
                    
                    # Calculate average robustness across all attacks
                    robust_accuracies = []
                    for attack_name, attack_results in robustness_results.get('robust_model', {}).items():
                        if isinstance(attack_results, dict) and 'accuracy' in attack_results:
                            robust_accuracies.append(attack_results['accuracy'])
                    
                    avg_robust_accuracy = np.mean(robust_accuracies) if robust_accuracies else 0.0
                    defense_score = (clean_accuracy + avg_robust_accuracy) / 2.0
                    
                    comparison['defense_score'] = defense_score
                    self.logger.info(f"🏆 Defense Score: {defense_score:.4f}")
                
                def to_safe(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    if isinstance(obj, dict):
                        return {k: to_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [to_safe(v) for v in obj]
                    return obj

                evaluation_results = to_safe({
                    'clean_results': clean_results,
                    'robustness_results': robustness_results,
                    'comparison': comparison
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
            
            self.logger.info("✅ Robust model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Robust model evaluation failed: {e}")
            raise
    
    def compare_with_baseline(self, robust_model: Any, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare robust model performance with baseline."""
        self.logger.info("📈 Comparing with baseline model...")
        
        try:
            comparison_results = {}
            
            # Load baseline model if available
            base_model_path = self.defense_config.get('base_model')
            if base_model_path and Path(base_model_path).exists():
                baseline_model = load_model(base_model_path, self.logger)
                
                X_test = data['X_test']
                y_test = data['y_test']
                
                # Evaluate baseline
                baseline_pred = baseline_model.predict(X_test)
                baseline_accuracy = np.mean(baseline_pred == y_test)
                
                # Evaluate robust model
                robust_pred = robust_model.predict(X_test)
                robust_accuracy = np.mean(robust_pred == y_test)
                
                # Calculate improvement
                accuracy_improvement = robust_accuracy - baseline_accuracy
                
                comparison_results = {
                    'baseline_accuracy': baseline_accuracy,
                    'robust_accuracy': robust_accuracy,
                    'accuracy_improvement': accuracy_improvement,
                    'improvement_percentage': (accuracy_improvement / baseline_accuracy) * 100
                }
                
                self.logger.info(f"📊 Performance Comparison:")
                self.logger.info(f"   Baseline accuracy: {baseline_accuracy:.4f}")
                self.logger.info(f"   Robust accuracy: {robust_accuracy:.4f}")
                self.logger.info(f"   Improvement: {accuracy_improvement:+.4f} ({comparison_results['improvement_percentage']:+.2f}%)")
                
            else:
                self.logger.warning("⚠️  Baseline model not available for comparison")
                comparison_results = {'comparison_available': False}
            
            return comparison_results
            
        except Exception as e:
            self.logger.warning(f"⚠️  Baseline comparison failed: {e}")
            return {'comparison_available': False, 'error': str(e)}
    
    def save_defense_artifacts(self, robust_model: Any, training_history: Dict[str, Any],
                             evaluation_results: Dict[str, Any], 
                             comparison_results: Dict[str, Any]) -> bool:
        """Save trained robust model and all results."""
        try:
            # Create output directory
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'adversarial_defense',
                self.logger
            )
            
            # Save robust model
            model_path = output_dir / 'simple_robust_rf.joblib'
            def to_safe(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: to_safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [to_safe(v) for v in obj]
                return obj

            model_metadata = to_safe({
                'model_type': 'RobustRandomForest',
                'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'defense_strategy': self.defense_config.get('strategy', 'adversarial_training'),
                'training_config': self.defense_config.get('adversarial_training', {}),
                'training_history': training_history,
                'evaluation_summary': evaluation_results,
                'comparison_results': comparison_results
            })
            
            save_model(robust_model, str(model_path), model_metadata, self.logger)
            
            # Save detailed results
            results_path = output_dir / 'defense_results.json'
            combined_results = to_safe({
                'training_history': training_history,
                'evaluation_results': evaluation_results,
                'comparison_results': comparison_results,
                'configuration': self.defense_config
            })
            save_results(combined_results, str(results_path), self.logger)
            
            # Update models directory if specified
            models_dir = self.models_config.get('save_dir')
            if models_dir:
                models_path = Path(models_dir)
                models_path.mkdir(parents=True, exist_ok=True)
                
                # Copy to models directory
                import shutil
                shutil.copy2(model_path, models_path / 'simple_robust_rf.joblib')
                
                # Save metadata separately
                metadata_path = models_path / 'simple_robust_rf_metadata.json'
                save_results(model_metadata, str(metadata_path), self.logger)
                
                self.logger.info(f"✅ Robust model copied to: {models_path}")
            
            # Generate defense report
            if self.defense_config.get('evaluation', {}).get('generate_defense_reports', True):
                self.generate_defense_report(training_history, evaluation_results, 
                                           comparison_results, output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save defense artifacts: {e}")
            return False
    
    def generate_defense_report(self, training_history: Dict[str, Any],
                              evaluation_results: Dict[str, Any],
                              comparison_results: Dict[str, Any], 
                              output_dir: Path):
        """Generate comprehensive defense report."""
        report_path = output_dir / 'defense_report.md'
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Adversarial Defense Evaluation Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Training summary
                f.write("## Training Summary\n\n")
                f.write(f"- **Training Time**: {training_history.get('training_time', 'N/A'):.2f}s\n")
                f.write(f"- **Features Used**: {training_history.get('features_used', 'N/A')}\n")
                f.write(f"- **Training Samples**: {training_history.get('training_samples', 'N/A')}\n")
                f.write(f"- **Training Rounds**: {len(training_history.get('round_results', []))}\n\n")
                
                # Performance comparison
                if comparison_results.get('comparison_available', True):
                    f.write("## Performance Comparison\n\n")
                    f.write("| Metric | Baseline | Robust | Improvement |\n")
                    f.write("|--------|----------|--------|-------------|\n")
                    
                    baseline_acc = comparison_results.get('baseline_accuracy', 0)
                    robust_acc = comparison_results.get('robust_accuracy', 0)
                    improvement = comparison_results.get('accuracy_improvement', 0)
                    
                    f.write(f"| Accuracy | {baseline_acc:.4f} | {robust_acc:.4f} | {improvement:+.4f} |\n\n")
                
                # Defense evaluation
                f.write("## Defense Evaluation\n\n")
                defense_score = evaluation_results.get('defense_score', 'N/A')
                f.write(f"**Defense Score**: {defense_score}\n\n")
                
                # Robustness against attacks
                if 'attack_evaluations' in evaluation_results:
                    f.write("### Robustness Against Attacks\n\n")
                    f.write("| Attack Type | Robustness Score |\n")
                    f.write("|-------------|------------------|\n")
                    
                    for attack_type, results in evaluation_results['attack_evaluations'].items():
                        robustness = results.get('average_robustness', 0)
                        f.write(f"| {attack_type} | {robustness:.4f} |\n")
                
                f.write("\n")
                
            self.logger.info(f"✅ Defense report generated: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to generate defense report: {e}")
    
    def execute(self) -> bool:
        """Execute complete adversarial defense pipeline."""
        print_experiment_header("ADVERSARIAL DEFENSE TRAINING", self.config, self.logger)
        
        try:
            # Validate configuration
            if not self.validate_defense_config():
                return False
            
            # Load base model and data
            self.logger.info("📂 Loading base model and training data...")
            base_model, base_scaler, data = self.load_base_model_and_data()
            
            # Setup adversarial trainer
            trainer = self.setup_adversarial_trainer(data)
            
            # Train robust model
            robust_model, training_history = self.train_robust_model(trainer, data)
            
            # Evaluate robust model
            evaluation_results = self.evaluate_robust_model(robust_model, data)
            
            # Compare with baseline
            comparison_results = self.compare_with_baseline(robust_model, data)
            
            # Save artifacts
            if not self.save_defense_artifacts(robust_model, training_history, 
                                             evaluation_results, comparison_results):
                return False
            
            # Log final summary
            self.logger.info("🛡️  DEFENSE TRAINING SUMMARY:")
            
            if comparison_results.get('comparison_available', True):
                improvement = comparison_results.get('accuracy_improvement', 0)
                improvement_pct = comparison_results.get('improvement_percentage', 0)
                self.logger.info(f"   Accuracy improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
            
            defense_score = evaluation_results.get('defense_score', 'N/A')
            self.logger.info(f"   Defense score: {defense_score}")
            
            training_time = training_history.get('training_time', 0)
            self.logger.info(f"   Training time: {training_time:.2f}s")
            
            print_experiment_footer("ADVERSARIAL DEFENSE TRAINING", True, self.logger)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Defense training pipeline failed: {e}")
            print_experiment_footer("ADVERSARIAL DEFENSE TRAINING", False, self.logger)
            return False
