"""
Defend command for the Adversarial 5G IDS CLI.

Provides defense evaluation and robustness testing capabilities.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import argparse
import json

from src.cli.utils.output import CLIOutput, Icons, Colors, ProgressBar

class DefendCommand:
    """Command for evaluating defenses and model robustness."""
    
    def __init__(self):
        self.output = CLIOutput()
    
    def add_parser(self, subparsers):
        """Add defend command parser."""
        parser = subparsers.add_parser(
            'defend',
            help='Evaluate defense mechanisms and model robustness',
            description='Test and analyze the effectiveness of adversarial defenses.'
        )
        
        parser.add_argument(
            '--evaluate',
            action='store_true',
            help='Run defense evaluation'
        )
        
        parser.add_argument(
            '--compare-models',
            action='store_true',
            help='Compare baseline vs robust model performance'
        )
        
        parser.add_argument(
            '--robustness-test',
            action='store_true',
            help='Run comprehensive robustness testing'
        )
        
        parser.add_argument(
            '--noise-levels',
            type=str,
            default='0.05,0.1,0.2,0.3,0.5',
            help='Comma-separated noise levels for testing (default: 0.05,0.1,0.2,0.3,0.5)'
        )
        
        parser.add_argument(
            '--samples', '-n',
            type=int,
            default=100,
            help='Number of samples for evaluation (default: 100)'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file for results (JSON format)'
        )
        
        parser.add_argument(
            '--detailed-report',
            action='store_true',
            help='Generate detailed analysis report'
        )
        
        parser.add_argument(
            '--save-metrics',
            type=str,
            help='Save detailed metrics to file'
        )
        
        return parser
    
    def execute(self, args, config, output):
        """Execute the defend command."""
        self.output = output
        self.config = config
        
        try:
            self.output.header("🛡️ Defense Evaluation Suite", "Analyzing adversarial robustness and defense effectiveness")
            
            # Parse noise levels
            noise_levels = [float(x.strip()) for x in args.noise_levels.split(',')]
            
            # Load test data
            X, y = self._load_test_data(args.samples)
            
            # Load models
            models = self._load_models()
            
            results = {}
            
            # Run requested evaluations
            if args.evaluate or args.compare_models:
                comparison_results = self._compare_models(X, y, models, noise_levels)
                results['model_comparison'] = comparison_results
            
            if args.robustness_test:
                robustness_results = self._robustness_testing(X, y, models, noise_levels)
                results['robustness_analysis'] = robustness_results
            
            # If no specific test requested, run all
            if not any([args.evaluate, args.compare_models, args.robustness_test]):
                comparison_results = self._compare_models(X, y, models, noise_levels)
                robustness_results = self._robustness_testing(X, y, models, noise_levels)
                results['model_comparison'] = comparison_results
                results['robustness_analysis'] = robustness_results
            
            # Display results
            self._display_defense_results(results, args.detailed_report)
            
            # Save results if requested
            if args.output:
                self._save_results(results, args.output)
            
            if args.save_metrics:
                self._save_detailed_metrics(results, args.save_metrics)
            
            return 0
            
        except Exception as e:
            self.output.error(f"Defense evaluation failed: {str(e)}")
            return 1
    
    def _load_test_data(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for defense evaluation."""
        test_data_path = self.config.get('paths.test_data')
        test_labels_path = self.config.get('paths.test_labels')
        
        if not Path(test_data_path).exists() or not Path(test_labels_path).exists():
            raise FileNotFoundError("Test data not found. Run baseline model training first.")
        
        X = np.load(test_data_path)
        y = np.load(test_labels_path)
        
        # Limit samples if requested
        if max_samples < len(X):
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        self.output.success(f"Loaded {len(X)} samples for defense evaluation")
        return X, y
    
    def _load_models(self) -> Dict[str, Any]:
        """Load both baseline and robust models."""
        models = {}
        
        # Load baseline model
        baseline_path = self.config.get('paths.baseline_model')
        transformers_path = self.config.get('paths.transformers')
        
        if Path(baseline_path).exists():
            baseline_model = joblib.load(baseline_path)
            transformers = joblib.load(transformers_path) if Path(transformers_path).exists() else None
            
            models['baseline'] = {
                'model': baseline_model,
                'transformers': transformers,
                'info': self.config.get_model_info('baseline'),
                'name': 'Baseline Random Forest'
            }
            self.output.success("Loaded baseline model")
        else:
            self.output.warning("Baseline model not found")
        
        # Load robust model
        robust_path = self.config.get('paths.robust_model')
        
        if Path(robust_path).exists():
            robust_model = joblib.load(robust_path)
            
            models['robust'] = {
                'model': robust_model,
                'transformers': None,  # Robust model uses raw features
                'info': self.config.get_model_info('robust'),
                'name': 'Adversarially Trained Random Forest'
            }
            self.output.success("Loaded robust model")
        else:
            self.output.warning("Robust model not found")
        
        if not models:
            raise ValueError("No models could be loaded")
        
        return models
    
    def _compare_models(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any], 
                       noise_levels: List[float]) -> Dict[str, Any]:
        """Compare baseline and robust model performance."""
        self.output.subheader("Model Performance Comparison")
        
        comparison_results = {
            'noise_levels': noise_levels,
            'models': {}
        }
        
        for model_name, model_data in models.items():
            self.output.info(f"Evaluating {model_data['name']}")
            
            model = model_data['model']
            transformers = model_data['transformers']
            
            # Transform features if needed
            if transformers is not None:
                X_input = transformers.transform(X)
                self.output.verbose("Applied feature transformations")
            else:
                X_input = X.copy()
            
            # Clean performance
            clean_accuracy = model.score(X_input, y)
            clean_predictions = model.predict(X_input)
            
            # Per-class clean performance
            clean_class_accuracy = {}
            for class_id in np.unique(y):
                mask = y == class_id
                if np.sum(mask) > 0:
                    class_acc = np.mean(clean_predictions[mask] == y[mask])
                    class_name = self.config.get_class_name(int(class_id))
                    clean_class_accuracy[class_name] = float(class_acc)
            
            # Robustness testing across noise levels
            robustness_scores = []
            noise_accuracies = {}
            
            progress = ProgressBar(len(noise_levels), f"Testing {model_name} robustness")
            
            for noise_level in noise_levels:
                # Generate noisy examples
                X_noisy = self._add_noise(X_input, noise_level)
                
                # Test on noisy data
                noisy_accuracy = model.score(X_noisy, y)
                robustness_scores.append(noisy_accuracy)
                noise_accuracies[str(noise_level)] = float(noisy_accuracy)
                
                progress.update(1)
            
            progress.finish(f"{model_name} robustness testing complete")
            
            # Calculate overall robustness score
            avg_robustness = float(np.mean(robustness_scores))
            
            model_results = {
                'name': model_data['name'],
                'clean_accuracy': float(clean_accuracy),
                'clean_class_accuracy': clean_class_accuracy,
                'robustness_score': avg_robustness,
                'noise_accuracies': noise_accuracies,
                'robustness_scores': [float(x) for x in robustness_scores]
            }
            
            comparison_results['models'][model_name] = model_results
        
        # Calculate defense effectiveness if both models available
        if 'baseline' in comparison_results['models'] and 'robust' in comparison_results['models']:
            baseline_results = comparison_results['models']['baseline']
            robust_results = comparison_results['models']['robust']
            
            clean_improvement = robust_results['clean_accuracy'] - baseline_results['clean_accuracy']
            robustness_improvement = robust_results['robustness_score'] - baseline_results['robustness_score']
            
            # Defense effectiveness score (clean improvement + robustness improvement)
            defense_effectiveness = clean_improvement + robustness_improvement
            
            comparison_results['defense_effectiveness'] = {
                'clean_improvement': float(clean_improvement),
                'robustness_improvement': float(robustness_improvement),
                'defense_score': float(defense_effectiveness),
                'rating': self._rate_defense_effectiveness(defense_effectiveness)
            }
        
        return comparison_results
    
    def _robustness_testing(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any], 
                           noise_levels: List[float]) -> Dict[str, Any]:
        """Comprehensive robustness testing."""
        self.output.subheader("Comprehensive Robustness Analysis")
        
        robustness_results = {
            'test_types': ['gaussian_noise', 'uniform_noise', 'feature_dropout'],
            'models': {}
        }
        
        for model_name, model_data in models.items():
            self.output.info(f"Running robustness tests on {model_data['name']}")
            
            model = model_data['model']
            transformers = model_data['transformers']
            
            # Transform features if needed
            if transformers is not None:
                X_input = transformers.transform(X)
            else:
                X_input = X.copy()
            
            model_robustness = {
                'gaussian_noise': self._test_gaussian_noise(model, X_input, y, noise_levels),
                'uniform_noise': self._test_uniform_noise(model, X_input, y, noise_levels),
                'feature_dropout': self._test_feature_dropout(model, X_input, y, [0.1, 0.2, 0.3, 0.4, 0.5])
            }
            
            # Calculate overall robustness metrics
            all_scores = []
            for test_type, scores in model_robustness.items():
                all_scores.extend(scores['accuracies'])
            
            model_robustness['overall'] = {
                'avg_robustness': float(np.mean(all_scores)),
                'min_robustness': float(np.min(all_scores)),
                'std_robustness': float(np.std(all_scores))
            }
            
            robustness_results['models'][model_name] = model_robustness
        
        return robustness_results
    
    def _test_gaussian_noise(self, model, X: np.ndarray, y: np.ndarray, 
                            noise_levels: List[float]) -> Dict[str, Any]:
        """Test robustness against Gaussian noise."""
        accuracies = []
        
        for noise_level in noise_levels:
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            # Apply feature bounds
            bounds = self.config.get('constraints.feature_bounds', [-3.0, 3.0])
            X_noisy = np.clip(X_noisy, bounds[0], bounds[1])
            
            accuracy = model.score(X_noisy, y)
            accuracies.append(float(accuracy))
        
        return {
            'noise_levels': noise_levels,
            'accuracies': accuracies,
            'avg_accuracy': float(np.mean(accuracies))
        }
    
    def _test_uniform_noise(self, model, X: np.ndarray, y: np.ndarray, 
                           noise_levels: List[float]) -> Dict[str, Any]:
        """Test robustness against uniform noise."""
        accuracies = []
        
        for noise_level in noise_levels:
            X_noisy = X + np.random.uniform(-noise_level, noise_level, X.shape)
            # Apply feature bounds
            bounds = self.config.get('constraints.feature_bounds', [-3.0, 3.0])
            X_noisy = np.clip(X_noisy, bounds[0], bounds[1])
            
            accuracy = model.score(X_noisy, y)
            accuracies.append(float(accuracy))
        
        return {
            'noise_levels': noise_levels,
            'accuracies': accuracies,
            'avg_accuracy': float(np.mean(accuracies))
        }
    
    def _test_feature_dropout(self, model, X: np.ndarray, y: np.ndarray, 
                             dropout_rates: List[float]) -> Dict[str, Any]:
        """Test robustness against feature dropout."""
        accuracies = []
        
        for dropout_rate in dropout_rates:
            X_dropout = X.copy()
            
            # Randomly set features to zero
            for i in range(len(X_dropout)):
                n_features = X_dropout.shape[1]
                n_dropout = int(n_features * dropout_rate)
                dropout_indices = np.random.choice(n_features, n_dropout, replace=False)
                X_dropout[i, dropout_indices] = 0
            
            accuracy = model.score(X_dropout, y)
            accuracies.append(float(accuracy))
        
        return {
            'dropout_rates': dropout_rates,
            'accuracies': accuracies,
            'avg_accuracy': float(np.mean(accuracies))
        }
    
    def _add_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to input data."""
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        
        # Apply feature bounds
        bounds = self.config.get('constraints.feature_bounds', [-3.0, 3.0])
        X_noisy = np.clip(X_noisy, bounds[0], bounds[1])
        
        return X_noisy
    
    def _rate_defense_effectiveness(self, score: float) -> str:
        """Rate defense effectiveness based on score."""
        if score >= 0.02:
            return "EXCELLENT"
        elif score >= 0.01:
            return "GOOD"
        elif score >= 0.005:
            return "MODERATE"
        elif score >= 0.0:
            return "WEAK"
        else:
            return "POOR"
    
    def _display_defense_results(self, results: Dict[str, Any], detailed: bool):
        """Display defense evaluation results."""
        
        # Model comparison results
        if 'model_comparison' in results:
            self._display_model_comparison(results['model_comparison'], detailed)
        
        # Robustness analysis results
        if 'robustness_analysis' in results:
            self._display_robustness_analysis(results['robustness_analysis'], detailed)
    
    def _display_model_comparison(self, comparison: Dict[str, Any], detailed: bool):
        """Display model comparison results."""
        self.output.header("📊 Model Performance Comparison")
        
        # Performance table
        if comparison['models']:
            self.output.subheader("Clean Performance")
            
            widths = [20, 15, 15, 15]
            self.output.table_row(['Model', 'Clean Accuracy', 'Robustness', 'Improvement'], widths)
            self.output.table_separator(widths)
            
            baseline_acc = None
            baseline_rob = None
            
            for model_name, model_results in comparison['models'].items():
                clean_acc = model_results['clean_accuracy'] * 100
                robustness = model_results['robustness_score'] * 100
                
                if model_name == 'baseline':
                    baseline_acc = clean_acc
                    baseline_rob = robustness
                    improvement = "-"
                else:
                    if baseline_acc is not None:
                        acc_improvement = clean_acc - baseline_acc
                        rob_improvement = robustness - baseline_rob
                        improvement = f"+{acc_improvement:.1f}%/+{rob_improvement:.1f}%"
                    else:
                        improvement = "-"
                
                acc_color = Colors.GREEN if clean_acc > 60 else Colors.YELLOW
                rob_color = Colors.GREEN if robustness > 60 else Colors.YELLOW
                
                self.output.table_row([
                    model_results['name'],
                    f"{acc_color}{clean_acc:.1f}%{Colors.RESET}",
                    f"{rob_color}{robustness:.1f}%{Colors.RESET}",
                    improvement
                ], widths)
        
        # Defense effectiveness
        if 'defense_effectiveness' in comparison:
            effectiveness = comparison['defense_effectiveness']
            
            self.output.subheader("Defense Effectiveness Analysis")
            
            clean_imp = effectiveness['clean_improvement'] * 100
            rob_imp = effectiveness['robustness_improvement'] * 100
            defense_score = effectiveness['defense_score']
            rating = effectiveness['rating']
            
            rating_color = Colors.GREEN if rating == "EXCELLENT" else Colors.YELLOW if rating in ["GOOD", "MODERATE"] else Colors.RED
            
            self.output.bullet(f"Clean Accuracy Improvement: {Colors.CYAN}{clean_imp:+.1f}%{Colors.RESET}")
            self.output.bullet(f"Robustness Improvement: {Colors.CYAN}{rob_imp:+.1f}%{Colors.RESET}")
            self.output.bullet(f"Defense Effectiveness Score: {Colors.BOLD}{defense_score:.4f}{Colors.RESET}")
            self.output.bullet(f"Rating: {rating_color}{rating}{Colors.RESET}")
        
        # Noise level performance if detailed
        if detailed and comparison['models']:
            self.output.subheader("Robustness Across Noise Levels")
            
            noise_levels = comparison['noise_levels']
            widths = [15] + [12] * len(noise_levels)
            
            header = ['Model'] + [f'ε={x}' for x in noise_levels]
            self.output.table_row(header, widths)
            self.output.table_separator(widths)
            
            for model_name, model_results in comparison['models'].items():
                row = [model_results['name']]
                
                for noise_level in noise_levels:
                    accuracy = model_results['noise_accuracies'][str(noise_level)] * 100
                    color = Colors.GREEN if accuracy > 50 else Colors.YELLOW if accuracy > 30 else Colors.RED
                    row.append(f"{color}{accuracy:.1f}%{Colors.RESET}")
                
                self.output.table_row(row, widths)
    
    def _display_robustness_analysis(self, robustness: Dict[str, Any], detailed: bool):
        """Display robustness analysis results."""
        self.output.header("🔬 Comprehensive Robustness Analysis")
        
        for model_name, model_results in robustness['models'].items():
            model_data = self.config.get_model_info(model_name)
            model_display_name = model_data.get('name', model_name.title())
            
            self.output.subheader(f"{model_display_name} Robustness Profile")
            
            overall = model_results['overall']
            avg_robustness = overall['avg_robustness'] * 100
            min_robustness = overall['min_robustness'] * 100
            std_robustness = overall['std_robustness'] * 100
            
            robustness_color = Colors.GREEN if avg_robustness > 50 else Colors.YELLOW if avg_robustness > 30 else Colors.RED
            
            self.output.bullet(f"Average Robustness: {robustness_color}{avg_robustness:.1f}%{Colors.RESET}")
            self.output.bullet(f"Minimum Robustness: {min_robustness:.1f}%")
            self.output.bullet(f"Robustness Std Dev: {std_robustness:.1f}%")
            
            if detailed:
                # Show performance for each test type
                for test_type in ['gaussian_noise', 'uniform_noise', 'feature_dropout']:
                    test_results = model_results[test_type]
                    avg_acc = test_results['avg_accuracy'] * 100
                    
                    test_color = Colors.GREEN if avg_acc > 50 else Colors.YELLOW if avg_acc > 30 else Colors.RED
                    
                    self.output.bullet(f"{test_type.replace('_', ' ').title()}: {test_color}{avg_acc:.1f}%{Colors.RESET}")
            
            self.output.blank_line()
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.output.success(f"Defense evaluation results saved to: {output_path}")
            
        except Exception as e:
            self.output.error(f"Failed to save results: {str(e)}")
    
    def _save_detailed_metrics(self, results: Dict[str, Any], output_path: str):
        """Save detailed metrics to file."""
        try:
            # Extract detailed metrics
            detailed_metrics = {
                'timestamp': str(np.datetime64('now')),
                'evaluation_summary': {},
                'detailed_results': results
            }
            
            # Add summary metrics
            if 'model_comparison' in results:
                comparison = results['model_comparison']
                if 'defense_effectiveness' in comparison:
                    effectiveness = comparison['defense_effectiveness']
                    detailed_metrics['evaluation_summary']['defense_effectiveness'] = effectiveness
            
            with open(output_path, 'w') as f:
                json.dump(detailed_metrics, f, indent=2)
            
            self.output.success(f"Detailed metrics saved to: {output_path}")
            
        except Exception as e:
            self.output.error(f"Failed to save detailed metrics: {str(e)}")
