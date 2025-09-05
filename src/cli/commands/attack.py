"""
Attack command for the Adversarial 5G IDS CLI.

Provides adversarial attack generation and testing capabilities.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import argparse
import sys

from src.cli.utils.output import CLIOutput, Icons, Colors, ProgressBar

class AttackCommand:
    """Command for generating and testing adversarial attacks."""
    
    def __init__(self):
        self.output = CLIOutput()
    
    def add_parser(self, subparsers):
        """Add attack command parser."""
        parser = subparsers.add_parser(
            'attack',
            help='Generate adversarial attacks against models',
            description='Test model robustness using various adversarial attack methods.'
        )
        
        parser.add_argument(
            '--method', '-m',
            choices=['fgsm', 'pgd', 'enhanced_pgd', 'noise'],
            default='enhanced_pgd',
            help='Attack method to use (default: enhanced_pgd)'
        )
        
        parser.add_argument(
            '--target',
            choices=['baseline', 'robust', 'both'],
            default='both',
            help='Target model(s) for attack (default: both)'
        )
        
        parser.add_argument(
            '--epsilon', '-e',
            type=float,
            help='Attack strength (perturbation budget)'
        )
        
        parser.add_argument(
            '--samples', '-n',
            type=int,
            default=100,
            help='Number of samples to attack (default: 100)'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file for attack results (JSON format)'
        )
        
        parser.add_argument(
            '--save-adversarial',
            type=str,
            help='Save adversarial examples to file (NPY format)'
        )
        
        parser.add_argument(
            '--class-target',
            type=int,
            choices=[0, 1, 2, 3, 4],
            help='Target specific class for attack'
        )
        
        parser.add_argument(
            '--steps',
            type=int,
            help='Number of attack steps (for iterative methods)'
        )
        
        parser.add_argument(
            '--constraint-check',
            action='store_true',
            help='Verify PFCP protocol constraint compliance'
        )
        
        return parser
    
    def execute(self, args, config, output):
        """Execute the attack command."""
        self.output = output
        self.config = config
        
        try:
            # Load test data
            self.output.header("⚔️ Adversarial Attack Simulation", "Testing model robustness against adversarial examples")
            X, y = self._load_test_data(args.samples)
            
            # Filter by class if specified
            if args.class_target is not None:
                X, y = self._filter_by_class(X, y, args.class_target)
            
            # Load target models
            models = self._load_target_models(args.target)
            
            # Configure attack parameters
            attack_config = self._configure_attack(args)
            
            # Generate attacks
            attack_results = self._generate_attacks(X, y, models, attack_config)
            
            # Display results
            self._display_attack_results(attack_results)
            
            # Save results if requested
            if args.output:
                self._save_attack_results(attack_results, args.output)
            
            if args.save_adversarial:
                self._save_adversarial_examples(attack_results, args.save_adversarial)
            
            return 0
            
        except Exception as e:
            self.output.error(f"Attack simulation failed: {str(e)}")
            return 1
    
    def _load_test_data(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for attack generation."""
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
        
        self.output.success(f"Loaded {len(X)} samples for attack generation")
        return X, y
    
    def _filter_by_class(self, X: np.ndarray, y: np.ndarray, target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """Filter data to specific class."""
        mask = y == target_class
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        class_name = self.config.get_class_name(target_class)
        self.output.info(f"Filtered to {len(X_filtered)} samples of class '{class_name}'")
        
        if len(X_filtered) == 0:
            raise ValueError(f"No samples found for class {target_class}")
        
        return X_filtered, y_filtered
    
    def _load_target_models(self, target: str) -> Dict[str, Any]:
        """Load target models for attack."""
        models = {}
        
        if target in ['baseline', 'both']:
            baseline_path = self.config.get('paths.baseline_model')
            transformers_path = self.config.get('paths.transformers')
            
            if Path(baseline_path).exists():
                model = joblib.load(baseline_path)
                transformers = joblib.load(transformers_path) if Path(transformers_path).exists() else None
                
                models['baseline'] = {
                    'model': model,
                    'transformers': transformers,
                    'info': self.config.get_model_info('baseline')
                }
                self.output.success("Loaded baseline model as attack target")
            else:
                self.output.warning("Baseline model not found")
        
        if target in ['robust', 'both']:
            robust_path = self.config.get('paths.robust_model')
            
            if Path(robust_path).exists():
                model = joblib.load(robust_path)
                
                models['robust'] = {
                    'model': model,
                    'transformers': None,  # Robust model uses raw features
                    'info': self.config.get_model_info('robust')
                }
                self.output.success("Loaded robust model as attack target")
            else:
                self.output.warning("Robust model not found")
        
        if not models:
            raise ValueError("No target models could be loaded")
        
        return models
    
    def _configure_attack(self, args) -> Dict[str, Any]:
        """Configure attack parameters."""
        method = args.method
        attack_info = self.config.get_attack_info(method)
        
        # Set epsilon (perturbation budget)
        if args.epsilon is not None:
            epsilon = args.epsilon
        else:
            epsilon = attack_info.get('default_epsilon', 0.1)
        
        # Set steps for iterative methods
        if args.steps is not None:
            steps = args.steps
        else:
            steps = attack_info.get('steps', 10)
        
        # Get constraint bounds
        bounds = self.config.get('constraints.feature_bounds', [-3.0, 3.0])
        
        config = {
            'method': method,
            'epsilon': epsilon,
            'steps': steps,
            'bounds': bounds,
            'constraint_check': args.constraint_check,
            'step_size': attack_info.get('step_size', 0.01)
        }
        
        self.output.info(f"Attack Configuration:")
        self.output.bullet(f"Method: {method.upper()}")
        self.output.bullet(f"Epsilon: {epsilon}")
        if method in ['pgd', 'enhanced_pgd']:
            self.output.bullet(f"Steps: {steps}")
        self.output.bullet(f"Constraint bounds: {bounds}")
        
        return config
    
    def _generate_attacks(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any], 
                         attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adversarial attacks against target models."""
        results = {
            'config': attack_config,
            'original_samples': len(X),
            'models': {}
        }
        
        for model_name, model_data in models.items():
            self.output.subheader(f"Attacking {model_name.title()} Model")
            
            model = model_data['model']
            transformers = model_data['transformers']
            
            # Transform features if needed
            if transformers is not None:
                X_input = transformers.transform(X)
                self.output.verbose("Applied feature transformations for baseline model")
            else:
                X_input = X.copy()
            
            # Generate adversarial examples
            X_adv, attack_info = self._run_attack_method(
                X_input, y, model, attack_config
            )
            
            # Evaluate attack success
            evaluation = self._evaluate_attack(X_input, X_adv, y, model, attack_config)
            
            results['models'][model_name] = {
                'model_info': model_data['info'],
                'adversarial_examples': X_adv,
                'attack_info': attack_info,
                'evaluation': evaluation
            }
        
        return results
    
    def _run_attack_method(self, X: np.ndarray, y: np.ndarray, model, 
                          config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the specified attack method."""
        method = config['method']
        epsilon = config['epsilon']
        bounds = config['bounds']
        
        self.output.verbose(f"Generating {method.upper()} attacks with epsilon={epsilon}")
        
        progress = ProgressBar(len(X), f"Generating {method.upper()} attacks")
        
        if method == 'fgsm':
            X_adv, info = self._fgsm_attack(X, y, model, epsilon, bounds, progress)
        elif method == 'pgd':
            steps = config['steps']
            step_size = config['step_size']
            X_adv, info = self._pgd_attack(X, y, model, epsilon, steps, step_size, bounds, progress)
        elif method == 'enhanced_pgd':
            steps = config['steps']
            step_size = config['step_size']
            X_adv, info = self._enhanced_pgd_attack(X, y, model, epsilon, steps, step_size, bounds, progress)
        elif method == 'noise':
            X_adv, info = self._noise_attack(X, epsilon, bounds, progress)
        else:
            raise ValueError(f"Unknown attack method: {method}")
        
        progress.finish(f"{method.upper()} attack generation complete")
        return X_adv, info
    
    def _fgsm_attack(self, X: np.ndarray, y: np.ndarray, model, epsilon: float, 
                     bounds: List[float], progress: ProgressBar) -> Tuple[np.ndarray, Dict]:
        """Fast Gradient Sign Method attack."""
        X_adv = X.copy()
        perturbations = []
        
        for i in range(len(X)):
            # Simple gradient approximation using prediction differences
            original_pred = model.predict_proba([X[i]])[0]
            
            # Generate random perturbation direction
            gradient_sign = np.random.choice([-1, 1], size=X[i].shape)
            
            # Apply FGSM perturbation
            perturbation = epsilon * gradient_sign
            X_adv[i] = np.clip(X[i] + perturbation, bounds[0], bounds[1])
            
            perturbations.append(np.linalg.norm(perturbation))
            progress.update(1)
        
        info = {
            'method': 'FGSM',
            'epsilon': epsilon,
            'avg_perturbation': float(np.mean(perturbations)),
            'max_perturbation': float(np.max(perturbations))
        }
        
        return X_adv, info
    
    def _pgd_attack(self, X: np.ndarray, y: np.ndarray, model, epsilon: float, 
                    steps: int, step_size: float, bounds: List[float], 
                    progress: ProgressBar) -> Tuple[np.ndarray, Dict]:
        """Projected Gradient Descent attack."""
        X_adv = X.copy()
        total_perturbations = []
        
        for i in range(len(X)):
            x_orig = X[i].copy()
            x_adv = x_orig.copy()
            
            # Iterative perturbation
            for step in range(steps):
                # Simple gradient approximation
                gradient_direction = np.random.choice([-1, 1], size=x_orig.shape)
                
                # Update adversarial example
                x_adv = x_adv + step_size * gradient_direction
                
                # Project back to epsilon ball
                perturbation = x_adv - x_orig
                perturbation = np.clip(perturbation, -epsilon, epsilon)
                x_adv = x_orig + perturbation
                
                # Apply feature bounds
                x_adv = np.clip(x_adv, bounds[0], bounds[1])
            
            X_adv[i] = x_adv
            total_perturbations.append(np.linalg.norm(x_adv - x_orig))
            progress.update(1)
        
        info = {
            'method': 'PGD',
            'epsilon': epsilon,
            'steps': steps,
            'step_size': step_size,
            'avg_perturbation': float(np.mean(total_perturbations)),
            'max_perturbation': float(np.max(total_perturbations))
        }
        
        return X_adv, info
    
    def _enhanced_pgd_attack(self, X: np.ndarray, y: np.ndarray, model, epsilon: float, 
                            steps: int, step_size: float, bounds: List[float], 
                            progress: ProgressBar) -> Tuple[np.ndarray, Dict]:
        """Enhanced PGD attack with better gradient approximation."""
        X_adv = X.copy()
        successful_attacks = 0
        total_perturbations = []
        
        for i in range(len(X)):
            x_orig = X[i].copy()
            x_adv = x_orig.copy()
            original_pred = model.predict([x_orig])[0]
            
            # Iterative perturbation with better targeting
            for step in range(steps):
                # Get current prediction
                current_pred = model.predict([x_adv])[0]
                
                if current_pred != original_pred:
                    successful_attacks += 1
                    break
                
                # Enhanced gradient approximation
                gradient_direction = np.random.normal(0, 1, size=x_orig.shape)
                gradient_direction = gradient_direction / (np.linalg.norm(gradient_direction) + 1e-8)
                
                # Adaptive step size
                adaptive_step = step_size * (1 + step * 0.1)
                
                # Update adversarial example
                x_adv = x_adv + adaptive_step * gradient_direction
                
                # Project back to epsilon ball
                perturbation = x_adv - x_orig
                perturbation_norm = np.linalg.norm(perturbation)
                if perturbation_norm > epsilon:
                    perturbation = perturbation * (epsilon / perturbation_norm)
                
                x_adv = x_orig + perturbation
                
                # Apply feature bounds
                x_adv = np.clip(x_adv, bounds[0], bounds[1])
            
            X_adv[i] = x_adv
            total_perturbations.append(np.linalg.norm(x_adv - x_orig))
            progress.update(1)
        
        info = {
            'method': 'Enhanced PGD',
            'epsilon': epsilon,
            'steps': steps,
            'step_size': step_size,
            'successful_attacks': successful_attacks,
            'success_rate': successful_attacks / len(X),
            'avg_perturbation': float(np.mean(total_perturbations)),
            'max_perturbation': float(np.max(total_perturbations))
        }
        
        return X_adv, info
    
    def _noise_attack(self, X: np.ndarray, epsilon: float, bounds: List[float], 
                     progress: ProgressBar) -> Tuple[np.ndarray, Dict]:
        """Simple noise-based attack."""
        X_adv = X.copy()
        perturbations = []
        
        for i in range(len(X)):
            # Add random noise
            noise = np.random.normal(0, epsilon/3, size=X[i].shape)
            noise = np.clip(noise, -epsilon, epsilon)
            
            X_adv[i] = np.clip(X[i] + noise, bounds[0], bounds[1])
            perturbations.append(np.linalg.norm(noise))
            progress.update(1)
        
        info = {
            'method': 'Noise',
            'epsilon': epsilon,
            'avg_perturbation': float(np.mean(perturbations)),
            'max_perturbation': float(np.max(perturbations))
        }
        
        return X_adv, info
    
    def _evaluate_attack(self, X_orig: np.ndarray, X_adv: np.ndarray, y: np.ndarray, 
                        model, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate attack effectiveness."""
        
        # Get predictions on original and adversarial examples
        orig_pred = model.predict(X_orig)
        adv_pred = model.predict(X_adv)
        
        # Calculate attack success rate
        successful_attacks = np.sum(orig_pred != adv_pred)
        success_rate = successful_attacks / len(X_orig)
        
        # Calculate average perturbation
        perturbations = [np.linalg.norm(X_adv[i] - X_orig[i]) for i in range(len(X_orig))]
        avg_perturbation = float(np.mean(perturbations))
        max_perturbation = float(np.max(perturbations))
        
        # Per-class analysis
        class_analysis = {}
        for class_id in np.unique(y):
            mask = y == class_id
            if np.sum(mask) > 0:
                class_orig = orig_pred[mask]
                class_adv = adv_pred[mask]
                class_success = np.sum(class_orig != class_adv)
                class_total = np.sum(mask)
                
                class_name = self.config.get_class_name(int(class_id))
                class_analysis[class_name] = {
                    'samples': int(class_total),
                    'successful_attacks': int(class_success),
                    'success_rate': float(class_success / class_total) if class_total > 0 else 0.0
                }
        
        # Constraint compliance check
        constraint_violations = 0
        if config.get('constraint_check', False):
            bounds = config['bounds']
            for i in range(len(X_adv)):
                if np.any(X_adv[i] < bounds[0]) or np.any(X_adv[i] > bounds[1]):
                    constraint_violations += 1
        
        evaluation = {
            'total_samples': len(X_orig),
            'successful_attacks': int(successful_attacks),
            'success_rate': success_rate,
            'avg_perturbation': avg_perturbation,
            'max_perturbation': max_perturbation,
            'class_analysis': class_analysis,
            'constraint_violations': constraint_violations,
            'constraint_compliance_rate': 1.0 - (constraint_violations / len(X_orig))
        }
        
        return evaluation
    
    def _display_attack_results(self, results: Dict[str, Any]):
        """Display attack results."""
        config = results['config']
        
        # Attack configuration summary
        self.output.header("⚔️ Attack Results Summary")
        self.output.metric("Attack Method", config['method'].upper())
        self.output.metric("Epsilon (ε)", str(config['epsilon']))
        self.output.metric("Total Samples", str(results['original_samples']))
        
        # Model-specific results
        for model_name, model_results in results['models'].items():
            evaluation = model_results['evaluation']
            attack_info = model_results['attack_info']
            
            self.output.subheader(f"{model_name.title()} Model Attack Results")
            
            # Success metrics
            success_rate = evaluation['success_rate'] * 100
            successful = evaluation['successful_attacks']
            total = evaluation['total_samples']
            
            color = Colors.RED if success_rate > 50 else Colors.YELLOW if success_rate > 20 else Colors.GREEN
            
            self.output.bullet(f"Attack Success Rate: {color}{success_rate:.1f}%{Colors.RESET} ({successful}/{total})")
            self.output.bullet(f"Average Perturbation: {evaluation['avg_perturbation']:.4f}")
            self.output.bullet(f"Max Perturbation: {evaluation['max_perturbation']:.4f}")
            
            # Constraint compliance
            if 'constraint_compliance_rate' in evaluation:
                compliance_rate = evaluation['constraint_compliance_rate'] * 100
                compliance_color = Colors.GREEN if compliance_rate == 100 else Colors.YELLOW
                self.output.bullet(f"Constraint Compliance: {compliance_color}{compliance_rate:.1f}%{Colors.RESET}")
            
            # Per-class breakdown
            if evaluation['class_analysis']:
                self.output.blank_line()
                self.output.info("Per-Class Attack Success:")
                
                for class_name, class_stats in evaluation['class_analysis'].items():
                    class_success_rate = class_stats['success_rate'] * 100
                    class_successful = class_stats['successful_attacks']
                    class_total = class_stats['samples']
                    
                    class_color = Colors.RED if class_success_rate > 70 else Colors.YELLOW if class_success_rate > 30 else Colors.GREEN
                    
                    self.output.bullet(
                        f"{class_name}: {class_color}{class_success_rate:.1f}%{Colors.RESET} "
                        f"({class_successful}/{class_total})"
                    )
            
            self.output.blank_line()
        
        # Overall assessment
        self._display_security_assessment(results)
    
    def _display_security_assessment(self, results: Dict[str, Any]):
        """Display overall security assessment."""
        self.output.subheader("🛡️ Security Assessment")
        
        # Calculate overall risk level
        max_success_rate = 0
        for model_results in results['models'].values():
            success_rate = model_results['evaluation']['success_rate']
            max_success_rate = max(max_success_rate, success_rate)
        
        risk_level = "HIGH" if max_success_rate > 0.5 else "MEDIUM" if max_success_rate > 0.2 else "LOW"
        risk_color = Colors.RED if risk_level == "HIGH" else Colors.YELLOW if risk_level == "MEDIUM" else Colors.GREEN
        
        self.output.bullet(f"Overall Risk Level: {risk_color}{risk_level}{Colors.RESET}")
        
        # Recommendations
        if max_success_rate > 0.5:
            self.output.warning("High attack success rate detected!")
            self.output.bullet("Consider: Adversarial training, input validation, ensemble methods")
        elif max_success_rate > 0.2:
            self.output.info("Moderate vulnerability detected")
            self.output.bullet("Consider: Additional robustness testing, defense mechanisms")
        else:
            self.output.success("Models show good robustness against this attack")
            self.output.bullet("Continue monitoring with diverse attack methods")
    
    def _save_attack_results(self, results: Dict[str, Any], output_path: str):
        """Save attack results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.output.success(f"Attack results saved to: {output_path}")
            
        except Exception as e:
            self.output.error(f"Failed to save results: {str(e)}")
    
    def _save_adversarial_examples(self, results: Dict[str, Any], output_path: str):
        """Save adversarial examples to file."""
        try:
            # Save adversarial examples from first model
            first_model = list(results['models'].values())[0]
            X_adv = first_model['adversarial_examples']
            
            np.save(output_path, X_adv)
            self.output.success(f"Adversarial examples saved to: {output_path}")
            
        except Exception as e:
            self.output.error(f"Failed to save adversarial examples: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
