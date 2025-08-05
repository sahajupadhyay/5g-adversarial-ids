"""
Constraint-Aware FGSM and PGD Attacks for 5G PFCP IDS
Implements adversarial attacks while respecting PFCP protocol constraints
"""

import numpy as np
import sys
import os
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append('src')
from attacks.pfcp_constraints import (
    project_to_pfcp_constraints, 
    validate_pfcp_constraints,
    get_feature_bounds,
    calculate_constraint_violations
)
from attacks.attack_utils import evaluate_attack_success, print_attack_summary

class ConstraintAwareFGSM:
    """
    Fast Gradient Sign Method with PFCP protocol constraints
    """
    
    def __init__(self, model, epsilon=0.1, constraints_enabled=True):
        """
        Initialize FGSM attack
        
        Args:
            model: Target classifier
            epsilon: Attack strength (L∞ bound)
            constraints_enabled: Whether to enforce PFCP constraints
        """
        self.model = model
        self.epsilon = epsilon
        self.constraints_enabled = constraints_enabled
        self.min_bounds, self.max_bounds = get_feature_bounds()
        
    def _compute_gradient(self, X, y):
        """
        Compute gradient of loss w.r.t. input features for Random Forest
        Uses gradient approximation via decision tree surrogate
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels
            
        Returns:
            numpy array: Approximate gradients
        """
        # For Random Forest, we use a gradient approximation method
        # based on feature importance and prediction confidence
        
        n_samples, n_features = X.shape
        gradients = np.zeros_like(X)
        
        # Get model predictions and probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
        else:
            # For models without predict_proba, use distance-based approximation
            predictions = self.model.predict(X)
            probabilities = np.eye(len(np.unique(y)))[predictions]
        
        # Feature importance-based gradient approximation
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        else:
            feature_importance = np.ones(n_features) / n_features
        
        for i in range(n_samples):
            true_label = y[i]
            pred_label = predictions[i]
            
            if hasattr(self.model, 'predict_proba'):
                confidence = probabilities[i][pred_label]
                true_confidence = probabilities[i][true_label] if true_label < len(probabilities[i]) else 0
            else:
                # Simple confidence approximation
                confidence = 1.0 if pred_label == true_label else 0.5
                true_confidence = confidence
            
            # Gradient approximation: direction that increases loss
            if pred_label == true_label:
                # Correctly classified: gradient reduces confidence
                direction = -1.0 * feature_importance
            else:
                # Misclassified: gradient increases confidence in wrong prediction
                direction = 1.0 * feature_importance
            
            # Scale by confidence difference
            scale = abs(confidence - true_confidence + 0.1)  # Add small constant
            gradients[i] = direction * scale
            
        return gradients
    
    def _finite_difference_gradient(self, X, y, delta=1e-4):
        """
        Compute gradient using finite differences (more accurate but slower)
        
        Args:
            X: Input features
            y: Target labels  
            delta: Finite difference step size
            
        Returns:
            numpy array: Gradients
        """
        n_samples, n_features = X.shape
        gradients = np.zeros_like(X)
        
        # Get baseline predictions
        if hasattr(self.model, 'predict_proba'):
            baseline_probs = self.model.predict_proba(X)
        else:
            baseline_preds = self.model.predict(X)
            n_classes = len(np.unique(y))
            baseline_probs = np.eye(n_classes)[baseline_preds]
        
        # Compute finite differences for each feature
        for i in range(n_features):
            # Perturb feature i
            X_pert_pos = X.copy()
            X_pert_neg = X.copy()
            X_pert_pos[:, i] += delta
            X_pert_neg[:, i] -= delta
            
            # Get perturbed predictions
            if hasattr(self.model, 'predict_proba'):
                probs_pos = self.model.predict_proba(X_pert_pos)
                probs_neg = self.model.predict_proba(X_pert_neg)
            else:
                preds_pos = self.model.predict(X_pert_pos)
                preds_neg = self.model.predict(X_pert_neg)
                n_classes = len(np.unique(y))
                probs_pos = np.eye(n_classes)[preds_pos]
                probs_neg = np.eye(n_classes)[preds_neg]
            
            # Compute gradient for each sample
            for j in range(n_samples):
                true_label = y[j]
                if true_label < baseline_probs.shape[1]:
                    # Loss gradient: decrease true class probability
                    loss_pos = -np.log(max(probs_pos[j, true_label], 1e-8))
                    loss_neg = -np.log(max(probs_neg[j, true_label], 1e-8))
                    gradients[j, i] = (loss_pos - loss_neg) / (2 * delta)
                else:
                    # Fallback for label issues
                    gradients[j, i] = 0.0
        
        return gradients
    
    def generate_adversarial_samples(self, X, y, method='approximation'):
        """
        Generate adversarial samples using FGSM
        
        Args:
            X: Clean input samples (n_samples, n_features)
            y: True labels
            method: 'approximation' or 'finite_difference'
            
        Returns:
            tuple: (adversarial_samples, attack_info)
        """
        print(f"🔄 Generating FGSM adversarial samples (ε={self.epsilon})...")
        
        # Compute gradients
        if method == 'finite_difference':
            gradients = self._finite_difference_gradient(X, y)
            print("   Using finite difference gradients")
        else:
            gradients = self._compute_gradient(X, y)
            print("   Using importance-based gradient approximation")
        
        # FGSM perturbation: x_adv = x + epsilon * sign(gradient)
        perturbations = self.epsilon * np.sign(gradients)
        X_adversarial = X + perturbations
        
        print(f"   Generated perturbations: L∞={np.max(np.abs(perturbations)):.3f}")
        
        # Apply constraints if enabled
        constraint_violations_before = 0
        if self.constraints_enabled:
            constraint_violations_before = 0
            for i in range(X_adversarial.shape[0]):
                is_valid, _ = validate_pfcp_constraints(X_adversarial[i])
                if not is_valid:
                    constraint_violations_before += 1
            
            print(f"   Constraint violations before projection: {constraint_violations_before}")
            
            # Project to constraint manifold
            for i in range(X_adversarial.shape[0]):
                X_adversarial[i] = project_to_pfcp_constraints(X_adversarial[i])
            
            # Verify constraint compliance after projection
            constraint_violations_after = 0
            for i in range(X_adversarial.shape[0]):
                is_valid, _ = validate_pfcp_constraints(X_adversarial[i])
                if not is_valid:
                    constraint_violations_after += 1
            
            print(f"   Constraint violations after projection: {constraint_violations_after}")
        
        # Calculate final perturbation statistics
        final_perturbations = X_adversarial - X
        attack_info = {
            'epsilon': self.epsilon,
            'method': method,
            'constraints_enabled': self.constraints_enabled,
            'perturbation_stats': {
                'l_inf_max': np.max(np.abs(final_perturbations)),
                'l_inf_mean': np.mean(np.abs(final_perturbations)),
                'l_2_mean': np.mean(np.linalg.norm(final_perturbations, axis=1))
            },
            'constraint_violations_before': constraint_violations_before,
            'constraint_violations_after': constraint_violations_after if self.constraints_enabled else 0
        }
        
        print(f"✅ FGSM attack complete: {X_adversarial.shape[0]} samples generated")
        
        return X_adversarial, attack_info

class ConstraintAwarePGD:
    """
    Projected Gradient Descent with PFCP protocol constraints
    """
    
    def __init__(self, model, epsilon=0.1, alpha=None, num_steps=10, constraints_enabled=True):
        """
        Initialize PGD attack
        
        Args:
            model: Target classifier
            epsilon: Attack strength (L∞ bound)
            alpha: Step size (default: epsilon/steps)
            num_steps: Number of PGD iterations
            constraints_enabled: Whether to enforce PFCP constraints
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / num_steps
        self.num_steps = num_steps
        self.constraints_enabled = constraints_enabled
        self.min_bounds, self.max_bounds = get_feature_bounds()
        
        # Initialize FGSM for gradient computation
        self.fgsm = ConstraintAwareFGSM(model, self.alpha, constraints_enabled=False)
    
    def generate_adversarial_samples(self, X, y, method='approximation'):
        """
        Generate adversarial samples using PGD
        
        Args:
            X: Clean input samples (n_samples, n_features)
            y: True labels
            method: Gradient computation method
            
        Returns:
            tuple: (adversarial_samples, attack_info)
        """
        print(f"🔄 Generating PGD adversarial samples (ε={self.epsilon}, steps={self.num_steps})...")
        
        # Initialize adversarial samples with random noise
        X_adversarial = X + np.random.uniform(-self.epsilon, self.epsilon, X.shape)
        
        # Clip to valid bounds initially
        X_adversarial = np.clip(X_adversarial, self.min_bounds, self.max_bounds)
        
        successful_attacks = np.zeros(X.shape[0], dtype=bool)
        iteration_info = []
        
        for step in range(self.num_steps):
            # Compute gradients using FGSM
            if method == 'finite_difference':
                gradients = self.fgsm._finite_difference_gradient(X_adversarial, y)
            else:
                gradients = self.fgsm._compute_gradient(X_adversarial, y)
            
            # Take PGD step
            X_adversarial = X_adversarial + self.alpha * np.sign(gradients)
            
            # Project to L∞ ball around original samples
            perturbations = X_adversarial - X
            perturbations = np.clip(perturbations, -self.epsilon, self.epsilon)
            X_adversarial = X + perturbations
            
            # Apply constraints if enabled
            if self.constraints_enabled:
                for i in range(X_adversarial.shape[0]):
                    X_adversarial[i] = project_to_pfcp_constraints(X_adversarial[i])
            
            # Check for early stopping (successful attacks)
            if hasattr(self.model, 'predict'):
                current_preds = self.model.predict(X_adversarial)
                newly_successful = (current_preds != y) & (~successful_attacks)
                successful_attacks |= newly_successful
                
                success_rate = np.mean(successful_attacks)
                iteration_info.append({
                    'step': step + 1,
                    'success_rate': success_rate,
                    'l_inf_max': np.max(np.abs(X_adversarial - X))
                })
                
                if step % 3 == 0 or step == self.num_steps - 1:
                    print(f"   Step {step+1}/{self.num_steps}: {success_rate:.1%} success rate")
        
        # Final constraint validation
        constraint_violations = 0
        if self.constraints_enabled:
            for i in range(X_adversarial.shape[0]):
                is_valid, _ = validate_pfcp_constraints(X_adversarial[i])
                if not is_valid:
                    constraint_violations += 1
        
        # Calculate final statistics
        final_perturbations = X_adversarial - X
        attack_info = {
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_steps': self.num_steps,
            'method': method,
            'constraints_enabled': self.constraints_enabled,
            'perturbation_stats': {
                'l_inf_max': np.max(np.abs(final_perturbations)),
                'l_inf_mean': np.mean(np.abs(final_perturbations)),
                'l_2_mean': np.mean(np.linalg.norm(final_perturbations, axis=1))
            },
            'constraint_violations': constraint_violations,
            'iteration_info': iteration_info,
            'final_success_rate': np.mean(successful_attacks)
        }
        
        print(f"✅ PGD attack complete: {np.mean(successful_attacks):.1%} success rate")
        
        return X_adversarial, attack_info

def run_attack_evaluation(model, X_test, y_test, epsilon_values=[0.01, 0.1, 0.3]):
    """
    Run comprehensive attack evaluation with multiple epsilon values
    
    Args:
        model: Target classifier
        X_test: Test samples
        y_test: Test labels
        epsilon_values: List of epsilon values to test
        
    Returns:
        dict: Complete evaluation results
    """
    print("🚀 COMPREHENSIVE ATTACK EVALUATION")
    print("="*50)
    
    all_results = []
    
    for epsilon in epsilon_values:
        print(f"\n🎯 Testing epsilon = {epsilon}")
        print("-" * 30)
        
        # FGSM Attack
        fgsm = ConstraintAwareFGSM(model, epsilon=epsilon, constraints_enabled=True)
        X_fgsm, fgsm_info = fgsm.generate_adversarial_samples(X_test, y_test)
        
        fgsm_results = evaluate_attack_success(
            model, X_test, X_fgsm, y_test, 
            attack_name=f"FGSM (ε={epsilon})"
        )
        fgsm_results['attack_info'] = fgsm_info
        all_results.append(fgsm_results)
        print_attack_summary(fgsm_results)
        
        # PGD Attack
        pgd = ConstraintAwarePGD(model, epsilon=epsilon, num_steps=10, constraints_enabled=True)
        X_pgd, pgd_info = pgd.generate_adversarial_samples(X_test, y_test)
        
        pgd_results = evaluate_attack_success(
            model, X_test, X_pgd, y_test,
            attack_name=f"PGD (ε={epsilon})"
        )
        pgd_results['attack_info'] = pgd_info
        all_results.append(pgd_results)
        print_attack_summary(pgd_results)
    
    # Summary of best results
    best_result = max(all_results, key=lambda x: x['evasion_rate'])
    print(f"\n🏆 BEST ATTACK: {best_result['attack_name']}")
    print(f"   Evasion Rate: {best_result['evasion_rate']:.1%}")
    print(f"   Constraint Violations: {best_result['attack_info'].get('constraint_violations_after', 0)}")
    
    # Success criteria check
    target_achieved = best_result['evasion_rate'] >= 0.8
    print(f"\n🎯 TARGET ≥80% EVASION: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    return {
        'all_results': all_results,
        'best_result': best_result,
        'target_achieved': target_achieved,
        'summary': {
            'total_attacks': len(all_results),
            'best_evasion_rate': best_result['evasion_rate'],
            'epsilon_values_tested': epsilon_values
        }
    }

if __name__ == "__main__":
    # Test the attack implementations
    print("🔧 CONSTRAINT-AWARE ATTACK ENGINE TEST")
    print("="*50)
    
    try:
        # Load baseline artifacts
        from attacks.attack_utils import load_baseline_artifacts
        model, transformers, scaler, X_test, y_test = load_baseline_artifacts()
        
        # Run attack evaluation with smaller test set for demonstration
        test_size = min(100, len(X_test))  # Use first 100 samples for quick test
        X_test_small = X_test[:test_size]
        y_test_small = y_test[:test_size]
        
        print(f"Testing on {test_size} samples...")
        
        # Run comprehensive evaluation
        evaluation_results = run_attack_evaluation(
            model, X_test_small, y_test_small, 
            epsilon_values=[0.1, 0.3]  # Test key epsilon values
        )
        
        print("\n✅ ATTACK ENGINE TEST COMPLETE")
        print(f"Best evasion rate: {evaluation_results['best_result']['evasion_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
